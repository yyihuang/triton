import argparse

import torch
import triton
import triton.language as tl
import triton.tools.experimental_descriptor
import triton.profiler as proton
from contextlib import contextmanager

from typing import Optional

NUM_SMS = (
    torch.cuda.get_device_properties("cuda").multi_processor_count // 2
)  # test 50% of the sms


def matmul_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "GROUP_SIZE_M": 8,
            },
            num_stages=s,
            num_warps=w,
        )
        for BM in [128]
        for BN in [128, 256]
        for BK in [64, 128]
        for s in ([3, 4])
        for w in [4, 8]
    ]


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.autotune(
    configs=matmul_get_configs(),
    key=["M", "N", "K"],
)
@triton.jit(launch_metadata=_matmul_launch_metadata)
def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    # NOTE: There is currently a bug in blackwell pipelining that means it can't handle a value being
    # used in both the prologue and epilogue, so we duplicate the counters as a work-around.
    tile_id_c = start_pid - NUM_SMS

    offs_k_for_mask = tl.arange(0, BLOCK_SIZE_K)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    #
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
        pid_m, pid_n = _compute_pid(
            tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        start_m = pid_m * BLOCK_SIZE_M
        start_n = pid_n * BLOCK_SIZE_N
        offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
        offs_am = tl.where(offs_am < M, offs_am, 0)
        offs_bn = tl.where(offs_bn < N, offs_bn, 0)
        offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            offs_k = ki * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + (
                offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_ptr + (
                offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
            )

            a = tl.load(
                a_ptrs, mask=offs_k_for_mask[None, :] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            b = tl.load(
                b_ptrs, mask=offs_k_for_mask[:, None] < K - ki * BLOCK_SIZE_K, other=0.0
            )
            accumulator = tl.dot(a, b, accumulator)

        tile_id_c += NUM_SMS
        pid_m, pid_n = _compute_pid(
            tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_SMS
        )
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        if c_ptr.dtype.element_ty == tl.float8e4nv:
            c = accumulator.to(tl.float8e4nv)
        else:
            c = accumulator.to(tl.float16)

        c = alpha * c + beta * tl.load(c_ptrs, mask=c_mask)
        # if beta != 0.0:
        #     c = alpha * c + beta * tl.load(c_ptrs, mask=c_mask)
        # else:
        #     c = (alpha * c).to(c.dtype)

        tl.store(c_ptrs, c, mask=c_mask)


def matmul_persistent(a, b, c = None, alpha=1.0, beta=0.0):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = a.shape
    K, N = b.shape
    dtype = a.dtype
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=dtype) if c is None else c
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        min(
            NUM_SMS,
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        ),
    )

    matmul_kernel_persistent[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        alpha=alpha,
        beta=beta,
        NUM_SMS=NUM_SMS,
    )
    return c


def torch_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    bytes_per_elem = a.element_size()
    flops_str = f"flops{bytes_per_elem * 8}"
    with proton.scope(
        f"torch [M={M}, N={N}, K={K}]",
        {"bytes": bytes_per_elem * (M * K + N * K + M * N), flops_str: 2.0 * M * N * K},
    ):
        c = torch.matmul(a, b.T)
    return c


def torch_gemm(a, b, c, alpha=1.0, beta=0.0):
    # Perform matrix multiplication
    x = torch.matmul(a, b.T)
    # Scale the result by alpha
    c = beta * c + alpha * x
    return c


def validate(M, N, K, alpha, beta, dtype: torch.dtype = torch.float16):
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)
    b = b.T.contiguous()

    if beta == 0.0:
        torch_result_matmul = torch_matmul(a, b)
        persistent_result_matmul = matmul_persistent(a, b.T, alpha=alpha, beta=beta)

        if torch_result_matmul is not None:
            persistent_vs_torch_matmul = (
                "✅"
                if torch.allclose(
                    persistent_result_matmul.to(torch.float16),
                    torch_result_matmul.to(torch.float16),
                    atol=1.0,
                )
                else "❌"
            )
        print(f"persistent_vs_torch_matmul: {persistent_vs_torch_matmul}")
    else:
        c = torch.randn((M, N), device="cuda", dtype=dtype)
        torch_result_gemm = torch_gemm(a, b, c, alpha=alpha, beta=beta)
        persistent_result_gemm = matmul_persistent(a, b.T, c = c, alpha=alpha, beta=beta)

        if torch_result_gemm is not None:
            persistent_vs_torch_gemm = (
                "✅"
                if torch.allclose(
                    persistent_result_gemm.to(torch.float16),
                    torch_result_gemm.to(torch.float16),
                    atol=1.0,
                )
                else "❌"
            )
        print(f"persistent_vs_torch_gemm: {persistent_vs_torch_gemm}")


if __name__ == "__main__":
    validate(1024, 1024, 1024, 1.0, 0.0, dtype=torch.float16)
    validate(1024, 1024, 1024, 1.0, 0.0, dtype=torch.float32)

    # alpha and beta
    alpha = 2.0
    beta = 3.0
    validate(1024, 1024, 1024, alpha, beta, dtype=torch.float16)
    validate(1024, 1024, 1024, alpha, beta, dtype=torch.float32)
