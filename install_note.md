# Install Note

## Version Check
`g++ --version`: g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
`python --version`: Python 3.10.16

## Install From Source
Before installing triton, install torch.
`pip install torch`: version does not matter, for now.

*Triton Installation Steps Refer to steps in readme.md:*
```bash
cd triton
pip install ninja cmake wheel pybind11
pip install -e python --no-build-isolation  -v # build based on local environment 
```

## Issues
No ....so: `conda install -c conda-forge libstdcxx-ng=12 -y`
Component mismatch: `pip install -e python --no-build-isolation  -v` instead of `pip install -e python -v`
