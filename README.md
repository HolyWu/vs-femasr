# FeMaSR
Real-World Blind Super-Resolution via Feature Matching with Implicit High-Resolution Priors, based on https://github.com/chaofengc/FeMaSR.


## Dependencies
- [NumPy](https://numpy.org/install)
- [PyTorch](https://pytorch.org/get-started) 1.13
- [VapourSynth](http://www.vapoursynth.com/) R55+

`trt` requires additional runtime libraries:
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 11.7
- [cuDNN](https://developer.nvidia.com/cudnn) 8.6
- [TensorRT](https://developer.nvidia.com/tensorrt) 8.5.2.2

For ease of installation on Windows, you can download the 7z file on [Releases](https://github.com/HolyWu/vs-femasr/releases) which contains required runtime libraries and Python wheel file. Either add the unzipped directory to your system `PATH` or copy the DLL files to a directory which is already in your system `PATH`. Finally pip install the Python wheel file.


## Installation
```
pip install -U vsfemasr
python -m vsfemasr
```


## Usage
```python
from vsfemasr import femasr

ret = femasr(clip)
```

See `__init__.py` for the description of the parameters.
