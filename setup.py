from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_extension',
    ext_modules=[
        CUDAExtension('my_extension', ['layers/temporal_combination/qrnn_module.cu']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
