from setuptools import setup
from torch.utils import cpp_extension

setup(name='smooth_sampler_cpp',
      ext_modules=[cpp_extension.CUDAExtension('smooth_sampler_cpp', ['smooth_sampler.cpp', 'smooth_sampler_kernel.cu'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})