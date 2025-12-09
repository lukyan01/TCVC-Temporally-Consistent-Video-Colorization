#!/usr/bin/env python3
import os
import torch

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++14']

nvcc_args = [
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75',  # Turing (RTX 20)
    '-gencode', 'arch=compute_80,code=sm_80',  # Ampere (A100, RTX 30)
    '-gencode', 'arch=compute_86,code=sm_86',  # Ampere (RTX 30 mobile, works on RTX 40)
    '-gencode', 'arch=compute_86,code=compute_86'
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension('correlation_cuda', [
            'correlation_cuda.cc',
            'correlation_cuda_kernel.cu'
        ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
