#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="epi_attention",
    packages=['epi_attention'],
    ext_modules=[
        CUDAExtension(
            name="epi_attention._C",
            sources=[
            "attn_module.cu",
            "attn_mask.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": []})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
