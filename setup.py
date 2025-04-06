import os
import torch
import glob
import sys

from setuptools import setup, find_packages

from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

library_name = "nanogpt"


def get_extensions():
    debug_mode = os.getenv("DEBUG", 0) == "1"
    extra_link_args = []
    extra_compile_args = {
        "cxx": [
            "-O3" if not debug_mode else "-O0",
            "-fdiagnostics-color=always",
            "-DPy_LIMITED_API=0x03090000",
        ],
        "nvcc": ["-O3" if not debug_mode else "-O0"],
    }
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    this_dir = os.path.dirname(os.path.curdir)
    extensions_dir = os.path.join(this_dir, library_name, "csrc")
    cpp_sources = list(glob.glob(os.path.join(extensions_dir, "*.cpp")))
    cuda_sources = list(glob.glob(os.path.join(extensions_dir, "*.cu")))

    sources = cpp_sources + cuda_sources

    ext_modules = [
        CUDAExtension(
            f"{library_name}._C",
            sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    return ext_modules


setup(
    name=library_name,
    version="0.0.1",
    packages=find_packages(),
    ext_modules=get_extensions(),
    install_requires=["torch"],
    cmdclass={"build_ext", BuildExtension},
)
