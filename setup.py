import os
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(".git"):
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)


setup(
    name="nanogpt",
    packages=find_packages(exclude=("csrc", "data", "img", "records", "logs")),
    ext_modules=[
        CUDAExtension(
            name="nanogpt_cuda",
            sources=[
                "csrc/nanogpt/newton_schulz.cpp",
                "csrc/nanogpt/newton_schulz.cu",
                # "csrc/nanogpt/muon.cpp",
                # "csrc/nanogpt/muon.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                ],
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "nanogpt",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
