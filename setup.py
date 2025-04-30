import os
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

this_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.isdir(".git"):
    subprocess.run(["git", "submodule", "update", "--init", "csrc/cutlass"], check=True)


setup(
    name="muon",
    packages=find_packages(exclude=("csrc", "data", "img", "records", "logs")),
    ext_modules=[
        CUDAExtension(
            name="muon",
            sources=[
                "csrc/muon/muon.cpp",
                "csrc/muon/newton_schulz_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-w", "-O3", "-std=c++17"],
                "nvcc": [
                    "-w",
                    "-O3",
                    "-std=c++17",
                    "--extended-lambda",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U_CUDA_NO_HALF2_OPERATORS__",
                ],
            },
            include_dirs=[
                Path(this_dir) / "csrc" / "muon",
                Path(this_dir) / "csrc" / "cutlass" / "include",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",
    ],
    setup_requires=[
        "ninja",
    ],
)
