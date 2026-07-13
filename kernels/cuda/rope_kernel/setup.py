import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

# RTX 3050 Laptop GPU is compute capability 8.6.
# An externally provided value will override this.
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")


setup(
    name="rope_cuda",

    ext_modules=[
        CUDAExtension(
            name="rope_cuda",

            sources=[
                "binding.cpp",
                "rope_kernel.cu",
            ],

            include_dirs=[
                ".",
                "../common",
            ],

            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-std=c++17",
                ],

                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-lineinfo",

                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],

    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True
        )
    },

    zip_safe=False,
)