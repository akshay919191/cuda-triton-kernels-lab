import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

setup(
    name="layernorm_cuda",
    ext_modules=[
        CUDAExtension(
            name="layernorm_cuda",
            sources=[
                "binding.cpp",
                "layernorm_kernel.cu",
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
                    "--use_fast_math",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)