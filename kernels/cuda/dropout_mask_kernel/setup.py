from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="dropout_cuda",
    ext_modules=[
        CUDAExtension(
            name="dropout_cuda",
            sources=[
                "binding.cpp",
                "dropout_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                ],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)