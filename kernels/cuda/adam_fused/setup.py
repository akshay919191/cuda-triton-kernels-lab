from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="adam_fused",
    ext_modules=[
        CUDAExtension(
            name="adam_fused",
            sources=[
                "binding.cpp",
                "fused_adam_kernel.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                ],
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