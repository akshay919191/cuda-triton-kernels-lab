from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="topk_cuda",
    ext_modules=[
        CUDAExtension(
            name="topk_cuda",
            sources=[
                "binding.cpp",
                "ksampling_kernel.cu",
            ],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    },
)