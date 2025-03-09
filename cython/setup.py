# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy as np
import os
import subprocess

# Set CUDA paths
CUDA_HOME = "/usr/local/cuda"
CUDA_INCLUDE_DIR = os.path.join(CUDA_HOME, "include")
CUDA_LIB_DIR = os.path.join(CUDA_HOME, "lib64")
NVCC = os.path.join(CUDA_HOME, "bin", "nvcc")

# List of CUDA source files
cuda_sources = [
    "/content/cython/digitrevorder_kernel.cu",
    "/content/cython/fhtseq_inv_gpu_kernel.cu"
]

# Custom build class to compile CUDA separately
class CustomBuildExt(build_ext):
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')

        # Compile each CUDA source file to an object file
        for source in cuda_sources:
            self.compile_cuda(source)

        super().build_extensions()

    def compile_cuda(self, source):
        obj_file = os.path.splitext(source)[0] + ".o"
        cmd = [
            NVCC,
            "-c", source,
            "-o", obj_file,
            "-Xcompiler", "-fPIC",
            "-arch=sm_60",  # Change to your GPU architecture if needed
            "-O3"
        ]
        print("Compiling CUDA source:", " ".join(cmd))
        subprocess.check_call(cmd)

# Define extension module without the .cu files in sources
ext_modules = [
    Extension(
        name="cython_CUDA_FHSPI",
        sources=[
            "/content/cython/cython_CUDA_FHSPI.pyx"
        ],
        extra_objects=[
            "/content/cython/digitrevorder_kernel.o",
            "/content/cython/fhtseq_inv_gpu_kernel.o"
        ],
        include_dirs=[
            np.get_include(),
            CUDA_INCLUDE_DIR,
            os.path.dirname(__file__)
        ],
        library_dirs=[CUDA_LIB_DIR],
        libraries=["cudart"],
        language="c++",
        runtime_library_dirs=[CUDA_LIB_DIR],
        extra_compile_args=["-O3"],
        extra_link_args=["-lcudart"],
    ),
]

# Setup with custom build_ext
setup(
    name="cython_CUDA_FHSPI",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
    cmdclass={"build_ext": CustomBuildExt},
)
