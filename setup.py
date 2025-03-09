# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        name="cython_CUDA_FHSPI",
        sources=["cython_CUDA_FHSPI.pyx"],
        libraries=["cudart"],
        library_dirs=["/usr/local/cuda/lib64"],
        include_dirs=[numpy.get_include(), "/usr/local/cuda/include"],
        language="c++",
        extra_compile_args=['-O3'],
        extra_link_args=['-lcudart', '-L/usr/local/cuda/lib64'],
    ),
]

setup(
    name="cython_cuda_fft",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
)
