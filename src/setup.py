from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "triangle_hash",
        ["triangle_hash.pyx"],
        include_dirs=[np.get_include()],
        language="c++",
    )
]

setup(ext_modules=cythonize(extensions), include_dirs=[np.get_include()])
