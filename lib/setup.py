import setuptools 
from Cython.Build import cythonize
import numpy

setuptools.setup(
    ext_modules=cythonize("testpair.pyx"),
    include_dirs=[numpy.get_include()]
)
