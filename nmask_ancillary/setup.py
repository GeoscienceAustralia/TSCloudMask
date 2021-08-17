import setuptools 
from Cython.Build import cythonize
import numpy

setuptools.setup(
    ext_modules=cythonize("nmask_cmod.pyx"),
    include_dirs=[numpy.get_include()]
)
