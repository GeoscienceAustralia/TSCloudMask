import setuptools 
from Cython.Build import cythonize
import numpy

setuptools.setup(
    name = "nmask-ancillary",
    ext_modules = cythonize("nmask_ancillary/nmask_cmod.pyx"),
    include_dirs = [numpy.get_include()],
    packages = ['nmask_ancillary'],
)
