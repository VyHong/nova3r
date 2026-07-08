# python setup.py build_ext --inplace
from setuptools import setup, Extension
import numpy as np

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

source = "_marching_cubes_lewiner_cy.pyx" if cythonize else "_marching_cubes_lewiner_cy.cpp"

ext_modules = [
    Extension(
        name="_marching_cubes_lewiner_cy",
        sources=[source],
        language="c++",
        include_dirs=[np.get_include()],
    )
]

setup(
    name="My MC",
    ext_modules=cythonize(ext_modules) if cythonize else ext_modules,
)
