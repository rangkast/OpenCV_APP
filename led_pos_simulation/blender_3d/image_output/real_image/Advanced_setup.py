from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("Advanced_Cython_Functions.pyx")
)
