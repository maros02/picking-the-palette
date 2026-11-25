from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

cy_extension = Extension("_shortest_path", ["_shortest_path.pyx"], include_dirs=[np.get_include()])

setup(ext_modules=cythonize(cy_extension, language_level="3"))
