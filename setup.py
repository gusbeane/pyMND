from setuptools import setup
from distutils.extension import Extension
import numpy
from Cython.Build import cythonize

import os
import sys
import re
import glob

import numpy # not sure how to include numpy directory before it is installed!?

dir_path = os.path.dirname(os.path.realpath(__file__))

init_string = open(os.path.join(dir_path, 'pyMND', '__init__.py')).read()
VERS = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VERS, init_string, re.M)
__version__ = mo.group(1)

ext = cythonize([Extension("pyMND.forcetree", ["pyMND/forcetree.pyx"], include_dirs=[numpy.get_include()]),
                 Extension("pyMND.force", ["pyMND/force.pyx"], include_dirs=[numpy.get_include()]),
                 Extension("pyMND.hernquist", ["pyMND/hernquist.pyx"], include_dirs=[numpy.get_include()])])

setup(
    name="pyMND",
    url="https://github.com/gusbeane/pyMND",
    version=__version__,
    author="Angus Beane",
    author_email="angus.beane@cfa.harvard.edu",
    packages=["pyMND"],
    # package_dir={'':'pyMND/'},
    description=("Initial conditions generator for isolated disk galaxies."),
    install_requires=["numpy", "scipy", "numba"],
    keywords=["initial conditions", "disk galaxies", "astronomy", "galaxies", "simulations"],
    classifiers=["Development Status :: 3 - Alpha",
                 "Natural Language :: English",
                 "Programming Language :: Python :: 3.6",
                 "Operating System :: OS Independent",
                 "Topic :: Scientific/Engineering :: Astronomy",
                 "Intended Audience :: Science/Research"],
    ext_modules=ext
)

