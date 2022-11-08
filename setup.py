# Copyright 2018-2022 Carnegie Mellon University

from setuptools import setup
from pyactup import __version__

DESCRIPTION = open("README.md").read()

setup(name="pyactup",
      version=__version__,
      description="A lightweight Python implementation of a subset of the ACT-R cognitive architecture’s Declarative Memory",
      author="Don Morrison",
      author_email="dfm2@cmu.edu",
      url="https://github.com/dfmorrison/pyactup/",
      platforms=["any"],
      long_description=DESCRIPTION,
      long_description_content_type="text/markdown",
      py_modules=["pyactup"],
      install_requires=[
          "numpy",
          "pylru",
          "prettytable"],
      tests_require=["pytest"],
      python_requires=">=3.8",
      classifiers=["Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3 :: Only",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10",
                   "Programming Language :: Python :: 3.11",
                   "Operating System :: OS Independent"])
