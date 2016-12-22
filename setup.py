from __future__ import print_function

import os
import subprocess

from setuptools import setup, find_packages

setup(name="raynomics",
      version="0.0.1",
      packages=find_packages(),
      install_requires=["numpy",
                        "scipy"],
      license="Apache 2.0")
