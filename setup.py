#!/usr/bin/env python

import os
import sys
import setuptools
import subprocess

from Cython.Build import cythonize
import Cython.Distutils.extension as cython_extension


if sys.version_info < (3, 10, 0):
  # It should work even with 2.7, just never really tested
  raise RuntimeError("statmoments requires Python 3.10 or later")

basedir = os.path.abspath(os.path.dirname(__file__))
USE_CYTHON = os.path.isfile(os.path.join(basedir, "statmoments/_native.pyx"))


def get_cython_compile_time_env():
  cythonize_env = {'USE_CUPY_CUDA': 0}

  try:
    import cupy as cp  # noqa: F401
    import nvmath.bindings.cublas as nvmath_cublas  # noqa: F401
    # Compilation settings for cupy and _native.pyx
    cythonize_env = {
          'CUPY_CUDA_VERSION': 124,
          'CUPY_HIP_VERSION': 0,
          'CUPY_USE_CUDA_PYTHON': 0,
          'USE_CUPY_CUDA': 1
      }
    print("GPU supported: detected nvmath-python and cupy, enabling CUDA compilation")
  except ModuleNotFoundError:
    print("Tip: To use GPU install cupy and nvmath-python; Here nvmath is not installed")

  return cythonize_env


def make_ext(modname, filename):
  # This function required for in-place pyximport compilation over pyxbld
  compile_args, link_args = [], []

  if False:
    pass
  elif sys.platform == 'linux':
    compile_args.extend(['-flto'])      # Link-time code generation
    link_args.extend(['-flto'])         # Link-time code generation
    pass
  elif sys.platform == 'win32':
    # Debugging
    # compile_args.extend(['/Zi'])        # Simple PDB format
    # link_args.extend(['/DEBUG'])        # Output PDB in link time
    pass

  ext = cython_extension.Extension(
      modname,
      [filename],
      extra_compile_args=compile_args,
      extra_link_args=link_args,
      cython_compile_time_env=get_cython_compile_time_env(),
      # cython_gdb = True,
      # force = True   # always rebuild
      )
  return ext


def store_git_hash():
  try:
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip().decode("utf-8")
  except subprocess.CalledProcessError:
    return False
  with open("statmoments/GIT_VERSION.txt", "w") as h:
    h.write(commit_hash + "\n")
  return True


def build_extensions():
  # if store_git_hash():
  #   kwargs["package_data"] = {"statmoments": ["GIT_VERSION.txt"]}

  extensions = []
  if USE_CYTHON:
    # Cythonize
    extensions = cythonize('statmoments/_native.pyx', compile_time_env=get_cython_compile_time_env())
  else:
    # Compile C code
    extensions = [make_ext("statmoments._native", 'statmoments/_native.c')]
  return extensions


# PEP 517 / pip install from sdist: setuptools imports this module and reads ext_modules.
setuptools.setup(ext_modules=build_extensions())
