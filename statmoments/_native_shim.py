use_debug, use_pyrex = False, False  # Deployment through wheels
# use_debug, use_pyrex = False, True   # cython code is compiled on import
# use_debug, use_pyrex = True, False   # cython code is interpreted

assert not (use_debug and use_pyrex), "Flags use_debug and use_pyrex cannot be True simultaneously"

if use_pyrex:
  import pyximport

  # TODO: Adding CUDA
  # from pyximport import pyxbuild
  # from setup import build_ext_cupy
  # pyxbuild.build_ext = build_ext_cupy

  import numpy as np

  pyximport.install(setup_args={
      "include_dirs": np.get_include(),
      "script_args": ["--verbose"]},
      build_in_temp=False, inplace=True, reload_support=True
  )
elif use_debug:
  # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
  from sys import modules
  from os import path as os_path
  from importlib import machinery, util

  # Enable spec for pyx
  ext = '.pyx'
  machinery.SOURCE_SUFFIXES.append(ext)

  module_name, file_name = 'statmoments._native', '_native' + ext
  module_path = os_path.abspath(os_path.join(os_path.dirname(__file__), file_name))
  spec = util.spec_from_file_location(module_name, module_path)
  modules[module_name] = util.module_from_spec(spec)
  spec.loader.exec_module(modules[module_name])
  machinery.SOURCE_SUFFIXES.pop()

from statmoments._native import *
