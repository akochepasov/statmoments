#!/usr/bin/env python

import os
import sys
import setuptools

from Cython.Build import cythonize


if sys.version_info < (3, 6, 0):
  raise RuntimeError("statmoments requires Python 3.6 or later")

basedir = os.path.abspath(os.path.dirname(__file__))
USE_CYTHON = os.path.isfile(os.path.join(basedir, "statmoments/_native.pyx"))


def make_ext(modname, filename):
  # This function is used for in-place pyximport compilation over pyxbld
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

  ext = setuptools.Extension(modname,
                             [filename],
                             extra_compile_args=compile_args,
                             extra_link_args=link_args)
  return ext


def get_version():
  _version_dict = {}
  version_path = os.path.join(basedir, 'statmoments/_version.py')
  with open(version_path) as h:
    exec(h.read(), None, _version_dict)
  return _version_dict['__version__']

# TODO: Add CUDA
# from Cython.Distutils import build_ext
# class build_ext_cupy(build_ext):
#  def __init__(self, *argc, **kw):
#    super().__init__(*argc, **kw)
#    cythonize_options = {
#      'CUPY_CUDA_VERSION'    : 115,
#      'CUPY_HIP_VERSION'     : 0,
#      'CUPY_USE_CUDA_PYTHON' : 0,
#    }

#    self.cython_compile_time_env = cythonize_options
#    self.force = True # TODO: delete this later

#  setuptools.setup(
#    cmdclass = {'build_ext' : build_ext_cupy},


kwargs = {}


def main():
  ext = '.pyx' if USE_CYTHON else '.c'
  extensions = [make_ext("statmoments._native", 'statmoments/_native' + ext)]
  extensions = cythonize('statmoments/_native' + ext) if USE_CYTHON else extensions
  setuptools.setup(
      name='statmoments',
      author='Anton Kochepasov',
      author_email='akss@me.com',
      version=get_version(),
      ext_modules=extensions,
      **kwargs)


if __name__ == '__main__':
  main()
