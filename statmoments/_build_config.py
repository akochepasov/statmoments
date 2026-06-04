import sys

from Cython.Distutils.extension import Extension


def get_cython_compile_time_env():
  env = {'USE_CUPY_CUDA': 0}

  try:
    import cupy as cp  # noqa: F401
    import nvmath.bindings.cublas as nvmath_cublas  # noqa: F401
  except ModuleNotFoundError:
    print("Unable to use GPU: nvmath is not installed")
  else:
    env = {
        'CUPY_CUDA_VERSION': 124,
        'CUPY_HIP_VERSION': 0,
        'CUPY_USE_CUDA_PYTHON': 0,
        'USE_CUPY_CUDA': 1
    }
    print("GPU support: detected nvmath-python and cupy, enabling CUDA compilation")

  return env


def make_ext(modname, filename):
  # Shared by setup.py builds and pyximport import-time builds.
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

  return Extension(
      modname,
      [filename],
      extra_compile_args=compile_args,
      extra_link_args=link_args,
      cython_compile_time_env=get_cython_compile_time_env(),
      # cython_gdb = True,  # Debugging
      # force = True  # Always rebuild
  )
