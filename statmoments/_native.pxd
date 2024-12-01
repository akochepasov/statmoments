# cython: language_level=3

cimport scipy.linalg.cython_blas as cython_blas
from libc.math cimport sqrt

# IF USE_CUPY_CUDA:
#   # fatal error C1083: Cannot open include file: 'driver_types.h': No such file or directory
  # from nvmath.bindings cimport cublas as nv_cublas
  # from cupy_backends.cuda.libs cimport cublas as cython_cublas
  # from cupy.cuda cimport device as cupy_device
