# cython: language_level=3

from scipy.linalg cimport cython_blas
from libc.math cimport sqrt
from libc.stdint cimport intptr_t, uintmax_t

#from cupy_backends.cuda.libs cimport cublas as cython_cublas
#from cupy_backends.cuda.api cimport runtime as cublas_runtime
#from cupy.cuda cimport device as cublas_device

