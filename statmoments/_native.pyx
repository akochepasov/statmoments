#!python
#cython: boundscheck=False, cdivision=True, initializedcheck=False, nonecheck=False, overflowcheck=False, profile=False, wraparound=False

# Papers to research numerical stability:
#  W. Kahan "Further remarks on reducing truncation errors", 1965
#  D.H.D. West "Updating mean and Variance Estimates: An improved method", 1979

import cython
import numpy as np
import scipy.linalg.blas as scipy_blas
from scipy.special import binom

cython.declare(USE_VTK = cython.int)
USE_VTK = 0
cython.declare(USE_GPU = cython.int)
USE_GPU = 0

def is_vtk_installed():
  if not USE_VTK:
    return False

  try:
    import vtk
    print("VTK is installed and used")
    return True
  except ModuleNotFoundError:
    print("VTK is not installed")
    return False

#if USE_GPU:
#    from cupy_backends.cuda.libs import cupy_cublas
#    from cupy import cublas as cupy_cublas


################################ BLAS INTEROP ################################

@cython.cfunc
@cython.inline
@cython.returns(cython.void)
@cython.locals(da=cython.double, dx='double[::1]')
def dscal(da, dx):
  """ dx *= da """
  cython.declare(n=cython.int, incx=cython.int)
  n = cython.cast(cython.int, dx.shape[0])
  incx = 1
  if cython.compiled:
    cython_blas.dscal(cython.address(n), cython.address(da), cython.address(dx[0]), cython.address(incx))
  else:
    scipy_blas.dscal(da, dx, n, 0, incx)

@cython.cfunc
@cython.inline
@cython.returns(cython.void)
@cython.locals(a='double[::1]', b='double[::1]', da=cython.double)
def daxpy(a, b, da=1.0):
  """ b += da * a """
  cython.declare(n=cython.int, incx=cython.int)
  incx = 1
  n = cython.cast(cython.int, min(a.shape[0], b.shape[0]))

  if cython.compiled:
    cython_blas.daxpy(cython.address(n), cython.address(da),
                      cython.address(a[0]), cython.address(incx), cython.address(b[0]), cython.address(incx))
  else:
    scipy_blas.daxpy(a, b, n, da)

@cython.cfunc
@cython.inline
@cython.returns(cython.void)
@cython.locals(A='double[::1]', x='double[::1]', y='double[::1]', alpha=cython.double, beta=cython.double,
               uplo=cython.char, k=cython.int, lda=cython.int, incx=cython.int, incy=cython.int)
def dsbmv(A, x, y, alpha = 1.0, beta = 0.0, uplo = b'L', k = 0, lda = 1, incx = 1, incy = 1):
  """ y := alpha*A*x + beta*y """
  cython.declare(n=cython.int)
  n = cython.cast(cython.int, min(A.shape[0], x.shape[0], y.shape[0]))
  # lda = A.shape[1]

  if cython.compiled:
    cython_blas.dsbmv(cython.address(uplo), cython.address(n), cython.address(k),
                      cython.address(alpha), cython.address(A[0]), cython.address(lda),
                      cython.address(x[0]), cython.address(incx),
                      cython.address(beta), cython.address(y[0]), cython.address(incy))
  else:
    A = A.reshape(len(A), -1)
    scipy_blas.dsbmv(k, alpha, A.T, x, 1, 0, beta, y, 1, 0, 1 if uplo == b'L' else 0, 1)

@cython.cfunc
@cython.inline
@cython.returns(cython.void)
@cython.locals(A='double[:,::1]', x='double[::1]', y='double[::1]', alpha=cython.double,
               incx=cython.int, incy=cython.int)
def dger(A, x, y, alpha=1.0, incx = 1, incy = 1):
  """ A := alpha*x*trans(y) + A """
  cython.declare(m=cython.int, n=cython.int)
  m = cython.cast(cython.int, A.shape[0])
  n = cython.cast(cython.int, A.shape[1])
  if cython.compiled:
    cython_blas.dger(cython.address(n), cython.address(m),
                     cython.address(alpha), cython.address(y[0]), cython.address(incy),
                     cython.address(x[0]), cython.address(incx), cython.address(A[0, 0]), cython.address(n))
  else:
    A = A.reshape(len(A), -1)
    scipy_blas.dger(alpha, y, x, incy, incx, A.T, 1, 1, 1)

@cython.cfunc
@cython.inline
@cython.returns(cython.void)
@cython.locals(A='float[:,::1]', C='float[:,::1]', uplo=cython.char,
               trans=cython.char, alpha=cython.float, beta=cython.float)
def ssyrk(A, C, uplo, trans=b'N', alpha=1.0, beta=1.0):
  """ C = alpha * trans(A * A^T) + beta * C """
  cython.declare(n=cython.int, k=cython.int, lda=cython.int, ldc=cython.int)
  n = cython.cast(cython.int, C.shape[0])
  k = cython.cast(cython.int, A.shape[0] if trans == b'N' else A.shape[1])
  lda = cython.cast(cython.int, A.shape[1])
  ldc = cython.cast(cython.int, C.shape[1])
  # assert (A.shape[1] if trans == b'N' else A.shape[0]) == n
  # assert C.shape[1] == n

  # !!! uplo 'L' and 'U' mixed up !!!
  if cython.compiled:
    cython_blas.ssyrk(cython.address(uplo), cython.address(trans), cython.address(n), cython.address(k),
                      cython.address(alpha), cython.address(A[0, 0]), cython.address(lda),
                      cython.address(beta), cython.address(C[0, 0]), cython.address(ldc))
  else:
    scipy_blas.ssyrk(alpha, A.T, beta, C.T, 1 if trans != b'N' else 0, 1 if uplo != b'U' else 0, 1)

@cython.cfunc
@cython.inline
@cython.returns(cython.void)
@cython.locals(A='double[:,::1]', C='double[:,::1]', uplo=cython.char,
               trans=cython.char, alpha=cython.double, beta=cython.double)
def dsyrk(A, C, uplo, trans=b'N', alpha=1.0, beta=1.0):
  """ C = alpha * trans(A * A^T) + beta * C """
  cython.declare(n=cython.int, k=cython.int, lda=cython.int, ldc=cython.int)
  n = cython.cast(cython.int, C.shape[0])
  k = cython.cast(cython.int, A.shape[0] if trans == b'N' else A.shape[1])
  lda = cython.cast(cython.int, A.shape[1])
  ldc = cython.cast(cython.int, C.shape[1])
  # assert (A.shape[1] if trans == b'N' else A.shape[0]) == n
  # assert C.shape[1] == n

  if cython.compiled:
    #    if not USE_GPU:
    uplo = b'U' if uplo != b'U' else b'L'
    cython_blas.dsyrk(cython.address(uplo), cython.address(trans), cython.address(n), cython.address(k),
                      cython.address(alpha), cython.address(A[0, 0]), cython.address(lda),
                      cython.address(beta), cython.address(C[0, 0]), cython.address(ldc))
  else:
    # A and C have to be transposed to comply with F-order
    scipy_blas.dsyrk(alpha, A.T, beta, C.T, 1 if trans != b'N' else 0, 0 if uplo != b'U' else 1, 1)

@cython.cfunc
@cython.inline
@cython.returns(cython.void)
@cython.locals(a='double[:,::1]', b='double[:,::1]', c='double[:,::1]',
               transa=cython.char, transb=cython.char,
               alpha=cython.double, beta=cython.double)
def dgemm(a, b, c, transa=b'N', transb=b'N', alpha=1.0, beta=1.0):
  """ C = alpha * transa(A) * transb(B) + beta * C """
  cython.declare(n=cython.int, m=cython.int, k=cython.int, lda=cython.int, ldb=cython.int, ldc=cython.int)
  n = cython.cast(cython.int, c.shape[0])
  m = cython.cast(cython.int, c.shape[1])
  k = cython.cast(cython.int, a.shape[0] if transa == b'N' else a.shape[1])
  lda = cython.cast(cython.int, a.shape[1])
  ldb = cython.cast(cython.int, b.shape[1])
  ldc = cython.cast(cython.int, c.shape[1])
  # assert (b.shape[0] if transb == b'N' else b.shape[1]) == n
  # assert (a.shape[1] if transa == b'N' else a.shape[0]) == m
  # assert (b.shape[1] if transb == b'N' else b.shape[0]) == k

  if cython.compiled:
    cython_blas.dgemm(cython.address(transa), cython.address(transb),
                      cython.address(m), cython.address(n), cython.address(k),
                      cython.address(alpha), cython.address(a[0, 0]), cython.address(lda),
                      cython.address(b[0, 0]), cython.address(ldb),
                      cython.address(beta), cython.address(c[0, 0]), cython.address(ldc))
  else:
    scipy_blas.dgemm(alpha, a.T, b.T, beta, c.T, 1 if transa != b'N' else 0, 1 if transb != b'N' else 0, 1)

################################ LOCAL HELPERS ################################
@cython.cfunc
@cython.locals(i=cython.Py_ssize_t, tr_len=cython.Py_ssize_t)
def _block_index(i, tr_len):
  j = slice(i, tr_len)
  k = slice(tr_len * i - (i - 1) * i // 2, tr_len * (i + 1) - (i + 1) * i // 2)
  return j, k


############################### COMMON CLASS ##################################
class _AccBase(object):
  def __init__(self, tr_len, cl_len, moment=2, normalize=True, **kwargs):
    self.total_count = 0
    self.moment = moment
    self.normalize = normalize
    self.trace_len = tr_len
    self.classifiers_len = cl_len
    self.acc_min_count = kwargs.pop('acc_min_count', 10)

  @staticmethod
  def estimate_mem_size(tr_len, cl_len=1, moment=2):
    raise NotImplementedError("Must be implemented in an inherited class")

  def memory_size(self):
    raise NotImplementedError("Must be implemented in an inherited class")

  def update(self, traces, classifiers):
    raise NotImplementedError("Must be implemented in an inherited class")

  def counts(self, i):
    raise NotImplementedError("Must be implemented in an inherited class")

  def moments(self, moments=None, normalize=None):
    """Return a generator yielding a pair of univariate statistical moments for each classifier"""
    moments = moments if moments is not None else [self.moment]
    normalize = normalize if normalize is not None else self.normalize
    return self._moments(moments, normalize)

  def _ensure_ret(self, mlen):
    if self._retm.shape[1] < mlen:
      tr_len = self.trace_len
      self._retm = np.empty((2, mlen, tr_len * (tr_len + 1) // 2), dtype=np.float64)

    return self._retm

  def comoments(self, moments=None, normalize=None):
    """Return a generator yielding the requested set of bivariate comoments for each classifier"""
    moments = moments if moments is not None else np.broadcast_to(self.moment, (2, 1))
    normalize = normalize if normalize is not None else self.normalize
    if len(moments[0]) != len(moments[1]):
      raise ValueError("The input moment lists should have equal lengths.")

    if self.moment < np.max(moments):
      raise ValueError("The input moment should be less or equal than indicated in constructor.")

    self._ensure_ret(len(moments[0]))

    return self._comoments(moments, normalize)


########################## BIVARIATE IMPLEMENTATIONS ##########################

@cython.cfunc
@cython.returns(cython.void)
@cython.locals(M = 'double[:, ::1]', moment='int', tmpbuf = 'double[:, ::1]')
def _rmoms2cmoms1D(M, moment, tmpbuf):
  """Convert raw moments to central moments"""
  # Non-Horner method
  cython.declare(k='int', p='int')
  cython.declare(terms='double[::1]')
  cython.declare(M0p='double[:, ::1]')
  terms = tmpbuf[2]
  M0p = tmpbuf[:2]
  for k in range(moment, 1, -1):
    terms[:] = 0.0
    M0p[(k + 1) % 2, :] = 1.0
    for p in range(k - 1, 1, -1):
      # terms += (-1) ** (k - p) * binom(k, p) * M[p - 1] * M0p[p % 2]
      # M0p[(p + 1)%2, :] = M[0] * M0p[p % 2]
      dsbmv(M0p[p % 2], M[p - 1], terms, (-1.0) ** (k - p) * binom(k, p), 1.0)
      dsbmv(M[0], M0p[p % 2], M0p[(p + 1) % 2], 1.0)
    p = 1
    # terms += (-1) ** (k - p) * (k - p) * M[p - 1] * M0p[p % 2]
    # M[k - 1] += M[0] * terms
    dsbmv(M0p[p % 2], M[p - 1], terms, (-1.0) ** (k - p) * (k - p), 1.0)
    dsbmv(M[0], terms, M[k - 1], 1.0, 1.0)

def _rmoms2cmoms2D_general(raw, ave, n, k, l, i, j):
  """Convert raw co-moments to central co-moments in a general fashion"""
  ave_i = ave[i]  # i is a scalar
  ave_j = ave[j]  # j is a slice such that indices (j, j) add diagonal!
  com = (-1) ** (k + l) * (1 - k - l) * n * ave_i ** k * ave_j ** l

  for p in range(2, k + 1):
    edge_val = raw[0, :, p - 2, :][i, i]
    com += (-1) ** (k + l - p) * binom(k, p) * edge_val * ave_j ** l * ave_i ** (k - p)
  for q in range(2, l + 1):
    edge_val = raw[0, :, q - 2, :][j, j].diagonal()
    com += (-1) ** (k + l - q) * binom(l, q) * edge_val * ave_i ** k * ave_j ** (l - q)
  for p in range(1, k + 1):
    for q in range(1, l + 1):
      Mpq = raw[p - 1, :, q - 1, :][i, j] if p <= q else raw[q - 1, :, p - 1, :][j, i]
      com += (-1) ** (k + l - p - q) * binom(k, p) * binom(l, q) * Mpq * ave_i ** (k - p) * ave_j ** (l - q)

  return com

@cython.cfunc
@cython.locals(n=cython.double, lm=cython.int, rm=cython.int, i=cython.int)
def _rmoms2cmoms2D(raw, ave, n, lm, rm, i, j):
  """Convert raw co-moments to central co-moments

  This coversion is optimized for a few co-moments, e.(1, 1) and (2, 2).
  Others are calculated more slowly in a general fashion"""

  inv_n = 1.0 / n
  ave_i = ave[i]  # i is a scalar
  ave_j = ave[j]  # j is a slice such that indices (j, j) add diagonal!

  # Expressions for conversions raw moments to central moments in the
  # Horner representation auto-generated with sympy, group order (mu_j, mu_i)
  if False:
    pass # Unifies formatting idents for cases
  elif lm == 1 and rm == 1:  # E(X Y)
    M_11 = raw[0, :, 0, :]   # 0
    m = M_11[i, j] - n * ave_i * ave_j
  elif lm == 1 and rm == 2 or lm == 2 and rm == 1:  # E(XX Y)
    M_11 = raw[0, :, 0, :]   # 0
    M_12 = raw[0, :, 1, :]
    m = ave_j * (2 * n * ave_i * ave_j - 2 * M_11[i, j]) \
        - ave_i * M_11[j, j].diagonal() \
        + M_12[i, j]
  elif lm == 1 and rm == 3 or lm == 3 and rm == 1:  # E(XXX Y)
    M_11 = raw[0, :, 0, :]   # 0
    M_12 = raw[0, :, 1, :]
    M_13 = raw[0, :, 2, :]
    m = ave_j * (3 * ave_i * M_11[j, j].diagonal() - 3 * M_12[i, j] \
                + ave_j * (3 * M_11[i, j] - 3 * n * ave_i * ave_j)) \
        - ave_i * M_12[j, j].diagonal() \
        + M_13[i, j]
  elif lm == 2 and rm == 2:  # E(XX YY)
    M_11 = raw[0, :, 0, :]   # 0
    M_12 = raw[0, :, 1, :]
    M_22 = raw[1, :, 1, :]   # 1
    m = ave_j * (4 * ave_i * M_11[i, j] - 2 * M_12[j, i] \
                + ave_j * (M_11[i, i] - 3 * n * ave_i * ave_i)) \
        + ave_i * (ave_i * M_11[j, j].diagonal() - 2 * M_12[i, j]) \
        + M_22[i, j]
  elif lm == 4 and rm == 4:  # E(XXXX YYYY)
    M_11 = raw[0, :, 0, :]   # 0
    M_12 = raw[0, :, 1, :]
    M_13 = raw[0, :, 2, :]
    M_14 = raw[0, :, 3, :]
    M_22 = raw[1, :, 1, :]   # 1
    M_23 = raw[1, :, 2, :]
    M_24 = raw[1, :, 3, :]
    M_33 = raw[2, :, 2, :]   # 2
    M_34 = raw[2, :, 3, :]
    M_44 = raw[3, :, 3, :]   # 3
    m = ave_j * (-4 * M_34[j, i] \
                + ave_i * (16 * M_33[i, j] \
                          + ave_i * (-24 * M_23[i, j] + ave_i * (-4 * M_12[j, j].diagonal() * ave_i + 16 * M_13[i, j]))) \
                + ave_j * (6 * M_24[j, i] \
                          + ave_i * (-24 * M_23[j, i] \
                                    + ave_i * (36 * M_22[i, j] \
                                              + ave_i * ((6 * ave_i) * M_11[j, j].diagonal() - 24 * M_12[i, j]))) \
                          + ave_j * (-4 * M_14[j, i] \
                                    + ave_i * (16 * M_13[j, i] + ave_i * (16 * ave_i * M_11[i, j] - 24 * M_12[j, i])) \
                                    + ave_j * (M_13[i, i] \
                                              + ave_i * (-4 * M_12[i, i] + ave_i * (6 * M_11[i, i] - 7* n * ave_i ** 2)))))) \
        + ave_i * (-4 * M_34[i, j] + ave_i * (6 * M_24[i, j] + ave_i * (M_13[j, j].diagonal() * ave_i - 4 * M_14[i, j]))) \
        + M_44[i, j]
  else:
    m = _rmoms2cmoms2D_general(raw, ave, n, lm, rm, i, j)

  m *= inv_n
  return m


class _bivar_sum_base(_AccBase):
  def __init__(self, trace_len, classifier_len, **kwargs):
    super().__init__(trace_len, classifier_len, **kwargs)

    moment = self.moment
    exp_traces = 50  # Guesstimate the number of traces in the input batch
    self._dtype = kwargs.pop('acctype', np.float64)

    # Buffers for operations to avoid reallocations
    self._layout = np.empty((exp_traces, moment * trace_len + 1), dtype=self._dtype)
    self._retm   = np.empty((2, 2, trace_len * (trace_len + 1) // 2), dtype=np.float64)
    self._buf    = np.empty((3, trace_len), dtype=self._retm.dtype)  # BLAS ops buffer

    # Accumulators
    # acc0 is upper triangle of accs[:,:-1], acc1 is lower triangle of accs[:, 1:]
    self._accs   = np.zeros((classifier_len, moment * trace_len + 2, moment * trace_len + 1), dtype=self._dtype)

  @staticmethod
  def estimate_mem_size(tr_len, cl_len=1, moment=2, acc_dtype=np.float64):
    # Approximate memory consumption for 1 classifier, 1 moment:
    # 10k: 5GB,  20k: 15GB,  50k: 80GB
    exp_traces = 50  # Expected number of traces in the input batch
    bufsz      = 3 * tr_len
    lytsz      = exp_traces * (moment * tr_len + 1)
    accssz     = cl_len * (moment * tr_len + 1) ** 2
    ttsz       = 2 * 2 * tr_len * tr_len // 2
    return (lytsz + accssz) * np.finfo(acc_dtype).bits // 8 + (bufsz + ttsz) * 8

  def memory_size(self):
    mem_class = sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))
    # ttest memory included in mem_class or std normalization computed one line at a time
    mem_ttest = 0  if self.moment < 3 else self.trace_len
    return mem_class + mem_ttest * 8

  def update(self, traces, classifiers):
    moment, batch_cnt = self.moment, len(traces)
    tr_len, cl_len = self.trace_len, self.classifiers_len

    self.total_count += batch_cnt

    if len(self._layout) < batch_cnt:
      self._layout = np.empty((batch_cnt, moment * tr_len + 1), dtype=self._layout.dtype)

    tr_lyt = self._layout[:batch_cnt]
    tr_lyt[:, 0] = 1
    tr_lyt_view = tr_lyt[:, 1:].reshape(batch_cnt, moment, tr_len)
    for i in range(cl_len):
      n0, n1 = 0, 0
      # Sort traces by classifiers
      for j, tr in enumerate(traces):
        if classifiers[j][i]:
          tr_lyt_view[batch_cnt - n1 - 1, 0, :] = tr
          n1 += 1
        else:
          tr_lyt_view[n0, 0, :] = tr
          n0 += 1

      # Step 1: Create layout matrix up to moment degree
      for j in range(1, moment):
        np.multiply(tr_lyt_view[:, 0, :], tr_lyt_view[:, j - 1, :], tr_lyt_view[:, j, :])

      # Step 2: Gram multiplication accumulates sums of all combinations of all degrees
      if n0 > 0:
        # syrk: acc0 += T0 * T0.T
        if tr_lyt.dtype == np.float64:
          dsyrk(tr_lyt[:n0], self._accs[i, 1:], b'L')
        else:
          ssyrk(tr_lyt[:n0], self._accs[i, 1:], b'L')
      if n1 > 0:
        # syrk: acc1 += T1 * T1.T
        if tr_lyt.dtype == np.float64:
          dsyrk(tr_lyt[n0:], self._accs[i, :-1], b'U')
        else:
          ssyrk(tr_lyt[n0:], self._accs[i, :-1], b'U')

  def counts(self, i):
    accs = self._accs[i]
    acc1, acc0 = accs, accs[1:].T
    return acc0[0, 0], acc1[0, 0]

  def _moments(self, moments, normalize):
    maxm = np.max(moments)
    if self.moment * 2 < maxm:
      raise ValueError("The moment should be less or equal than indicated in constructor.")

    moments = [m - 1 for m in moments]
    ma, min_cnt = self.moment, self.acc_min_count
    tr_len, cl_len = self.trace_len, self._accs.shape[0]
    # 2D stat should return a separate piece of memory to avoid buffer corruption
    # while normalizing coskeweness and higher moments
    cm, buf = np.empty((2, maxm, tr_len), dtype=np.float64), self._buf

    for ii in range(cl_len):
      n0, n1 = self.counts(ii)

      acc1, acc0 = self._accs[ii], self._accs[ii, 1:].T

      # Convert to double if accumulator is float
      acc0 = np.asarray(acc0, dtype=np.float64)
      acc1 = np.asarray(acc1, dtype=np.float64)

      # Reshape accumulator to a co-moment tensor
      raw0 = acc0[1:acc0.shape[0] - 0, 1:].reshape(ma, tr_len, ma, tr_len)
      raw1 = acc1[1:acc1.shape[0] - 1, 1:].reshape(ma, tr_len, ma, tr_len)

      for _i in range(2):
        nn = n0 if _i == 0 else n1
        if nn < min_cnt:
          cm[_i, moments] = 0.0
          continue

        raw = raw0 if _i == 0 else raw1
        acc = acc0 if _i == 0 else acc1

        # Get mean
        cm[_i, 0] = acc[0, 1: 1 + tr_len]

        # and higher raw moments
        for i in range(maxm - 1):
          cm[_i, i + 1] = raw[i // 2, :, i // 2 + i % 2, :].diagonal()

        cm[_i] *= 1.0 / nn

        _rmoms2cmoms1D(cm[_i], maxm, buf)

        if maxm >= 3 and normalize:
          sd = np.sqrt(cm[_i, 1])
          for i, cmi in enumerate(cm[_i, 2:maxm], 3):
            cmi[:] /= sd ** i

      yield cm[:, moments]

  def _comoments(self, moments, normalize):
    retm = self._retm
    tr_len, cl_len = self.trace_len, self._accs.shape[0]
    ma, min_cnt = self.moment, self.acc_min_count

    for ii in range(cl_len):
      n0, n1 = self.counts(ii)

      # Convert to double if accumulator is float
      acc0 = np.asarray(self._accs[ii, 1:].T, dtype=np.float64)
      acc1 = np.asarray(self._accs[ii, 0:],   dtype=np.float64)

      # Reshape accumulator to a 4D co-sum tensor
      raw0 = acc0[1:acc0.shape[0] - 0, 1:].reshape(ma, tr_len, ma, tr_len)
      raw1 = acc1[1:acc1.shape[0] - 1, 1:].reshape(ma, tr_len, ma, tr_len)

      for _i in range(2):
        nn = n0 if _i == 0 else n1
        if nn < min_cnt:
          retm[_i, :len(moments[0])] = 0.0
          continue

        # The left and the right degree of the product terms, e.g. xi^1 * xj^3
        for jj, (lm, rm) in enumerate(zip(*moments)):
          raw = raw0 if _i == 0 else raw1
          acc = acc0 if _i == 0 else acc1
          for i in range(tr_len):
            j, k = _block_index(i, tr_len)
            m1 = 1.0 / nn * acc[0, 1: 1 + tr_len]
            retm[_i, jj, k] = _rmoms2cmoms2D(raw, m1, nn, lm, rm, i, j)

            if (lm + rm) >= 3 and normalize:
              m2 = 1.0 / nn * raw[0, :, 0, :].diagonal()
              m1 = 1.0 / nn * acc[0, 1: 1 + tr_len]
              sd = np.sqrt(m2 - m1 * m1)
              retm[_i, jj, k] /= sd[i] ** lm * sd[j] ** rm

      yield self._retm[:, :len(moments[0])]


class bivar_sum(_bivar_sum_base):
  def __init__(self, trace_len, classifier_len, **kwargs):
    super().__init__(trace_len, classifier_len, acctype=np.float64, **kwargs)


class bivar_sum_mix(_bivar_sum_base):
  def __init__(self, trace_len, classifier_len, **kwargs):
    super().__init__(trace_len, classifier_len, acctype=np.float32, **kwargs)


class bivar_sum_detrend(_bivar_sum_base):
  def __init__(self, trace_len, classifier_len, **kwargs):
    super().__init__(trace_len, classifier_len, **kwargs)
    self._dbuf = np.zeros((1, self.trace_len), dtype=self._dtype)[1:]
    self._cbuf = np.asarray([bytearray(b'2') * classifier_len], dtype=np.uint8)[1:]  # 2 is non existing element
    self._detrend_cnt = kwargs.pop('detrend_cnt', 10)

  def update(self, traces, classifiers):
    tr_copy = np.asarray(traces, dtype=self._dtype)
    cl_copy = np.asarray(classifiers, dtype=np.uint8)

    if self._detrend_cnt > 0:
      db_part = tr_copy[:self._detrend_cnt]
      cl_part = cl_copy[:self._detrend_cnt]
      self._dbuf = np.vstack((self._dbuf, db_part))
      self._cbuf = np.vstack((self._cbuf, cl_part))
      self._detrend_cnt -= len(db_part)
      if self._detrend_cnt > 0:
        return  # Not enough input data for detrending
      if self._detrend_cnt == 0:
        tr_copy = np.vstack((self._dbuf, tr_copy[len(db_part):]))
        cl_copy = np.vstack((self._cbuf, cl_copy[len(cl_part):]))
        self._dbuf = np.mean(self._dbuf, axis=0)
        self._cbuf = None
    super().update(tr_copy - self._dbuf, cl_copy)

  def _moments(self, moments, normalize):
    m1idx = [i for i, m in enumerate(moments) if m == 1]
    has_m1 = len(m1idx) != 0
    dbuf = self._dbuf if len(self._dbuf) > 0 else 0
    for res in super()._moments(moments, normalize):
      if has_m1:
        # Restore offset if mean requested
        res[:, m1idx] += dbuf
      yield res


def _triuflatten_gen(n):
  _ndx = np.triu_indices_from(np.empty((n, n)))
  def _triu(tr2d):
    return tr2d[_ndx]
  return _triu

def _compute_by_rows(tr_len, f):
  # Compute by rows upper triangle part of matrix from function f
  # This function is required for optimized memory consumption
  com = np.empty(tr_len * (tr_len + 1) // 2, dtype=np.float64)
  for i in range(tr_len):
    j, k = _block_index(i, tr_len)
    com[k] = f(i, j)
  return com

def update_moments(acc2d, acc1d, n1, moment, traces):
  # The update method:
  # x - ave2 = x + (ave1 - ave1) - ave2 = x - ave1 + (ave1 - ave2)
  # Assume: delta = ave1 - ave2
  # Therefore: x - ave2 = (x - ave1) + delta
  # A complete update step is:
  #  (x[i] - ave2[i])^k * (x[j] - ave2[j])^m =
  # ((x[i] - ave1[i]) + delta[i])^k * ((x[j] - ave1[j]) + delta[j])^m
  # delta can be found as in Welford's method, check for Meng's for higher moments:
  # ave1 - ave2 =
  #   (ave1*n1*n2 - n1*sum2) / (n1*(n1 + n2)) = (ave1*n2 - sum2) / (n1 + n2)
  # This function is slow and should be vectorized

  n2 = len(traces)
  delta = np.empty_like(acc2d[0, 0])  # 2m x tr_len
  np.sum(traces, axis=0, out=delta[0])
  delta[0] = (acc1d * n2 - delta[0]) / (n1 + n2)

  for j in range(1, moment):
    dsbmv(delta[0], delta[j - 1], delta[j])

  acc1d[:] -= delta[0]
  temp_diag = np.empty_like(acc1d)
  for k in range(moment - 1, -1, -1):
    for l in range(moment - 1, -1, -1):

      for p in range(k - 1, -1, -1):
        for q in range(l - 1, -1, -1):
          da = binom(k + 1, p + 1) * binom(l + 1, q + 1)
          acc2d[k, :, l, :] += da * acc2d[p, :, q, :] * np.outer(delta[k - p - 1], delta[l - q - 1])

      for p in range(k - 1, -1, -1):
        acc2d[k, :, l, :] += acc2d[p, :, l, :] * (binom(k + 1, p + 1) * delta[k - p - 1, :, np.newaxis])
      for q in range(l - 1, -1, -1):
        acc2d[k, :, l, :] += acc2d[k, :, q, :] * (binom(l + 1, q + 1) * delta[l - q - 1, np.newaxis, :])

      if k > 0:
        temp_diag[:] = np.diag(acc2d[k - 1, :, 0, :])
        for p in range(k - 1, 0, -1):
          temp_diag += np.diag(acc2d[p - 1, :, 0, :]) * (binom(k + 1, p + 1) * delta[k - p - 1])
        acc2d[k, :, l, :] += np.outer(temp_diag, delta[l])

      if l > 0:
        temp_diag[:] = np.diag(acc2d[0, :, l - 1, :])
        for q in range(l - 1, 0, -1):
          temp_diag += np.diag(acc2d[0, :, q - 1, :]) * (binom(l + 1, q + 1) * delta[l - q - 1])
        acc2d[k, :, l, :] += np.outer(delta[k], temp_diag)

      acc2d[k, :, l, :] += n1 * np.outer(delta[k], delta[l])

def _calc_comoments(MM, lm, rm, normalize):
  # Find the coefficients with the required degree for the left part and the right part
  # lm - left_moment, rm - right_moment
  m = MM[lm - 1, :, rm - 1, :]
  if (lm + rm) >= 3 and normalize:
    sd = np.sqrt(np.diag(MM[0, :, 0, :]))
    m /= np.outer(sd ** lm, sd ** rm)
  return m


class bivar_cntr(_AccBase):
  """A class, working with 2D accs, which track central moments"""
  def __init__(self, trace_len, classifier_len, **kwargs):
    super().__init__(trace_len, classifier_len, **kwargs)

    # Accumulators, storing full rectangles
    moment = self.moment
    self._accs1d = np.zeros((classifier_len, 2, trace_len), dtype=np.float64)
    self._accs2d = np.zeros((classifier_len, 2, moment, trace_len, moment, trace_len), dtype=np.float64)
    self._layout = np.empty((50, moment, trace_len), dtype=np.float64)
    self._cls_count = np.zeros((classifier_len, 2), np.uint32)
    self._retm = np.empty((2, 2, trace_len * (trace_len + 1) // 2))

  @staticmethod
  def estimate_mem_size(tr_len, cl_len=1, moment=2):
    exp_traces = 50  # Expected number of traces in the input batch
    acc_dtype  = np.float64
    lytsz      = exp_traces * moment * tr_len
    accs1dsz   = cl_len * 2 * tr_len
    accs2dsz   = cl_len * 2 * (moment * tr_len) ** 2
    ttsz       = 2 * 2 * tr_len * (tr_len + 1) // 2
    clssz      = cl_len * 2
    # One additional square for std normalization for moments > 2
    ttstdsz    = 0  if moment < 3 else tr_len + tr_len ** 2
    return (lytsz + accs1dsz + accs2dsz) * np.finfo(acc_dtype).bits // 8 + (ttsz + ttstdsz) * 8 + clssz * 4

  def memory_size(self):
    mem_class = sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))
    # ttest memory included in mem_class or one additional square for std normalization
    mem_ttest = 0  if self.moment < 3 else self.trace_len + self.trace_len ** 2
    return mem_class + mem_ttest * 8

  def update(self, traces, classifiers):
    moment, batch_cnt = self.moment, len(traces)
    tr_len, cl_len = self.trace_len, len(classifiers[0])

    self.total_count += batch_cnt

    if len(self._layout) < batch_cnt:
      self._layout = np.empty((batch_cnt, moment, self._accs1d.shape[2]), dtype=np.float64)

    tr_layout = self._layout[:batch_cnt]
    for i in range(cl_len):
      n0, n1 = 0, 0
      # Sort traces by classifiers
      for tr, cl in zip(traces, classifiers):
        if cl[i]:
          tr_layout[batch_cnt - n1 - 1, 0, :] = tr
          n1 += 1
        else:
          tr_layout[n0, 0, :] = tr
          n0 += 1

      if n0 > 0:
        update_moments(self._accs2d[i, 0], self._accs1d[i, 0], self._cls_count[i, 0], moment, tr_layout[:n0, 0])
      if n1 > 0:
        update_moments(self._accs2d[i, 1], self._accs1d[i, 1], self._cls_count[i, 1], moment, tr_layout[n0:, 0])

      self._cls_count[i, 0] += n0
      self._cls_count[i, 1] += n1

      tr_layout[:n0, 0] -= self._accs1d[i, 0]
      tr_layout[n0:, 0] -= self._accs1d[i, 1]
      for j in range(1, moment):
        np.multiply(tr_layout[:, 0, :], tr_layout[:, j - 1, :], tr_layout[:, j, :])

      ar_len = tr_layout.shape[1] * tr_layout.shape[2]
      if n0 > 0:
        dgemm(tr_layout[:n0].reshape((-1, ar_len)), tr_layout[:n0].reshape((-1, ar_len)),
              self._accs2d[i, 0].reshape(ar_len, ar_len), transa=b'N', transb=b'T')
      if n1 > 0:
        dgemm(tr_layout[n0:].reshape((-1, ar_len)), tr_layout[n0:].reshape((-1, ar_len)),
              self._accs2d[i, 1].reshape(ar_len, ar_len), transa=b'N', transb=b'T')

  def counts(self, i):
    return self._cls_count[i]

  def _moments(self, moments, normalize):
    maxm, min_cnt = np.max(moments), self.acc_min_count
    if self.moment * 2 < maxm:
      raise ValueError("The moment should be less or equal than indicated in constructor.")

    moments = [m - 2 for m in moments]
    tr_len, cl_len = self.trace_len, self._accs2d.shape[0]

    # 2D stat should return a separate piece of memory to avoid buffer corruption
    # while normalizing coskeweness and higher moments
    retm = np.empty((2, len(moments), tr_len), dtype=np.float64)

    for ii in range(cl_len):
      nn = self._cls_count[ii]
      if min(nn) < min_cnt:
        retm[:] = 0
        yield retm
        continue

      for _i in range(2):
        cm = retm[_i]
        for jj, m in enumerate(moments):
          if m == -1:
            cm[jj] = self._accs1d[ii][_i]
          else:
            cm[jj] = 1 / nn[_i] * self._accs2d[ii][_i, m // 2, :, m // 2 + m % 2, :].diagonal()

            if m >= 1 and normalize:
              sd = np.sqrt(1 / nn[_i] * self._accs2d[ii][_i, 0, :, 0, :].diagonal())
              cm[jj] /= sd ** (m + 2)

      yield retm

  def _comoments(self, moments, normalize):
    retm = self._retm
    tr_len, min_cnt = self.trace_len, self.acc_min_count
    triuflatten = _triuflatten_gen(tr_len)

    for accs2d, accs_count in zip(self._accs2d, self._cls_count):
      for _i in range(2):
        nn = accs_count[_i]
        if nn < min_cnt:
          retm[_i, :] = 0
          continue
        for jj, (lm, rm) in enumerate(zip(*moments)):
          C = _calc_comoments(1.0 / nn * accs2d[_i], lm, rm, normalize)
          retm[_i, jj] = triuflatten(C)
      yield self._retm[:, :len(moments[0])]


########################## BIVAR 2pass and text book ##########################
def _std_normalize(arr):
  return arr / np.std(arr, axis=0)

def _sort_meanfree(ret, normalize=False):
  if len(ret) < 1:
    return ret
  ret[:] -= np.mean(ret, axis=0)
  if normalize:
    ret[:] = (ret / np.std(ret, axis=0))
  return ret

@cython.cfunc
@cython.returns('double[:,::1]')
@cython.locals(tmp='double[:,::1]', m1=cython.int, m2=cython.int)
def _uni2bivar(traces, tmp, ret, m1, m2):
  # 2D traces for equal l-and r-moments
  cython.declare(m=cython.Py_ssize_t, n=cython.Py_ssize_t)
  cython.declare(C='double[:,::1]')
  m = cython.cast(cython.int, traces.shape[0])
  n = cython.cast(cython.int, traces.shape[1])
  triuflatten, C = _triuflatten_gen(n), tmp
  for j in range(m):
    dsyrk(traces[j, np.newaxis] ** m1, C, b'U', b'N', 1.0, 0.0)                               # m1 == m2
    if cython.compiled:
      ret[j, :] = triuflatten(C.base)
    else:
      ret[j, :] = triuflatten(C)
  return ret

@cython.cfunc
@cython.returns('double[:,::1]')
@cython.locals(tmp='double[:,::1]', m1=cython.int, m2=cython.int)
def _uni2bivar_neq(traces, tmp, ret, m1, m2):
  # 2D traces for non-equal l-and r-moments
  cython.declare(m=cython.Py_ssize_t, n=cython.Py_ssize_t)
  cython.declare(C='double[:,::1]')
  m = cython.cast(cython.int, traces.shape[0])
  n = cython.cast(cython.int, traces.shape[1])
  triuflatten, C = _triuflatten_gen(n), tmp
  for j in range(m):
    dgemm(traces[j, np.newaxis] ** m2, traces[j, np.newaxis] ** m1, C, b'N', b'T', 1.0, 0.0)  # m1 != m2
    if cython.compiled:
      ret[j, :] = triuflatten(C.base)
    else:
      ret[j, :] = triuflatten(C)
  return ret
  # np.vstack and np.outer are 3x slower
  # return np.vstack(list(triuflatten(np.outer(tr**m1, tr**m2)) for tr in traces)) # 3x slower


class _BivarNpassBase(_AccBase):
  def __init__(self, tr_len, cl_len, **kwargs):
    super().__init__(tr_len, cl_len, **kwargs)
    self.traces = np.empty((1, tr_len))[1:]
    self.cls    = np.asarray([bytearray(b'2' * cl_len)], dtype=np.uint8)[1:]  # 2 is non existing element
    self._tmpsq = np.empty((tr_len, tr_len))  # Has to be square for syrk
    self._retm  = np.empty((2, 2, tr_len * (tr_len + 1) // 2))

  @staticmethod
  def estimate_mem_size(tr_len, cl_len=1, moment=2):
    mem_class = tr_len * tr_len + 2 * 2 * tr_len * (tr_len + 1) // 2
    return mem_class * 8

  def memory_size(self):
    mem_class = sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))
    return mem_class

  def update(self, traces, classifiers):
    self.total_count += len(traces)

    tr_copy = np.asarray(traces, dtype=np.float64)
    cl_copy = np.asarray(classifiers, dtype=np.uint8)
    self.traces = np.vstack((self.traces, tr_copy))
    self.cls    = np.vstack((self.cls, cl_copy))

  def counts(self, i):
    c1 = np.sum(self.cls[:, i], dtype=np.uint32)
    return self.total_count - c1, c1

  def _moments(self, moments, normalize):
    min_cnt = self.acc_min_count
    tr_len, cl_len = self.trace_len, self.classifiers_len

    # 2D stat should return a separate piece of memory to avoid buffer corruption
    # while normalizing coskeweness and higher moments
    retm = np.empty((2, len(moments), tr_len), dtype=np.float64)

    for ii in range(cl_len):
      cls0 = (self.cls[:, ii] == 0)
      traces_cl = [self.traces[cls0], self.traces[~cls0]]
      n0, n1 = [len(ts) for ts in traces_cl]

      for _i in range(2):
        nn = n0 if _i == 0 else n1
        if nn < min_cnt:
          retm[_i, :len(moments)] = 0.0
          continue

        cm = retm[_i]
        tr_set = traces_cl[_i]
        raw = np.mean(tr_set, axis=0)
        for jj, m in enumerate(moments):
          if m == 1:
            cm[jj] = raw
          else:
            np.sum((tr_set - raw) ** m, axis=0, out=cm[jj])
            dscal(1.0 / nn, cm[jj])
            if m >= 3 and normalize:
              sd = np.sum((tr_set - raw) ** 2, axis=0)
              dscal(1.0 / nn, sd)
              np.sqrt(sd, out=sd)
              cm[jj] /= sd ** m

      yield retm[:, :len(moments)]


class bivar_2pass(_BivarNpassBase):
  @staticmethod
  def estimate_mem_size(tr_len, cl_len=1, moment=2):
    mem_base  = _BivarNpassBase.estimate_mem_size(tr_len, cl_len, moment)
    # One additional square for std normalization for moments > 2
    mem_ttstd = 0  if moment < 3 else tr_len + tr_len ** 2
    return mem_base + mem_ttstd * 8

  def memory_size(self):
    mem_base = super().memory_size()
    # One additional square for std normalization for moments > 2
    mem_ttstd = 0  if self.moment < 3 else self.trace_len + self.trace_len ** 2
    return mem_base + mem_ttstd * 8

  def _comoments(self, moments, normalize):
    min_cnt, C = self.acc_min_count, self._tmpsq
    tr_len, cl_len = self.trace_len, self.classifiers_len
    for ii in range(cl_len):
      for _i in range(2):
        mft = _sort_meanfree(self.traces[self.cls[:, ii] == _i], False)
        m, n = mft.shape
        retm = self._retm[_i]
        triuflatten = _triuflatten_gen(n)
        if m < min_cnt:
          retm[:] = 0
          continue

        for jj, (lm, rm) in enumerate(zip(*moments)):
          if lm == rm:
            dsyrk(mft ** lm,            C, b'U', b'N', 1.0, 0.0)
          else:
            dgemm(mft ** rm, mft ** lm, C, b'N', b'T', 1.0, 0.0)
          retm[jj] = triuflatten(C) / m
          if (lm + rm >= 3) and normalize:
            triuflatten = _triuflatten_gen(tr_len)
            sd = np.std(mft, axis=0)
            retm[jj] /= triuflatten(np.outer(sd ** lm, sd ** rm))

      yield self._retm[:, :len(moments[0])]


class bivar_txtbk(_BivarNpassBase):
  def __init__(self, trace_len, classifier_len, **kwargs):
    super().__init__(trace_len, classifier_len, **kwargs)
    m, n = self.acc_min_count, trace_len
    self._tri = np.empty((m, n * (n + 1) // 2))  # Handle the triangle dynamically

  @staticmethod
  def estimate_mem_size(tr_len, cl_len=1, moment=2):
    exp_traces = 10  # Expected number of traces in the input batch
    mem_base  = _BivarNpassBase.estimate_mem_size(tr_len, cl_len, moment)
    mem_tri   = exp_traces * tr_len * (tr_len + 1) // 2
    mem_ttstd = 0  if moment < 3 else tr_len + tr_len ** 2
    return mem_base + (mem_tri  + mem_ttstd)* 8

  def memory_size(self):
    mem_base = super().memory_size()
    # One additional square for std normalization for moments > 2
    mem_ttstd = 0  if self.moment < 3 else self.trace_len + self.trace_len ** 2
    return mem_base + mem_ttstd * 8

  def _realloc_tri(self, m):
    if len(self._tri) < m:
      # Greater batch received, dealloc the previous triangle and alloc new one
      n = self.traces.shape[1]
      self._tri = None
      self._tri = np.empty((m, n * (n + 1) // 2))
    return self._tri[:m]

  def _comoments(self, moments, normalize):
    min_cnt, C = self.acc_min_count, self._tmpsq
    tr_len, cl_len = self.trace_len, self.classifiers_len
    for ii in range(cl_len):
      cl_set = [(self.cls[:, ii] == 0)]
      cl_set.append(~(cl_set[0]))
      for _i in range(2):
        mft  = _sort_meanfree(self.traces[cl_set[_i]], False)
        m, n = mft.shape
        retm = self._retm[_i]
        tri  = self._realloc_tri(len(mft))
        if m < min_cnt:
          retm[:] = 0
          continue

        for jj, (lm, rm) in enumerate(zip(*moments)):
          tri = _uni2bivar(mft, C, tri, lm, rm) if lm == rm else _uni2bivar_neq(mft, C, tri, lm, rm)
          # Cut two-fold, to fit into memory
          np.mean(tri[:, :n // 2], 0, out=retm[jj, :n // 2])
          np.mean(tri[:, n // 2:], 0, out=retm[jj, n // 2:])
          if (lm + rm >= 3) and normalize:
            triuflatten = _triuflatten_gen(tr_len)
            sd = np.std(mft, axis=0)
            retm[jj] /= triuflatten(np.outer(sd ** lm, sd ** rm))

      yield self._retm[:, :len(moments[0])]


##################################### VTK #####################################
# Used for benchmarking only
if is_vtk_installed():
  import vtk
  from vtk.util.numpy_support import numpy_to_vtk as np2vtk, vtk_to_numpy as vtk2np


class bivar_vtk(object):
  """A class, using vtk multivariate kernel. Covariance ONLY!!!"""
  def __init__(self, tr_len, classifier_len, moment=2, **kwargs):
    self.moment = moment # In fact, only 1
    self.trace_len = tr_len
    self.classifiers_len = classifier_len
    self.acc_min_count = kwargs.pop('acc_min_count', 10)
    self._stats = [[vtk.vtkMultiCorrelativeStatistics() for i in range(2)] for _ in range(classifier_len)]
    self._counts = [[0, 0] for i in range(classifier_len)]
    self._retm   = np.empty((2, 2, tr_len * (tr_len + 1) // 2))
    for mcss in self._stats:
      for mcs in mcss:
        mcs.SetLearnOption(True)
        mcs.SetDeriveOption(True)
        mcs.SetAssessOption(False)
        mcs.SetTestOption(False)

  def memory_size(self):
    def _ms_helper(instl):
      return sum(i.GetActualMemorySize() for i in instl)

    intbl0  = [mcs.GetInputDataObject(0, 0) for mcss in self._stats for mcs in mcss]
    outtbl0 = [mcs.GetOutputDataObject(0)   for mcss in self._stats for mcs in mcss]
    outds1  = [mcs.GetOutputDataObject(1)   for mcss in self._stats for mcs in mcss]

    mem_ttest = 2 * 2 * self.trace_len * (self.trace_len + 1) // 2 * 8

    return sum(map(_ms_helper, (intbl0, outtbl0, outds1))) * 1024 + mem_ttest

  @property
  def total_count(self):
    return sum(self._counts[0])

  def update(self, traces, classifiers):
    traces = np.array(np.asarray(traces, dtype=np.float64).T, copy=True)
    cl_copy = np.asarray(classifiers, dtype=np.uint8)

    for i in range(len(cl_copy[0])):
      cls0 = (cl_copy[:, i] == 0)
      traces_cl = [traces[:, cls0], traces[:, ~cls0]]
      self._counts[i][0] += len(traces_cl[0][0])
      self._counts[i][1] += len(traces_cl[1][0])
      for mcs, tr_set in zip(self._stats[i], traces_cl):
        if tr_set.shape[1] == 0:
          continue
        tbl = vtk.vtkTable()
        tbl.SetNumberOfRows(len(tr_set[0]))
        mcs.SetInputData(tbl)
        for j, tr in enumerate(tr_set):
          vtk_arr = np2vtk(tr)
          vtk_arr.SetName('P{}'.format(j))
          tbl.AddColumn(vtk_arr)
          mcs.SetColumnStatus(vtk_arr.GetName(), 1)
        mcs.Update()

  def counts(self, i):
    # dbs = (mcs.GetOutputDataObject(1).GetBlock(1) for mcs in self._stats[i])
    # return [vtk2np(dd.GetColumn(1))[dd.GetNumberOfRows()-1] for dd in dbs]
    return self._counts[i]

  def moments(self, moments, normalize):
    maxm = np.max(moments)
    if self.moment * 2 < maxm:
      raise ValueError("The moment should be less or equal than indicated in constructor.")

    if normalize:
      raise NotImplementedError("Not implemented")

    tr_len, min_cnt = self.trace_len, self.acc_min_count
    triuflatten = _triuflatten_gen(tr_len)

    retm = np.empty((2, len(moments), tr_len), dtype=np.float64)

    for ii, mcss in enumerate(self._stats):
      for _i, mcs in enumerate(mcss):
        dd = mcs.GetOutputDataObject(1).GetBlock(1)
        n, cm = dd.GetNumberOfRows() - 1, retm[_i]
        nn = vtk2np(dd.GetColumn(1))[n]
        for jj, m in enumerate(moments):
          if m == 1:
            cm[:] = vtk2np(dd.GetColumn(1))[:n]
          else: # VTK cannot find moments higher
            for k in range(n):
              cm[jj, k] = vtk2np(dd.GetColumn(k + 2))[k]
            # Restore to ddof=0 to ensure correctness
            cm[jj, :] *= (nn - 1.0) / nn

    yield retm[:, :len(moments)]

  def comoments(self, moments, normalize):
    tr_len, min_cnt = self.trace_len, self.acc_min_count
    triuflatten = _triuflatten_gen(tr_len)

    for ii, mcss in enumerate(self._stats):
      for _i, mcs in enumerate(mcss):
        retm = self._retm[_i]
        dd = mcs.GetOutputDataObject(1).GetBlock(1)
        n = dd.GetNumberOfRows() - 1
        nn = vtk2np(dd.GetColumn(1))[n]
        cov_arr = np.empty((n, n))
        # VTK cannot find co-moments higher than covariance
        for jj, m in enumerate(moments):
          for k in range(n):
            cov_arr[k, :] = vtk2np(dd.GetColumn(k + 2))[:n]
          # Restore to ddof=0 to ensure correctness
          cov_arr *= (nn - 1.0) / nn
          retm[jj] = triuflatten(cov_arr.T)
    yield self._retm[:, :len(moments[0])]

########################## UNIVARIATE IMPLEMENTATIONS #########################

@cython.cfunc
@cython.returns('void')
@cython.locals(rm='double[::1]', sums1='double[::1]', n='int')
def _rsums2rmoms_acc0(sums1, rm, n):
  """Convert raw sums to raw moments"""
  daxpy(sums1, rm, -1.0)
  dscal(1.0 / n, rm)

@cython.cfunc
@cython.returns('void')
@cython.locals(rm1='double[::1]', sums1='double[::1]', n='int')
def _rsums2rmoms_acc1(sums1, rm1, n):
  """Convert raw sums to raw moments"""
  daxpy(sums1, rm1, 1.0 / n)


class univar_sum(_AccBase):
  def __init__(self, trace_len, classifier_len, **kwargs):
    super().__init__(trace_len, classifier_len, **kwargs)

    moment = self.moment
    exp_traces = 50  # Guesstimate the number of traces in the input batch
    self._dtype = kwargs.pop('acctype', np.float64)

    # Buffers for operations to avoid reallocations
    self._layout = np.empty((exp_traces, moment * trace_len + 1), dtype=self._dtype)
    self._retm   = np.empty((2, moment, trace_len), dtype=np.float64)
    self._buf    = np.empty((3, trace_len), dtype=self._retm.dtype)  # BLAS ops buffer

    # Accumulators
    self._accs   = np.zeros((classifier_len + 1, moment * trace_len + 1), dtype=self._dtype)

  @staticmethod
  def estimate_mem_size(tr_len, cl_len=1, moment=2):
    exp_traces = 50  # Expected number of traces in the input batch
    acc_dtype  = np.float64
    lytsz      = exp_traces * (moment * tr_len + 1)
    accssz     = (cl_len + 1) * (moment * tr_len + 1)
    bufsz      = 3 * tr_len
    ttsz       = 2 * moment * tr_len
    return (lytsz + accssz) * np.finfo(acc_dtype).bits // 8 + (bufsz + ttsz) * 8

  def memory_size(self):
    mem_class = sum(v.nbytes for v in vars(self).values() if isinstance(v, np.ndarray))
    mem_ttest = 0  # included in mem_class
    return mem_class + mem_ttest * 8

  def counts(self, i):
    c1 = self._accs[i + 1, 0]
    return self._accs[0, 0] - c1, c1

  def _moments(self, moments, normalize):
    maxm = np.max(moments)
    if 2 * self.moment < maxm:
      raise ValueError("The moment should be less or equal than indicated in constructor.")

    # TODO: slice cm up to maxm
    # Ensure that 2nd moment (variance) is in a separate piece of memory
    # to avoid buffer corruption while normalizing skeweness and higher moments
    cm, buf = self._retm, self._buf
    moments = [m - 1 for m in moments]
    nT, accT = self._accs[0, 0], self._accs[0, 1:].reshape(self._retm.shape[1:])

    for acc1 in self._accs[1:]:
      n1, acc1 = acc1[0], acc1[1:]
      nn = [nT - n1, n1]

      cm[0][:] = accT         #  cm0 = (accT - acc[ii]) / nn[0] # Has to be copied out
      cm[1][:] = 0.0          #  cm1 =         acc[ii]  / nn[1] # Has to be zeroed out

      # TODO: Shorten the accumulator to convert
      _rsums2rmoms_acc0(acc1.ravel(), cm[0].ravel(), max(1, nn[0]))
      _rsums2rmoms_acc1(acc1.ravel(), cm[1].ravel(), max(1, nn[1]))

      _rmoms2cmoms1D(cm[0], maxm, buf)
      _rmoms2cmoms1D(cm[1], maxm, buf)

      if maxm >= 3 and normalize:
        for _i in range(2):
          if nn[_i] > 0:
            sd = np.sqrt(cm[_i, 1])
            for i, cmi in enumerate(cm[_i, 2:maxm], 3):
              cmi[:] /= sd ** i

      yield self._retm[:, moments]

  def update(self, traces, classifiers):
    moment, batch_cnt = self.moment, len(traces)
    tr_len, cl_len = self.trace_len, self.classifiers_len

    self.total_count += batch_cnt

    if len(self._layout) < batch_cnt:
      self._layout = np.empty((batch_cnt, moment * tr_len + 1), dtype=self._layout.dtype)

    tr_lyt = self._layout[:batch_cnt]
    tr_lyt[:, 0] = 1
    tr_lyt_view = tr_lyt[:, 1:].reshape(batch_cnt, moment, tr_len)

    # Step 1: Create trace layout matrix up to moment degree
    tr_lyt_view[:, 0, :] = traces
    for j in range(1, moment):
      np.multiply(tr_lyt_view[:, 0, :], tr_lyt_view[:, j - 1, :], tr_lyt_view[:, j, :])

    # Step 2: Create classifier layout matrix
    _cl_layout = np.empty((len(classifiers), len(classifiers[0]) + 1), dtype=self._layout.dtype)
    cl_lyt = _cl_layout[:batch_cnt]
    cl_lyt[:, 0] = 1
    cl_lyt_view = cl_lyt[:, 1:]
    cl_lyt_view[:] = classifiers

    # Step 3: Multiply and accumulate
    dgemm(tr_lyt, cl_lyt, self._accs.reshape((cl_lyt.shape[1], -1)), b'N', b'T')


class univar_sum_detrend(univar_sum):
  def __init__(self, trace_len, classifier_len, **kwargs):
    super().__init__(trace_len, classifier_len, **kwargs)
    self._dbuf = np.zeros((1, self.trace_len), dtype=self._dtype)[1:]
    self._cbuf = np.asarray([bytearray(b'2') * classifier_len], dtype=np.uint8)[1:]  # 2 is non existing element
    self._detrend_cnt = kwargs.pop('detrend_cnt', 10)

  def update(self, traces, classifiers):
    tr_copy = np.asarray(traces, dtype=self._dtype)
    cl_copy = np.asarray(classifiers, dtype=np.uint8)

    if self._detrend_cnt > 0:
      db_part = tr_copy[:self._detrend_cnt]
      cl_part = cl_copy[:self._detrend_cnt]
      self._dbuf = np.vstack((self._dbuf, db_part))
      self._cbuf = np.vstack((self._cbuf, cl_part))
      self._detrend_cnt -= len(db_part)
      if self._detrend_cnt > 0:
        return  # Not enough input data for detrending
      if self._detrend_cnt == 0:
        tr_copy = np.vstack((self._dbuf, tr_copy[len(db_part):]))
        cl_copy = np.vstack((self._cbuf, cl_copy[len(cl_part):]))
        self._dbuf = np.mean(self._dbuf, axis=0)
        self._cbuf = None
    super().update(tr_copy - self._dbuf, cl_copy)

  def _moments(self, moments, normalize):
    m1idx = [i for i, m in enumerate(moments) if m == 1]
    has_m1 = len(m1idx) != 0
    dbuf_nonempty = len(self._dbuf) != 0
    for res in super()._moments(moments, normalize):
      if has_m1 and dbuf_nonempty:
        # Restore offset if mean requested
        res[:, m1idx] += self._dbuf
      yield res


################################ STAT TESTS ##################################

#@cython.cfunc
#@cython.returns('double[:, ::1]')
#@cython.locals(M = 'double[:, ::1]', m='int')
def _preprocvar(M, m, variance=None):
  if m > 2:
    M[1][:] = (M[1] - M[0] ** 2) / variance
    M[0][:] = M[0] / np.sqrt(variance)
  elif m == 2:
    M[1] = M[1] - M[0] ** 2
    M[0] = M[0]

  # Do nothing for m == 1
  return M

@cython.cfunc
@cython.returns('double[::1]')
@cython.locals(n0=cython.int, n1=cython.int, m0='double[::1]', m1='double[::1]', v0='double[::1]', v1='double[::1]')
def _ttest_ne(n0, n1, m0, m1, v0, v1):
  """ Non-equal vars """
  # This function modifies the buffers of given data
  # This _ttest adds about
  #   2-5% of CM time for ttest order 1
  # 0.8-1% of CM time for ttest order 2
  # Divide to (n-1) to compensate division to n in finding CMs
  daxpy(m1, m0, -1.0)
  dscal(1.0 / (n0 - 1), v0)
  daxpy(v1, v0, 1.0 / (n1 - 1))
  return m0 / np.sqrt(v0)

@cython.cfunc
#@cython.locals(n0 = cython.int, n1 = cython.int, m0 = 'double[::1]', m1 = 'double[::1]', v0 = 'double[::1]', v1 = 'double[::1]')
def _ttest_eq(n0, n1, m0, m1, v0, v1):
  """ Equal vars """
  # This function modifies the buffers of given data
  denom = (1.0 / n0 + 1.0 / n1) / (n0 + n1 - 2.0)
  daxpy(m1, m0, -1.0)
  dscal(n0 * denom, v0)
  daxpy(v1, v0, n1 * denom)
  return m0 / np.sqrt(v0)

def ttest(n0, n1, m10, m11, m20, m21, veq):
  """Compute t-test"""

  return _ttest_ne(n0, n1, m10, m11, m20, m21) if not veq else _ttest_eq(n0, n1, m10, m11, m20, m21)
