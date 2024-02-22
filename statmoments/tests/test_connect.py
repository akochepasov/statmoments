import pytest

import numpy as np
import numpy.testing as nt

import statmoments

# np.random.seed(1)

eng1d_list = [
    statmoments.univar_sum,
    statmoments.univar_sum_detrend
]

eng2d_list = [
    statmoments.bivar_2pass,
    statmoments.bivar_txtbk,
    statmoments.bivar_sum,
    statmoments.bivar_cntr,
    statmoments.bivar_sum_detrend
]


@pytest.mark.parametrize("kernel1d", eng1d_list)
@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_moments1D(kernel1d, kernel2d):
  m, cl_len = 4, 1
  tr_count, tr_len = 50, 10
  traces0 = np.random.uniform(0, 100, (tr_count // 2, tr_len)).astype(np.uint8)
  traces1 = (traces0 + np.sin(traces0) * 3).astype(np.uint8)
  all_traces = np.vstack((traces0, traces1))
  all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
  all_cls[len(traces0):] = 1

  eng1d = statmoments.Univar(tr_len, cl_len, kernel=kernel1d, moment=2 * m, acc_min_count=3)
  eng1d.update(all_traces[:tr_count // 3], all_cls[:tr_count // 3])
  eng1d.update(all_traces[tr_count // 3:], all_cls[tr_count // 3:])
  mom1d = [next(eng1d.moments(mi)).copy() for mi in range(1, m + 1)]

  eng2d = statmoments.Bivar(tr_len,  cl_len, kernel=kernel2d, moment=m, acc_min_count=3)
  eng2d.update(all_traces[:tr_count // 2], all_cls[:tr_count // 2])
  eng2d.update(all_traces[tr_count // 2:], all_cls[tr_count // 2:])
  mom2d = [next(eng2d.moments(mi)).copy() for mi in range(1, m + 1)]

  # Ensure 1D moments are equal to the ones, calculated with 2D
  nt.assert_allclose(mom1d, mom2d)


@pytest.mark.parametrize("kernel1d", eng1d_list)
@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_moments2D_diag(kernel1d, kernel2d):
  m, cl_len = 4, 1
  tr_count, tr_len = 50, 10
  traces0 = np.random.uniform(0, 100, (tr_count // 2, tr_len)).astype(np.uint8)
  traces1 = (traces0 + np.sin(traces0) * 3).astype(np.uint8)
  all_traces = np.vstack((traces0, traces1))
  all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
  all_cls[len(traces0):] = 1

  eng1d = statmoments.Univar(tr_len, cl_len, kernel=kernel1d, moment=2 * m, acc_min_count=3)
  eng1d.update(all_traces[:tr_count // 2], all_cls[:tr_count // 2])
  eng1d.update(all_traces[tr_count // 2:], all_cls[tr_count // 2:])
  ll = [next(eng1d.moments(mi)).copy() for mi in range(2, 2 * m + 1, 2)]
  momdiag1d = np.asarray(ll)[:, :, 0, :]

  momdiag2d = []
  tmp = np.zeros((2, tr_len, tr_len))
  eng2d = statmoments.Bivar(tr_len,  cl_len, kernel=kernel2d, moment=m, acc_min_count=3)
  eng2d.update(all_traces[:tr_count // 3], all_cls[:tr_count // 3])
  eng2d.update(all_traces[tr_count // 3:], all_cls[tr_count // 3:])
  for mi in range(1, m + 1):
    for cm in eng2d.comoments(mi):
      tmp[0][np.triu_indices(tr_len)] = cm[0]
      tmp[1][np.triu_indices(tr_len)] = cm[1]
      momdiag2d.append(tmp.T.diagonal().copy())

  # Ensure even 1D moments are equal to 2D diagonals
  nt.assert_allclose(momdiag1d, momdiag2d)


@pytest.mark.parametrize("kernel1d", eng1d_list)
@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_trivial(kernel1d, kernel2d):
  m, cl_len = 4, 1
  traces0 = np.array([
      [2,  4,  5],
      [4,  6,  7],
      [7,  3,  9],
      [8,  5,  6],
      [2,  7,  3],
      [9,  1,  8],
      [4,  8,  0],
      [1,  2,  4],
  ], dtype=np.uint8)
  # traces0 = np.random.uniform(0, 20, (50, 10)).astype(np.uint8)
  traces1 = (traces0 + np.sin(traces0) * 3).astype(np.uint8)
  all_traces = np.vstack((traces0, traces1))
  tr_count, tr_len = all_traces.shape
  all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
  all_cls[len(traces0):] = 1

  eng1d = statmoments.Univar(tr_len, cl_len, kernel=kernel1d, moment=2 * m, acc_min_count=3)
  eng2d = statmoments.Bivar(tr_len,  cl_len, kernel=kernel2d, moment=m, acc_min_count=3)

  eng1d.update(all_traces, all_cls)
  eng2d.update(all_traces, all_cls)

  # Test for moments
  mom1d = [next(eng1d.moments(mi)).copy() for mi in range(1, m + 1)]
  mom2d = [next(eng2d.moments(mi)).copy() for mi in range(1, m + 1)]
  nt.assert_almost_equal(mom1d, mom2d)


# Entrance point
if __name__ == '__main__':
  pytest.main([__file__ + "::test_moments2D_diag"])
