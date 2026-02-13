import unittest
import functools

import numpy as np
import numpy.testing as nt

import statmoments

# np.random.seed(1)


def engines1D(tr_len, cl_len, moment=2, **kwargs):
  return [
      statmoments.Univar(tr_len, cl_len, kernel=statmoments.univar_sum,    moment=moment, acc_min_count=3),
  ]


def engines2D(tr_len, cl_len, moment=2, **kwargs):
  return [
      statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_2pass,        moment=moment, acc_min_count=3),
      statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_txtbk,        moment=moment, acc_min_count=3),
      statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_sum,          moment=moment, acc_min_count=3),
      statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_cntr,         moment=moment, acc_min_count=3),
      statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_sum_detrend,  moment=moment, acc_min_count=3),
  ]


class Test_connect(unittest.TestCase):

  def test_moments1D(self):
    m, cl_len = 4, 1
    tr_count, tr_len = 50, 10
    traces0 = np.random.uniform(0, 100, (tr_count // 2, tr_len)).astype(np.uint8)
    traces1 = (traces0 + np.sin(traces0) * 3).astype(np.uint8)
    all_traces = np.vstack((traces0, traces1))
    all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
    all_cls[len(traces0):] = 1

    eng_1d = engines1D(tr_len, cl_len, m * 2)
    eng_2d = engines2D(tr_len, cl_len, m)

    for eng in eng_1d:
      eng.update(all_traces[:tr_count // 3], all_cls[:tr_count // 3])
      eng.update(all_traces[tr_count // 3:], all_cls[tr_count // 3:])

    for eng in eng_2d:
      eng.update(all_traces[:tr_count // 2], all_cls[:tr_count // 2])
      eng.update(all_traces[tr_count // 2:], all_cls[tr_count // 2:])

    # Ensure 1D moments are equal to the ones, calculated with 2D
    mom1d = []
    for eng in eng_1d:
      mom1d.append([next(eng.moments(mi)).copy() for mi in range(1, m + 1)])

    for eng in eng_2d:
      mom1d.append([next(eng.moments(mi)).copy() for mi in range(1, m + 1)])

    test_all = functools.partial(nt.assert_allclose, mom1d[0])
    list(map(test_all, mom1d))

  def test_moments2D_diag(self):
    m, cl_len = 4, 1
    tr_count, tr_len = 50, 10
    traces0 = np.random.uniform(0, 100, (tr_count // 2, tr_len)).astype(np.uint8)
    traces1 = (traces0 + np.sin(traces0) * 3).astype(np.uint8)
    all_traces = np.vstack((traces0, traces1))
    all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
    all_cls[len(traces0):] = 1

    eng_1d = engines1D(tr_len, cl_len, m * 2)
    eng_2d = engines2D(tr_len, cl_len, m)

    for eng in eng_1d:
      eng.update(all_traces[:tr_count // 2], all_cls[:tr_count // 2])
      eng.update(all_traces[tr_count // 2:], all_cls[tr_count // 2:])

    for eng in eng_2d:
      eng.update(all_traces[:tr_count // 3], all_cls[:tr_count // 3])
      eng.update(all_traces[tr_count // 3:], all_cls[tr_count // 3:])

    # Ensure even 1D moments are equal to 2D diagonals
    mom2d = []
    for eng in eng_1d:
      ll = [next(eng.moments(mi)).copy() for mi in range(2, 2 * m + 1, 2)]
      mom2d.append(np.asarray(ll)[:, :, 0, :])

    aa = np.zeros((2, tr_len, tr_len))
    for eng in eng_2d:
      mom2d.append([])
      for mi in range(1, m + 1):
        for cm in eng.comoments(mi):
          aa[0][np.triu_indices(tr_len)] = cm[0]
          aa[1][np.triu_indices(tr_len)] = cm[1]
          mom2d[-1].append(aa.T.diagonal().copy())

    test_all = functools.partial(nt.assert_allclose, mom2d[0])
    list(map(test_all, mom2d))

  def test_trivial(self):
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

    eng_1d = engines1D(tr_len, cl_len, m * 2)
    eng_2d = engines2D(tr_len, cl_len, m)

    for eng in eng_1d:
      eng.update(all_traces, all_cls)

    for eng in eng_2d:
      eng.update(all_traces, all_cls)

    # Test for moments
    mom1d = []
    for eng in eng_1d:
      mom1d.append([next(eng.moments(mi)).copy() for mi in range(1, m + 1)])

    for eng in eng_2d:
      mom1d.append([next(eng.moments(mi)).copy() for mi in range(1, m + 1)])

    test_all = functools.partial(nt.assert_almost_equal, mom1d[0])
    list(map(test_all, mom1d))


def run_one():
  tsuite = unittest.TestSuite()
  tsuite.addTest(Test_connect('test_trivial'))
  unittest.TextTestRunner(verbosity=2).run(tsuite)


# Entrance point
if __name__ == '__main__':
  # run_one()
  unittest.main(argv=['first-arg-is-ignored'])
