
import unittest
import functools

import numpy as np
import numpy.testing as nt

from scipy.stats import skew, kurtosis

import statmoments

from statmoments.tests.test_stattest import wttest
from statmoments._statmoments_impl import meanfree


def all_engines_1d(tr_len, cl_len, **kwargs):
  return [
      statmoments.Univar(tr_len, cl_len, kernel=statmoments.univar_sum,             **kwargs),
      statmoments.Univar(tr_len, cl_len, kernel=statmoments.univar_sum_detrend,     **kwargs),
  ]


def kurtosis_pearson(x, **args):
  return kurtosis(x, fisher=False, **args)


def calc_mom(traces, m, normalized):
  # First 4 moments, with sigma-normalized skewness and kurtosis
  if m < 5:
    mom_list = [None, np.mean, np.var, skew, kurtosis_pearson]
    mom_res = mom_list[m](traces, axis=0)
  else:
    # Just in case
    mom_res = np.sum(traces, axis=0) / len(traces)
    mom_res = np.sum((traces - mom_res) ** m, axis=0) / len(traces)
    mom_res /= np.std(traces, axis=0) ** m

  if m >= 3 and not normalized:
    mom_res *= np.std(traces, axis=0) ** m

  return mom_res


def _ensure_mom(engine, traces0, traces1, normalized, **kwargs):
  nt.assert_equal(engine.total_count, len(traces0) + len(traces1))

  for m in range(1, engine.moment + 1):
    emom = [calc_mom(traces0, m, normalized), calc_mom(traces1, m, normalized)]
    amom = ((em[0][0], em[1][0]) for em in engine.moments(m))
    amom = [em[0].copy() for em in zip(*amom)]
    nt.assert_allclose(amom, emom, atol=1e-08, err_msg='error in moment order {}'.format(m))


class Test_univar(unittest.TestCase):
  def test_init(self):
    engine = statmoments.Univar(10, 16, 2)

    self.assertEqual(engine.total_count, 0)

    amoms = [m.copy() for m in engine.moments(1)]
    self.assertEqual(np.max(np.abs(amoms)), 0)
    self.assertEqual(len(amoms), 16)

  def test_engine_choice(self):
    engine1 = statmoments.Univar(10, 16, 1)
    engine3 = statmoments.Univar(10, 16, 3)

    self.assertTrue(isinstance(engine1._impl, statmoments.univar_sum))
    self.assertTrue(isinstance(engine3._impl, statmoments.univar_sum_detrend))

  def _test_first4_moments(self, normalized):
    moment = 4
    tr_len, cl_len = 5, 1
    n0, n1 = 789, 1235

    hi, lo = -100, 100
    traces0 = np.random.randint(hi, lo, (n0, tr_len))
    traces1 = np.random.randint(hi, lo, (n1, tr_len))

    engines = all_engines_1d(tr_len, cl_len, moment=moment, normalize=normalized)
    for eng in engines:
      eng.update(traces0, ['0'] * n0)
      eng.update(traces1, ['1'] * n1)

    for eng in engines:
      _ensure_mom(eng, traces0, traces1, normalized)

  def test_first4_moments_non_normalized(self):
    self._test_first4_moments(False)

  def test_first4_moments_normalized(self):
    self._test_first4_moments(True)

  def test_nist_skeweness(self):
    # From https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/skewness.htm
    traces0 = [62.3,42.0,47.3,54.8,48.5,61.3,43.1,55.6,51.3,46.3,61.8,66.1,53.5,40.7,46.0,54.0,57.0,57.0,45.1,59.1]  # noqa:E231, E501
    traces0 = [[e] for e in traces0]  # make 2D, 1 sample point, N observations

    engines = all_engines_1d(1, 1, moment=3, normalize=True)
    for eng in engines:
      eng.update(traces0, ['0'] * len(traces0))

    skew_nist = calc_mom(traces0, 3, True)
    nt.assert_almost_equal(skew_nist, 0.0329, 4)
    askew = [m[0][0][0].copy() for eng in engines for m in eng.moments(3)]
    nt.assert_almost_equal(askew, 0.0329, 4)

  def test_trivial(self):
    m, cl_len = 6, 1
    traces0 = np.array([
        [2,  4,  5],
        [4,  6,  7],
        [7,  3,  9],
        [8,  5,  6],
        [2,  7,  3],
        [9,  1,  8],
        [4,  8,  4],
        [1,  2,  0],
    ], dtype=np.uint8)
    # traces0 = np.random.uniform(0, 20, (50, 1000)).astype(np.uint8)
    traces1 = (traces0 + np.sin(traces0) * 3).astype(np.uint8)
    all_traces = np.vstack((traces0, traces1))
    tr_count, tr_len = all_traces.shape
    all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
    all_cls[len(traces0):] = 1

    normalize = False
    engines = all_engines_1d(tr_len, cl_len, moment=m, acc_min_count=3, normalize=normalize)

    for eng in engines:
      eng.update(traces0, ['0'] * len(traces0))
      eng.update(traces1, [[1]] * len(traces1))

    for eng in engines:
      _ensure_mom(eng, traces0, traces1, normalize)

    # mem10 = statmoments.univar_sum.estimate_mem_size(all_traces.shape[1], cl_len, m)

    # Testing partial update
    # for eng in engines:
    #  eng.update(all_traces[:6], all_cls[:6])
    #  eng.update(all_traces[6:], all_cls[6:])

    # for eng in engines:
    #  eng.update(all_traces, all_cls)

    # all_traces = traces0 = traces1 = None

    ett3 = wttest((meanfree(traces0) / np.std(traces0, axis=0))**3, (meanfree(traces1) / np.std(traces1, axis=0))**3)
    tt30 = next(statmoments.stattests.ttests(engines[0])).copy()
    tt31 = next(statmoments.stattests.ttests(engines[1])).copy()
    nt.assert_allclose(ett3, tt30)
    nt.assert_allclose(tt31, tt30)

    # mem00 = engines[0].memory_size  # sum

    att = [next(statmoments.stattests.ttests(eng)).copy() for eng in engines]

    test_all = functools.partial(nt.assert_allclose, att[0])
    list(map(test_all, att))


def run_one():
  tsuite = unittest.TestSuite()
  tsuite.addTest(Test_univar('test_nist_skeweness'))
  unittest.TextTestRunner(verbosity=2).run(tsuite)


# Entrance point
if __name__ == '__main__':
  # run_one()
  unittest.main(argv=['first-arg-is-ignored'])
