import pytest

import numpy as np
import numpy.testing as nt
from scipy.stats import skew, kurtosis

import statmoments
from statmoments.tests.test_stattest import wttest
from statmoments.common import meanfree


eng1d_list = [
    statmoments.univar_sum,
    statmoments.univar_sum_detrend
]


@pytest.mark.parametrize("krnl1d", eng1d_list)
def test_init(krnl1d):
  engine = statmoments.Univar(10, 16, 2, kernel=krnl1d)
  assert engine.total_count == 0
  amoms = [m.copy() for m in engine.moments(1)]
  assert np.max(np.abs(amoms)) == 0
  assert len(amoms) == 16


def test_engine_choice():
  engine1 = statmoments.Univar(10, 16, 1)
  engine3 = statmoments.Univar(10, 16, 3)

  assert isinstance(engine1._impl, statmoments.univar_sum)
  assert isinstance(engine3._impl, statmoments.univar_sum_detrend)


def calc_mom(traces, m, normalized):
  # First 4 moments, with sigma-normalized skewness and kurtosis
  if m < 5:
    kurtosis_pearson = lambda x, **a: kurtosis(x, fisher=False, **a)  # noqa: E731
    mom_list = [None, np.mean, np.var, skew, kurtosis_pearson]
    mom_res = mom_list[m](traces, axis=0)
  else:  # Just in case
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


@pytest.mark.parametrize("krnl1d", eng1d_list)
@pytest.mark.parametrize("normalized", [False, True])
def test_first4_moments(krnl1d, normalized):
  moment = 4
  tr_len, cl_len = 5, 1
  n0, n1 = 789, 1235
  eng = statmoments.Univar(tr_len, cl_len, moment, kernel=krnl1d, normalize=normalized)

  hi, lo = -100, 100
  traces0 = np.random.randint(hi, lo, (n0, tr_len))
  traces1 = np.random.randint(hi, lo, (n1, tr_len))
  eng.update(traces0, ['0'] * n0)
  eng.update(traces1, [[1]] * n1)

  _ensure_mom(eng, traces0, traces1, normalized)


# From https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/skewness.htm
@pytest.mark.parametrize("normalized", [True])
@pytest.mark.parametrize("krnl1d", eng1d_list)
def test_nist_skewness(krnl1d, normalized):
  moment = 3
  traces0 = [62.3,42.0,47.3,54.8,48.5,61.3,43.1,55.6,51.3,46.3,61.8,66.1,53.5,40.7,46.0,54.0,57.0,57.0,45.1,59.1]  # noqa:E231, E501
  traces0 = [[e] for e in traces0]  # make 2D, 1 sample point, N observations

  eng = statmoments.Univar(1, 1, moment, kernel=krnl1d, normalize=normalized)
  eng.update(traces0, ['0'] * len(traces0))

  skew_nist = calc_mom(traces0, moment, normalized)
  nt.assert_almost_equal(skew_nist, 0.0329, 4)
  askew = next(eng.moments(moment))[0][0][0].copy()
  nt.assert_almost_equal(askew, 0.0329, 4)


@pytest.fixture
def trivial_traces():
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
  return traces0, traces1


@pytest.mark.parametrize("krnl1d", eng1d_list)
@pytest.mark.parametrize("normalized", [False, True])
def test_trivial_bypack(trivial_traces, krnl1d, normalized):
  m, cl_len = 3, 1
  traces0, traces1 = trivial_traces
  tr_len = len(trivial_traces[0][0])
  eng = statmoments.Univar(tr_len, cl_len, 2 * m, kernel=krnl1d, normalize=normalized, acc_min_count=3)

  eng.update(traces0, ['0'] * len(traces0))
  eng.update(traces1, [[1]] * len(traces1))

  _ensure_mom(eng, traces0, traces1, normalized)

  # Ensure t-test
  ett3 = wttest((meanfree(traces0) / np.std(traces0, axis=0))**m, (meanfree(traces1) / np.std(traces1, axis=0))**m)
  att3 = next(statmoments.stattests.ttests(eng)).copy()
  nt.assert_allclose(ett3, att3)

  # mem10 = statmoments.univar_sum.estimate_mem_size(all_traces.shape[1], cl_len, m)


@pytest.mark.parametrize("krnl1d", eng1d_list)
@pytest.mark.parametrize("normalized", [False, True])
def test_trivial_partial(trivial_traces, krnl1d, normalized):
  m, cl_len = 3, 1
  tr_len = len(trivial_traces[0][0])
  eng = statmoments.Univar(tr_len, cl_len, 2 * m, kernel=krnl1d, normalize=normalized, acc_min_count=3)

  traces0, traces1 = trivial_traces
  all_traces = np.vstack(trivial_traces)
  tr_count, tr_len = all_traces.shape
  all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
  all_cls[len(traces0):] = 1
  eng.update(all_traces[:6], all_cls[:6])
  eng.update(all_traces[6:], all_cls[6:])

  # eng.update(all_traces, all_cls)  # The whole pack

  # all_traces = traces0 = traces1 = None
  # mem00 = eng.memory_size

  ett3 = wttest((meanfree(traces0) / np.std(traces0, axis=0))**m, (meanfree(traces1) / np.std(traces1, axis=0))**m)
  att3 = next(statmoments.stattests.ttests(eng)).copy()
  nt.assert_allclose(ett3, att3)


# Entrance point
if __name__ == '__main__':
  pytest.main(["-v", __file__ + "::test_trivial_bypack"])
