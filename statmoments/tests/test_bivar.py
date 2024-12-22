import pytest

from itertools import combinations_with_replacement as _combu

import numpy as np
import numpy.testing as nt

from scipy.stats import skew, kurtosis, moment

import statmoments
from statmoments.common import meanfree, triu_flatten, uni2bivar

# np.random.seed(1)


def calc_mom(data, k, normalize=False):
  """ Central moments, numpy and scipy """
  # First 5 moments, with sigma-normalized skewness, kurtosis and 5th
  kurtosis_pearson = lambda x, **a: kurtosis(x, fisher=False, **a)  # noqa: E731
  stat_moment = lambda x, **a: moment(x, moment=k, **a) / np.std(data, **a)**k  # noqa: E731
  mom_funcs = [np.sum, np.mean, np.var, skew, kurtosis_pearson]
  mom_funcs += [stat_moment] * 4

  mom_res = mom_funcs[k](data, axis=0)

  return mom_res if k < 3 or normalize else mom_res * np.std(data, axis=0) ** k


def calc_comom(traces, mm, normalized):
  lm, rm = mm
  if lm == 1 and rm == 1:
    ecm2 = np.cov(traces.T, bias=True, ddof=0)
    ecm2 = triu_flatten(ecm2)
  else:
    emf = meanfree(traces)
    ecm2 = np.dot(emf.T ** lm, emf ** rm) / len(emf)
    ecm2 = triu_flatten(ecm2)
    ecmt = uni2bivar(traces, *mm, False)
    ecmt = np.mean(ecmt, axis=0)
    nt.assert_allclose(ecm2, ecmt, atol=1e-08, err_msg=f'error in comoment order {mm}')
    if normalized:
      sd = np.std(traces, axis=0)
      ecm2 /= triu_flatten(np.outer(sd ** lm, sd ** rm))
  return ecm2


def _ensure_comom(eng, traces0, traces1, normalized=True):
  n0, n1 = len(traces0), len(traces1)

  # Test counts
  nt.assert_equal(list(eng.counts(0)), [n0, n1])

  # Test 2D
  max_moment = eng.moment

  # Get list of all possible comoments
  lmm = list(_combu(range(1, max_moment + 1), r=2))

  # Get at once
  ecm = [(calc_comom(traces0, mm, normalized), calc_comom(traces1, mm, normalized)) for mm in lmm]
  ecm = list(zip(*ecm))
  acm = next(eng.comoments(moments=lmm))
  nt.assert_allclose(acm, ecm, atol=1e-08, err_msg='error in comoment array')

  # One by one
  for mm in lmm:
    ecm = [calc_comom(traces0, mm, normalized), calc_comom(traces1, mm, normalized)]
    acm = [cm[:, 0].copy() for cm in eng.comoments(moments=mm)][0]  # Only one classifier
    nt.assert_allclose(acm, ecm, atol=1e-08, err_msg=f'error in comoment order {mm}')


def _ensure_mom(eng, traces0, traces1, normalized=True):
  n0, n1 = len(traces0), len(traces1)

  # Test counts
  nt.assert_equal(list(eng.counts(0)), [n0, n1])

  # Test 1D
  max_moment = eng.moment * eng.dim

  # Raw moment
  m = 1
  emom = [calc_mom(traces0, m, normalized), calc_mom(traces1, m, normalized)]
  amom = [_m[:, 0].copy() for _m in eng.moments(moments=m)][0]  # Only one classifier
  nt.assert_allclose(emom, amom, atol=1e-07, err_msg='error in engine {} moment order {}'.format(eng._impl, m))

  # Central moment
  m = 2
  emom = [calc_mom(traces0, m, normalized), calc_mom(traces1, m, normalized)]
  amom = [_m[:, 0].copy() for _m in eng.moments(moments=[m])][0]  # Only one classifier
  nt.assert_allclose(emom, amom, err_msg='error in engine {} moment order {}'.format(eng._impl, m))

  # Standartized moments
  for m in range(3, max_moment + 1):
    emom = [calc_mom(traces0, m, normalized), calc_mom(traces1, m, normalized)]
    amom = [_m[:, 0].copy() for _m in eng.moments(moments=m)][0]  # Only one classifier
    nt.assert_allclose(emom, amom, atol=1e-07, err_msg='error in engine {} moment order {}'.format(eng._impl, m))

  mlist = list(range(max_moment - 1, max_moment + 1))
  amom = next(eng.moments(moments=mlist))
  for i, mm01 in enumerate(amom.transpose(1, 0, 2), np.min(mlist)):
    emom = [calc_mom(traces0, i, normalized), calc_mom(traces1, i, normalized)]
    nt.assert_allclose(emom, mm01, atol=1e-07, err_msg='error in engine {} moment order {}'.format(eng._impl, m))


eng2d_list = [
    statmoments.bivar_2pass,
    statmoments.bivar_txtbk,
    statmoments.bivar_sum,
    statmoments.bivar_cntr,
    statmoments.bivar_sum_detrend
]


@pytest.mark.parametrize("m", [2, 4])
@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_init(kernel2d, m):
  engine = statmoments.Bivar(10, 16, m, kernel=kernel2d)
  assert engine.total_count == 0

  amoms = [m.copy() for m in engine.moments()]
  assert np.max(np.abs(amoms)) < 1e-16
  assert len(amoms) == 16

  with nt.suppress_warnings() as sup:
    # Commoments issue warnings
    sup.filter(RuntimeWarning, "Mean of empty slice")
    sup.filter(RuntimeWarning, "divide by zero")
    sup.filter(RuntimeWarning, "invalid value")

    acomoms = [m.copy() for m in engine.comoments()]
    assert np.max(np.abs(acomoms)) < 1e-16
    assert len(acomoms) == 16

    att = [tt.copy() for tt in statmoments.stattests.ttests(engine)]
    assert np.max(np.abs(att)) < 1e-16
    assert len(att) == 16


@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_est_krnl_memsize(kernel2d):
  tr_len, cl_len = 10000, 1

  ms = kernel2d.estimate_mem_size(tr_len, cl_len, moment=2)
  assert ms > 1024


def test_est_obj_memsize():
  tr_len, cl_len = 10000, 1
  ms = statmoments.Bivar.estimate_mem_size(tr_len, cl_len, moment=4)
  assert ms > 10 * 1024 * 1024


@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_act_memory_size(kernel2d):
  moment = 4
  tr_len, cl_len = 1000, 2

  ems = statmoments.Bivar.estimate_mem_size(tr_len, cl_len, moment, kernel=kernel2d)
  ams = statmoments.Bivar(tr_len, cl_len, kernel=kernel2d, moment=moment).memory_size

  assert abs(ems - ams) <= 1 << 20  # The memory estimation is within 1MB of actual size


@pytest.mark.parametrize("kernel2d", eng2d_list)
@pytest.mark.parametrize("normalized", [False, True])
def test_comoments_all_kernels(kernel2d, normalized):
  tr_len, cl_len = 3, 1
  n0, n1 = 114, 180
  traces0 = np.random.randint(0, 256, (n0, tr_len))
  traces1 = np.random.randint(0, 256, (n1, tr_len))
  eng = statmoments.Bivar(tr_len, cl_len, kernel=kernel2d, moment=4, normalize=normalized)

  eng.update(traces0, ['0'] * n0)
  eng.update(traces1, [b'1'] * n1)

  _ensure_mom(eng, traces0, traces1, normalized)
  _ensure_comom(eng, traces0, traces1, normalized)


@pytest.fixture
def trivial_traces():
  traces0 = np.array([
      [2,  4,   5],
      [4,  6,  -2],
      [7,  3,   0],
      [8,  5,   1],
      [2,  7,   6],
      [9,  1,   2],
      [4,  8,  -4],
      [1,  2,   8],
  ], dtype=np.int8)
  # traces0 = np.random.uniform(0, 20, (50, 10)).astype(np.int8)
  traces1 = (traces0 + np.sin(traces0) * 3).astype(np.int8)
  traces0[:, 0] = (3 * traces0[:, 1] / 2 + 20).astype(np.int8)
  return traces0, traces1


@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_trivial(trivial_traces, kernel2d):
  m, cl_len = 4, 1
  traces0, traces1 = trivial_traces

  all_traces = np.vstack((traces0, traces1))
  tr_count, tr_len = all_traces.shape
  all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
  all_cls[len(traces0):] = 1

  eng = statmoments.Bivar(tr_len, cl_len, m, kernel=kernel2d, acc_min_count=3)

  # mem10 = statmoments.bivar_sum.estimate_mem_size(all_traces.shape[1],     cl_len, m)
  # mem11 = statmoments.bivar_sum_mix.estimate_mem_size(all_traces.shape[1], cl_len, m)
  # mem12 = statmoments.bivar_2pass.estimate_mem_size(all_traces.shape[1],   cl_len, m)
  # mem13 = statmoments.bivar_txtbk.estimate_mem_size(all_traces.shape[1],   cl_len, m)

  # Pass one batch after another
  # eng.update(all_traces[:6], all_cls[:6])
  # eng.update(all_traces[6:], all_cls[6:])
  eng.update(all_traces, all_cls)  # VTK cannot learn streamingly

  _ensure_mom(eng, traces0, traces1)
  _ensure_comom(eng, traces0, traces1)

  # mem00 = engines[0].memory_size  # sum
  # mem01 = engines[1].memory_size  # 2pass mf mul
  # mem02 = engines[2].memory_size  # txtbk


# Entrance point
if __name__ == '__main__':
  pytest.main(["-v", __file__ + "::test_init"])
