import pytest

import functools
from itertools import combinations_with_replacement as _combu

import numpy as np
import numpy.testing as nt
from numpy.random import normal as _rand

from scipy.stats import ttest_ind, skew, kurtosis, moment

import statmoments
from statmoments.common import meanfree, triu_flatten, uni2bivar


def mom_3pass(data, k, normalize=False):
  """ Central moments, 3 pass """
  kwargs = {'axis': 0}
  if k == 1:
    x = data
  else:
    x = meanfree(data)
    if k >= 3 and normalize:
      x = x / np.std(data, **kwargs)
    x = x ** k
  return np.mean(x, **kwargs)


def mom_libs(data, k, normalize=False):
  """ Central moments, numpy and scipy """
  # First 5 moments, with sigma-normalized skewness, kurtosis and 5th
  kurtosis_pearson = lambda x, **a: kurtosis(x, fisher=False, **a)  # noqa: E731
  stat_moment = lambda x, **a: moment(x, moment=k, **a) / np.std(data, **a)**k  # noqa: E731
  mom_funcs = [np.sum, np.mean, np.var, skew, kurtosis_pearson]
  mom_funcs += [stat_moment] * 4

  mom_res = mom_funcs[k](data, axis=0)

  return mom_res if k < 3 or normalize else mom_res * np.std(data, axis=0) ** k


def mom_sums(data, mom, normalize=False):
  """ Central moments, by definition """
  kwargs = {'axis': 0}
  data_mean = np.mean(data, **kwargs)
  if mom == 1:
    return data_mean

  res = np.sum((data - data_mean)**mom, **kwargs) / len(data)
  if normalize and mom >= 3:
    res = res / np.std(data, **kwargs)**mom
  return res


def wttest(tr0, tr1):
  """ t-test shortcut """
  return ttest_ind(tr0, tr1, equal_var=False).statistic


def find_maxtt(traces_gen, eng, iter_cnt=10):
  traces = np.vstack(traces_gen())
  classifiers = ['0'] * (len(traces) // 2) + ['1'] * (len(traces) // 2)

  for _ in range(iter_cnt):
      traces = np.vstack(list(traces_gen()))
      eng.update(traces, classifiers)

  act_tt = [tt.copy() for tt in statmoments.stattests.ttests(eng)]
  return [np.nanmax(np.abs(np.vstack(act_tt)), axis=1)]


def ensure_ttest_1d(eng, traces0, traces1):
  n0, n1 = len(traces0), len(traces1)

  # Test scalar
  nt.assert_equal(eng.total_count, n0 + n1)

  # Test 1D
  max_ttmoment = eng.moment // 2

  # t-test order 1
  exp_tt = wttest(traces0, traces1)
  act_tt = [tt.copy() for tt in statmoments.stattests.ttests(eng, moment=1, dim=1)]
  nt.assert_allclose(exp_tt, act_tt[0])
  nt.assert_allclose(-1 * exp_tt, act_tt[1])

  if max_ttmoment < 2:
    return

  # t-test order 2
  exp_tt = wttest(meanfree(traces0)**2, meanfree(traces1)**2)
  act_tt = [tt.copy() for tt in statmoments.stattests.ttests(eng, moment=2, dim=1)]
  nt.assert_allclose(exp_tt, act_tt[0])
  nt.assert_allclose(-1 * exp_tt, act_tt[1])

  if max_ttmoment < 3:
    return

  # t-test orders 3 and higher
  for m in range(3, max_ttmoment + 1):
    trstd0 = (meanfree(traces0) / np.std(traces0, axis=0))**m
    trstd1 = (meanfree(traces1) / np.std(traces1, axis=0))**m
    exp_tt = wttest(trstd0, trstd1)
    act_tt = [tt.copy() for tt in statmoments.stattests.ttests(eng, moment=m, dim=1)]
    nt.assert_almost_equal(exp_tt, act_tt[0], decimal=5)
    nt.assert_almost_equal(-1 * exp_tt, act_tt[1], decimal=5)


def ensure_ttest_2d(eng, traces0, traces1):
  n0, n1 = len(traces0), len(traces1)

  # Test scalar
  nt.assert_equal(eng.total_count, n0 + n1)

  # Test 2D
  max_moment = eng.moment // 2
  for mm in _combu(range(1, max_moment + 1), r=2):
    ett = wttest(uni2bivar(traces0, *mm), uni2bivar(traces1, *mm))
    att = [_tt.copy() for _tt in statmoments.stattests.ttests(eng, moment=mm)]
    nt.assert_allclose(att[0], ett, err_msg=f'Error in ttest order {mm}')
    nt.assert_allclose(att[1], -1 * ett, err_msg=f'Error in ttest order {mm}')
    if np.sum(mm) >= 3:
      ettnorm = wttest(uni2bivar(traces0, *mm), uni2bivar(traces1, *mm))
      nt.assert_almost_equal(att[0], ettnorm, err_msg=f'Error in normalized ttest order {mm}')
      nt.assert_almost_equal(att[1], -1 * ettnorm, err_msg=f'Error in normalized ttest order {mm}')


eng2d_list = [
    statmoments.bivar_txtbk,
    statmoments.bivar_2pass,
    statmoments.bivar_sum,
    statmoments.bivar_cntr,
    statmoments.bivar_sum_detrend
]

eng1d_list = [
    statmoments.univar_sum,
    statmoments.univar_sum_detrend
]


@pytest.mark.parametrize("kernel1d", eng1d_list)
def test_init1(kernel1d):
  engine = statmoments.Univar(10, 16, 2, kernel=kernel1d)

  assert engine.total_count == 0

  att = [tt.copy() for tt in statmoments.stattests.ttests(engine)]
  assert np.max(np.abs(att)) == 0
  assert len(att) == 16


@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_init2(kernel2d):
  engine = statmoments.Bivar(10, 16, 2, kernel=kernel2d)

  assert engine.total_count == 0

  # TODO: Fix this
  # att = [tt.copy() for tt in statmoments.stattests.ttests(engine)]
  # assert np.max(np.abs(att)) == 0
  # assert len(att) == 16


@pytest.mark.parametrize("kernel1d", eng1d_list)
def test_ttest_1d(kernel1d):
  max_moment = 4
  tr_len, cl_len = 5, 2
  n0, n1 = 1017, 1124
  traces0 = np.random.randint(0, 256, (n0, tr_len))
  # Insert different distribution into some points of one batch
  traces0[:, 2:4] = np.random.normal(30, 7, (n0, 2)).astype(traces0.dtype)
  traces1 = np.random.randint(0, 256, (n1, tr_len))
  eng = statmoments.Univar(tr_len, cl_len, moment=2 * max_moment, kernel=kernel1d)

  eng.update(traces0, [b'01'] * n0)
  eng.update(traces1, [[1, 0]] * n1)

  ensure_ttest_1d(eng, traces0, traces1)

  # Ensure t-test finds stat diff
  # Find different means
  for tt1 in statmoments.stattests.ttests(eng, moment=1):
    nt.assert_array_less(np.abs(tt1[0:2]), 3.5)
    assert np.all(np.abs(tt1[2:4]) > 30)
  # Find different vars
  for tt2 in statmoments.stattests.ttests(eng, moment=2):
    nt.assert_array_less(np.abs(tt2[0:2]), 3.5)
    assert np.all(np.abs(tt2[2:4]) > 30)
  # Find no different skews
  for tt3 in statmoments.stattests.ttests(eng, moment=3):
    nt.assert_array_less(np.abs(tt3), 2.5)
  # Find different kurtoses
  for tt4 in statmoments.stattests.ttests(eng, moment=4):
    nt.assert_array_less(np.abs(tt4[0:2]), 3.5)
    assert np.all(np.abs(tt4[2:4]) > 2.5)


# textbk is very slow
@pytest.mark.parametrize("kernel2d",
                         [e for e in eng2d_list if e is not statmoments.bivar_txtbk])
def test_ttest_2d(kernel2d):
  max_moment = 4
  tr_len, cl_len = 4, 2
  n0, n1 = 843, 934
  traces0 = np.random.normal(100, 20, (n0, tr_len))
  traces1 = np.random.normal(100, 20, (n1, tr_len))
  # Insert co-dependence to some point of one batch
  traces0[:, 2] = (3 * traces0[:, 3] / 2 - 10)
  eng = statmoments.Bivar(tr_len, cl_len, moment=max_moment, kernel=kernel2d)

  eng.update(traces0, [[0, 1]] * n0)
  eng.update(traces1, ['10'] * n1)

  ensure_ttest_1d(eng, traces0, traces1)
  ensure_ttest_2d(eng, traces0, traces1)

  # Ensure t-test finds the inserted correlation
  # Find different covars
  for tt2 in statmoments.stattests.ttests(eng, moment=(1, 1)):
    nt.assert_array_less(np.abs(tt2[0:7]), 3.5)
    nt.assert_array_less(8, np.abs(tt2[7:-1]))
  # Find no different co-skews
  for tt3 in statmoments.stattests.ttests(eng, moment=(1, 2)):
    nt.assert_array_less(np.abs(tt3), 3.5)
  # Find different co-kurtoses
  for tt4 in statmoments.stattests.ttests(eng, moment=(2, 2)):
    nt.assert_array_less(np.abs(tt4[0:8]), 2.5)
    nt.assert_array_less(5, np.abs(tt2[8]))


@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_mean_neg(kernel2d):
  # different mean, same variance
  cl_len = 1
  m0, m1 = -4.0, 4.0
  tr_len, tr_cnt = 5, 31
  traces_gen = lambda: [_rand(loc=m0, size=(tr_cnt, tr_len)), _rand(loc=m1, size=(tr_cnt, tr_len))]  # noqa: E731
  eng = statmoments.Bivar(tr_len, cl_len, kernel=kernel2d)

  maxsd = find_maxtt(traces_gen, eng)
  nt.assert_array_less(maxsd, 5)


@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_same_moms_neg(kernel2d):
  # same mean, same variance
  cl_len = 1
  s0, s1 = 2.0, 6.0  # noqa: F841 s1 not used
  tr_len, tr_cnt = 5, 53
  traces_gen = lambda: [_rand(scale=s0, size=(tr_cnt, tr_len)), _rand(scale=s0, size=(tr_cnt, tr_len))]  # noqa: E731
  eng = statmoments.Bivar(tr_len, cl_len, kernel=kernel2d)

  maxsd = find_maxtt(traces_gen, eng)
  nt.assert_array_less(maxsd, 5)


@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_2d_pos(kernel2d):
  cl_len = 1
  m0, m1 = -3.0, 3.0
  tr_len, tr_cnt = 5, 101
  _rand = np.random.normal
  traces_gen1 = lambda: [_rand(loc=m0, size=(tr_cnt, tr_len)), _rand(loc=m1, size=(tr_cnt, tr_len))]  # noqa: E731

  def traces_gen_corr():
    rnorm = traces_gen1()
    rnorm[1][:, 1] += rnorm[1][:, 0]
    return rnorm

  eng = statmoments.Bivar(tr_len, cl_len, kernel=kernel2d)

  maxsd = find_maxtt(traces_gen_corr, eng)
  nt.assert_array_less(5, maxsd)

  test_all = functools.partial(nt.assert_allclose, maxsd[0])
  list(map(test_all, maxsd))


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
  return traces0, traces1


@pytest.mark.parametrize("mom", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("normalized", [True, False])
def test_moments(trivial_traces, mom, normalized):
  for traces in trivial_traces:
    cm = [mom_3pass(traces, mom, normalized), mom_libs(traces, mom, normalized), mom_sums(traces, mom, normalized)]
    test_all = functools.partial(nt.assert_almost_equal, cm[0])
    list(map(test_all, cm))


@pytest.mark.parametrize("kernel1d", eng1d_list)
def test_nist(kernel1d):
  # ttest from https://www.itl.nist.gov/div898/handbook/eda/section3/eda353.htm
  # data  from https://www.itl.nist.gov/div898/handbook/eda/section3/eda3531.htm

  traces = [
      np.array([18,15,18,16,17,15,14,14,14,15,15,14,15,14,22,18,21,21,10,10,11,9,28,25,19,16,17,19,18,14,14,14,14,12,13,13,18,22,19,18,23,26,25,20,21,13,14,15,14,17,11,13,12,13,15,13,13,14,22,28,13,14,13,14,15,12,13,13,14,13,12,13,18,16,18,18,23,11,12,13,12,18,21,19,21,15,16,15,11,20,21,19,15,26,25,16,16,18,16,13,14,14,14,28,19,18,15,15,16,15,16,14,17,16,15,18,21,20,13,23,20,23,18,19,25,26,18,16,16,15,22,22,24,23,29,25,20,18,19,18,27,13,17,13,13,13,30,26,18,17,16,15,18,21,19,19,16,16,16,16,25,26,31,34,36,20,19,20,19,21,20,25,21,19,21,21,19,18,19,18,18,18,30,31,23,24,22,20,22,20,21,17,18,17,18,17,16,19,19,36,27,23,24,34,35,28,29,27,34,32,28,26,24,19,28,24,27,27,26,24,30,39,35,34,30,22,27,20,18,28,27,34,31,29,27,24,23,38,36,25,38,26,22,36,27,27,32,28,31])[:, np.newaxis],  # noqa: E231,E501
      np.array([24,27,27,25,31,35,24,19,28,23,27,20,22,18,20,31,32,31,32,24,26,29,24,24,33,33,32,28,19,32,34,26,30,22,22,33,39,36,28,27,21,24,30,34,32,38,37,30,31,37,32,47,41,45,34,33,24,32,39,35,32,37,38,34,34,32,33,32,25,24,37,31,36,36,34,38,32,38,32])[:, np.newaxis],  # noqa: E231,E501
  ]

  tr_len, cl_len, m = 1, 2, 1
  eng = statmoments.Univar(tr_len, cl_len, moment=2 * m, kernel=kernel1d)

  eng.update(traces[0], [[0, 1]] * len(traces[0]))
  eng.update(traces[1], [[1, 0]] * len(traces[1]))

  nt.assert_equal(list(eng.counts(0)), [249, 79])
  nt.assert_equal(list(eng.counts(1)), [79, 249])

  # Request both moments at once for both classifiers
  allmoms = [m.copy() for m in eng.moments((1, 2))]

  # Ensure means
  classifier, moment, rvar = 0, 0, 0
  nt.assert_allclose(allmoms[classifier][:, moment][:, rvar], [20.14458, 30.48101])
  classifier, moment, rvar = 1, 0, 0
  nt.assert_allclose(allmoms[classifier][:, moment][:, rvar], [30.48101, 20.14458])

  # Ensure standard deviations
  classifier, moment, rvar = 0, 1, 0
  nc = np.array(eng.counts(classifier))
  sd = np.sqrt(allmoms[classifier][:, moment][:, rvar] * nc / (nc - 1))
  nt.assert_almost_equal(sd, [6.41470, 6.10771], decimal=5)

  classifier, moment, rvar = 1, 1, 0
  nc = np.array(eng.counts(classifier))
  sd = np.sqrt(allmoms[classifier][:, moment][:, rvar] * nc / (nc - 1))
  nt.assert_almost_equal(sd, [6.10771, 6.41470], decimal=5)

  # Ensure t-test
  exptt_eq = ttest_ind(traces[0], traces[1], equal_var=True).statistic           # NIST
  alltt_eq = [tt.copy() for tt in statmoments.stattests.ttests(eng, equal_var=True)]
  nt.assert_almost_equal(exptt_eq, -12.62059, decimal=5)
  nt.assert_almost_equal(alltt_eq, [[-12.62059], [12.62059]], decimal=5)

  alltt_ne = [m.copy() for m in statmoments.stattests.ttests(eng, equal_var=False)]  # Welch
  exptt_ne = ttest_ind(traces[0], traces[1], equal_var=False).statistic
  nt.assert_almost_equal(exptt_ne, -12.94627, decimal=5)
  nt.assert_almost_equal(alltt_ne, [[-12.94627], [12.94627]], decimal=5)


@pytest.mark.parametrize("kernel1d", eng1d_list)
def test_trivial_1d(trivial_traces, kernel1d):
  m, cl_len = 4, 1
  traces0, traces1 = trivial_traces

  n0, n1 = len(traces0), len(traces1)

  eng = statmoments.Univar(len(traces0[0]), cl_len, moment=2 * m, kernel=kernel1d, acc_min_count=3)
  eng.update(traces0, ['0'] * n0)
  eng.update(traces1, ['1'] * n1)

  # tt 1 ord
  i = 1
  tt_exp = wttest(traces0, traces1)
  tt_act = next(statmoments.stattests.ttests(eng, moment=i)).copy()
  nt.assert_almost_equal(tt_exp, tt_act)

  # tt 2 ord
  i = 2
  tt_act = next(statmoments.stattests.ttests(eng, moment=i)).copy()

  mm2 = mom_3pass(traces0, i), mom_3pass(traces1, i)
  mm4 = mom_3pass(traces0, 2 * i), mom_3pass(traces1, 2 * i)
  mm4[0][:] = mm4[0] - mm2[0] ** 2
  mm4[1][:] = mm4[1] - mm2[1] ** 2
  tt_exp1 = (mm2[0] - mm2[1]) / np.sqrt(mm4[0] / (n0 - 1) + mm4[1] / (n1 - 1))

  mf_traces = [meanfree(traces0), meanfree(traces1)]
  tt_exp2 = wttest(mf_traces[0] ** 2, mf_traces[1] ** 2)

  nt.assert_almost_equal(tt_exp1, tt_exp2)
  nt.assert_almost_equal(tt_exp1, tt_act)

  # tt 3-4 ord, standartized
  std = [np.sqrt(mm2[0]), np.sqrt(mm2[1])]
  for i in [3, 4]:
    sm20 = (mom_3pass(traces0, 2 * i) - mom_3pass(traces0, i) ** 2) / std[0] ** (2 * i)
    sm21 = (mom_3pass(traces1, 2 * i) - mom_3pass(traces1, i) ** 2) / std[1] ** (2 * i)
    sm10 = mom_3pass(traces0, i, True)
    sm11 = mom_3pass(traces1, i, True)
    tt_exp1 = (sm10 - sm11) / np.sqrt(sm20 / (n0 - 1) + sm21 / (n1 - 1))
    tt_exp2 = wttest((mf_traces[0] / std[0]) ** i, (mf_traces[1] / std[1]) ** i)
    tt_act = next(statmoments.stattests.ttests(eng, moment=i)).copy()

    nt.assert_almost_equal(tt_exp1, tt_exp2)
    nt.assert_almost_equal(tt_exp1, tt_act)


@pytest.mark.parametrize("kernel2d", eng2d_list)
def test_trivial_2d(trivial_traces, kernel2d):
  m, cl_len = 4, 1
  traces0, traces1 = trivial_traces
  n0, n1 = len(traces0), len(traces1)

  eng = statmoments.Bivar(len(traces0[0]), cl_len, moment=m, kernel=kernel2d, acc_min_count=3)
  eng.update(traces0, ['0'] * n0)
  eng.update(traces1, ['1'] * n1)

  # tt 1-1 ord
  m1, m2 = 1, 1
  tt_exp = wttest(uni2bivar(traces0, m1, m2), uni2bivar(traces1, m1, m2))
  tt_act = next(statmoments.stattests.ttests(eng, moment=m1)).copy()
  nt.assert_almost_equal(tt_exp, tt_act)

  # tt 2-2 ord
  m1, m2 = 2, 2
  std2d0 = np.var(traces0, axis=0, ddof=0)  # std ** 2 = var
  std2d1 = np.var(traces1, axis=0, ddof=0)  # std ** 2 = var

  tr2d_std0 = uni2bivar(traces0, m1, m2, False) / triu_flatten(np.outer(std2d0, std2d0))
  tr2d_std1 = uni2bivar(traces1, m1, m2, False) / triu_flatten(np.outer(std2d1, std2d1))

  tt_exp = wttest(tr2d_std0, tr2d_std1)
  tt_act = next(statmoments.stattests.ttests(eng, moment=m1)).copy()

  nt.assert_almost_equal(tt_exp, tt_act)


# Entrance point
if __name__ == '__main__':
  pytest.main(["-v", __file__ + "::test_ttest_1d"])
