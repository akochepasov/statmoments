import unittest
from itertools import combinations_with_replacement as _combu

import numpy as np
import numpy.testing as nt

from scipy.stats import skew, kurtosis

import statmoments
from statmoments._statmoments_impl import meanfree, triu_flatten, uni2bivar

# np.random.seed(1)


def all_engines_2d(tr_len, cl_len, moment=2, **kwargs):
  return [
    statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_sum,         moment=moment, **kwargs),
    statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_cntr,        moment=moment, **kwargs),
    statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_txtbk,       moment=moment, **kwargs),
    statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_2pass,       moment=moment, **kwargs),
    statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_sum_detrend, moment=moment, **kwargs),
  ]


def kurtosis_pearson(x, **args):
  return kurtosis(x, fisher=False, **args)


def calc_mom(traces, m, normalized):
  # skewness and kurtosis here are sigma-normalized
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
    nt.assert_allclose(ecm2, ecmt, atol=1e-08, err_msg='error in comoment order {}'.format(mm))
    if normalized:
      sd = np.std(traces, axis=0)
      ecm2 /= triu_flatten(np.outer(sd ** lm, sd ** rm))
  return ecm2


def _ensure_comom(engines, traces0, traces1, normalized=True):
  n0, n1 = len(traces0), len(traces1)

  for eng in engines:
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
      nt.assert_allclose(acm, ecm, atol=1e-08, err_msg='error in comoment order {}'.format(mm))


def _ensure_mom(engines, traces0, traces1, normalized=True):
  n0, n1 = len(traces0), len(traces1)

  for eng in engines:
    # Test counts
    nt.assert_equal(list(eng.counts(0)), [n0, n1])

    # Test 1D
    max_moment = eng.moment * eng.dim

    # Raw moment
    m = 1
    emom = [calc_mom(traces0, m, normalized), calc_mom(traces1, m, normalized)]
    amom = [_m[:, 0].copy() for _m in eng.moments(moments=m)][0]  # Only one classifier
    nt.assert_allclose(emom, amom, atol=1e-07, err_msg='error in engine {} moment order {}'.format(eng._impl, m))

    if max_moment < 3:
      return

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


class Test_bivar(unittest.TestCase):
  def test_init(self):
    engine = statmoments.Bivar(10, 16)

    att = list(statmoments.stattests.ttests(engine))
    self.assertEqual(np.max(np.abs(att)), 0)

    self.assertEqual(len(att), 16)
    self.assertEqual(engine.total_count, 0)

  def test_proj_krnl_memsize(self):
    moment = 2
    tr_len, cl_len = 10000, 1

    kernels = [
      statmoments.bivar_sum,
      statmoments.bivar_cntr,
      statmoments.bivar_2pass,
    ]
    for kr in kernels:
      self.assertGreater(kr.estimate_mem_size(tr_len, cl_len, moment), 1024)

    ms = statmoments.Bivar.estimate_mem_size(tr_len, cl_len, moment=4)
    self.assertGreater(ms, 10 * 1024 * 1024)

  def test_comoments_all_kernels(self):
    tr_len, cl_len = 3, 1
    n0, n1 = 567, 890
    traces0 = np.random.randint(0, 256, (n0, tr_len))
    traces1 = np.random.randint(0, 256, (n1, tr_len))
    engines = all_engines_2d(tr_len, cl_len, moment=4)

    for eng in engines:
      eng.update(traces0, ['0'] * n0)
      eng.update(traces1, ['1'] * n1)

    _ensure_mom(engines, traces0, traces1)
    _ensure_comom(engines, traces0, traces1)

  def test_comoments_all_kernels_normalized(self):
    normalize = True
    tr_len, cl_len = 3, 1
    n0, n1 = 764, 980
    traces0 = np.random.randint(0, 256, (n0, tr_len))
    traces1 = np.random.randint(0, 256, (n1, tr_len))
    engines = all_engines_2d(tr_len, cl_len, moment=4, normalize=normalize)

    for eng in engines:
      eng.update(traces0, ['0'] * n0)
      eng.update(traces1, ['1'] * n1)

    _ensure_mom(engines, traces0, traces1, normalize)
    _ensure_comom(engines, traces0, traces1, normalize)


  def test_memory_size(self):
    moment = 4
    tr_len, cl_len = 1000, 2
    engines = all_engines_2d(tr_len, cl_len, moment=moment)

    act_memsz = []
    for eng in engines:
      ams = eng.memory_size
      act_memsz.append(ams)
    est_memsz = []
    for eng in engines:
      kernel_type = type(eng._impl)
      ems = statmoments.Bivar.estimate_mem_size(tr_len, cl_len, moment, kernel=kernel_type)
      est_memsz.append(ems)

    for m1, m2 in zip(act_memsz, est_memsz):
      self.assertLessEqual(abs(m1 - m2), 1 << 20)


  def test_trivial(self):
    m, cl_len = 4, 1
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

    all_traces = np.vstack((traces0, traces1))
    tr_count, tr_len = all_traces.shape
    all_cls = np.zeros((tr_count, cl_len), dtype=np.uint8)
    all_cls[len(traces0):] = 1

    engines = all_engines_2d(tr_len, cl_len, moment=m, acc_min_count=3)
    # engines = [statmoments.Bivar(tr_len, cl_len, kernel=statmoments.bivar_sum,      moment=m, acc_min_count=3)]

    # mem10 = statmoments.bivar_sum.estimate_mem_size(all_traces.shape[1],     cl_len, m)
    # mem11 = statmoments.bivar_sum_mix.estimate_mem_size(all_traces.shape[1], cl_len, m)
    # mem12 = statmoments.bivar_2pass.estimate_mem_size(all_traces.shape[1],   cl_len, m)
    # mem13 = statmoments.bivar_txtbk.estimate_mem_size(all_traces.shape[1],   cl_len, m)

    for eng in engines:
      # Pass one batch after another
      eng.update(all_traces[:6], all_cls[:6])
      eng.update(all_traces[6:], all_cls[6:])
      # eng.update(all_traces, all_cls)  # VTK cannot learn streamingly

    _ensure_mom(engines, traces0, traces1)
    _ensure_comom(engines, traces0, traces1)

    # mem00 = engines[0].memory_size  # sum
    # mem01 = engines[1].memory_size  # 2pass mf mul
    # mem02 = engines[2].memory_size  # txtbk


def run_one():
  tsuite = unittest.TestSuite()
  tsuite.addTest(Test_bivar('test_memory_size'))
  unittest.TextTestRunner(verbosity=2).run(tsuite)


# Entrance point
if __name__ == '__main__':
  run_one()
  # unittest.main(argv=['first-arg-is-ignored'])
