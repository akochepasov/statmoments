import sys

import numpy as np
import scipy

import statmoments
import statmoments.benchmark.benchlib as bl

from statmoments._native import is_vtk_installed


def bivar_benchmark(debug_run=False):
  params = bl.make_trace_ushort, bl.make_hypotheses, statmoments.Bivar

  kernels = [statmoments.bivar_sum, statmoments.bivar_2pass, statmoments.bivar_txtbk]
  if is_vtk_installed():
    kernels.append(statmoments.bivar_vtk)

  engines = [bl.EngineFactory(*params, kernel=k, moment=m)
             for k in kernels
             for m in [2, 3, 4]]

  traces_length, traces_count = (50, 300) if debug_run else (500, 5000)

  def test_filter(engine_name, tr_cnt, tr_len, _cl_len, batch_size):
    if engine_name == 'bivar_txtbk' and (batch_size is not None or tr_cnt >= 2500 or tr_len >= 2500):
      # txtbk is too slow for larger tests
      return False
    if engine_name != 'bivar_sum' and tr_cnt * tr_len >= 40000 * 5000:
      return False
    return True

  steps = [1, 2, 5, 10, 20]
  bl.benchmark(test_filter,
               [
                 ("trlen", [engines,
                            [traces_count],
                            [n * traces_length for n in steps],
                            [1],  # ttest class counts
                            [None]]),  # batch size
                 ("trcnt", [engines,
                            [n * traces_count for n in steps],
                            [traces_length],
                            [1],
                            [None]]),
                 ("batch", [engines,
                            [n * traces_count for n in steps],
                            [1000, 5000],
                            [1],
                            [500, 1000]])])


def run_benchmark():
  print("-" * 80)
  print(f"Python: {sys.version}")
  print(f"numpy : {np.__version__}")
  print(f"scipy : {scipy.__version__}")
  print("-" * 80)

  print(' ===== Bivar benchmark ==== ')
  np.random.seed(1)
  # np.seterr(all='ignore')
  bivar_benchmark()


if __name__ == '__main__':
  run_benchmark()
