import sys

import numpy as np
import scipy

import statmoments
import statmoments.benchmark.benchlib as bl

# Setup
debug_run = False
np.random.seed(1)
np.seterr(all='ignore')


def have_vtk():
  # try:
  #   import vtk
  #   return True
  # except ModuleNotFoundError:
    return False


def bivar_benchmark():
  params = bl.make_trace_ushort, bl.make_hypotheses, statmoments.Bivar

  kernels = [statmoments.bivar_sum, statmoments.bivar_2pass, statmoments.bivar_txtbk]
  if have_vtk():
    kernels.append(statmoments.bivar_vtk)

  engines = {(k.__name__, m): bl.EngineFactory(*params, kernel=k, moment=m)
             for k in kernels
             for m in [2, 3, 4]}

  print(' ===== Bivar benchmark ==== ')
  print('\nVarying trace lengths.')
  cl_len = 1
  tr_len, tr_cnt = (50, 300) if debug_run else (500, 5000)
  bl.benchmark([(f"bivar_m{moment}", [[engine], [tr_cnt], [n * tr_len], [cl_len]])
                for n in [1, 2, 5, 10, 20]
                for (impl, moment), engine in engines.items()
                if not (impl == "bivar_txtbk" and n >= 4)])

  print('\nVarying trace count.')
  cl_len = 1
  tr_len, tr_cnt = (50, 60) if debug_run else (1000, 5000)
  bl.benchmark([(f"bivar_m{moment}", [[engine], [n * tr_cnt], [tr_len], [cl_len]])
                for n in [1, 2, 5, 10, 20]
                for (impl, moment), engine in engines.items()
                if not (impl == "bivar_txtbk" and n >= 4)])


def run_benchmark():
  print("-" * 80)
  print(f"Python: {sys.version}")
  print(f"numpy : {np.__version__}")
  print(f"scipy : {scipy.__version__}")
  print("-" * 80)

  bivar_benchmark()


if __name__ == '__main__':
  run_benchmark()
