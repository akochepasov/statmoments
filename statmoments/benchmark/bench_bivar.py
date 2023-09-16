
import sys

import numpy as np
import scipy

import statmoments
import statmoments.benchmark.benchlib as bl

# Setup
np.random.seed(1)
np.seterr(all='ignore')


def bivar_benchmark():
  debug_run = True
  params = bl.make_trace_ushort, bl.make_hypotheses, statmoments.Bivar

  bivar_fctr11 = bl.engine_factory(*params, kernel=statmoments.bivar_sum)
  bivar_fctr12 = bl.engine_factory(*params, kernel=statmoments.bivar_2pass)
  bivar_fctr13 = bl.engine_factory(*params, kernel=statmoments.bivar_txtbk)
  bivar_fctr14 = bl.engine_factory(*params, kernel=statmoments.bivar_vtk)
  bivar_fctr21 = bl.engine_factory(*params, kernel=statmoments.bivar_sum,    moment=2)
  bivar_fctr22 = bl.engine_factory(*params, kernel=statmoments.bivar_2pass,  moment=2)
  bivar_fctr23 = bl.engine_factory(*params, kernel=statmoments.bivar_txtbk,  moment=2)
  bivar_fctr31 = bl.engine_factory(*params, kernel=statmoments.bivar_sum,    moment=3)
  bivar_fctr32 = bl.engine_factory(*params, kernel=statmoments.bivar_2pass,  moment=3)
  bivar_fctr33 = bl.engine_factory(*params, kernel=statmoments.bivar_txtbk,  moment=3)
  bivar_fctr41 = bl.engine_factory(*params, kernel=statmoments.bivar_sum,    moment=4)
  bivar_fctr42 = bl.engine_factory(*params, kernel=statmoments.bivar_2pass,  moment=4)
  bivar_fctr43 = bl.engine_factory(*params, kernel=statmoments.bivar_txtbk,  moment=4)

  print(' ===== Bivar benchmark ==== ')
  print('Bivar trace length benchmark')
  cl_len = 1
  tr_len, tr_cnt = (50, 61) if debug_run else (500, 1000)
  bl.benchmark([
    # ("bivar_m1", [[bivar_fctr11],    [5*tr_cnt],  [ 1*tr_len],  [cl_len]]),  # len  0.5k
    # ("bivar_m1", [[bivar_fctr11],    [5*tr_cnt],  [ 2*tr_len],  [cl_len]]),  # len  1.0k
    # ("bivar_m1", [[bivar_fctr11],    [5*tr_cnt],  [ 4*tr_len],  [cl_len]]),  # len  2.0k
    # ("bivar_m1", [[bivar_fctr11],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
    # ("bivar_m1", [[bivar_fctr11],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
    # ("bivar_m1", [[bivar_fctr11],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k

    # ("bivar_m1", [[bivar_fctr12],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m1", [[bivar_fctr12],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m1", [[bivar_fctr12],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m1", [[bivar_fctr12],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
    # ("bivar_m1", [[bivar_fctr12],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
    # ("bivar_m1", [[bivar_fctr12],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k

    # ("bivar_m1", [[bivar_fctr13],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m1", [[bivar_fctr13],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m1", [[bivar_fctr13],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m1", [[bivar_fctr13],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
#    ("bivar_m1", [[bivar_fctr13],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k # too large
#    ("bivar_m1", [[bivar_fctr13],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k # too large

    # ("bivar_m1", [[bivar_fctr14],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m1", [[bivar_fctr14],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m1", [[bivar_fctr14],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m1", [[bivar_fctr14],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
#    ("bivar_m1", [[bivar_fctr14],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
#    ("bivar_m1", [[bivar_fctr14],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k  # too slow

    # ("bivar_m2", [[bivar_fctr21],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m2", [[bivar_fctr21],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m2", [[bivar_fctr21],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    ("bivar_m2", [[bivar_fctr21],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
    # ("bivar_m2", [[bivar_fctr21],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
    # ("bivar_m2", [[bivar_fctr21],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k

    # ("bivar_m2", [[bivar_fctr22],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m2", [[bivar_fctr22],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m2", [[bivar_fctr22],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    ("bivar_m2", [[bivar_fctr22],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
    # ("bivar_m2", [[bivar_fctr22],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
    # ("bivar_m2", [[bivar_fctr22],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k

    # ("bivar_m2", [[bivar_fctr23],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m2", [[bivar_fctr23],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m2", [[bivar_fctr23],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m2", [[bivar_fctr23],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
#    ("bivar_m2", [[bivar_fctr23],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k # too large
#    ("bivar_m2", [[bivar_fctr23],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k # too large

    # ("bivar_m3", [[bivar_fctr31],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m3", [[bivar_fctr31],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m3", [[bivar_fctr31],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m3", [[bivar_fctr31],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
    # ("bivar_m3", [[bivar_fctr31],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
    # ("bivar_m3", [[bivar_fctr31],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k

    # ("bivar_m3", [[bivar_fctr32],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m3", [[bivar_fctr32],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m3", [[bivar_fctr32],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m3", [[bivar_fctr32],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
    # ("bivar_m3", [[bivar_fctr32],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
    # ("bivar_m3", [[bivar_fctr32],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k

    # ("bivar_m3", [[bivar_fctr33],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m3", [[bivar_fctr33],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m3", [[bivar_fctr33],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m3", [[bivar_fctr33],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
#    ("bivar_m3", [[bivar_fctr33],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k # too large
#    ("bivar_m3", [[bivar_fctr33],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k # too large

    # ("bivar_m4", [[bivar_fctr41],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m4", [[bivar_fctr41],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m4", [[bivar_fctr41],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m4", [[bivar_fctr41],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
    # ("bivar_m4", [[bivar_fctr41],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
    # ("bivar_m4", [[bivar_fctr41],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k

    # ("bivar_m4", [[bivar_fctr42],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m4", [[bivar_fctr42],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m4", [[bivar_fctr42],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m4", [[bivar_fctr42],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
    # ("bivar_m4", [[bivar_fctr42],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k
    # ("bivar_m4", [[bivar_fctr42],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k

    # ("bivar_m4", [[bivar_fctr43],    [5*tr_cnt],  [1*tr_len],   [cl_len]]),  # len  0.5k
    # ("bivar_m4", [[bivar_fctr43],    [5*tr_cnt],  [2*tr_len],   [cl_len]]),  # len  1.0k
    # ("bivar_m4", [[bivar_fctr43],    [5*tr_cnt],  [4*tr_len],   [cl_len]]),  # len  2.0k
    # ("bivar_m4", [[bivar_fctr43],    [5*tr_cnt],  [10*tr_len],  [cl_len]]),  # len  5.0k
#    ("bivar_m4", [[bivar_fctr43],    [5*tr_cnt],  [20*tr_len],  [cl_len]]),  # len 10.0k # too large
#    ("bivar_m4", [[bivar_fctr33],    [5*tr_cnt],  [40*tr_len],  [cl_len]]),  # len 20.0k # too large
  ])

  return  # don't benchmark counts
  print('Bivar trace count benchmark')
  cl_len = 1
  tr_len, tr_cnt = (50, 61) if debug_run else (1000, 5000)
  bl.benchmark([
    ("bivar_m1", [[bivar_fctr11],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m1", [[bivar_fctr11],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m1", [[bivar_fctr11],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m1", [[bivar_fctr11],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m1", [[bivar_fctr11],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m1", [[bivar_fctr11],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m1", [[bivar_fctr12],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m1", [[bivar_fctr12],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m1", [[bivar_fctr12],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m1", [[bivar_fctr12],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m1", [[bivar_fctr12],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m1", [[bivar_fctr12],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m1", [[bivar_fctr13],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m1", [[bivar_fctr13],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m1", [[bivar_fctr13],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m1", [[bivar_fctr13],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m1", [[bivar_fctr13],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
#    ("bivar_m1", [[bivar_fctr13],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k # Too big

    ("bivar_m1", [[bivar_fctr14],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m1", [[bivar_fctr14],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m1", [[bivar_fctr14],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m1", [[bivar_fctr14],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m1", [[bivar_fctr14],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m1", [[bivar_fctr14],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m2", [[bivar_fctr21],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m2", [[bivar_fctr21],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m2", [[bivar_fctr21],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m2", [[bivar_fctr21],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m2", [[bivar_fctr21],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m2", [[bivar_fctr21],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m2", [[bivar_fctr22],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m2", [[bivar_fctr22],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m2", [[bivar_fctr22],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m2", [[bivar_fctr22],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m2", [[bivar_fctr22],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m2", [[bivar_fctr22],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m2", [[bivar_fctr23],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m2", [[bivar_fctr23],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m2", [[bivar_fctr23],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m2", [[bivar_fctr23],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m2", [[bivar_fctr23],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
#   ("bivar_m2", [[bivar_fctr23],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m3", [[bivar_fctr31],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m3", [[bivar_fctr31],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m3", [[bivar_fctr31],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m3", [[bivar_fctr31],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m3", [[bivar_fctr31],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m3", [[bivar_fctr31],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m3", [[bivar_fctr32],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m3", [[bivar_fctr32],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m3", [[bivar_fctr32],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m3", [[bivar_fctr32],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m3", [[bivar_fctr32],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m3", [[bivar_fctr32],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m3", [[bivar_fctr33],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m3", [[bivar_fctr33],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m3", [[bivar_fctr33],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m3", [[bivar_fctr33],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m3", [[bivar_fctr33],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
#    ("bivar_m3", [[bivar_fctr33],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m4", [[bivar_fctr41],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m4", [[bivar_fctr41],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m4", [[bivar_fctr41],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m4", [[bivar_fctr41],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m4", [[bivar_fctr41],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m4", [[bivar_fctr41],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m4", [[bivar_fctr42],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m4", [[bivar_fctr42],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m4", [[bivar_fctr42],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m4", [[bivar_fctr42],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m4", [[bivar_fctr42],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
    ("bivar_m4", [[bivar_fctr42],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k

    ("bivar_m4", [[bivar_fctr43],    [ 1*tr_cnt],  [tr_len],  [cl_len]]),   #   5k
    ("bivar_m4", [[bivar_fctr43],    [ 2*tr_cnt],  [tr_len],  [cl_len]]),   #  10k
    ("bivar_m4", [[bivar_fctr43],    [ 5*tr_cnt],  [tr_len],  [cl_len]]),   #  25k
    ("bivar_m4", [[bivar_fctr43],    [10*tr_cnt],  [tr_len],  [cl_len]]),   #  50k
    ("bivar_m4", [[bivar_fctr43],    [20*tr_cnt],  [tr_len],  [cl_len]]),   # 100k
#    ("bivar_m4", [[bivar_fctr43],    [40*tr_cnt],  [tr_len],  [cl_len]]),   # 200k
  ])


def run_benchmark():
  print("-" * 80)
  print("Python: {0}".format(sys.version))
  print("numpy : {0}".format(np.__version__))
  print("scipy : {0}".format(scipy.__version__))
  print("-" * 80)

  bivar_benchmark()


if __name__ == '__main__':
  run_benchmark()
