import numpy as np

import statmoments


def bivar_ttest():
  # Input data parameters
  tr_count = 100   # M input waveforms
  tr_len = 5       # N features or points in the input waveforms
  cl_len = 4       # L hypotheses how to split input waveforms
  # Create engine
  bveng = statmoments.Bivar(tr_len, cl_len, moment=4)

  # Input data
  traces0 = np.random.normal(0, 10, (tr_count, tr_len)).astype(np.int8)
  # Insert correlation for sample points 0-1 for the full batch
  traces0[:,   0] = (2 * traces0[:,   1]   + 10).astype(np.int8)
  # and for interlaced waveforms, sample points 2-3
  traces0[::2, 2] = (3 * traces0[::2, 3]/2 + 20).astype(np.int8)

  # Generate sorting classification (data partitioning hypotheses)
  # 0: the input batch belongs to dataset 0
  # 1: the input batch belongs to dataset 1
  # 2: data interlaced from 0
  # 3: data interlaced from 1
  cl0 = [[0, 1, i % 2, (i+1) % 2] for i in range(len(traces0))]

  # Process input
  bveng.update(traces0, cl0)

  traces1 = np.random.normal(0, 10, (tr_count, tr_len)).astype(np.int8)
  # Insert correlation for interlaces waveforms, sample points 2-3
  traces1[::2, 2] = (3 * traces1[::2, 3]/2 + 20).astype(np.int8)
  # Generate sorting classification (data partitioning hypotheses)
  # 0: the input batch belongs to dataset 1
  # 1: the input batch belongs to dataset 0
  # 2: data interlaced from 0
  # 3: data interlaced from 1
  cl1 = [[1, 0, i % 2, (i+1) % 2] for i in range(len(traces1))]
  bveng.update(traces1, cl1)

  # All generator returned data must be copied out
  # since it points to an internal buffer

  # ======== Bivariate moments by one (simpler) ========
  covariance    = [cm.copy() for cm in bveng.comoments(moments=(1, 1))]
  coskeweness   = [cm.copy() for cm in bveng.comoments(moments=(1, 2))]
  cokurtosis22  = [cm.copy() for cm in bveng.comoments(moments=(2, 2))]
  cokurtosis13  = [cm.copy() for cm in bveng.comoments(moments=(1, 3))]

  # ======== Covariance and cokurtosis in batch (faster) ========
  moments = [(1, 1), (2, 2)]  # E(X^1 * Y^1) and E(X^2 * Y*2)
  comoms_11_22 = [cm.copy() for cm in bveng.comoments(moments)]

  # ======== Univariate moments (simpler) ========
  mean       = [cm.copy() for cm in bveng.moments(moments=1)]
  variance   = [cm.copy() for cm in bveng.moments(moments=2)]
  skeweness  = [cm.copy() for cm in bveng.moments(moments=3)]
  kurtosis   = [cm.copy() for cm in bveng.moments(moments=4)]

  # ======== mean and kurtosis in batch (faster) ========
  moments = (1, 4)  # E(X^1) and E(X^4)
  moments_1_4 = [cm.copy() for cm in bveng.moments(moments)]

  # ======== Bivariate t-test ========
  # Highest (cokurtoses)
  for i, tt in enumerate(statmoments.stattests.ttests(bveng)):
    if np.any(np.abs(tt) > 5):
      print(f"Found stat diff in the split {i}")

  # Second (covariances)
  for i, tt in enumerate(statmoments.stattests.ttests(bveng, moment=(1,1))):
    if np.any(np.abs(tt) > 5):
      print(f"Found stat diff in the split {i}")

  # First univariate (means)
  for i, tt in enumerate(statmoments.stattests.ttests(bveng, moment=1, dim=1)):
    if np.any(np.abs(tt) > 5):
      print(f"Found stat diff in the split {i}")


if __name__ == '__main__':
  bivar_ttest()
