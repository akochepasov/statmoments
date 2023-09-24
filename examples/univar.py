
import numpy as np

import statmoments


def bivar_ttest():
  # Input data parameters
  tr_count = 100   # M input waveforms
  tr_len = 5       # N features or points in the input waveforms
  cl_len = 2       # L hypotheses how to split input waveforms

  # Create engine
  uveng = statmoments.Univar(tr_len, cl_len, moment=4)

  # Generate input data
  wforms = np.random.normal(0, 7, (2, tr_count, tr_len)).astype(np.int8)
  # Insert different distribution into some points of one batch
  wforms[0][:, 2:4] = np.random.normal(50, 15, (100, 2)).astype(np.int8)
  wforms = wforms.reshape(2 * tr_count, tr_len)

  # Generate sorting classification (hypotheses how to split the input data)
  # 1 assumes data as is, 2 assumes the data waveforms are interlaced
  cl0 = [(0, i % 2) for i in range(tr_count)]
  cl1 = [(1, i % 2) for i in range(tr_count)]

  # Process input
  uveng.update(wforms, cl0 + cl1)

  # All generator returned data must be copied out
  # since it points to an internal buffer

  # ======== Univariate moments by one (simpler) ========
  mean       = [cm.copy() for cm in uveng.moments(moments=1)]
  variance   = [cm.copy() for cm in uveng.moments(moments=2)]
  skeweness  = [cm.copy() for cm in uveng.moments(moments=3)]
  kurtosis   = [cm.copy() for cm in uveng.moments(moments=4)]

  # ======== Mean and kurtosis in a batch (faster) ========
  moments = (1, 4)  # E(X^1) and E(X^4)
  moments_1_4 = [mom.copy() for mom in uveng.moments(moments)]

  # ======== Univariate t-test ========
  # Detect statistical differences in the highest t-test order
  # Variances: 2nd order
  for i, tt in enumerate(statmoments.stattests.ttests(uveng)):
    if np.any(np.abs(tt) > 5):
      print(f"Data split {i} has different variances")

  # Detect statistical differences in the first t-test order
  # means: 1st order
  for i, tt in enumerate(statmoments.stattests.ttests(uveng, moment=1)):
    if np.any(np.abs(tt) > 5):
      print(f"Data split {i} has different means")


if __name__ == '__main__':
  bivar_ttest()
