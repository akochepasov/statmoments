
from itertools import repeat as _repeat

import numpy as np

from ._native_shim import ttest
from ._native import _preprocvar  # Is not imported otherwise

from .common import triu_flatten


def ttests(engine, moment=None, dim=None, equal_var=False):
  """Return a generator yielding the independed samples Welch's t-test for each classifier"""

  test_dim = dim    if dim    is not None else engine.dim
  test_mom = moment if moment is not None else engine.moment // 2

  if engine.moment < 2 or engine.moment // 2 < np.max(test_mom):
    emsg = "The engine cannot provide required statistical moments. Increase the engine moment"
    raise ValueError(emsg)

  mom_conv = [
      {  # Empty
      },
      {  # Univariate case
          1: (1, 2),
          2: (2, 4),
          3: (3, 6),
          4: (4, 8),
      },
      {  # Bivariate case
          1: ((1, 1), (2, 2)),
          2: ((2, 2), (4, 4)),
          3: ((3, 3), (6, 6)),
          4: ((4, 4), (8, 8)),
          (1, 1): ((1, 1), (2, 2)),
          (1, 2): ((1, 2), (2, 4)),
          (2, 2): ((2, 2), (4, 4)),
      }
  ]

  mlist = test_mom
  if isinstance(mlist, int) or isinstance(mlist, tuple):
    mlist = mom_conv[test_dim][test_mom]

  test_mom = np.broadcast_to(test_mom, test_dim)
  tmd = np.sum(test_mom)
  momgen = (None, engine.moments, engine.comoments)[test_dim]

  # The standartization denominator (std** 2 = var)
  stddgen = engine.moments(2) if tmd > 2 else _repeat([1, 1])

  # Moments are standardized _not_ here, but later
  # _preprocvar to save FLOPS on sqrt
  for i, (cm, stdd) in enumerate(zip(momgen(mlist, False), stddgen)):
    n0, n1 = engine.counts(i)
    if min(n0, n1) < engine.acc_min_count:
      tr_len = engine.trace_len
      tr_len = [tr_len, tr_len * (tr_len + 1) // 2][test_dim - 1]
      yield np.zeros(tr_len)
      continue

    if tmd > 2:
      if False:
        pass
      elif test_dim == 1:
        stdd = stdd ** test_mom
      elif test_dim == 2:
        lm, rm = test_mom
        stdl, stdr = stdd
        stdd = [triu_flatten(np.outer(stdl ** lm, stdl ** rm)), triu_flatten(np.outer(stdr ** lm, stdr ** rm))]

    # Convert to preprocessed variance and standartize
    _preprocvar(cm[0], tmd, stdd[0])
    _preprocvar(cm[1], tmd, stdd[1])

    m10, m20 = cm[0]
    m11, m21 = cm[1]
    yield ttest(n0, n1, m10, m11, m20, m21, equal_var)
