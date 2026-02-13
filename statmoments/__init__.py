
"""A bivariate statistical library.

Example:
>>> python -m statmoments.bivar
"""

from ._version import __version__

# Compiled or interpreted data from _native.pyx
from . import stattests

# Functionality
from ._statmoments_impl import bivar_sum, bivar_cntr, bivar_2pass   # primary kernels
from ._statmoments_impl import bivar_sum_mix, bivar_sum_detrend     # experimental
from ._statmoments_impl import bivar_txtbk, bivar_vtk               # baseline test kernels

from ._statmoments_impl import univar_sum               # primary kernels
from ._statmoments_impl import univar_sum_detrend       # experimental

from ._statmoments_impl import Bivar, Univar

__name__      = "statmoments"
