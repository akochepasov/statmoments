import os
import logging
import argparse
from abc import ABC, abstractmethod
from itertools import islice
from timeit import default_timer as _timer
from psutil import virtual_memory as _vmem
from multiprocessing import cpu_count as _cpu_count

import h5py
import numpy as np

from ._version import __version__ as _pkg_version

# common
from .common import str2bytes

# bivar
from ._native_shim import bivar_cntr, bivar_sum
from ._native_shim import bivar_sum_mix, bivar_sum_detrend
from ._native_shim import bivar_2pass, bivar_txtbk, bivar_vtk

# univar
from ._native_shim import univar_sum, univar_sum_detrend


logging.basicConfig(level=logging.INFO, format='%(message)s')


def get_mm(mm):
    try:
      next(iter(mm))  # mm is iterable
      return mm
    except TypeError:
      pass            # mm is just a number
    return [mm] if mm is not None else None


def get_lmrm(mm):
  try:
    try:
      next(iter(mm[0]))               # mm is 2D: ((1,1), (2,2))
      return [m for m in zip(*iter(mm))]
    except TypeError:
      return [[m] for m in iter(mm)]  # mm is 1D
  except TypeError:
    pass                              # mm is just a number
  return np.broadcast_to(mm, (2, 1)) if mm is not None else None


# ============================= ENGINE INTERFACE ============================= #
class _BaseImpl(object):
  @classmethod
  def estimate_mem_size(cls, tr_len, cl_len, moment=2):
    """Return an approximate amount of memory, required for processing."""
    raise NotImplementedError("Must be implemented in the inherited class")

  @property
  def memory_size(self):
    """The actual amount of memory, used by the statistical kernel."""
    return self._impl.memory_size()

  @property
  def trace_len(self):
    """The trace length"""
    return self._impl.trace_len

  @property
  def classifiers_len(self):
    """The number of hypoteses in the accumulator."""
    return self._impl.classifiers_len

  @property
  def moment(self):
    """The maximal available statistical moment."""
    return self._impl.moment

  @property
  def total_count(self):
    """The total number of traces processed by the time."""
    return self._impl.total_count

  @property
  def acc_min_count(self):
    """The minimum amount of traces required to obtain statistical moments."""
    return self._impl.acc_min_count

  def update(self, traces, classifiers):
    """Update the accumulator with a given set of traces, sorted according to classifiers."""
    if len(traces) < 1:
      return

    assert len(traces) == len(classifiers), "Number of traces and classifiers should be equal"
    assert len(traces[0]) == self.trace_len, "Trace has too many features (too long)"
    assert len(classifiers[0]) == self.classifiers_len, "Classifier length should correspond the initial"

    if False:
      pass
    elif isinstance(classifiers[0], (u"".__class__, "".__class__)):
      classifiers = np.asarray(list(map(str2bytes, classifiers)), dtype=np.int8) - ord('0')
    elif isinstance(classifiers[0], bytes):
      if classifiers[0][0] in b'01':
        classifiers = np.asarray(list(map(list, classifiers)), dtype=np.int8) - ord('0')

    self._impl.update(traces, classifiers)

  def counts(self, i):
    """Return counts for each of two sets for i-th hypotesis"""
    return self._impl.counts(i)

  def moments(self, moments=None, normalize=None):
    """Return a generator, yielding univariate moments for each hypotesis"""
    return self._impl.moments(get_mm(moments), normalize)

  def comoments(self, moments=None, normalize=None):
    """Return a generator, yielding flattened upper triangles of bivariate comoments for each hypotesis"""
    return self._impl.comoments(get_lmrm(moments), normalize)


class Bivar(_BaseImpl):
  """
  Streaming processing statistical tool to find univariate and bivariate moments for input data sets.

  Usage example:
  >>> bivar = Bivar(5000, 2)  # 5k sample points, 2 hypotheses
  >>> bivar.update(traces, classifiers)
  >>> ttests = ttest(bivar)
  """

  def __init__(self, tr_len, cl_len, moment=2, use_central=None, **kwargs):
    """Create the tool.

    Parameters
    ----------
    tr_len : int
      Number of random variables in a trace
    cl_len : int
      Number of hypoteses to process
    moment : int
      The max degree in a product moment (the default is 2, i.e. E(X^2*X^2) kurtosis)
    kernel : kind of the processing implementaion
      Specifies which calculation kernel to use (default - heuristic one)
    """
    impl_type = kwargs.pop('kernel', None)
    if impl_type is None:
      impl_type = self._fit_kernel(tr_len, cl_len, moment, use_central)
    self._impl = impl_type(tr_len, cl_len, moment=moment, **kwargs)
    self.dim = 2

  @staticmethod
  def _fit_kernel(_1, _2, moment=2, use_central=None):
    use_central = use_central if use_central is not None else False  # moment > 2
    return bivar_cntr if use_central else bivar_sum_detrend

  @classmethod
  def estimate_mem_size(cls, tr_len, cl_len, moment=2, use_central=None, **kwargs):
    """Return an coarse-grained amount of memory, required by the processing kernel."""
    impl_type = kwargs.pop('kernel', None)
    if impl_type is None:
      impl_type = cls._fit_kernel(tr_len, cl_len, moment, use_central)
    return impl_type.estimate_mem_size(tr_len, cl_len, moment)


class Univar(_BaseImpl):
  """
  Streaming processing statistical tool to find univariate moments for input data sets.

  Usage example:
  >>> univar = Univar(5000, 1024)   # 5k sample points, 1024 hypotheses
  >>> univar.update(traces, classifiers)
  >>> ttests = ttest(univar)
  """

  def __init__(self, tr_len, cl_len, moment=2, use_central=None, **kwargs):
    """Create the tool.

    Parameters
    ----------
    tr_len : int
      Number of random variables in a trace
    cl_len : int
      Number of hypoteses to process
    moment : int
      The max degree in a moment (the default is 2, i.e. E(X^2) variance)
    kernel : kind of the processing implementaion
      Specifies which calculation kernel to use (default - heuristic one)
    """
    impl_type = kwargs.pop('kernel', None)
    if impl_type is None:
      impl_type = self._fit_kernel(tr_len, cl_len, moment, use_central)
    self._impl = impl_type(tr_len, cl_len, moment=moment, **kwargs)
    self.dim = 1

  @staticmethod
  def _fit_kernel(_1, _2, moment=2, use_central=None):
    use_central = use_central if use_central is not None else moment > 2
    return univar_sum if not use_central else univar_sum_detrend

  @classmethod
  def estimate_mem_size(cls, tr_len, cl_len, moment=2, use_central=None):
    """Return an coarse-grained amount of memory, required by the processing kernel."""
    impl_type = cls._fit_kernel(tr_len, cl_len, moment, use_central)
    return impl_type.estimate_mem_size(tr_len, cl_len, moment)


# ============================= CLI INTERFACE ============================= #

class _CliCommon(ABC):
  def iterate_dataset(self, infile, outdir, **kwargs):
    engine, engine_memsz = None, None

    roi    = kwargs.pop('roi')
    online = kwargs.pop('online')
    moment = kwargs.get('moment', 2)

    h5f = h5py.File(infile, 'r')
    assert len(h5f.values()) > 0, f"Input data file {infile} is empty."

    roi = slice(*roi)
    if not roi.stop:
      # The whole trace is used if range is not set
      roi = slice(0, len(h5f[next(iter(h5f.keys()))]['trace'][:]))

    start, batchsz = _timer(), 4 * _cpu_count()
    count, online_save = 0, online

    group_gen = ((g['trace'][:], g['offsets'][:], g['classifiers'][:]) for g in h5f.values())
    line_views = ((tr[off:][roi], cl) for tr, offsets, cl_line in group_gen for off, cl in zip(offsets, cl_line))

    for traces, classifiers in iter(lambda: tuple(zip(*islice(line_views, batchsz))), ()):
      if not engine:
        tr_len, cl_len = len(traces[0]), len(classifiers[0])
        virt_memsz = _vmem().available // 1024**3
        engine = self.create_engine(1, 1)
        engine_memsz = engine.estimate_mem_size(tr_len, cl_len, moment=moment) // 1024**3
        if virt_memsz < engine_memsz:
          raise MemoryError(f"There is not enough memory for this processing. Required: {engine_memsz} GB")
        engine = self.create_engine(tr_len, cl_len, **kwargs)

      engine.update(traces, classifiers)

      count = engine.total_count
      k_traces = count // 1000

      if k_traces > 0:
          if (online > 1) and (k_traces >= online_save):
            output = "Online status: {0} traces processed in: {1:.2f} sec".format(count, _timer() - start)
            self.save_results(engine, outdir, {'!{0}k.proc.log'.format(k_traces): output})
            logging.info(output)
            online_save += online

      if not engine_memsz:
        virt_memsz, engine_memsz = _vmem().available, engine.memory_size
        batchsz2 = (virt_memsz - engine_memsz) // engine.trace_len
        batchsz = max(min(batchsz2, 16 * _cpu_count()), batchsz)

    time_elapsed = _timer() - start
    logging.info("Processing completed in {0}".format(time_elapsed))

    if engine:
      engine_memsz = engine.memory_size // (1024**3)
      output = [
          "Processed traces: {}".format(count),
          "Memory required:  {} GB".format(engine_memsz),
          "Time elapsed:     {:.2f} sec.".format(time_elapsed),
          "Trace rate:       {:.2f} tr/sec".format(count / time_elapsed),
      ]
      output = '\n'.join(output)
      self.save_results(engine, outdir, {f'!{k_traces}k.proc.log': output})
      logging.info(output)

  @abstractmethod
  def create_engine(self, tr_len, cl_len, **options):
    pass

  @abstractmethod
  def create_parser(self):
    class JsonParam(argparse.Action):
      def __call__(self, parser, namespace, items, option_string=None):
        for k, v in items.items():
          setattr(namespace, k, v)

    parser = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--version', action='version', version='%(prog)s ' + _pkg_version)

    parser.epilog = """Examples:
      Find t-test statistics from data.h5 for a range [0:200):
      > %(prog)s -i data.h5 -r 0:300
      Find t-test statistics for 3rd statistical moment:
      > %(prog)s -i data.h5 -m 3"""

    parser.add_argument('-i', '--ifile', metavar='{filename}.h5',
                        help='input HDF5 dataset', required=True)
    parser.add_argument('-o', '--odir', default='ttest.result', type=lambda arg: arg.rstrip('/'),
                        help='ttest output directory (default: ttest.result)')
    parser.add_argument('-m', '--moment', default=2, type=int,
                        help='max moment to calculate (default: 2, i.e. enough for t-test of means)')
    parser.add_argument('-r', '--roi', default=[None, None], metavar='FROM:TO',
                        type=lambda arg: map(int, map(float, arg.split(':'))),
                        help='analysis range as FROM:TO')
    parser.add_argument('--online', default=10, type=int,
                        help='Number of Ktraces to process before updating results')
    parser.add_argument('--use-central', type=int, nargs='?', const=1, choices=[0, 1],
                        help='1 = use central moments, 0 = use raw moments (default: heuristic)')

    return parser

  @abstractmethod
  def save_results(self, engine, dirname, meta={}):
    os.makedirs(dirname, exist_ok=True)
    # Mean of the whole dataset
    # save_npy(os.path.join(dirname, 'mean'), engine.mean())

    for fn, content in meta.items():
      with open(os.path.join(dirname, fn), 'w') as f:
        f.write(content)

  def run(self):
    params = vars(self.create_parser().parse_args())

    inputhdf5, outputdir = params.pop('ifile'), params.pop('odir')
    self.iterate_dataset(inputhdf5, outputdir, **params)
