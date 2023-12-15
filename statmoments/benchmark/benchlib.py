import gc
from itertools import product
from os import urandom as _urandom
from binascii import b2a_hex
from time import perf_counter

import numpy as np

from statmoments.stattests import ttests


def make_trace_double(size):
  return np.random.uniform(-1, 1, size=size)


def make_trace_ushort(size):
  return np.random.randint(0, 255 * 255, size=size, dtype=np.uint16)


def bytes2classifiers(byte_line):
  hex_line = b2a_hex(byte_line).decode('ascii')
  return bin(int(hex_line, 16))[2:].zfill(len(byte_line) * 8)


def make_hypotheses(hlen):
  return bytes2classifiers(_urandom((hlen + 7) // 8))[:hlen]


class EngineFactory(object):
  def __init__(self, make_trace, make_classifiers, factory, **options):
    self._gen_trace = make_trace
    self._gen_classifiers = make_classifiers
    self._factory = factory
    self._options = options

  def create_data(self, tr_count, tr_len, cl_len):
    blk_sz = 8
    traces = np.empty((tr_count // blk_sz * blk_sz, tr_len))
    traces[:len(traces) // blk_sz, :] = self._gen_trace((len(traces) // blk_sz, tr_len))
    traces.reshape(blk_sz, -1, tr_len)[:] = traces[:len(traces) // blk_sz]
    traces = np.random.permutation(traces)
    classifiers = [self._gen_classifiers(cl_len) for _ in range(len(traces))]
    return traces, classifiers

  def create_engine(self, tr_len, cl_len):
    return self._factory(tr_len, cl_len, **self._options)


def benchmark(benchset):
  batch_count, repeat = 1, 1

  print('{:12}{:20}{:>7}{:>8}{:>9}{:>8}{:>8}{:>9}{:>11}'.format(
        'Name', 'Implementation', 'MB', 'tr_cnt', 'tr_len', 'cl_cnt', 'tr/sec', 'upd_time', 'ttest_time'))

  for name, params in benchset:
    for engine_factory, tr_count, tr_len, cl_count in product(*params):
      update_times, ttest_times = [], []
      assert repeat > 0
      for _ in range(repeat):
        traces, classifiers = engine_factory.create_data(tr_count, tr_len, cl_count)
        engine = engine_factory.create_engine(tr_len, cl_count)
        for _ in range(batch_count):
          start = perf_counter()
          engine.update(traces, classifiers)  # Streaming: layout and accumulator
          update_times.append(perf_counter() - start)

          start = perf_counter()
          for _ in ttests(engine):  # On-demand: t-tests
            pass
          ttest_times.append(perf_counter() - start)

      min_update, min_tt = min(update_times), min(ttest_times)
      max_mom = str(engine.moment) * 2
      kname = '{}(m{})'.format(type(engine._impl).__name__, max_mom)
      print("{:12}{:20}{:7d}{:8d}{:9d}{:8d}{:>8d}{:>9.1f}{:>9.1f}".format(
          name, kname, engine.memory_size >> 20, tr_count, tr_len, cl_count,
          int(tr_count / min_update), min_update, min_tt))
      # Force garbage collection
      del traces
      del engine
      gc.collect()

  print()
