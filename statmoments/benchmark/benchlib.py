import csv
import datetime
import gc
from binascii import b2a_hex
from itertools import product
from os import urandom as _urandom
from time import perf_counter

import numpy as np

from statmoments.stattests import ttests


def is_vtk_installed():
  try:
    import vtk
    return True
  except ModuleNotFoundError:
    return False


def make_trace_double(size):
  return np.random.uniform(-1, 1, size=size)


def make_trace_ushort(size):
  return np.random.randint(0, 255 * 255, size=size, dtype=np.uint16)


def bytes2classifiers(byte_line):
  hex_line = b2a_hex(byte_line).decode('ascii')
  return bin(int(hex_line, 16))[2:].zfill(len(byte_line) * 8)


def make_hypotheses(hlen):
  return bytes2classifiers(_urandom((hlen + 7) // 8))[:hlen]


class EngineFactory:
  def __init__(self, make_trace, make_classifiers, factory, **options):
    self._gen_trace = make_trace
    self._gen_classifiers = make_classifiers
    self._factory = factory
    self._options = options

  def name(self):
    return self._options['kernel'].__name__

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


def report_filename():
  return f"benchmark-report-{datetime.datetime.now().isoformat().replace(':', '_')}.csv"


def benchmark(test_filter, benchset, repeat=1):
  report_name = report_filename()
  with open(report_name, newline='\n', mode='w') as report_file:
    report_writer = csv.writer(report_file)
    report_writer.writerow(['test', 'kname', 'memory_b',
                            'trace_count', 'trace_length', 'classifier_count', 'batch_size',
                            'avg_trace_time_s',
                            'total_update_time_s', 'min_update_time_s', 'max_update_time_s',
                            'total_ttest_time_s', 'min_ttest_time_s', 'max_ttest_time_s',
                            'run_time_s'])
    print(f"Writing report to {report_name}\n")
    print('Peak performance estimation:')
    print('{:18}{:>7}{:>8}{:>9}{:>8}{:>6}{:>10}{:>9}{:>11}'.format(
      'Implementation', 'MB', 'tr_cnt', 'tr_len', 'cl_cnt', 'batsz', 'tr/sec', 'upd_time', 'ttest_time'))
    for name, params in benchset:
      for engine_factory, tr_count, tr_len, cl_count, batch_size in product(*params):
        if not test_filter(engine_factory.name(), tr_count, tr_len, cl_count, batch_size):
          continue
        batch_size = tr_count if batch_size is None else batch_size
        assert batch_size > 0
        batch_count = tr_count // batch_size
        assert batch_count > 0
        update_times, ttest_times = [], []
        assert repeat > 0
        run_start = perf_counter()
        for _ in range(repeat):
          traces, classifiers = engine_factory.create_data(tr_count, tr_len, cl_count)
          assert len(traces) == len(classifiers)
          assert len(traces) == tr_count
          assert batch_count * batch_size == tr_count
          engine = engine_factory.create_engine(tr_len, cl_count)
          for batch in range(batch_count):
            batch_slice = slice(batch * batch_size, (batch + 1) * batch_size)
            start = perf_counter()
            engine.update(traces[batch_slice], classifiers[batch_slice])  # Streaming: layout and accumulator
            update_times.append(perf_counter() - start)

            start = perf_counter()
            for _ in ttests(engine):  # On-demand: t-tests
              pass
            ttest_times.append(perf_counter() - start)

        min_update, min_tt = min(update_times), min(ttest_times)
        max_mom = str(engine.moment) * 2
        kname = '{}(m{})'.format(type(engine._impl).__name__, max_mom)
        print("{:18}{:7d}{:8d}{:9d}{:8d}{:6d}{:>10d}{:>9.1f}{:>9.1f}".format(
          kname, engine.memory_size >> 20, tr_count, tr_len, cl_count, batch_size,
          int(tr_count / min_update), min_update, min_tt))
        report_writer.writerow([name, kname, engine.memory_size, tr_count, tr_len, cl_count, batch_size,
                                (sum(update_times) + sum(ttest_times)) / tr_count,
                                sum(update_times), min(update_times), max(update_times),
                                sum(ttest_times), min(ttest_times), max(ttest_times),
                                perf_counter() - run_start # Note that this includes data generation
                                ])
        report_file.flush()
        # Force garbage collection
        del traces
        del engine
        gc.collect()
    print()
