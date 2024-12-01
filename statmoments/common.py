import os
import shutil

import numpy as np


def str2bits(hexdata):
  if not hexdata.startswith('0x'):
    return hexdata  # binary data

  # Convert hexadecimal
  return bin(int(hexdata, 16))[2:].zfill((len(hexdata) - 2) * 4)


def str2bytes(strline):
  return bytearray(strline, 'ascii')


def save_npy(filename, trace):
  np.save(filename, trace, False, False)


def remove_dir(dirname):
  if not os.path.exists(dirname):
      return

  for fn in os.listdir(dirname):
    fullname = os.path.join(dirname, fn)
    if os.path.isfile(fullname):
      os.remove(fullname)
    elif os.path.isdir(fullname):
      remove_dir(fullname)

  shutil.rmtree(dirname, ignore_errors=True)


def triu_flatten(sqmat):
  return sqmat[np.triu_indices_from(sqmat)]


def meanfree(traces):
  return traces - np.mean(traces, axis=0)


def uni2bivar(data, lm=1, rm=1, normalize=True):
  sd = 1
  if normalize and (lm + rm) >= 3:
    sd = np.std(data, axis=0, ddof=0)
  res = [triu_flatten(np.outer((tr / sd) ** lm, (tr / sd) ** rm)) for tr in meanfree(data)]
  return res
