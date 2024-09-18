#!/usr/bin/env python

from os import path as os_path
from itertools import combinations_with_replacement as _combu

from statmoments.common import save_npy
from statmoments._statmoments_impl import Bivar
from statmoments._statmoments_impl import _CliCommon

from statmoments.stattests import ttests


class BivarCMD(_CliCommon):
  def create_engine(self, tr_len, cl_len, **options):
    return Bivar(tr_len, cl_len, **options)

  def create_parser(self):
    parser = super().create_parser()
    parser.prog = 'python -m statmoments.bivar'
    return parser

  def save_results(self, engine, dirname, meta={}):
    super().save_results(engine, dirname, meta)

    # 1D: cl-mom-01
    fname_pttrn = os_path.join(dirname, 'mom1d{:04d}_{}_{}')
    for m in range(1, engine.moment + 1):
      for i, mom in enumerate(engine.moments(m)):
        save_npy(fname_pttrn.format(i, m, 0), mom[0])
        save_npy(fname_pttrn.format(i, m, 1), mom[1])

    # 2D: cl-comom-01
    max_mm = engine.moment  # max degree in E[(x-a)^lm (x-b)^rm]
    fname_pttrn = os_path.join(dirname, 'comom2d{:04d}_{}_{}')
    for mm in _combu(range(1, max_mm + 1), r=2):
      mstr = ''.join(map(str, mm))
      for i, cm in enumerate(engine.comoments(mm)):
        save_npy(fname_pttrn.format(i, mstr, 0), cm[0, 0])
        save_npy(fname_pttrn.format(i, mstr, 1), cm[1, 0])

    # 2D: cl-tt
    max_mm = engine.moment // 2  # max(lm, rm) in E[(x-a)^lm (x-b)^rm]
    fname_pttrn = os_path.join(dirname, 'tt2d{:04d}_{}')
    for mm in _combu(range(1, max_mm + 1), r=2):
      mstr = ''.join(map(str, mm))
      for i, tt in enumerate(ttests(engine, mm)):
        save_npy(fname_pttrn.format(i, mstr), tt)


if __name__ == '__main__':
  BivarCMD().run()
