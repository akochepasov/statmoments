#!/usr/bin/env python

from os import path as os_path


from statmoments.common import save_npy
from statmoments._statmoments_impl import Univar
from statmoments._statmoments_impl import _CliCommon

from statmoments.stattests import ttests


class UnivarCMD(_CliCommon):
  def create_engine(self, tr_len, cl_len, **options):
    return Univar(tr_len, cl_len, **options)

  def create_parser(self):
    parser = super().create_parser()
    parser.prog = 'python -m statmoments.univar'
    return parser

  def save_results(self, engine, dirname, meta={}):
    super().save_results(engine, dirname, meta)

    # 1D: cl-mom-[01]
    name_pttrn = 'mom{{i:0{}d}}_{{m}}_{{}}'.format(len(str(engine.classifiers_len)))
    fname_pttrn = os_path.join(dirname, name_pttrn)
    for m in range(1, engine.moment + 1):
      for i, mom in enumerate(engine.moments(m)):
        save_npy(fname_pttrn.format(0, i=i, m=m), mom[0])
        save_npy(fname_pttrn.format(1, i=i, m=m), mom[1])

    max_m = engine.moment // 2 + 1  # max degree in E[(x-a)^m]
    name_pttrn = 'tt{{i:0{}d}}_{{m}}'.format(len(str(engine.classifiers_len)))
    fname_pttrn = os_path.join(dirname, name_pttrn)
    for m in range(1, max_m):
      for i, tt in enumerate(ttests(engine, m)):
        save_npy(fname_pttrn.format(i=i, m=m), tt)


if __name__ == '__main__':
  UnivarCMD().run()
