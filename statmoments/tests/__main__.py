
import sys
import unittest


def run_alltests():
  suite = unittest.TestLoader().discover('statmoments.tests', pattern='test_*.py')
  return unittest.TextTestRunner(verbosity=1).run(suite)


# Entrance point
if __name__ == '__main__':
  res = run_alltests()
  sys.exit(res.wasSuccessful() is False)
