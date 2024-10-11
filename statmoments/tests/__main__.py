
import os
import sys
import pytest


# Entrance point
if __name__ == '__main__':
  sys.exit(pytest.main(["-v", "-s", os.path.dirname(__file__)]))
