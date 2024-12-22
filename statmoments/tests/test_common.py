import pytest

import numpy as np
import numpy.testing as nt

from statmoments.common import str2bits, str2bytes, triu_flatten, meanfree


def test_str2bits():
    assert str2bits('0x1') == '0001'
    assert str2bits('0xA') == '1010'
    assert str2bits('0x10') == '00010000'
    assert str2bits('binarydata') == 'binarydata'


def test_str2bytes():
    assert str2bytes('hello') == bytearray(b'hello')
    assert str2bytes('123') == bytearray(b'123')
    assert str2bytes('!@#') == bytearray(b'!@#')


def test_triu_flatten():
    sqmat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = np.array([1, 2, 3, 5, 6, 9])
    nt.assert_array_equal(triu_flatten(sqmat), expected)


def test_meanfree():
    traces = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = np.array([[-3, -3, -3], [0, 0, 0], [3, 3, 3]])
    nt.assert_array_equal(meanfree(traces), expected)


if __name__ == "__main__":
    pytest.main()
