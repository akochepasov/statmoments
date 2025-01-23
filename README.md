# statmoments

Fast streaming univariate and bivariate moments and t-statistics.

statmoments is a high-performance library for computing univariate and bivariate statistical moments in a single pass over large waveform datasets with thousands of sample points. It can produce Welch's t-test statistics for hypothesis testing on arbitrary data partitions.

## Features

- Streaming processing for both univariate and bivariate analysis
- Efficient memory usage through dense matrix representation
- High numerical accuracy
- Command-line interface for analysis of existing datasets

## How is it different?

When input data differences are subtle, millions of waveforms may have to be processed to find the statistically significant difference, requiring efficient algorithms. In addition to that, the high-order moment computation need multiple passes and may require starting over once new data appear. With thousands of sample points per waveform, the problem becomes more complex.

A streaming algorithm processes sequences of inputs in a single pass as they are collected. When fast enough, it's suitable for real-time sources like oscilloscopes, sensors, and financial markets, as well as for large datasets that don't fit in memory. The dense matrix representation of an intermediate accumulator reduces memory requirements. The accumulator can be converted to co-moments and Welch's t-test statistics on demand. Data batches can be iteratively processed to increase precision and then discarded. The library handles significant input streams, processing hundreds of megabytes per second.

Yet another dimension can be added when the data split is unknown. In other words, which bucket the input waveform belongs to. This library solves this with given pre-classification of the input data and computing moments for all the requested data splits.

Some of the benefits of streaming computation include:

- Real-time insights for trend identification and anomaly detection
- Reduced data processing latency, crucial for time-sensitive applications
- Scalability to handle large data volumes, essential for data-intensive research in fields like astrophysics and financial analysis

## Numeric accuracy

The numeric accuracy of results depends on the coefficient of variation (COV) of a sample point in the input waveforms. With a COV of about 5%, the computed (co-)kurtosis has about 10 correct significant digits for 10,000 waveforms, sufficient for Welch's t-test. Increasing data by 100x loses one more significant digit.

## Examples

### Performing univariate data analysis

```python
  # Input data parameters
  tr_count = 100   # M input waveforms
  tr_len   = 5     # N features or points in the input waveforms
  cl_len   = 2     # L hypotheses how to split input waveforms

  # Create engine, which can compute up to kurtosis
  uveng = statmoments.Univar(tr_len, cl_len, moment=4)

  # Process input data and split hypotheses
  uveng.update(wforms1, classification1)

  # Process more input data and split hypotheses
  uveng.update(wforms2, classification2)

  # Get statistical moments
  mean       = [cm.copy() for cm in uveng.moments(moments=1)]  # E(X)
  skeweness  = [cm.copy() for cm in uveng.moments(moments=3)]  # E(X^3)

  # Detect statistical differences in the first-order t-test
  for i, tt in enumerate(statmoments.stattests.ttests(uveng, moment=1)):
    if np.any(np.abs(tt) > 5):
      print(f"Data split {i} has different means")

  # Process more input data and split hypotheses
  uveng.update(wforms3, classification3)

  # Get updated statistical moments and t-tests
  # with statmoments.stattests.ttests(uveng, moment=1)
```

### Performing bivariate data analysis

```python
  # Input data parameters
  tr_count = 100   # M input waveforms
  tr_len = 5       # N features or points in the input waveforms
  cl_len = 2       # L hypotheses how to split input waveforms

  # Create bivariate engine, which can compute up to co-kurtosis
  bveng = statmoments.Bivar(tr_len, cl_len, moment=4)

  # Process input data and split hypotheses
  bveng.update(wforms1, classification1)

  # Process more input data and split hypotheses
  bveng.update(wforms2, classification2)

  # Get bivariate moments
  covariance    = [cm.copy() for cm in bveng.comoments(moments=(1, 1))]  # E(X Y)
  cokurtosis22  = [cm.copy() for cm in bveng.comoments(moments=(2, 2))]  # E(X^2 Y^2)
  cokurtosis13  = [cm.copy() for cm in bveng.comoments(moments=(1, 3))]  # E(X^1 Y^3)

  # univariate statistical moments are also can be obtained
  variance   = [cm.copy() for cm in bveng.moments(moments=2)]  # E(X^2)

  # Detect statistical differences in the second order t-test (covariances)
  for i, tt in enumerate(statmoments.stattests.ttests(bveng, moment=(1,1))):
    if np.any(np.abs(tt) > 5):
      print(f"Found stat diff in the split {i}")

  # Process more input data and split hypotheses
  bveng.update(wforms3, classification3)

  # Get updated statistical moments and t-tests
  # with statmoments.stattests.ttests(bveng, moment=(1,1))
```

### Performing data analysis from the command line

```shell
# Find univariate t-test statistics of skeweness for the first
# 5000 waveform sample points, stored in a HDF5 dataset
python -m statmoments.univar -i data.h5 -m 3 -r 0:5000

# Find bivariate t-test statistics of covariance for the first
# 1000 waveform sample points, stored in a HDF5 dataset
python -m statmoments.bivar -i data.h5 -r 0:1000
```

More examples can be found in the examples and tests directories.

## Implementation Notes

statmoments uses top BLAS implementations, including GPU based on [nvmath-python](https://github.com/NVIDIA/nvmath-python) if available, for the best peformance on Windows, Linux and Macs,to maximize computational efficiency.

Due to RAM limits, results are produced one at a time for each input classifier as the set of statistical moments. Each classifier's output moment has dimensions 2 x M x L, where M is an index of the requested classifier and L is the region length.

The bivariate  results, co-moments and t-tests, are represented by the **upper triangle** of the symmetric matrix as 1D array for each classifier.

## Installation

```shell
pip install statmoments
```

## References

Anton Kochepasov, Ilya Stupakov, "An Efficient Single-pass Online Computation of Higher-Order Bivariate Statistics", 2024 IEEE International Conference on Big Data (BigData), 2024, pp. 123-129, [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10825659).

```bibtex
@INPROCEEDINGS{10825659,
  author={Stupakov, Ilya and Kochepasov, Anton},
  booktitle={2024 IEEE International Conference on Big Data (BigData)},
  title={An Efficient Single-pass Online Computation of Higher-Order Bivariate Statistics},
  year={2024},
  pages={123-129},
  doi={10.1109/BigData62323.2024.10825659}
}
```

[![PyPi Version](https://img.shields.io/pypi/v/statmoments.svg?style=flat-square)](https://pypi.org/project/statmoments/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/statmoments.svg?style=flat-square)](https://pypi.org/project/statmoments/)
