# statmoments

Fast streaming univariate and bivariate moments and t-statistics.

statmoments is a high-performance library to compute higher-order univariate and bivariate statistical moments in a single pass over large waveform datasets with thousands of sample points. The moments can be converted to Welch's t-test statistics for hypothesis testing on arbitrary data partition.

## Features

- Streaming processing for both univariate and bivariate analysis
- Covariance, coskewness, cokurtosis with possibility to compute even higher moments
- Efficient memory usage through dense matrix representation
- High numerical accuracy
- NVIDIA GPU acceleration for intensive operations, with CPU fallback
- Command-line interface for analysis of existing HDF5 datasets

## How is it different?

statmoments performs single-pass streaming analysis of higher-order moments for individual sample points and joint central moments for pairs of sample points.

When differences are subtle, analysis may require millions of waveforms with thousands of sample points each. High-order moment computations traditionally require multiple passes and may necessitate restarting analysis when new data arrives. With thousands of sample points per waveform, the computational complexity grows substantially, as these applications typically require millions to billions of data observations because dependencies are often weak, masked by noise, or only apparent across large populations or extended time periods, making single-pass streaming algorithms essential for practical analysis. In this library, input batches can be processed and discarded, so the complete dataset need not fit in memory. The dense twofold matrix representation of intermediate accumulators reduces memory requirements. The accumulators can be converted to co-moments and Welch's t-test statistics on demand. The library handles significant input streams, processing hundreds of megabytes per second. When fast enough, it's suitable for real-time sources like oscilloscopes, sensors, and financial markets.

When the relevant data split is not known in advance, multiple candidate splits can be analyzed at once. Each incoming waveform is pre-classified into one of two groups for each requested split, and the library computes moments for all of them.

Some of the benefits of streaming computation include:

- Real-time insights for trend identification and anomaly detection
- Reduced data processing latency, crucial for time-sensitive applications
- Scalability to handle large data volumes, essential for data-intensive research in fields like particle physics, signal processing and side-channel analysis

## Numeric accuracy

The numeric accuracy of results depends on the coefficient of variation (COV) of a sample point in the input waveforms. With a COV of about 5%, the computed (co-)kurtosis has about 10 correct significant digits for 10,000 waveforms, sufficient for Welch's t-test. Increasing number of waveforms by the factor of 100 loses one more significant digit.

## Examples

### Performing univariate data analysis

```python
  # Input data parameters
  tr_count = 100   # M input waveforms
  tr_len   = 5     # N features or points in the input waveforms
  cl_len   = 2     # C hypotheses how to split input waveforms

  # Create engine, which can compute up to kurtosis
  uveng = statmoments.Univar(tr_len, cl_len, moment=4)

  # Process input data and split hypotheses
  uveng.update(wforms1, classification1)

  # Process more input data and split hypotheses
  uveng.update(wforms2, classification2)

  # Get statistical moments
  mean       = [cm.copy() for cm in uveng.moments(moments=1)]  # E(X)
  skewness   = [cm.copy() for cm in uveng.moments(moments=3)]  # E(X^3)

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
  cl_len = 2       # C hypotheses how to split input waveforms

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
# Find univariate t-test statistics of skewness for the first
# 5000 waveform sample points, stored in a HDF5 dataset
python -m statmoments.univar -i data.h5 -m 3 -r 0:5000

# Find bivariate t-test statistics of covariance for the first
# 1000 waveform sample points, stored in a HDF5 dataset
python -m statmoments.bivar -i data.h5 -r 0:1000
```

More examples with specific details can be found in `examples` and `tests` directories.

## Implementation Notes

statmoments uses fastest BLAS implementations for computationally intensive matrix operations. When CuPy and [nvmath-python](https://github.com/NVIDIA/nvmath-python) are available, supported operations run on NVIDIA GPU. If GPU dependencies are absent, computation runs on the CPU.

Bivariate results -- co-moments and t-tests -- are represented by the **upper triangle** of the symmetric matrix as 1D array for each classifier for efficient memory use, similar to `numpy.triu_indices_from()`.

Due to RAM limits, the results are produced one at a time for each input classifier as the set of statistical moments. Each classifier's output co-moment has dimensions 2 x C x L, where C is an index of the requested classifier, and L is the length of the flattened upper triangle.

## Installation

- Optional NVIDIA GPU acceleration:
  ```shell
  pip install "statmoments[gpu]"
  ```
- Regular CPU installation:
  ```shell
  pip install statmoments
  ```

## Citation

If you use this software in academic work, please cite our conference paper.

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
[![GitHub License](https://img.shields.io/github/license/akochepasov/statmoments)](https://github.com/akochepasov/statmoments/blob/main/LICENSE)


[![numpy](https://img.shields.io/badge/numpy-required-blue?logo=numpy)](https://pypi.org/project/numpy/)
[![scipy](https://img.shields.io/badge/scipy-required-blue?logo=scipy)](https://pypi.org/project/scipy/)


[![CI](https://github.com/akochepasov/statmoments/actions/workflows/python-package.yml/badge.svg)](https://github.com/akochepasov/statmoments/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/akochepasov/statmoments/branch/main/graph/badge.svg)](https://codecov.io/gh/akochepasov/statmoments)
