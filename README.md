# statmoments

Fast streaming univariate and bivariate moments and t-statistics.

statmoments is a library for the fast streaming one-pass computation of univariate and bivariate moments for batches of multiple of waveforms or traces with thousands of sample points. Given the data sorting with classifiers, it can compute Welch's t-test statistics of various orders for arbitrary data partitioning to allow finding relationships and statistical differences among many data splits, which are unknown beforehand. statmoments uses best-of-class BLAS implementation and preprocesses input data to make the most of computational power and perform computations as fast as possible on Windows and Linux platforms.

## How is that different?

When the difference in input data is really subtle, millions of waveforms should be processed to find the statistically significant difference. Therefore, computationally efficient algorithms for analyzing such volumes of data are crucial for practical use. Due to their nature, computing high-order moments and statistics requires 2- or 3-pass computations. However, not all the data can fit into the available memory. In another case, after initial input has been processed, new data may appear, which requires starting the process over. When the input waveforms contain many thousands of sample points, it adds another dimension to an already complex problem.

This can be overcome with a streaming approach. A streaming algorithm examines a sequence of inputs in a single pass, it processes data as it is collected, without waiting for it to be pre-collected and stored in a persistent storage. Streaming computation is often used to process data from real-time sources, such as oscilloscopes, sensors as well as financial markets. It can also be used to process large datasets that are too large to fit in memory or to process data that is constantly changing. Another name for such an approach is an online algorithm.

This library implements streaming algorithms to avoid recomputation, and stores intermediate data in the accumulator. The algorithm also takes into account that covariance and higher matrices are represented by a symmetric matrix, decreasing the memory requirement two-fold. The data update is blazingly fast and at any moment the internal accumulator can be converted to (co-)moments on demand and the moments in turn can be converted to t-statistics with Welch's t-test. Once more data is collected, it can be iteratively processed, increasing the precision of the moments, and discarded. The computation is optimized to consume significant input streams, hundreds of megabytes per second of waveforms, which may contain thousands of points.
Yet another dimension can be added when the data split is unknown. In other words, which bucket the input waveform belongs to. This library solves this with pre-classification of the input data and computing moments for all the requested data splits.

Some of the benefits of streaming computation include:

- Real-time insights from the data, such as identifying trends and detecting anomalies.
- Reduced latency of data processing, as the data is processed as generated and then discarded. This is important for applications where time is critical, from financial trading to online gaming.
- Scalability: Streaming computation can handle large volumes of data. This is important for research that generates large amounts of data, such as astrophysics and financial analysis.

## Where is this needed?

Univariate statistics are used in various fields and contexts to analyze and describe the characteristics of a single variable or data set. Common applications can be:

- Descriptive Statistics: summarizing and describing the central tendency, dispersion, and shape of a dataset.
- Hypothesis Testing: testing hypothesis to determine if there are significant differences or relationships between groups or conditions.
- Finance and Economics: Examining the performance of financial assets, tracking market trends, and assessing risk in real-time.

In summary, univariate statistics are a fundamental tool in data analysis and are widely used across a range of fields to explore, summarize, and draw conclusions from single-variable data. They provide essential insights into the characteristics and behavior of individual variables, which can inform decision-making and further research.

Bivariate statistics is a tool for understanding the relationships between two variables. Researchers and practitioners use it in a wide range of fields to make informed decisions and improve outcomes. They can be used to answer a variety of questions, such as:

- Is there a statistically significant relationship between variables?
- Which data points are related?
- If any, how strong is the relationship?
- Can we use one variable to predict the other variable?

These statistical methods are used in medical and bioinformatics research, astrophysics, seismology, market predictions, and many more where the input data may be measured in hundreds of gigabytes.

## Numeric accuracy

The numeric accuracy of results is dependent on the coefficient of variation (COV) of the sample point in the input waveforms. With COV of about 5%, the computed (co-)kurtosis has about 10 correct significant digits for 10'000 waveforms, which is more than enough for the resulting t-test. Increasing data by about 100x additionally loses one more significant digit.

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
  mean       = [cm.copy() for cm in uveng.moments(moments=1)]
  skeweness  = [cm.copy() for cm in uveng.moments(moments=3)]

  # Detect statistical differences in the first-order t-test
  for i, tt in enumerate(statmoments.stattests.ttests(uveng, moment=1)):
    if np.any(np.abs(tt) > 5):
      print(f"Data split {i} has different means")

  # Process more input data and split hypotheses
  uveng.update(wforms3, classification3)

  # Get updated statistical moments and t-tests
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
  covariance    = [cm.copy() for cm in bveng.comoments(moments=(1, 1))]
  cokurtosis22  = [cm.copy() for cm in bveng.comoments(moments=(2, 2))]
  cokurtosis13  = [cm.copy() for cm in bveng.comoments(moments=(1, 3))]

  # univariate statistical moments are also can be obtained
  variance   = [cm.copy() for cm in bveng.moments(moments=2)]

  # Detect statistical differences in the second order t-test (covariances)
  for i, tt in enumerate(statmoments.stattests.ttests(bveng, moment=(1,1))):
    if np.any(np.abs(tt) > 5):
      print(f"Found stat diff in the split {i}")

  # Process more input data and split hypotheses
  bveng.update(wforms3, classification3)

  # Get updated statistical moments and t-tests
```

### Performing data analysis from the command line

```shell
# Find univariate t-test statistics of skeweness for
# the first 5000 waveform sample points
# taken from the HDF5 dataset
python -m statmoments.univar -i data.h5 -m 3 -r 0:5000

# Find bivariate t-test statistics of covariance for
# the first 1000 waveform sample points
# Taken from the HDF5 dataset
python -m statmoments.bivar -i data.h5 -r 0:1000
```

More examples can be found in the examples and tests directories.

## Implementation notes

Since the output data can exhaust the existing RAM, the results are the matrices of statistical moments for the requested region, which is produced in a one-at-a-time fashion for each input classifier. The output moment for each classifier has dimension 2 x M x L, where, M is an index of the requested classifier, L is the region length. t-test is represented by a 1D array for each classifier.
The **bivariate moments** are represented by the **upper triangle** of the symmetric matrix.


## Installation

```shell
pip install statmoments
```
