# statmoments
Fast streaming univariate and bivariate moments and t-statistics

statmoments is a library for fast streaming computation of univariate and bivariate moments and statistics. It allows finding variance, skewness and kurtosis as well as covariance, coskewness and cokurtosis. Along with that it can compute Welch's t-test statistics for (co-)moments of data split in sets. statmoments used BLAS matrix computation for data provided in batches to perform as fast as possible.

## Examples
Examples can be found in the examples directory and in tests

## Installation
pip install -e .
