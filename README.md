# statmoments
Fast streaming univariate and bivariate moments and t-statistics

statmoments is a library for fast streaming computation of univariate and bivariate moments and statistics for sample points of waveforms. Along with that, it can compute Welch's t-test statistics of various orders for arbitrary data partitioning. Given the data sorting with classifier, it allows finding relationships and statistical differences among many data splits, unknown beforehand. statmoments used BLAS matrix computation and can take data in batches to perform computations as fast as possible.

# How is that different?
Streaming computation is a paradigm to process data as it is generated, rather than waiting for it to be pre-collected and stored in a database or other persistent storage. Streaming computation is often used to process data from real-time sources, such as sensors, financial markets, and social media. It can also be used to process large datasets that are too large to fit in memory or to process data that is constantly changing.
When the difference in input data is really subtle, millions of waveforms should be processed to find the statistical difference. In case the input waveforms contain many thousands of sample points, which adds another dimension to an already complex problem.
This library avoids recomputation with a streaming approach, collecting data in the intermediate accumulator which can be converted to (co-)moments and they in turn can be converted to t-statistics with Welch's t-test. Once more data collected, it can be iteratively processed, increasing the precision of the moments, and discarded. The computation is optimized to consume significant input streams, hundreds of megabytes per second of waveforms, which may contain thousands of points.
Yet another dimension can be added when the data split is unknown, which bucket the input waveform belongs to. This library solves this with pre-classification of the input data and computing moments for all the requested data splits.

# Where is this needed?
Bivariate statistics is a powerful tool for understanding the relationships between variables. Researchers and practitioners use it in a wide range of fields to make informed decisions and improve outcomes.
can be used to answer a variety of questions, such as:
 - Is there a statistically significant relationship between variables?
 - Which data points are related?
 - If any, how strong is the relationship?
 - Can we use one variable to predict the other variable?

These statistical methods are used in medical and bioinformatics research, astrophysics, cryptography, seismology, market predictions, and many more where the input data is measured in hundreds of gigabytes.
Some of the benefits of streaming computation include:
 - Real-time insights: The computation can be used to find real-time insights from the data, such as identifying trends and detecting anomalies.
 - Reduced latency: It can help reduce the latency of data processing, as the data is processed as generated and then discarded. This is important for applications where time is critical, such as financial trading and online gaming.
 - Scalability: Streaming computation can handle large volumes of data. This is important for research that generates large amounts of data, such as cryptoanalysis and astrophysics.

## Examples
Examples can be found in the examples and tests directories.

## Installation
pip install -e .
