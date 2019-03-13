Benchmarks
==========

Introduction
------------

To provide an overview of the implemented models, a benchmark
is supplied for select algorithms. In total, 17 benchmark data are used for comparision which
can be downloaded at `ODDS <http://odds.cs.stonybrook.edu/#table1>`_.

For each dataset, it is first split into 60% for training and 40% for testing.
All experiments are repeated 10 times independently with random splits.
The mean of 10 trials is regarded as the final result. Three evaluation metrics
are provided:

- The area under receiver operating characteristic (ROC) curve
- Precision @ rank n (P@N)
- Execution time


You are welcome to replicate this process by running:
`benchmark.py <https://github.com/yzhao062/Pyod/blob/master/notebooks/benchmark.py>`_

ROC Performance
---------------

.. csv-table:: ROC Performances (average of 10 independent trials)
   :file: tables/roc.csv
   :header-rows: 1

P@N Performance
---------------

.. csv-table:: Precision @ N Performances (average of 10 independent trials)
   :file: tables/prc.csv
   :header-rows: 1


Execution Time
--------------

.. csv-table:: Time Elapsed in Seconds (average of 10 independent trials)
   :file: tables/time.csv
   :header-rows: 1

