# Python Outlier Detection (PyOD)
[![PyPI version](https://badge.fury.io/py/pyod.svg)](https://badge.fury.io/py/pyod) [![Documentation Status](https://readthedocs.org/projects/pyod/badge/?version=latest)](https://pyod.readthedocs.io/en/latest/?badge=latest) [![Build Status](https://travis-ci.org/yzhao062/Pyod.svg?branch=master)](https://travis-ci.org/yzhao062/Pyod) [![Coverage Status](https://coveralls.io/repos/github/yzhao062/Pyod/badge.svg?branch=master&service=github)](https://coveralls.io/github/yzhao062/Pyod?branch=master) 

--------------------------

PyOD is a **Python-based toolkit** to **identify outlying objects** in data with both unsupervised and supervised approaches. It strives to provide an unified APIs across various anomaly detection algorithms. This exciting yet challenging field is commonly referred as ***[Outlier Detection](https://en.wikipedia.org/wiki/Anomaly_detection)*** or ***[Anomaly Detection](https://en.wikipedia.org/wiki/Anomaly_detection)*** .

**PyOD has been successfully used in academic researches [4, 8] and under active development**. However, the purpose of the toolkit is quick exploration. Using it as the final output should be cautious, and fine-tunning may be needed to generate meaningful results. The authours can be reached out at yuezhao@cs.toronto.edu; comments, questions, pull requests and issues are welcome. **Enjoy catching outliers!**

**Table of Contents**:
<!-- TOC -->

- [Key Links & Resources](#key-links-resources)
- [Quick Introduction](#quick-introduction)
- [Installation](#installation)
- [API Cheatsheet & Reference](#api-cheatsheet-reference)
- [Quick Start for Outlier Detection](#quick-start-for-outlier-detection)
- [Quick Start for Combining Outlier Scores from Various Base Detectors](#quick-start-for-combining-outlier-scores-from-various-base-detectors)
- [Reference](#reference)

<!-- /TOC -->

------------------------------
# Key Links & Resources

- **[Documentation & API Reference](https://pyod.readthedocs.io)** [![Documentation Status](https://readthedocs.org/projects/pyod/badge/?version=latest)](https://pyod.readthedocs.io/en/latest/?badge=latest)

- **[Current version on PyPI](https://pypi.org/project/pyod/)** [![PyPI version](https://badge.fury.io/py/pyod.svg)](https://badge.fury.io/py/pyod) 

- **[Github repository with examples](https://github.com/yzhao062/Pyod/examples)** | **[Example Documentation](https://pyod.readthedocs.io/en/latest/example.html)**

- **Anomaly detection related resources**, e.g., books, papers and videos, can be found at **[anomaly-detection-resources.](https://github.com/yzhao062/anomaly-detection-resources)**

-----------------------------------

### Quick Introduction

PyOD toolkit consists of three major groups of functionalities: (i) **outlier detection algorithms**; (ii) **outlier ensemble frameworks** and (iii) **outlier detection utility functions**.

- Individual Detection Algorithms:  
  1. **Local Outlier Factor, LOF** [1]
  2. **Isolation Forest, iForest** [2]
  3. **One-Class Support Vector Machines** [3]
  4. **kNN** Outlier Detection (use the distance to the kth nearst neighbor as the outlier score)
  5. **Average KNN** Outlier Detection (use the average distance to k nearst neighbors as the outlier score)
  6. **Median KNN** Outlier Detection (use the median distance to k nearst neighbors as the outlier score)
  7. **Histogram-based Outlier Score, HBOS** [5]
  8. **Angle-Based Outlier Detection, ABOD** [7]
  9. **Fast Angle-Based Outlier Detection, FastABOD** [7]
  10. More to add...

- Outlier Ensemble Framework (Outlier Score Combination Frameworks)
  1. **Feature bagging**
  2. **Average of Maximum (AOM)** [6]
  3. **Maximum of Average (MOA)** [6]
  4. **Threshold Sum (Thresh)** [6]

- Utility functions:
   1. **score_to_lable()**: convert raw outlier scores to binary labels
   2. **precision_n_scores()**: one of the popular evaluation metrics for outlier mining (precision @ rank n)
   3. **generate_data()**: generate pseudo data for outlier detection experiment
   4. **wpearson()**: weighted pearson is useful in pseudo ground truth generation
------------

### Installation

It is advised to use **pip** for installation. Please make sure **the latest version** is installed since PyOD is currently updated on **a daily basis**:
````cmd
pip install pyod
pip install --upgrade pyod # make sure the latest version is installed!
````
or 
````cmd
pip install pyod==x.y.z  # (x.y.z) is the current version number
````
 Alternatively, [downloading/cloning the Github repository](https://github.com/yzhao062/Pyod) also works. You could unzip the files and execute the following command in the folder where the files get decompressed.

````cmd
python setup.py install
````
Library Dependency (work only with **Python 3.5+**,  e.g. 3.5 & 3.6):
- scipy>=0.19.1
- pandas>=0.21
- numpy>=1.13
- scikit_learn>=0.19.1
- matplotlib>=2.0.2 **(optional but required for running examples)**

------------
### API Cheatsheet & Reference

Full API Reference: (http://pyod.readthedocs.io/en/latest/api.html)

API cheatsheet:

- **fit(X)**: Fit detector.
- **fit_predict(X)**: Fit detector and predict if a particular sample is an outlier or not.
- **fit_predict_evaluate(X, y)**: Fit, predict and then evaluate with ROC and Precision @ rank n. 
- **decision_function(X)**: Return raw outlier scores of a sample.
- **predict(X)**: Predict if a particular sample is an outlier or not. The model must be fitted first.
- **predict_proba(X)**: Predict the probability of a sample being outlier. The model must be fitted first.
- **predict_rank(X)**: Predict the outlyingness rank of a sample.


Import outlier detection models, such like:
````python
from pyod.models.knn import KNN
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
...
````

Import utility functions:
````python
from pyod.util.utility import precision_n_scores
...
````

Full package structure can be found below:
- http://pyod.readthedocs.io/en/latest/genindex.html
- http://pyod.readthedocs.io/en/latest/py-modindex.html

------------

### Quick Start for Outlier Detection
See examples for more demos. "examples/knn_example.py" demonstrates the basic APIs of PyOD using kNN detector. **It is noted the APIs for other detectors are similar**.

0. Import models
    ````python
    from pyod.models.knn import KNN  # kNN detector

    from pyod.utils.load_data import generate_data
    from pyod.utils.utility import precision_n_scores
    from sklearn.metrics import roc_auc_score
    ````

1. Generate sample data first; normal data is generated by a 2-d Gaussian distribution, and outliers are generated by a 2-d uniform distribution.
    ````python
    contamination = 0.1  # percentage of outliers
    n_train = 1000  # number of training points
    n_test = 500  # number of testing points

    X_train, y_train, c_train, X_test, y_test, c_test = generate_data(
        n_train=n_train, n_test=n_test, contamination=contamination)
    ````

2. Initialize a kNN detector, fit the model, and make the prediction.
    ```python
    # train a k-NN detector (default parameters, k=5)
    clf = KNN()
    clf.fit(X_train)

    y_train_pred = clf.y_pred
    y_train_score = clf.decision_scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier label (0 or 1)
    y_test_score = clf.decision_function(X_test) 
    ```
3. Evaluate the prediction by ROC and Precision@rank *n* (p@n):
    ```python
    print(n_train.format(
        roc=roc_auc_score(y_train, y_train_score),
        prn=precision_n_scores(y_train, y_train_score)))

    print(n_train.format(
        roc=roc_auc_score(y_test, y_test_score),
        prn=precision_n_scores(y_test, y_test_score)))
    ```
    See a sample output:
    ````python
    Train ROC:0.9473, precision@n:0.7857
    Test ROC:0.992, precision@n:0.9
    ````
    
To check the result of the classification visually ([knn_figure](https://github.com/yzhao062/Pyod/blob/master/examples/example_figs/knn.png)):
![kNN example figure](https://github.com/yzhao062/Pyod/blob/master/examples/example_figs/knn.png)

---
### Quick Start for Combining Outlier Scores from Various Base Detectors

"examples/comb_example.py" is a quick demo for showing the API for combining multiple algorithms. Given we have *n* individual outlier detectors, each of them generates an individual score for all samples. The task is to combine the outputs from these detectors effectivelly.

**Key Step: conducting Z-score normalization on raw scores before the combination.**
Four combination mechanisms are shown in this demo:
1. Mean: use the mean value of all scores as the final output.
2. Max: use the max value of all scores as the final output.
3. Average of Maximum (AOM): first randomly split n detectors in to p groups. For each group, use the maximum within the group as the group output. Use the average of all group outputs as the final output.
4. Maximum of Average (MOA): similarly to AOM, the same grouping is introduced. However, we use the average of a group as the group output, and use maximum of all group outputs as the final output.
To better understand the merging techniques, refer to [6].

The walkthrough of the code example is provided:

0. Import models and generate sample data
    ````python
    from pyod.models.knn import Knn
    from pyod.models.combination import aom, moa # combination methods
    from pyod.utils.load_data import generate_data
    from pyod.utils.utility import precision_n_scores
    from pyod.utils.utility import standardizer
    from sklearn.metrics import roc_auc_score
    
    X, y, _ = generate_data(train_only=True)  # load data
    ````
    
1. First initialize 20 kNN outlier detectors with different k (10 to 200), and get the outlier scores:
    ```python
    # initialize 20 base detectors for combination
    k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                150, 160, 170, 180, 190, 200]

    train_scores = np.zeros([X_train.shape[0], n_clf])
    test_scores = np.zeros([X_test.shape[0], n_clf])

    for i in range(n_clf):
        k = k_list[i]

        clf = KNN(n_neighbors=k, method='largest')
        clf.fit(X_train_norm)

        train_scores[:, i] = clf.decision_scores.ravel()
        test_scores[:, i] = clf.decision_function(X_test_norm).ravel()
    ```
2. Then the output codes are standardized into zero mean and unit std before combination.
    ```python
    decision_scores
    train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)
    ```
3. Then four different combination algorithms are applied as described above:
    ```python
    comb_by_mean = np.mean(test_scores_norm, axis=1)
    comb_by_max = np.max(test_scores_norm, axis=1)
    comb_by_aom = aom(test_scores_norm, 5) # 5 groups
    comb_by_moa = moa(test_scores_norm, 5)) # 5 groups
    ```
4. Finally, all four combination methods are evaluated with 20 iterations:
    ````bash
    Combining 20 kNN detectors
    ite 1 comb by mean, ROC: 0.9014 precision@n_train: 0.4531
    ite 1 comb by max, ROC: 0.9014 precision@n_train: 0.5
    ite 1 comb by aom, ROC: 0.9081 precision@n_train: 0.5
    ite 1 comb by moa, ROC: 0.9052 precision@n_train: 0.4843
    ...
    
    Summary of 10 iterations
    comb by mean, ROC: 0.9196, precision@n: 0.5464
    comb by max, ROC: 0.9198, precision@n: 0.5532
    comb by aom, ROC: 0.9260, precision@n: 0.5630
    comb by moa, ROC: 0.9244, precision@n: 0.5523
    ````
---    

### Reference
[1] Breunig, M.M., Kriegel, H.P., Ng, R.T. and Sander, J., 2000, May. LOF: identifying density-based local outliers. In *ACM SIGMOD Record*, pp. 93-104. ACM.

[2] Liu, F.T., Ting, K.M. and Zhou, Z.H., 2008, December. Isolation forest. In *ICDM '08*, pp. 413-422. IEEE.

[3] Ma, J. and Perkins, S., 2003, July. Time-series novelty detection using one-class support vector machines. In *IJCNN' 03*, pp. 1741-1745. IEEE.

[4] Y. Zhao and M.K. Hryniewicki, "DCSO: Dynamic Combination of Detector Scores for Outlier Ensembles," *ACM SIGKDD Workshop on Outlier Detection De-constructed*, 2018. Submitted, under review.

[5] Goldstein, M. and Dengel, A., 2012. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm. In *KI-2012: Poster and Demo Track*, pp.59-63.

[6] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms for outlier ensembles.*ACM SIGKDD Explorations Newsletter*, 17(1), pp.24-47.

[7] Kriegel, H.P. and Zimek, A., 2008, August. Angle-based outlier detection in high-dimensional data. In *KDD '08*, pp. 444-452. ACM.

[8] Y. Zhao and M.K. Hryniewicki, "XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning," *IEEE International Joint Conference on Neural Networks*, 2018.

