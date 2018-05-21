# Python Outlier Detection (PyOD)

**[Current version: 0.1.3](https://pypi.org/project/pyod/)**

**Note: PyOD is still under development without full test coverages. However, it has been successfully used in the various academic research projects** [8, 9].

- [Python Outlier Detection (PyOD)](#python-outlier-detection-pyod)
    - [Quick Introduction](#quick-introduction)
    - [Installation (**Current version: 0.1.3**)](#installation-current-version-013)
    - [API Cheatsheet](#api-cheatsheet)
    - [Quick Start for Outlier Detection](#quick-start-for-outlier-detection)
    - [Quick Start for Combining Outlier Scores from Various Base Detectors](#quick-start-for-combining-outlier-scores-from-various-base-detectors)
    - [Reference](#reference)


More anomaly detection related resources, e.g., books, papers and videos, can be found at [anomaly-detection-resources.](https://github.com/yzhao062/anomaly-detection-resources)

### Quick Introduction
PyOD is a **Python-based toolkit** to identify outliers in data with both unsupervised and supervised algorithms. It strives to provide unified APIs across for different anomaly detection algorithms. The toolkit consists of three major groups of functionalities:
- Individual Detection Algorithms:  
  1. **Local Outlier Factor, LOF** (wrapped on sklearn implementation) [1]
  2. **Isolation Forest, iForest** (wrapped on sklearn implementation) [2]
  3. **One-Class Support Vector Machines** (wrapped on sklearn implementation) [3]
  4. **kNN** Outlier Detection (use the distance to the kth nearst neighbor as the outlier score)
  5. **Average KNN** Outlier Detection (use the average distance to k nearst neighbors as the outlier score)
  6. **Median KNN** Outlier Detection (use the median distance to k nearst neighbors as the outlier score)
  7. **Global-Local Outlier Score From Hierarchies** [4]
  8. **Histogram-based Outlier Score, HBOS** [5]
  9. **Angle-Based Outlier Setection, ABOD** [7]

- Ensemble Framework (Outlier Score Combination Frameworks)
  1. **Feature bagging**
  2. **Average of Maximum (AOM)** [6]
  3. **Maximum of Average (MOA)** [6]
  4. **Threshold Sum (Thresh)** [6]

- Utility functions:
   1. **scores_to_lables()**: converting raw outlier scores to binary labels
   2. **precision_n_scores()**: one of the popular evaluation metrics for outlier mining (precision @ rank n)
  
 Please be advised the purpose of the toolkit is for quick exploration. Using it as the final output should be understood with cautions. Fine-tunning may be needed to generate meaningful results. It is recommended to be used for the first-step data exploration only. Due to the restriction of time, the unit tests are not supplied but have been planned to implement.

### Installation (**[Current version: 0.1.3](https://pypi.org/project/pyod/)**)

It is advised to install with **pip** to manage the package:
````cmd
pip install pyod
````
Pypi can be unstable sometimes. Alternatively, [downloading/cloning the Github repository](https://github.com/yzhao062/Pyod) also works. You could unzip the files and execute the following command in the folder where the files get decompressed.

````cmd
python setup.py install
````
Library Dependency (work only with **Python 3**):
- scipy>=0.19.1
- pandas>=0.21
- numpy>=1.13
- scikit_learn>=0.19.1
- matplotlib>=2.0.2 **(optinal)**

------------
### API Cheatsheet
For all algorithms implemented/wrapped in PyOD, the similar API is forced for consistency.

- **fit()**: fitting the model with the training data
- **decision_function()**: return raw outlier scores for test data
- **predict()**: returning binary outlier labels of test data
- **predict_proba()**: returning outlier probability of test data (0 to 1)
- **predict_rank()**: returning outlier rank of test data (data outlyness rank in training data)

Import outlier detection models:
````python
from pyod.models.knn import Knn
from pyod.models.abod import ABOD
from pyod.models.hbos import HBOS
````

Import utility functions:
````python
from pyod.util.utility import precision_n_scores
````
Full package structure can be found below:

````
pyod
│  
├───data
│       load_data.py
|           generate_data(): generae and load sample data
|           load_cardio(): load cardio data
|           load_letter(): load letter data
├───examples
│       abod_example.py: Example of using ABOD for outlier detection
│       comb_example.py: Example of combining multiple base outlier scores
│       hbos_example.py: Example of using HBOS for outlier detection
│       knn_example.py: Example of using kNN for outlier detection
├───models
│       abod.py: 
|           class ABOD(), from pyod.models.abod import ABOD
│       combination.py
|           amo(), from pyod.models.combination import aom
|           moa(), from pyod.models.combination import moa
│       glosh.py: class Glosh(), from pyod.models.glosh import Glosh
│       hbos.py: class HBOS(), from pyod.models.hbos import HBOS
│       iforest: class IForest(), from pyod.models.iforest import IForest
│       knn.py: class Knn(), from pyod.models.knn import Knn
│       lof.py: class Lof(), from pyod.models.lof import Lof
│       ocsvm.py: class OCSVM(), from pyod.models.ocsvm import OCSVM
│
├───utils
        stat_models.py
        utility.py
            standardizer(): z- normalization function
            scores_to_lables(): turn raw outlier scores to binary labels (0 or 1)
            precision_n_scores(): Utlity function to calculate precision@n
````

------------

### Quick Start for Outlier Detection
See pyod/examples for more examples.

"examples/knn_example.py" demonstrates the basic APIs of PyOD with kNN detector. **It is noted the APIs for other detectors are similar**.

0. Import models
    ````python
    from pyod.data.load_data import generate_data
    from pyod.models.knn import Knn # kNN detector

    from pyod.utils.utility import precision_n_scores
    from sklearn.metrics import roc_auc_score
    ````

1. Generate sample data first; normal data is generated by a 2-d gaussian distribution, and outliers are generated by a 2-d uniform distribution.
    ````python
    contamination = 0.1  # percentage of outliers
    n_train = 1000  # number of training points
    n_test = 500  # number of testing points

    X_train, y_train, c_train, X_test, y_test, c_test = generate_data(
        n=n_train, contamination=contamination, n_test=n_test)
    ````

2. Initialize a kNN detector, fit the model, and make the prediction.
    ```python
    # train a k-NN detector (default parameters, k=10)
    clf = Knn()
    clf.fit(X_train)

    # get the prediction label and scores on the training data
    y_train_pred = clf.y_pred
    y_train_score = clf.decision_scores

    # get the prediction on the test data
    y_test_pred = clf.predict(X_test)  # outlier label (0 or 1)
    y_test_score = clf.decision_function(X_test)  # outlier scores
    ```
3. Evaluate the prediction by ROC and Precision@rank *n* (p@n):
    ```python
    print('Train ROC:{roc}, precision@n:{prn}'.format(
        roc=roc_auc_score(y_train, y_train_score),
        prn=precision_n_scores(y_train, y_train_score)))

    print('Test ROC:{roc}, precision@n:{prn}'.format(
        roc=roc_auc_score(y_test, y_test_score),
        prn=precision_n_scores(y_test, y_test_score)))
    ```
    See a sample output:
    ````python
    Train ROC:0.9473, precision@n:0.7857
    Test ROC:0.992, precision@n:0.9
    ````
    
To check the result of the classification visually ([knn_figure](https://github.com/yzhao062/Pyod/blob/master/pyod/examples/example_figs/knn.png)):
![kNN example figure](https://github.com/yzhao062/Pyod/blob/master/pyod/examples/example_figs/knn.png)

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

0. Import models
    ````python
    from pyod.data.load_data import load_cardio, load_letter
    from pyod.models.knn import Knn
    from pyod.models.combination import aom, moa # combination methods
    from pyod.utils.utility import precision_n_scores
    from pyod.utils.utility import standardizer
    from sklearn.metrics import roc_auc_score
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

        clf = Knn(n_neighbors=k, method='largest')
        clf.fit(X_train_norm)

        train_scores[:, i] = clf.decision_scores.ravel()
        test_scores[:, i] = clf.decision_function(X_test_norm).ravel()
    ```
2. Then the output codes are standardized into zero mean and unit std before combination.
    ```python
    # scores have to be normalized before combination
    train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores)
    ```
3. Then four different combination algorithms are applied as described above:
    ```python
    comb_by_mean = np.mean(test_scores_norm, axis=1)
    comb_by_max = np.max(test_scores_norm, axis=1)
    comb_by_aom = aom(test_scores_norm, 5, 20) # 5 groups
    comb_by_moa = moa(test_scores_norm, 5, 20)) # 5 groups
    ```
4. Finally, all four combination methods are evaluated with 20 iterations:
    ````bash
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

[4] Campello, R.J., Moulavi, D., Zimek, A. and Sander, J., 2015. Hierarchical density estimates for data clustering, visualization, and outlier detection. *TKDD*, 10(1), pp.5.

[5] Goldstein, M. and Dengel, A., 2012. Histogram-based outlier score (hbos): A fast unsupervised anomaly detection algorithm. In *KI-2012: Poster and Demo Track*, pp.59-63.

[6] Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms for outlier ensembles.*ACM SIGKDD Explorations Newsletter*, 17(1), pp.24-47.

[7] Kriegel, H.P. and Zimek, A., 2008, August. Angle-based outlier detection in high-dimensional data. In *KDD '08*, pp. 444-452. ACM.

[8] Y. Zhao and M.K. Hryniewicki, "XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning," *IEEE International Joint Conference on Neural Networks*, 2018.

[9] Y. Zhao and M.K. Hryniewicki, "DCSO: Dynamic Combination of Detector Scores for Outlier Ensembles," *ACM SIGKDD Workshop on Outlier Detection De-constructed*, 2018. Submitted, under review.
