Outlier Detection 101
=====================

Outlier detection broadly refers to the task of identifying observations which may be considered anomalous given the distribution of a sample.
Any observation belonging to the distribution is referred to as an inlier and any outlying point is referred to as an outlier. 

In the context of machine learning, there are three common approaches for this task: 

1. Unsupervised Outlier Detection
    - Training data (unlabelled) contains both normal and anomalous observations.
    - The model identifies outliers during the fitting process.
    - This approach is taken when outliers are defined as points that exist in low-density regions in the data. 
    - Any new observations that do not belong to high-density regions are considered outliers. 

2. Semi-supervised Novelty Detection
    - Training data consists only of observations describing normal behavior.
    - The model is fit on training data and then used to evaluate new observations. 
    - This approach is taken when outliers are defined as points differing from the distribution of the training data. 
    - Any new observations differing from the training data within a threshold, even if they form a high-density region, are considered outliers. 

3. Supervised Outlier Classification
    - The ground truth label (inlier vs outlier) for every observation is known.
    - The model is fit on imbalanced training data and then used to classify new observations. 
    - This approach is taken when ground truth is available and it is assumed that outliers will follow the same distribution as in the training set.
    - Any new observations are classified using the model.

The algorithms found in *PyOD* focus on the first two approaches which differ in terms of how the training data is defined and how the model's outputs are interpreted. If interested in learning more,
please refer to our `Anomaly Detection Resources <https://github.com/yzhao062/anomaly-detection-resources>`_ page for
relevant related books, papers, videos, and toolboxes.
