Outlier Detection 101 (WIP)
==================================

Outlier detection broadly refers to the task of identifying observations which may be considered anomalous considering the distribution of a sample.
Any observation belonging to the distribution is referred to as an inlier and any outlying point is referred to as an outlier. 

In the context of machine learning, there are three common approaches taken for this task: 
- Unsupervised outlier detection
    - Training data (unlabelled) contains both normal and anomalous observations.
    - This approach is taken when outliers are considered points that exist in low-density regions.  

- Semi-supervised novelty detection
    - Training data consists only of observations describing normal behavior
    - Any new observation sufficiently differing from the training data, even if it exists in a high-density region, as an outlier. 

- Supervised outlier classification
    - The ground truth label (inlier vs outlier) for every observation is known
    - Can be considered an imbalanced supervised classification problem

Outlier Detection, Anomaly Detection, and Novelty Detection
-----------------------------------------------------------

To finish, several links for now:

- https://medium.com/@mehulved1503/outlier-detection-and-anomaly-detection-with-machine-learning-caa96b34b7f6
- https://scikit-learn.org/stable/modules/outlier_detection.html
- https://en.wikipedia.org/wiki/Anomaly_detection
- https://www-users.cs.york.ac.uk/vicky/myPapers/Hodge+Austin_OutlierDetection_AIRE381.pdf