# Python Outlier Detection (PyOD)

### Note: the project is still under development as of Feb 7th 2018.
**A Python Toolkit for outlier Detection**
PyOD is a Python-based toolkit to identify anomalies in data with unsupervised and supervised approach. The toolkits consist of two major functionalities:
- Individual Algorithms
  1.  Local Outlier Factor (wrapped on sklearn implementation)
  2. Isolation Fores (wrapped on sklearn implementation)
  3. One-Class support vector machines (wrapped on sklearn implementation)
  4. TO ADD MORE
- Ensemble Framework
  1. Feature bagging
  2. TO ADD MORE
  
Before using the toolkit, please be advised the purpose of the tool is only for quick exploration. Using it as the final result should be understood with cautions. Fine-tunning may be needed to generate meaningful solution. I would recommend to use this as the first-step data exploration tool, and build your model/reuse the this model to get more accurate results.

One of the ensemble framework, Pyador, follows the procedures 
**It follows the following procedure**:
1. Take in the cleaned data in Pandas format.
2. Perform necessary data quality check to ensure it is compatible with the models. **Note: you are supposed to supply cleaned pandas dataframe with only numerical variables to the model**
   1. Missing value check and imputation
   2. Handle single value variables
   3. Convert categorical variables into numerical format
3. Leverage unsupervised anomaly detection methods in Sklearn, such as Isolation Forest
4. Build anomaly detection rule by decision tree visualization
   1. For example, we find 1000 potential anomalies in step 3, we will randomly sample another 1000 "normal" data points from the original dataset
   2. Combine "normal" data and outliers, and use it as the training data to build a decision tree classifier.
   3. Using the decision tree classification rule as the anomaly determination rule
   4. Output feature importance ranking based on the decision tree result
5. Provide related visuals
	1. Correlation Matrix
	2. 2-D data visulization using multi-dimensional scaling
6. TODO

