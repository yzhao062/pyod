# *Py*thon-*A*nomaly-*D*etect*or* (Pyador)

### Note: the project is still under development as of Oct 7th 2017.
**A Python Toolkit for Unsupervised Anomaly Detection**
Pyador is a Python-based toolkit to identify anamolies in data with unsupervised and supervised approach.
Before using the toolkit, please be advised the purpose of the tool is only for quick exploration. Using it as the final result should be understood with cautions. Fine-tunning may be needed to generate mature solution. I would recommend to use this as the first-step data exploration tool, and build your model/reuse the this model to get more accurate result.

**It follows the following procedure**:
1. Take in the cleaned data in Pandas format.
2. Perform necessary data quality check to ensure it is compatible with the models. **Note: you are supposed to supply cleaned pandas dataframe with only numerical variables to the model**
   1. Missing value check and imputation
   2. Handle single value variables
   3. Convert categorical variables into numerical format
3. Leverage unsupervised anamoly detection methods in Sklearn, such as Isolation Forest
4. Build anamoly detection rule by decision tree visualization
   1. For example, we find 1000 potential anamolies in step 3, we will randomly sample another 1000 "normal" data points from the original dataset
   2. Combine "normal" data and outliers, and use it as the training data to build a decision tree classifier.
   3. Using the decision tree classification rule as the anamoly determination rule
   4. Output feature importance ranking based on the decision tree result
5. Provide related visuals
	1. Correlation Matrix
	2. 2-D data visulization using multi-dimensional scaling
6. TODO

