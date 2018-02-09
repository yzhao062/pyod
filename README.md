# Python Outlier Detection (PyOD)

### Note: the project is still under development as of Feb 7th 2018.
**A Python Toolkit for outlier Detection**
PyOD is a Python-based toolkit to identify anomalies in data with unsupervised and supervised approach. The toolkits consist of two major functionalities:
- Individual Algorithms
  1.  Local Outlier Factor (wrapped on sklearn implementation)
  2. Isolation Forest (wrapped on sklearn implementation)
  3. One-Class support vector machines (wrapped on sklearn implementation)
  4. **KNN Outlier Detection (implemented)**
  5. **Average KNN Outlier Detection (implemented)**
  6. **Median KNN Outlier Detection (implemented)**
  7. **Global-Local Outlier Score From Hierarchies (implemented, under optimization)**
  8. More to add
- Ensemble Framework
  1. Feature bagging
  2. More to add
  
Before using the toolkit, please be advised the purpose of the tool is only for quick exploration. Using it as the final result should be understood with cautions. Fine-tunning may be needed to generate meaningful solution. I would recommend to use this as the first-step data exploration tool, and build your model/reuse the this model to get more accurate results.
