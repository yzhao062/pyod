Installation
============

It is recommended to use **pip** or **conda** for installation. Please make sure
**the latest version** is installed, as PyOD is updated frequently:

.. code-block:: bash

   pip install pyod            # normal install
   pip install --upgrade pyod  # or update if needed

.. code-block:: bash

   conda install -c conda-forge pyod

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/pyod.git
   cd pyod
   pip install .


**Required Dependencies**\ :


* Python 2.7, 3.5, 3.6, or 3.7
* combo>=0.0.8
* joblib
* numpy>=1.13
* numba>=0.35
* scipy>=0.20.0
* scikit_learn>=0.19.1
* statsmodels


**Optional Dependencies (see details below)**:

* combo (optional, required for models/combination.py and FeatureBagging)
* keras (optional, required for AutoEncoder, and other deep learning models)
* matplotlib (optional, required for running examples)
* pandas (optional, required for running benchmark)
* suod (optional, required for running SUOD model)
* tensorflow (optional, required for AutoEncoder, and other deep learning models)
* xgboost (optional, required for XGBOD)

.. warning::

    PyOD has multiple neural network based models, e.g., AutoEncoders, which are
    implemented in both PyTorch and Tensorflow. However, PyOD does **NOT** install DL libraries for you.
    This reduces the risk of interfering with your local copies.
    If you want to use neural-net based models, please make sure Keras and a backend library, e.g., TensorFlow, are installed.
    Instructions are provided: `neural-net FAQ <https://github.com/yzhao062/pyod/wiki/Setting-up-Keras-and-Tensorflow-for-Neural-net-Based-models>`_.
    Similarly, models depending on **xgboost**, e.g., XGBOD, would **NOT** enforce xgboost installation by default.


.. warning::

    PyOD contains multiple models that also exist in scikit-learn. However, these two
    libraries' API is not exactly the same--it is recommended to use only one of them
    for consistency but not mix the results. Refer `scikit-learn and PyOD <https://pyod.readthedocs.io/en/latest/issues.html>`_
    for more information.