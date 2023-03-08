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


* Python 3.6+
* joblib
* matplotlib
* numpy>=1.19
* numba>=0.51
* scipy>=1.5.1
* scikit_learn>=0.20.0
* six


**Optional Dependencies (see details below)**:

* combo (optional, required for models/combination.py and FeatureBagging)
* keras/tensorflow (optional, required for AutoEncoder, and other deep learning models)
* pandas (optional, required for running benchmark)
* suod (optional, required for running SUOD model)
* xgboost (optional, required for XGBOD)
* pythresh to use thresholding

.. warning::

    PyOD has multiple neural network based models, e.g., AutoEncoders, which are
    implemented in both Tensorflow and PyTorch. However, PyOD does **NOT** install these deep learning libraries for you.
    This reduces the risk of interfering with your local copies.
    If you want to use neural-net based models, please make sure these deep learning libraries are installed.
    Instructions are provided: `neural-net FAQ <https://github.com/yzhao062/pyod/wiki/Setting-up-Keras-and-Tensorflow-for-Neural-net-Based-models>`_.
    Similarly, models depending on **xgboost**, e.g., XGBOD, would **NOT** enforce xgboost installation by default.
