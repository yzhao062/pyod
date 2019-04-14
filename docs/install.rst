Installation
============

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as PyOD is updated frequently:

.. code-block:: bash

   pip install pyod
   pip install --upgrade pyod # make sure the latest version is installed!

Alternatively, install from github directly (\ **NOT Recommended**\ )

.. code-block:: bash

   git clone https://github.com/yzhao062/pyod.git
   python setup.py install

**Required Dependencies**\ :


* Python 2.7, 3.5, 3.6, or 3.7
* numpy>=1.13
* numba>=0.35
* scipy>=0.19.1
* scikit_learn>=0.19.1


**Optional Dependencies (see details below)**:

* Keras (optional, required for AutoEncoder)
* Matplotlib (optional, required for running examples)
* Tensorflow (optional, required for AutoEncoder, other backend works)
* XGBoost (optional, required for XGBOD)

.. warning::

    Running examples needs Matplotlib, which may throw errors in conda
    virtual environment on macOS. See
    `mac_matplotlib <https://github.com/yzhao062/Pyod/issues/6>`_.


.. warning::

    Keras and/or TensorFlow are listed as optional. However, they are
    both required if you want to use neural network based models, such as
    AutoEncoder. See reasons and solutions `neural-net FAQ <https://github.com/yzhao062/pyod/wiki/Setting-up-Keras-and-Tensorflow-for-Neural-net-Based-models>`_.

.. warning::

    xgboost is listed as optional. However, it is required to run XGBOD.
    Users are expected to install **xgboost** to use XGBOD model.