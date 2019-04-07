Installation
============

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as PyOD is updated frequently:

.. code-block:: bash

   pip install pyod            # normal install
   pip install --upgrade pyod  # or update if needed
   pip install --pre pyod      # or include pre-release version for new features

Alternatively, you could clone and run setup.py file:

.. code-block:: bash

   git clone https://github.com/yzhao062/pyod.git
   cd pyod
   pip install .


.. warning::

    To be consistent with the dependent libraries, e.g., scikit-learn,
    PyOD will stop supporting Python 2.7 soon (to be decided).
    We encourage you to move to Python 3.5 or newer for latest functions and
    bug fixes. More information could be found at
    `scikit-learn install page <https://scikit-learn.org/stable/install.html>`_.

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