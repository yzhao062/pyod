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

    The maintenance of Python 2.7 will be stopped by January 1, 2020 (see `official announcement <https://github.com/python/devguide/pull/344>`_).
    To be consistent with the Python change and PyOD's dependent libraries, e.g., scikit-learn, we will
    stop supporting Python 2.7 in the near future (dates are still to be decided). We encourage you to use
    Python 3.5 or newer for the latest functions and bug fixes. More information can
    be found at `Moving to require Python 3 <https://python3statement.org/>`_.

**Required Dependencies**\ :


* Python 2.7, 3.5, 3.6, or 3.7
* combo>=0.0.8
* joblib
* numpy>=1.13
* numba>=0.35
* pandas>=0.25
* scipy>=0.19.1
* scikit_learn>=0.19.1
* statsmodels


**Optional Dependencies (see details below)**:

* combo (optional, required for models/combination.py)
* keras (optional, required for AutoEncoder)
* matplotlib (optional, required for running examples)
* pandas (optional, required for running benchmark)
* tensorflow (optional, required for AutoEncoder, other backend works)
* xgboost (optional, required for XGBOD)

.. warning::

    PyOD has multiple neural network based models, e.g., AutoEncoders, which are
    implemented in Keras. However, PyOD does **NOT** install **keras** and/or
    **tensorflow** for you. This reduces the risk of interfering with your local copies.
    If you want to use neural-net based models, please make sure Keras and a backend library, e.g., TensorFlow, are installed.
    Instructions are provided: `neural-net FAQ <https://github.com/yzhao062/pyod/wiki/Setting-up-Keras-and-Tensorflow-for-Neural-net-Based-models>`_.
    Similarly, models depending on **xgboost**, e.g., XGBOD, would **NOT** enforce xgboost installation by default.


.. warning::

    Running examples needs **matplotlib**, which may throw errors in conda
    virtual environment on mac OS. See reasons and solutions `mac_matplotlib <https://github.com/yzhao062/pyod/issues/6>`_.


.. warning::

    PyOD contains multiple models that also exist in scikit-learn. However, these two
    libraries' API is not exactly the same--it is recommended to use only one of them
    for consistency but not mix the results. Refer `sckit-learn and PyOD <https://pyod.readthedocs.io/en/latest/issues.html>`_
    for more information.