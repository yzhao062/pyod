Installation
============

PyOD is designed for easy installation using either **pip** or **conda**.
We recommend using the latest version of PyOD due to frequent updates and enhancements:

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


* Python 3.6 or higher
* joblib
* matplotlib
* numpy>=1.19
* numba>=0.51
* scipy>=1.5.1
* scikit_learn>=0.22.0
* six


**Optional Dependencies (see details below)**:

* combo (optional, required for models/combination.py and FeatureBagging)
* keras/tensorflow (optional, required for AutoEncoder, and other deep learning models)
* suod (optional, required for running SUOD model)
* xgboost (optional, required for XGBOD)
* pythresh (optional, required for thresholding)

.. warning::

    PyOD includes several neural network-based models, such as AutoEncoders, implemented in Tensorflow and PyTorch. These deep learning libraries are not automatically installed by PyOD to avoid conflicts with existing installations. If you plan to use neural-net based models, please ensure these libraries are installed. See the `neural-net FAQ <https://github.com/yzhao062/pyod/wiki/Setting-up-Keras-and-Tensorflow-for-Neural-net-Based-models>`_ for guidance. Additionally, xgboost is not installed by default but is required for models like XGBOD.
