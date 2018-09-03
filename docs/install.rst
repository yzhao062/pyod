============
Installation
============

It is recommended to use **pip** for installation.
Please make sure **the latest version** is installed since PyOD is currently updated on **a daily basis**:

.. code-block:: bash

    pip install pyod
    pip install --upgrade pyod  # make sure the latest version is installed!

Alternatively, install from github directly (**not recommended**)

.. code-block:: bash

    git clone https://github.com/yzhao062/pyod.git
    python setup.py install

**Required Dependency**:

- Python 2.7, 3.5 or 3.6
- keras
- matplotlib (optional, required for running examples)
- nose
- numpy>=1.13
- numba>=0.35
- scipy>=0.19.1
- scikit_learn>=0.19.1
- tensorflow (optional, required if calling AutoEncoder, other backend works)

**Known Issue 1**: PyOD depends on matplotlib, which would throw errors in conda
virtual environment on mac OS. See causes and solutions `here <https://github.com/yzhao062/Pyod/issues/6>`_

**Known Issue 2**: PyOD builds on various packages, which most of them you should have
already installed. If you are installing PyOD in a fresh state (virtualenv),
downloading and installing the dependencies, e.g., TensorFlow, may take
**5-10 mins**.
