Installation
============

It is recommended to use **pip** for installation. Please make sure
**the latest version** is installed, as PyOD is updated frequently:

.. code-block:: bash

    pip install pyod
    pip install --upgrade pyod  # make sure the latest version is installed!

Alternatively, install from github directly (**not recommended**)

.. code-block:: bash

    git clone https://github.com/yzhao062/pyod.git
    python setup.py install

**Required Dependencies**:

- Python 2.7, 3.5, 3.6, or 3.7
- nose
- numpy>=1.13
- numba>=0.35
- scipy>=0.19.1
- scikit_learn>=0.19.1

**Optional Dependencies (required for running examples or AutoEncoder)**:

- keras (optional, required if calling AutoEncoder, other backend works)
- matplotlib (optional, required for running examples)
- tensorflow (optional, required if calling AutoEncoder, other backend works)

.. warning::

    PyOD examples depend on matplotlib, which would throw errors in conda
    virtual environment on mac OS. See
    `mac_matplotlib <https://github.com/yzhao062/Pyod/issues/6>`_.

.. warning::

    PyOD builds on various packages, which most of them you should have
    already installed. If you are installing PyOD in a fresh state (virtualenv),
    downloading and installing the dependencies, e.g., TensorFlow, may take
    3-5 mins.

.. warning::

    If you are willing to run examples, matplotlib is required. PyOD does
    not list it as a required package for eliminating the dependency.
    Similarly, Keras and TensorFlow are listed as optional. However, they are
    both required if you want to use neural network based models, such as
    AutoEncoder. See causes and solutions
    `dependency <https://github.com/yzhao062/Pyod/issues/19>`_.
