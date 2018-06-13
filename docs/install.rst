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

- Python 2.7, 3.4, 3.5 or 3.6
- numpy>=1.13
- scipy>=0.19.1
- scikit_learn>=0.19.1
- matplotlib
- nose

**Known Issue**: PyOD depends on matplotlib, which would throw errors in conda
virtual environment on mac OS. See causes and solutions `here <https://github.com/yzhao062/Pyod/issues/6>`_
