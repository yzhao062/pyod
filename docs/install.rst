============
Installation
============

It is advised to use **pip**.
Please make sure **the latest version** is installed since PyOD is currently updated on **a daily basis**:

.. code-block:: bash

    pip install pyod
    pip install --upgrade pyod  # make sure the latest version is installed!

or

.. code-block:: bash

    pip install pyod==x.y.z  # (x.y.z) is the current version number

Alternatively, downloading/cloning the `Github repository <https://github.com/yzhao062/Pyod/>`_ also works.
You could unzip the files and execute the following command in the folder where the files get decompressed.

.. code-block:: bash

    python setup.py install

Supported Python Version:

- Python 2: 2.7 only
- Python 3: 3.4, 3.5 or 3.6

Library Dependency:

.. code-block:: bash

    - matplotlib                       # needed for running examples
    - nose                             # needed for running tests
    - numpy>=1.13
    - pathlib2 ; python_version < '3'  # needed if python 2.7
    - pytest                           # needed for running tests
    - scipy>=0.19.1
    - scikit_learn>=0.19.1