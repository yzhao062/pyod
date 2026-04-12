Frequently Asked Questions
==========================

How to Contribute
^^^^^^^^^^^^^^^^^

Contributions are welcome. The workflow is:

* Check the `GitHub issue tracker <https://github.com/yzhao062/pyod/issues>`_ for "help wanted" items and leave a comment on one you would like to take on so it can be assigned to you.
* Fork the ``master`` branch and add your change on a feature branch.
* Open a pull request against the ``development`` branch and follow the `PR template <https://github.com/yzhao062/pyod/blob/master/PULL_REQUEST_TEMPLATE.md>`_.
* CI will run the full test suite on your PR. New modules must ship with corresponding tests.

For style conventions, refer to any of the well-established detector modules such as ``abod.py``, ``hbos.py``, or ``feature_bagging.py``.

You are also welcome to open an issue to share ideas or reach the maintainer at zhaoy@cmu.edu.


Inclusion Criteria
^^^^^^^^^^^^^^^^^^

Similar to `scikit-learn's inclusion criteria <https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms>`_, PyOD prioritizes well-established algorithms. A rule of thumb is at least two years since publication, 50+ citations, and practical usefulness on real-world anomaly detection tasks.

Authors of newly proposed detectors are welcome to contribute their implementations to improve accessibility and reproducibility, provided they can commit to at least two years of maintenance for the contributed model.
