Frequently Asked Questions
==========================

How to Contribute
^^^^^^^^^^^^^^^^^

PyOD accepts contributions through the issue-first process described in `CONTRIBUTING.md <https://github.com/yzhao062/pyod/blob/master/CONTRIBUTING.md>`_. In short:

* Open an issue, or comment on an existing one to ask to take it. The maintainer accepts and assigns the issue, declines it, or marks it for an in-house fix.
* After the issue is labelled ``accepted`` and assigned to you, branch from ``development``. Open one pull request against ``development`` that follows the `PR template <https://github.com/yzhao062/pyod/blob/master/PULL_REQUEST_TEMPLATE.md>`_.
* CI runs the full test suite on every pull request. New modules must ship with corresponding tests.

For style conventions, refer to established detector modules such as ``abod.py``, ``hbos.py``, or ``feature_bagging.py``.

To share an idea, open an issue or start a `GitHub Discussion <https://github.com/yzhao062/pyod/discussions>`_.


Inclusion Criteria
^^^^^^^^^^^^^^^^^^

Similar to `scikit-learn's inclusion criteria <https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms>`_, PyOD prioritizes well-established algorithms. A rule of thumb is at least two years since publication, 50+ citations, and practical usefulness on real-world anomaly detection tasks.

Authors of newly proposed detectors are welcome to contribute their implementations to improve accessibility and reproducibility, provided they can commit to at least two years of maintenance for the contributed model.
