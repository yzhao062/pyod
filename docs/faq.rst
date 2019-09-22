Frequently Asked Questions
==========================

----

What is the Next?
^^^^^^^^^^^^^^^^^

This is the central place to track important things to be fixed/added:

- GPU support (it is noted that keras with TensorFlow backend will automatically run on GPU; auto_encoder_example.py takes around 96.95 seconds on a RTX 2060 GPU).
- Installation efficiency improvement, such as using docker
- Add contact channel with `Gitter <https://gitter.im>`_
- Support additional languages, see `Manage Translations <https://docs.readthedocs.io/en/latest/guides/manage-translations.html>`_
- Fix the bug that numba enabled function may be excluded from code coverage
- Decide which Python interpreter should readthedocs use. 3.X invokes Python 3.7 which has no TF supported for now.

Feel free to open on issue report if needed.
See `Issues <https://github.com/yzhao062/pyod/issues>`_.

----

How to Contribute
^^^^^^^^^^^^^^^^^

You are welcome to contribute to this exciting project:


* Please first check Issue lists for "help wanted" tag and comment the one
  you are interested. We will assign the issue to you.

* Fork the master branch and add your improvement/modification/fix.

* Create a pull request to **development branch** and follow the pull request template `PR template <https://github.com/yzhao062/pyod/blob/master/PULL_REQUEST_TEMPLATE.md>`_

* Automatic tests will be triggered. Make sure all tests are passed. Please make sure all added modules are accompanied with proper test functions.


To make sure the code has the same style and standard, please refer to abod.py, hbos.py, or feature_bagging.py for example.

You are also welcome to share your ideas by opening an issue or dropping me an email at zhaoy@cmu.edu :)


Inclusion Criteria
^^^^^^^^^^^^^^^^^^

Similarly to `scikit-learn <https://scikit-learn.org/stable/faq.html#what-are-the-inclusion-criteria-for-new-algorithms>`_,
We mainly consider well-established algorithms for inclusion.
A rule of thumb is at least two years since publication, 50+ citations, and usefulness.

However, we encourage the author(s) of newly proposed models to share and add your implementation into PyOD
for boosting ML accessibility and reproducibility.
This exception only applies if you could commit to the maintenance of your model for at least two year period.