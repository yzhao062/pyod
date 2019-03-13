Todo & Contribution Guidance
============================

This is the central place to track important things to be fixed/added:

- GPU support
- Installation efficiency improvement, such as using docker
- Add contact channel with `Gitter <https://gitter.im>`_
- Support additional languages, see `Manage Translations <https://docs.readthedocs.io/en/latest/guides/manage-translations.html>`_
- Fix the bug that numba enabled function may be excluded from code coverage
- Decide which Python interpreter should readthedocs use. 3.X invokes Python 3.7 which has no TF supported for now.

Feel free to open on issue report if needed.
See `Issues <https://github.com/yzhao062/pyod/issues>`_.

----

How to Contribute
-----------------

You are welcome to contribute to this exciting project. A procedure is prepared
for effective collaboration.


* Please first check `Issue lists <https://github.com/yzhao062/pyod/issues>`_
  for "help wanted" tag and comment on the one that you are interested in.
  We will assign the issue to you.

* Fork the master branch and add your improvement/modification/fix.

* Create a pull request and follow the pull request template `PR template <https://github.com/yzhao062/pyod/blob/master/PULL_REQUEST_TEMPLATE.md>`_


To make sure the code has the same style and standard, please refer to models,
such as abod.py, hbos.py, or feature bagging for example.

You are also welcome to share your ideas by opening an issue or dropping me
an email at yuezhao@cs.toronto.edu :)
