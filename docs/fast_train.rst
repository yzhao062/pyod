Fast Train with SUOD
====================

**Fast training and prediction**: it is possible to train and predict with
a large number of detection models in PyOD by leveraging SUOD framework.
See  `SUOD Paper <https://www.andrew.cmu.edu/user/yuezhao2/papers/21-mlsys-suod.pdf>`_
and  `SUOD example <https://github.com/yzhao062/pyod/blob/master/examples/suod_example.py>`_.


.. code-block:: python

    from pyod.models.suod import SUOD

    # initialized a group of outlier detectors for acceleration
    detector_list = [LOF(n_neighbors=15), LOF(n_neighbors=20),
                     LOF(n_neighbors=25), LOF(n_neighbors=35),
                     COPOD(), IForest(n_estimators=100),
                     IForest(n_estimators=200)]

    # decide the number of parallel process, and the combination method
    # then clf can be used as any outlier detection model
    clf = SUOD(base_estimators=detector_list, n_jobs=2, combination='average',
               verbose=False)
