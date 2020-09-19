Model Save & Load
=================

PyOD takes a similar approach of sklearn regarding model persistence.
See `model persistence <https://scikit-learn.org/stable/modules/model_persistence.html>`_ for clarification.

In short, we recommend to use joblib or pickle for saving and loading PyOD models.
See `"examples/save_load_model_example.py" <https://github.com/yzhao062/pyod/blob/master/examples/save_load_model_example.py>`_ for an example.
In short, it is simple as below:

.. code-block:: python

    from joblib import dump, load

    # save the model
    dump(clf, 'clf.joblib')
    # load the model
    clf = load('clf.joblib')
