Negative Selection Algorithm Family
===================================

.. automodule:: pyod.models.nsa
    :members:
    :undoc-members:
    :show-inheritance:

Recommended imports::

    from pyod.models.nsa import VDetector, RNSA, BinaryNSA, GridNSA, MNSA

The module intentionally lives in ``pyod/models/nsa.py``. The package-level
``pyod.models`` namespace does not need to import these classes because PyOD
avoids importing all detectors at package load time.
