# -*- coding: utf-8 -*-
# Intentionally avoid package-level model imports; several detectors
# require optional extras (torch, combo, xgboost, suod, pythresh, ...)
# and importing them here would force every PyOD user to install every
# extra. Import each detector explicitly, e.g.
# `from pyod.models.iforest import IForest`.
# from .abod import ABOD
# from .auto_encoder import AutoEncoder
# from .cblof import CBLOF
# from .combination import aom, moa, average, maximization
# from .feature_bagging import FeatureBagging
# from .hbos import HBOS
# from .iforest import IForest
# from .knn import KNN
# from .lof import LOF
# from .mcd import MCD
# from .ocsvm import OCSVM
# from .pca import PCA
#
# __all__ = ['ABOD',
#            'AutoEncoder',
#            'CBLOF',
#            'aom', 'moa', 'average', 'maximization',
#            'FeatureBagging',
#            'HBOS',
#            'IForest',
#            'KNN',
#            'LOF',
#            'MCD',
#            'OCSVM',
#            'PCA']

__all__ = ['NSA']


def __getattr__(name):
    """Load explicitly exported estimators without eager model imports."""
    if name == 'NSA':
        from .nsa import NSA
        return NSA
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
