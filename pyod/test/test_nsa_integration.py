# -*- coding: utf-8 -*-
"""Public import and ADEngine integration for negative selection."""

import subprocess
import sys

import numpy as np
import pytest

from pyod.models.nsa import NSA
from pyod.utils.ad_engine import ADEngine


def test_package_export_is_lazy():
    """Importing the package must not load the newly exported detector."""
    code = (
        "import sys\n"
        "import pyod.models\n"
        "assert 'pyod.models.nsa' not in sys.modules\n"
        "from pyod.models import NSA\n"
        "from pyod.models.nsa import NSA as DirectNSA\n"
        "assert NSA is DirectNSA\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)


def test_unknown_package_attribute_raises():
    import pyod.models

    with pytest.raises(AttributeError, match='no attribute'):
        getattr(pyod.models, 'unknown_detector')


def test_adengine_lists_and_explains_nsa():
    engine = ADEngine()
    tabular = {entry['name']: entry
               for entry in engine.list_detectors(data_type='tabular')}
    assert tabular['NSA']['class_path'] == 'pyod.models.nsa.NSA'
    metadata = engine.explain_detector('NSA')
    assert metadata['status'] == 'shipped'
    assert metadata['preprocessing_mode'] == 'internal'
    assert metadata['requires'] == []
    assert metadata['benchmark_refs'] == []
    assert metadata['benchmark_rank'] == {}
    assert 'normal' in metadata['best_for']


def test_adengine_builds_and_fits_registered_defaults():
    engine = ADEngine(random_state=13)
    metadata = engine.explain_detector('NSA')
    model = engine.build_detector({
        'detector_name': 'NSA',
        'params': metadata['default_params'],
    })
    assert isinstance(model, NSA)
    assert model.random_state == 13
    X = np.random.RandomState(9).normal(size=(50, 3))
    assert model.fit(X) is model
    scores = model.decision_function(X)
    assert scores.shape == (len(X),)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, model.decision_scores_)


def test_adengine_preserves_explicit_constructor_parameters():
    model = ADEngine(random_state=13).build_detector({
        'detector_name': 'NSA',
        'params': {
            'strategy': 'fixed', 'n_detectors': 7, 'random_state': 23,
        },
    })
    assert model.strategy == 'fixed'
    assert model.n_detectors == 7
    assert model.random_state == 23
