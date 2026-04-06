# -*- coding: utf-8 -*-

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

from pyod.utils.auto_model_selector import (
    load_model_analyses_labels_only,
    _normalize_model_name,
    _MODEL_REGISTRY,
)


class TestModelAnalysesLoading(unittest.TestCase):
    """Test that JSON model analyses load correctly with normalized names."""

    def test_load_returns_nonempty(self):
        analyses, model_list = load_model_analyses_labels_only()
        self.assertGreater(len(analyses), 0)
        self.assertGreater(len(model_list), 0)

    def test_names_are_normalized(self):
        analyses, model_list = load_model_analyses_labels_only()
        # These names should be normalized (no hyphens or spaces)
        for name in model_list:
            self.assertNotIn('-', name,
                             f"Model name '{name}' contains a hyphen")
            self.assertNotIn(' ', name,
                             f"Model name '{name}' contains a space")

    def test_all_loaded_models_in_registry(self):
        analyses, model_list = load_model_analyses_labels_only()
        for name in model_list:
            self.assertIn(name, _MODEL_REGISTRY,
                          f"Model '{name}' not found in _MODEL_REGISTRY")

    def test_analyses_have_strengths_and_weaknesses(self):
        analyses, _ = load_model_analyses_labels_only()
        for name, info in analyses.items():
            self.assertIn('strengths', info, f"{name} missing strengths")
            self.assertIn('weaknesses', info, f"{name} missing weaknesses")
            self.assertIsInstance(info['strengths'], list)
            self.assertIsInstance(info['weaknesses'], list)


class TestNameNormalization(unittest.TestCase):
    """Test the name normalization mapping."""

    def test_hyphenated_names(self):
        self.assertEqual(_normalize_model_name('MO-GAAL'), 'MO_GAAL')
        self.assertEqual(_normalize_model_name('SO-GAAL'), 'SO_GAAL')

    def test_spaced_names(self):
        self.assertEqual(_normalize_model_name('Deep SVDD'), 'DeepSVDD')

    def test_already_normalized(self):
        self.assertEqual(_normalize_model_name('VAE'), 'VAE')
        self.assertEqual(_normalize_model_name('LUNAR'), 'LUNAR')


class TestModelRegistry(unittest.TestCase):
    """Test that the model registry can resolve all entries."""

    def test_registry_module_paths_exist(self):
        """Verify registry paths are valid.

        Deep-learning models require torch (optional). Skip entries
        whose module fails due to a missing backend, but fail on bad
        registry paths. We check e.name against known optional backends
        rather than string matching, so a typo like 'pydo.models...'
        will not be silently skipped.
        """
        import importlib
        _OPTIONAL_BACKENDS = {'torch', 'tensorflow', 'keras'}
        for name, (module_path, class_name, kwargs) in _MODEL_REGISTRY.items():
            try:
                mod = importlib.import_module(module_path)
            except ImportError as e:
                # e.name is the top-level module that failed to import.
                # If it is an optional backend, skip; otherwise fail.
                missing = getattr(e, 'name', '') or ''
                top_level = missing.split('.')[0]
                if top_level in _OPTIONAL_BACKENDS:
                    continue
                raise
            self.assertTrue(
                hasattr(mod, class_name),
                f"Module {module_path} has no class {class_name}")


class TestAutoModelSelectorSmoke(unittest.TestCase):
    """End-to-end smoke test with mocked OpenAI API."""

    def test_full_pipeline_mocked(self):
        """Mock the OpenAI call and verify the pipeline runs.

        openai is mocked via sys.modules so the real _check_openai_dependency
        sees it as present (covers the positive path).
        """
        mock_openai = MagicMock()

        # Simulate GPT responses
        tag_response = MagicMock()
        tag_response.choices = [MagicMock()]
        tag_response.choices[0].message.content = (
            '{"Data size": ["small"], "Data type": ["tabular data"], '
            '"Domain": ["technology"], '
            '"Characteristics": ["high dimensionality"], '
            '"Additional requirements": ["CPU"]}'
        )

        select_response = MagicMock()
        select_response.choices = [MagicMock()]
        select_response.choices[0].message.content = (
            '{"selected_model": "AutoEncoder", '
            '"reason": "Best fit for tabular data"}'
        )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            tag_response, select_response
        ]
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict('sys.modules', {'openai': mock_openai}):
            from pyod.utils.auto_model_selector import AutoModelSelector

            X = np.random.RandomState(42).randn(50, 5)
            # Explicit api_key: dotenv should NOT be required
            selector = AutoModelSelector(dataset=X, api_key="test-key")

            selected, reason = selector.model_auto_select()
            self.assertEqual(selected, 'AutoEncoder')
            self.assertIsNotNone(reason)

            clf = selector.get_top_clf()
            self.assertIsNotNone(clf)

    def test_missing_openai_raises(self):
        """Verify that missing openai gives a clear error."""
        from pyod.utils.auto_model_selector import _check_openai_dependency
        with patch.dict('sys.modules', {'openai': None}):
            with self.assertRaises(ImportError) as ctx:
                _check_openai_dependency()
            self.assertIn('openai', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
