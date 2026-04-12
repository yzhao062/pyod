# -*- coding: utf-8 -*-
"""Tests for ADEngine V3 session workflow."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.utils.ad_engine import ADEngine
from pyod.utils.investigation import InvestigationState, PHASES


class TestSessionStartPlan(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def test_start_returns_state(self):
        state = self.engine.start(self.X)
        assert isinstance(state, InvestigationState)
        assert state.phase == 'profiled'
        assert state.profile['data_type'] == 'tabular'
        assert state.profile['n_samples'] == 200
        assert state.next_action['action'] == 'plan'

    def test_start_with_data_type(self):
        state = self.engine.start(self.X, data_type='time_series')
        assert state.profile['data_type'] == 'time_series'

    def test_plan_returns_state(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        assert state.phase == 'planned'
        assert len(state.plans) >= 1
        assert len(state.plans) <= 3
        assert state.next_action['action'] == 'run'

    def test_plan_has_detector_names(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        for p in state.plans:
            assert 'detector_name' in p
            assert len(p['detector_name']) > 0

    def test_plan_with_exclude(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(
            state, constraints={'exclude_detectors': ['IForest']})
        names = [p['detector_name'] for p in state.plans]
        assert 'IForest' not in names

    def test_plan_max_detectors_1(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(
            state, constraints={'max_detectors': 1})
        assert len(state.plans) == 1

    def test_plan_max_detectors_2(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(
            state, constraints={'max_detectors': 2})
        assert len(state.plans) <= 2

    def test_history_tracking(self):
        state = self.engine.start(self.X)
        assert len(state.history) == 1
        assert state.history[0]['action'] == 'start'
        state = self.engine.plan(state)
        assert len(state.history) == 2
        assert state.history[1]['action'] == 'plan'


if __name__ == '__main__':
    unittest.main()
