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


class TestSessionRun(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def test_run_returns_state(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        assert state.phase == 'detected'
        assert len(state.results) > 0
        assert state.consensus is not None
        assert state.next_action['action'] == 'analyze'

    def test_results_have_scores(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        for r in state.results:
            if r['status'] == 'success':
                assert r['scores_train'] is not None
                assert len(r['scores_train']) == 200

    def test_consensus_scores(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        assert len(state.consensus['scores']) == 200
        assert len(state.consensus['labels']) == 200
        assert 0 <= state.consensus['agreement'] <= 1

    def test_consensus_single_detector(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(
            state, constraints={'exclude_detectors': [
                'ECOD', 'KNN', 'HBOS', 'LOF', 'COPOD', 'CBLOF',
                'PCA', 'INNE'], 'max_detectors': 1})
        state = self.engine.run(state)
        # Single detector: agreement = 0.5
        assert state.consensus['agreement'] == 0.5


class TestSessionAnalyze(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def _run_to_detected(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        return state

    def test_analyze_returns_state(self):
        state = self._run_to_detected()
        state = self.engine.analyze(state)
        assert state.phase == 'analyzed'
        assert state.analysis is not None
        assert state.quality is not None

    def test_quality_metrics(self):
        state = self._run_to_detected()
        state = self.engine.analyze(state)
        q = state.quality
        assert 0 <= q['separation'] <= 1
        assert 0 <= q['agreement'] <= 1
        assert 0 <= q['stability'] <= 1
        assert 0 <= q['overall'] <= 1
        assert q['verdict'] in ('high', 'medium', 'low')
        assert len(q['explanation']) > 0

    def test_analysis_has_best_detector(self):
        state = self._run_to_detected()
        state = self.engine.analyze(state)
        a = state.analysis
        assert 'best_detector' in a
        assert 'best_detector_index' in a
        assert 'consensus_analysis' in a
        assert 'per_detector_analysis' in a
        assert 'summary' in a

    def test_per_detector_aligned_with_results(self):
        state = self._run_to_detected()
        state = self.engine.analyze(state)
        assert len(state.analysis['per_detector_analysis']) == len(state.results)

    def test_next_action_after_analyze(self):
        state = self._run_to_detected()
        state = self.engine.analyze(state)
        assert state.next_action['action'] in (
            'report_to_user', 'iterate')

    def test_quality_separation_edge_case(self):
        """All same label -> separation = 0."""
        state = self._run_to_detected()
        state.consensus['labels'] = np.zeros(200, dtype=int)
        state = self.engine.analyze(state)
        assert state.quality['separation'] == 0.0


class TestSessionIterate(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def _run_to_analyzed(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        state = self.engine.analyze(state)
        return state

    def test_structured_adjust_contamination(self):
        state = self._run_to_analyzed()
        state = self.engine.iterate(
            state, {'action': 'adjust_contamination', 'value': 0.05})
        assert state.phase == 'planned'
        assert state.iteration == 1
        assert state.next_action['action'] == 'run'

    def test_structured_exclude(self):
        state = self._run_to_analyzed()
        excluded = state.plans[0]['detector_name']
        state = self.engine.iterate(
            state, {'action': 'exclude', 'detectors': [excluded]})
        names = [p['detector_name'] for p in state.plans]
        assert excluded not in names

    def test_structured_rerun(self):
        state = self._run_to_analyzed()
        old_plans = [p['detector_name'] for p in state.plans]
        state = self.engine.iterate(state, {'action': 'rerun'})
        new_plans = [p['detector_name'] for p in state.plans]
        assert old_plans == new_plans
        assert state.phase == 'planned'

    def test_nl_high_confidence(self):
        state = self._run_to_analyzed()
        state = self.engine.iterate(
            state, 'try without IForest')
        # Should either execute or ask confirmation
        assert state.next_action['action'] in ('run', 'confirm_with_user')

    def test_nl_low_confidence(self):
        state = self._run_to_analyzed()
        state = self.engine.iterate(
            state, 'hmm something seems off')
        # Ambiguous -> confirm
        assert state.next_action['action'] == 'confirm_with_user'

    def test_iteration_counter(self):
        state = self._run_to_analyzed()
        assert state.iteration == 0
        state = self.engine.iterate(state, {'action': 'rerun'})
        assert state.iteration == 1


class TestSessionReport(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def _run_to_analyzed(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        state = self.engine.analyze(state)
        return state

    def test_report_text(self):
        state = self._run_to_analyzed()
        report = self.engine.report(state, format='text')
        assert isinstance(report, str)
        assert 'Anomaly' in report
        assert 'consensus' in report.lower() or 'quality' in report.lower()

    def test_report_json(self):
        state = self._run_to_analyzed()
        report = self.engine.report(state, format='json')
        assert isinstance(report, dict)
        assert 'session' in report
        assert 'best_detector' in report

    def test_report_no_analysis_raises(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        # No analyze() called
        state.analysis = None
        with self.assertRaises(ValueError):
            self.engine.report(state)


class TestSessionInvestigate(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def test_investigate_returns_analyzed_state(self):
        state = self.engine.investigate(self.X)
        assert isinstance(state, InvestigationState)
        assert state.phase == 'analyzed'
        assert state.analysis is not None
        assert state.quality is not None
        assert len(state.results) > 0

    def test_investigate_with_data_type(self):
        state = self.engine.investigate(
            self.X, data_type='tabular')
        assert state.profile['data_type'] == 'tabular'


class TestSessionGuardrails(unittest.TestCase):
    """Tests for workflow enforcement and edge cases."""

    def setUp(self):
        self.engine = ADEngine()
        self.X = np.random.RandomState(42).randn(200, 10)

    def test_run_before_plan_raises(self):
        state = self.engine.start(self.X)
        with self.assertRaises(ValueError):
            self.engine.run(state)

    def test_analyze_before_run_raises(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        with self.assertRaises(ValueError):
            self.engine.analyze(state)

    def test_iterate_before_analyze_raises(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        with self.assertRaises(ValueError):
            self.engine.iterate(state, {'action': 'rerun'})

    def test_report_invalid_format_raises(self):
        state = self.engine.investigate(self.X)
        with self.assertRaises(ValueError):
            self.engine.report(state, format='csv')

    def test_report_before_analyze_raises(self):
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        with self.assertRaises(ValueError):
            self.engine.report(state)

    def test_include_respects_v1_cap(self):
        state = self.engine.investigate(self.X)
        state = self.engine.iterate(
            state, {'action': 'include',
                    'detectors': ['LOF', 'HBOS', 'COPOD']})
        assert len(state.plans) <= 3

    def test_include_at_cap_does_not_lie(self):
        """Include at cap: only claims detectors that were added."""
        state = self.engine.investigate(self.X)
        assert len(state.plans) == 3  # already at cap
        state = self.engine.iterate(
            state, {'action': 'include',
                    'detectors': ['COPOD']})
        # COPOD should not appear since we're at cap
        names = [p['detector_name'] for p in state.plans]
        if 'COPOD' not in names:
            assert 'Could not add' in state.next_action.get(
                'adjustment', state.next_action.get('reason', ''))


    def test_include_duplicate_not_claimed(self):
        """Including an already-present detector reports 'Already present'."""
        state = self.engine.investigate(self.X)
        existing = state.plans[0]['detector_name']
        state = self.engine.iterate(
            state, {'action': 'include', 'detectors': [existing]})
        adj = state.next_action.get(
            'adjustment', state.next_action.get('reason', ''))
        assert 'Already present' in adj

    def test_replan_clears_downstream(self):
        """Re-planning after analyze clears stale results."""
        state = self.engine.investigate(self.X)
        assert state.analysis is not None
        state = self.engine.plan(state)
        assert state.phase == 'planned'
        assert state.results == []
        assert state.consensus is None
        assert state.analysis is None
        assert state.quality is None


if __name__ == '__main__':
    unittest.main()
