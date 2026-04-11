# V3 Agentic Session API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add session-based workflow engine to ADEngine: `start → plan → run → analyze → iterate → report`, with multi-detector consensus, quality assessment, and actionable iteration.

**Architecture:** New `InvestigationState` dataclass in `pyod/utils/investigation.py`. Session methods added to existing `ADEngine` class in `pyod/utils/ad_engine.py`. All existing methods unchanged. Session methods wrap existing helpers (`plan_detection`, `run_detection`, `analyze_results`, `generate_report`).

**Tech Stack:** Python dataclasses, numpy, scipy.stats (spearmanr, rankdata — both already dependencies).

**Spec:** `docs/superpowers/specs/2026-04-12-v3-agentic-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `pyod/utils/investigation.py` | `InvestigationState` dataclass, `PHASES`, `ACTION_TYPES` enums, helper constructors |
| `pyod/test/test_ad_engine_v3.py` | Tests for session workflow methods |

### Modified files

| File | Change |
|------|--------|
| `pyod/utils/ad_engine.py` | Add 7 session methods: `start`, `plan`, `run`, `analyze`, `iterate`, `report`, `investigate` |
| `skills/od-expert/SKILL.md` | Update with V3 session workflow instructions |
| `CHANGES.txt` | Add V3 entry |

---

## Dependency graph

```
Task 1 (InvestigationState) → Task 2 (start + plan) → Task 3 (run + consensus)
→ Task 4 (analyze + quality) → Task 5 (iterate) → Task 6 (report + investigate)
→ Task 7 (CHANGES.txt) → Task 8 (od-expert skill)
```

All tasks are sequential — each builds on the previous.

---

### Task 1: InvestigationState dataclass

**Files:**
- Create: `pyod/utils/investigation.py`

- [ ] **Step 1: Create `investigation.py`**

```python
# pyod/utils/investigation.py
# -*- coding: utf-8 -*-
"""Investigation state for ADEngine session workflow."""

import time
from dataclasses import dataclass, field

PHASES = ('profiled', 'planned', 'detected', 'analyzed')

ACTION_TYPES = (
    'plan',
    'run',
    'analyze',
    'report_to_user',
    'confirm_with_user',
    'iterate',
    'done',
)


@dataclass
class InvestigationState:
    """Typed state object for an ADEngine investigation session.

    Tracks the full workflow: profiling, planning, detection,
    analysis, and iteration. Each session method updates the state
    and sets ``next_action`` to guide the agent.

    Attributes
    ----------
    phase : str
        One of ``PHASES``: 'profiled', 'planned', 'detected', 'analyzed'.
    iteration : int
        Current iteration (0 = first run).
    history : list
        List of HistoryEntry dicts.
    data : object
        Reference to input data (not copied).
    profile : dict
        Output of ``profile_data()``.
    plans : list
        List of DetectionPlan dicts (top-N).
    results : list
        List of DetectorResult dicts.
    consensus : dict or None
        ConsensusResult dict.
    analysis : dict or None
        InvestigationAnalysis dict.
    quality : dict or None
        QualityAssessment dict.
    next_action : dict
        NextAction dict guiding the agent.
    """
    phase: str
    iteration: int = 0
    history: list = field(default_factory=list)
    data: object = None
    profile: dict = field(default_factory=dict)
    plans: list = field(default_factory=list)
    results: list = field(default_factory=list)
    consensus: dict = None
    analysis: dict = None
    quality: dict = None
    next_action: dict = field(default_factory=dict)


def _make_history_entry(phase, action, iteration, detail=''):
    """Create a HistoryEntry dict."""
    return {
        'phase': phase,
        'action': action,
        'iteration': iteration,
        'timestamp': time.time(),
        'detail': detail,
    }
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from pyod.utils.investigation import InvestigationState, PHASES, ACTION_TYPES; print('OK', PHASES)"`
Expected: `OK ('profiled', 'planned', 'detected', 'analyzed')`

- [ ] **Step 3: Commit**

```bash
git add pyod/utils/investigation.py
git commit -m "feat: add InvestigationState dataclass for V3 session workflow"
```

---

### Task 2: start() and plan() session methods

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Create: `pyod/test/test_ad_engine_v3.py` (start with first tests)

- [ ] **Step 1: Write failing tests for start() and plan()**

Create `pyod/test/test_ad_engine_v3.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py -v`
Expected: FAIL — `ADEngine has no attribute 'start'`

- [ ] **Step 3: Implement start() and plan()**

Add to `pyod/utils/ad_engine.py`, at the end of the class (before the knowledge query section, around line 830):

```python
    # ------------------------------------------------------------------
    # V3 Session workflow
    # ------------------------------------------------------------------

    def start(self, X, data_type=None):
        """Start an investigation session.

        Profiles the data and returns an InvestigationState.

        Parameters
        ----------
        X : array-like, Data, list, or dict
            Input data (any modality).
        data_type : str or None
            Explicit type override.

        Returns
        -------
        state : InvestigationState
        """
        from .investigation import InvestigationState, _make_history_entry

        profile = self.profile_data(X, data_type=data_type)
        state = InvestigationState(
            phase='profiled',
            data=X,
            profile=profile,
            next_action={
                'action': 'plan',
                'reason': 'Data profiled as %s with %d samples. '
                          'Ready to select detectors.'
                          % (profile['data_type'],
                             profile.get('n_samples', 0)),
            },
        )
        state.history.append(_make_history_entry(
            'profiled', 'start', 0,
            'Profiled %s data' % profile['data_type']))
        return state

    def plan(self, state, priority='balanced', constraints=None):
        """Plan detection: select top-N detectors.

        Wraps ``plan_detection()`` and extracts primary + alternatives
        into ``state.plans`` (up to 3 detectors, v1 limit).

        Parameters
        ----------
        state : InvestigationState
        priority : str
        constraints : dict or None

        Returns
        -------
        state : InvestigationState
        """
        from .investigation import _make_history_entry

        constraints = constraints or {}
        result = self.plan_detection(
            state.profile, priority=priority, constraints=constraints)

        # Extract primary + alternatives into flat list
        plans = []
        if result.get('detector_name'):
            plans.append(result)
        for alt in result.get('alternatives', []):
            if alt.get('detector_name'):
                plans.append(alt)

        # Honor max_detectors (v1 cap at 3)
        max_det = max(1, min(
            int(constraints.get('max_detectors', 3)), 3))
        state.plans = plans[:max_det]
        state.phase = 'planned'
        names = [p['detector_name'] for p in state.plans]
        state.next_action = {
            'action': 'run',
            'reason': 'Top %d detectors selected: %s. Ready to run.'
                      % (len(state.plans), ', '.join(names)),
        }
        state.history.append(_make_history_entry(
            'planned', 'plan', state.iteration,
            'Selected %d detectors: %s' % (len(plans), ', '.join(names))))
        return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py -v`
Expected: 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine_v3.py
git commit -m "feat: add start() and plan() session methods to ADEngine"
```

---

### Task 3: run() method — multi-detector execution + consensus

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine_v3.py`

- [ ] **Step 1: Write failing tests for run()**

Add to `pyod/test/test_ad_engine_v3.py`:

```python
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
                'PCA', 'INNE']})
        state = self.engine.run(state)
        # Single detector: agreement = 0.5
        assert state.consensus['agreement'] == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py::TestSessionRun -v`
Expected: FAIL — `ADEngine has no attribute 'run'`

- [ ] **Step 3: Implement run()**

Add to `pyod/utils/ad_engine.py` after `plan()`:

```python
    def run(self, state):
        """Run detection with all planned detectors.

        Wraps ``run_detection()`` per plan. Computes consensus via
        rank normalization and majority vote. Records errors per
        detector without stopping.

        Parameters
        ----------
        state : InvestigationState

        Returns
        -------
        state : InvestigationState
        """
        from .investigation import _make_history_entry
        from scipy.stats import rankdata, spearmanr

        results = []
        for plan in state.plans:
            try:
                raw = self.run_detection(state.data, plan)
                entry = dict(raw)
                entry['detector_name'] = plan['detector_name']
                entry['status'] = 'success'
                entry['error'] = None
                results.append(entry)
            except Exception as e:
                results.append({
                    'detector_name': plan['detector_name'],
                    'status': 'error',
                    'error': str(e),
                    'plan': plan,
                })

        state.results = results
        state.phase = 'detected'

        # Compute consensus from successful detectors
        successful = [r for r in results if r['status'] == 'success']

        if len(successful) == 0:
            state.consensus = None
            state.next_action = {
                'action': 'confirm_with_user',
                'reason': 'All %d detectors failed. Check data format '
                          'or try a different detector family.'
                          % len(results),
            }
        elif len(successful) == 1:
            r = successful[0]
            state.consensus = {
                'scores': r['scores_train'],
                'labels': r['labels_train'],
                'n_detectors': 1,
                'agreement': 0.5,
                'disagreements': [],
            }
            state.next_action = {
                'action': 'analyze',
                'reason': 'Detection complete (1 detector).',
            }
        else:
            n_samples = len(successful[0]['scores_train'])
            # Rank-normalize scores per detector
            rank_scores = np.array([
                rankdata(r['scores_train']) / n_samples
                for r in successful
            ])
            consensus_scores = np.mean(rank_scores, axis=0)

            # Majority-vote labels
            all_labels = np.array([
                r['labels_train'] for r in successful])
            vote_count = np.sum(all_labels, axis=0)
            consensus_labels = (
                vote_count > len(successful) / 2).astype(int)

            # Pairwise Spearman agreement
            correlations = []
            for i in range(len(successful)):
                for j in range(i + 1, len(successful)):
                    rho, _ = spearmanr(
                        successful[i]['scores_train'],
                        successful[j]['scores_train'])
                    correlations.append(
                        max(0.0, rho) if np.isfinite(rho) else 0.0)
            agreement = float(np.mean(correlations)) if correlations else 0.5

            # Disagreements: indices where detectors disagree
            disagreements = []
            for idx in range(n_samples):
                votes = all_labels[:, idx]
                if not (votes.all() or not votes.any()):
                    disagreements.append(int(idx))

            state.consensus = {
                'scores': consensus_scores,
                'labels': consensus_labels,
                'n_detectors': len(successful),
                'agreement': agreement,
                'disagreements': disagreements,
            }
            state.next_action = {
                'action': 'analyze',
                'reason': 'Detection complete (%d detectors, '
                          'agreement=%.2f).'
                          % (len(successful), agreement),
            }

        state.history.append(_make_history_entry(
            'detected', 'run', state.iteration,
            '%d/%d detectors succeeded'
            % (len(successful), len(results))))
        return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py -v`
Expected: 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine_v3.py
git commit -m "feat: add run() session method with multi-detector consensus"
```

---

### Task 4: analyze() method — quality assessment

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine_v3.py`

- [ ] **Step 1: Write failing tests for analyze()**

Add to `pyod/test/test_ad_engine_v3.py`:

```python
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
        """All same label → separation = 0."""
        # Use very low contamination so likely all labeled 0
        state = self.engine.start(self.X)
        state = self.engine.plan(state)
        state = self.engine.run(state)
        # Force all labels to 0 for test
        state.consensus['labels'] = np.zeros(200, dtype=int)
        state = self.engine.analyze(state)
        assert state.quality['separation'] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py::TestSessionAnalyze -v`
Expected: FAIL

- [ ] **Step 3: Implement analyze()**

Add to `pyod/utils/ad_engine.py` after `run()`:

```python
    def analyze(self, state):
        """Analyze detection results with quality assessment.

        Computes per-detector analysis, consensus analysis, quality
        metrics (separation, agreement, stability), and selects
        the best detector.

        Parameters
        ----------
        state : InvestigationState

        Returns
        -------
        state : InvestigationState
        """
        from .investigation import _make_history_entry
        from scipy.stats import spearmanr

        state.phase = 'analyzed'

        # All-error path
        successful = [r for r in state.results
                      if r['status'] == 'success']
        if not successful:
            state.analysis = None
            state.quality = {
                'separation': 0.0, 'agreement': 0.0,
                'stability': 0.0, 'overall': 0.0,
                'verdict': 'low',
                'explanation': 'All detectors failed.',
            }
            state.next_action = {
                'action': 'confirm_with_user',
                'reason': 'All detectors failed. Check data format '
                          'or try a different detector family.',
            }
            state.history.append(_make_history_entry(
                'analyzed', 'analyze', state.iteration,
                'All detectors failed'))
            return state

        # Per-detector analysis (aligned with state.results)
        per_det = []
        for r in state.results:
            if r['status'] == 'success':
                try:
                    a = self.analyze_results(r, X=state.data)
                except Exception:
                    a = None
                per_det.append(a)
            else:
                per_det.append(None)

        # Consensus analysis (lightweight, not via analyze_results)
        c = state.consensus
        c_scores = c['scores']
        c_labels = c['labels']
        n_anomalies = int(c_labels.sum())
        n_samples = len(c_labels)
        top_k = min(10, n_samples)
        top_indices = np.argsort(c_scores)[::-1][:top_k]
        consensus_analysis = {
            'n_anomalies': n_anomalies,
            'anomaly_ratio': n_anomalies / max(n_samples, 1),
            'score_distribution': {
                'mean': float(np.mean(c_scores)),
                'std': float(np.std(c_scores)),
                'min': float(np.min(c_scores)),
                'max': float(np.max(c_scores)),
                'median': float(np.median(c_scores)),
                'q25': float(np.percentile(c_scores, 25)),
                'q75': float(np.percentile(c_scores, 75)),
            },
            'top_anomalies': [
                {'index': int(i), 'score': float(c_scores[i])}
                for i in top_indices],
            'summary': '%d anomalies detected out of %d samples '
                       '(%.1f%%) by consensus of %d detectors.'
                       % (n_anomalies, n_samples,
                          100 * n_anomalies / max(n_samples, 1),
                          c['n_detectors']),
        }

        # Best detector selection
        best_idx = self._select_best_detector(
            state.results, c_scores)

        state.analysis = {
            'consensus_analysis': consensus_analysis,
            'per_detector_analysis': per_det,
            'best_detector': state.results[best_idx]['detector_name'],
            'best_detector_index': best_idx,
            'summary': consensus_analysis['summary'],
        }

        # Quality metrics
        state.quality = self._compute_quality(
            c_scores, c_labels, state.results, c)
        state.analysis['summary'] += (
            ' Quality: %s (%.2f).'
            % (state.quality['verdict'], state.quality['overall']))

        # Next action based on quality
        if state.quality['overall'] >= 0.4:
            state.next_action = {
                'action': 'report_to_user',
                'reason': 'Results ready (quality=%s, %.2f).'
                          % (state.quality['verdict'],
                             state.quality['overall']),
                'summary': state.analysis['summary'],
                'confidence': state.quality['overall'],
            }
        else:
            state.next_action = {
                'action': 'iterate',
                'reason': 'Low result quality (%.2f). Consider '
                          'trying different detectors.'
                          % state.quality['overall'],
                'suggestion': 'Exclude lowest-agreement detector '
                              'and re-run.',
            }

        state.history.append(_make_history_entry(
            'analyzed', 'analyze', state.iteration,
            'Quality: %s (%.2f)' % (
                state.quality['verdict'],
                state.quality['overall'])))
        return state

    def _select_best_detector(self, results, consensus_scores):
        """Select best detector via Spearman with consensus.

        Fallback chain (per spec):
        1. Highest finite Spearman correlation
        2. If tied: highest plan confidence
        3. If still tied: fastest runtime
        4. If ALL correlations are NaN: first successful detector
        """
        from scipy.stats import spearmanr

        successful = [
            (i, r) for i, r in enumerate(results)
            if r['status'] == 'success']
        if len(successful) == 1:
            return successful[0][0]

        # Compute Spearman for each successful detector
        rhos = []
        for i, r in successful:
            rho, _ = spearmanr(r['scores_train'], consensus_scores)
            rhos.append(float(rho) if np.isfinite(rho) else None)

        # If ALL NaN: return first successful (spec rule 4)
        if all(rho is None for rho in rhos):
            return successful[0][0]

        # Find best by finite Spearman, then tie-break
        best_j = 0  # index into successful list
        best_rho = -1.0
        for j, (i, r) in enumerate(successful):
            rho = rhos[j]
            if rho is None:
                continue
            if rho > best_rho:
                best_rho = rho
                best_j = j
            elif rho == best_rho:
                # Tie-break: plan confidence
                curr_conf = r.get('plan', {}).get('confidence', 0)
                prev_conf = successful[best_j][1].get(
                    'plan', {}).get('confidence', 0)
                if curr_conf > prev_conf:
                    best_j = j
                elif curr_conf == prev_conf:
                    # Tie-break: fastest
                    if r.get('runtime_seconds', 999) < successful[
                            best_j][1].get('runtime_seconds', 999):
                        best_j = j
        return successful[best_j][0]

    def _compute_quality(self, scores, labels, results, consensus):
        """Compute quality metrics: separation, agreement, stability."""
        # Separation
        if labels.sum() == 0 or labels.sum() == len(labels):
            separation = 0.0
        else:
            anomaly_mean = float(np.mean(scores[labels == 1]))
            inlier_mean = float(np.mean(scores[labels == 0]))
            separation = float(np.clip(
                anomaly_mean / (inlier_mean + 1e-10) - 1, 0, 1))

        # Agreement (from consensus)
        agreement = float(consensus.get('agreement', 0.5))

        # Stability: Jaccard of top-k under +/-20% perturbation
        n_anomalies = int(labels.sum())
        n_samples = len(labels)
        if n_anomalies == 0:
            stability = 0.0
        else:
            k = n_anomalies
            k_low = max(1, int(k * 0.8))
            k_high = min(n_samples, int(k * 1.2))
            sorted_idx = np.argsort(scores)[::-1]
            top_k = set(sorted_idx[:k].tolist())
            top_low = set(sorted_idx[:k_low].tolist())
            top_high = set(sorted_idx[:k_high].tolist())

            def _jaccard(a, b):
                if not a and not b:
                    return 1.0
                return len(a & b) / len(a | b)

            stability = 0.5 * (
                _jaccard(top_k, top_low)
                + _jaccard(top_k, top_high))

        overall = float(np.mean([separation, agreement, stability]))
        if overall >= 0.7:
            verdict = 'high'
        elif overall >= 0.4:
            verdict = 'medium'
        else:
            verdict = 'low'

        return {
            'separation': separation,
            'agreement': agreement,
            'stability': stability,
            'overall': overall,
            'verdict': verdict,
            'explanation': 'Separation=%.2f, agreement=%.2f, '
                           'stability=%.2f.' % (
                               separation, agreement, stability),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py -v`
Expected: 18 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine_v3.py
git commit -m "feat: add analyze() with quality metrics and best-detector selection"
```

---

### Task 5: iterate() method — feedback handling

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine_v3.py`

- [ ] **Step 1: Write failing tests for iterate()**

Add to `pyod/test/test_ad_engine_v3.py`:

```python
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
        # Ambiguous → confirm
        assert state.next_action['action'] == 'confirm_with_user'

    def test_iteration_counter(self):
        state = self._run_to_analyzed()
        assert state.iteration == 0
        state = self.engine.iterate(state, {'action': 'rerun'})
        assert state.iteration == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py::TestSessionIterate -v`
Expected: FAIL

- [ ] **Step 3: Implement iterate()**

Add to `pyod/utils/ad_engine.py` after `_compute_quality()`:

```python
    def iterate(self, state, feedback):
        """Iterate based on feedback.

        Structured dicts execute immediately. NL strings are
        parsed with confidence; ambiguous feedback triggers
        ``'confirm_with_user'``.

        Parameters
        ----------
        state : InvestigationState
        feedback : str or dict

        Returns
        -------
        state : InvestigationState
        """
        from .investigation import _make_history_entry

        if isinstance(feedback, dict):
            return self._iterate_structured(state, feedback)
        return self._iterate_nl(state, str(feedback))

    def _iterate_structured(self, state, feedback):
        """Handle structured feedback dict."""
        from .investigation import _make_history_entry

        action = feedback.get('action', '')
        state.iteration += 1

        if action == 'adjust_contamination':
            value = feedback['value']
            for p in state.plans:
                params = dict(p.get('params', {}))
                params['contamination'] = value
                p['params'] = params
            detail = 'Adjusted contamination to %.3f' % value

        elif action == 'exclude':
            to_exclude = set(feedback.get('detectors', []))
            state.plans = [
                p for p in state.plans
                if p['detector_name'] not in to_exclude]
            if not state.plans:
                # Re-plan without excluded detectors
                result = self.plan_detection(
                    state.profile,
                    constraints={'exclude_detectors': list(to_exclude)})
                state.plans = [result]
                for alt in result.get('alternatives', []):
                    if alt.get('detector_name'):
                        state.plans.append(alt)
            detail = 'Excluded: %s' % ', '.join(to_exclude)

        elif action == 'include':
            to_include = feedback.get('detectors', [])
            existing = {p['detector_name'] for p in state.plans}
            for name in to_include:
                if name not in existing:
                    algo = self.kb.get_algorithm(name)
                    if algo and algo.get('status') in (
                            'shipped', 'experimental'):
                        state.plans.append(self._make_plan(
                            detector_name=name, params={},
                            reason='Added by user', confidence=0.5))
            detail = 'Included: %s' % ', '.join(to_include)

        elif action == 'rerun':
            detail = 'Re-running same plan'

        else:
            state.next_action = {
                'action': 'confirm_with_user',
                'reason': 'Unknown action: %s' % action,
            }
            return state

        state.phase = 'planned'
        state.results = []
        state.consensus = None
        state.analysis = None
        state.quality = None
        state.next_action = {
            'action': 'run',
            'reason': 'Plan adjusted. ' + detail,
            'adjustment': detail,
        }
        state.history.append(_make_history_entry(
            'planned', 'iterate', state.iteration, detail))
        return state

    def _iterate_nl(self, state, feedback):
        """Parse NL feedback into structured action."""
        from .investigation import _make_history_entry

        lower = feedback.lower()
        proposed = None
        confidence = 0.0

        # High-confidence patterns
        if 'without' in lower or 'exclude' in lower:
            # Try to extract detector name
            for r in state.results:
                name = r.get('detector_name', '')
                if name.lower() in lower:
                    proposed = {'action': 'exclude',
                                'detectors': [name]}
                    confidence = 0.9
                    break
            if proposed is None and ('without' in lower
                                     or 'exclude' in lower):
                proposed = {'action': 'exclude', 'detectors': []}
                confidence = 0.3

        elif ('false positive' in lower or 'too many' in lower):
            current = state.plans[0].get('params', {}).get(
                'contamination', 0.1) if state.plans else 0.1
            proposed = {'action': 'adjust_contamination',
                        'value': max(current * 0.5, 0.01)}
            confidence = 0.7

        elif ('missed' in lower or 'false negative' in lower):
            current = state.plans[0].get('params', {}).get(
                'contamination', 0.1) if state.plans else 0.1
            proposed = {'action': 'adjust_contamination',
                        'value': min(current * 1.5, 0.5)}
            confidence = 0.7

        elif 'rerun' in lower or 'again' in lower:
            proposed = {'action': 'rerun'}
            confidence = 0.9

        if proposed is None:
            proposed = {'action': 'rerun'}
            confidence = 0.0

        if confidence >= 0.8:
            return self._iterate_structured(state, proposed)

        # Low confidence → ask for confirmation
        state.next_action = {
            'action': 'confirm_with_user',
            'reason': 'Interpreted "%s" as: %s (confidence=%.1f).'
                      % (feedback, proposed.get('action', '?'),
                         confidence),
            'suggestion': 'Proposed: %s. Proceed?' % str(proposed),
            'proposed_change': proposed,
        }
        state.history.append(_make_history_entry(
            state.phase, 'iterate_nl', state.iteration,
            'NL feedback: "%s" → confidence=%.1f'
            % (feedback, confidence)))
        return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py -v`
Expected: 24 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine_v3.py
git commit -m "feat: add iterate() with structured and NL feedback handling"
```

---

### Task 6: report() and investigate() — output + convenience

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine_v3.py`

- [ ] **Step 1: Write failing tests**

Add to `pyod/test/test_ad_engine_v3.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py::TestSessionReport -v`
Expected: FAIL

- [ ] **Step 3: Implement report() and investigate()**

Add to `pyod/utils/ad_engine.py` after `_iterate_nl()`:

```python
    def report(self, state, format='text'):
        """Generate investigation report.

        Text format wraps ``generate_report()`` for best detector,
        prepending session-level context. JSON format returns a
        native dict.

        Parameters
        ----------
        state : InvestigationState
        format : str
            'text' or 'json'.

        Returns
        -------
        report : str or dict
        """
        if state.analysis is None:
            raise ValueError(
                "No successful detectors to report on. "
                "Use iterate() to adjust the plan.")

        best_idx = state.analysis['best_detector_index']
        best_result = state.results[best_idx]
        best_analysis = state.analysis['per_detector_analysis'][
            best_idx]

        if format == 'json':
            return {
                'session': {
                    'consensus': {
                        'scores': state.consensus[
                            'scores'].tolist(),
                        'labels': state.consensus[
                            'labels'].tolist(),
                        'n_detectors': state.consensus[
                            'n_detectors'],
                        'agreement': state.consensus[
                            'agreement'],
                        'disagreements': state.consensus[
                            'disagreements'],
                    },
                    'quality': state.quality,
                    'comparison': {
                        'agreement': state.consensus[
                            'agreement'],
                        'disagreements': state.consensus[
                            'disagreements'],
                    },
                },
                'best_detector': {
                    'name': best_result['detector_name'],
                    'scores': best_result[
                        'scores_train'].tolist(),
                    'labels': best_result[
                        'labels_train'].tolist(),
                    'threshold': best_result['threshold'],
                    'analysis': best_analysis,
                },
            }

        # Text format
        lines = []
        lines.append('# Investigation Report')
        lines.append('')

        # Session section
        lines.append('## Session Summary')
        c = state.consensus
        q = state.quality
        lines.append('- **Detectors run:** %d' % c['n_detectors'])
        lines.append('- **Detector agreement:** %.2f'
                     % c['agreement'])
        lines.append('- **Quality verdict:** %s (%.2f)'
                     % (q['verdict'], q['overall']))
        lines.append('- **Iterations:** %d' % state.iteration)
        if c['disagreements']:
            lines.append('- **Disagreements:** %d samples'
                         % len(c['disagreements']))
        lines.append('')

        # Best detector report (via generate_report)
        detector_report = self.generate_report(
            best_result, best_analysis, format='text')
        lines.append(detector_report)

        return '\n'.join(lines)

    def investigate(self, X, data_type=None, priority='balanced'):
        """One-shot investigation: start → plan → run → analyze.

        Parameters
        ----------
        X : array-like
            Input data.
        data_type : str or None
        priority : str

        Returns
        -------
        state : InvestigationState
        """
        state = self.start(X, data_type=data_type)
        state = self.plan(state, priority=priority)
        state = self.run(state)
        state = self.analyze(state)
        return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pyod/test/test_ad_engine_v3.py -v`
Expected: 29 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine_v3.py
git commit -m "feat: add report() and investigate() to complete V3 session API"
```

---

### Task 7: Documentation

**Files:**
- Modify: `CHANGES.txt`

- [ ] **Step 1: Add CHANGES.txt entry**

Append to `CHANGES.txt`:

```
v<2.2.0>, <04/12/2026> -- V3 Agentic Session API: add InvestigationState workflow engine to ADEngine. Session methods (start, plan, run, analyze, iterate, report, investigate) enable multi-detector comparison with rank-normalized consensus, result quality assessment (separation, agreement, stability), and actionable iteration with structured and natural-language feedback. One-shot investigate() runs the full expert workflow.
```

- [ ] **Step 2: Commit**

```bash
git add CHANGES.txt
git commit -m "docs: add V3 agentic session API to CHANGES.txt"
```

---

---

### Task 8: od-expert skill update

**Files:**
- Modify: `skills/od-expert/SKILL.md`

- [ ] **Step 1: Update od-expert skill to use session API**

Update `skills/od-expert/SKILL.md` to instruct the agent to use the V3 session workflow. The key change: instead of calling individual methods (`profile_data`, `plan_detection`, `run_detection`, etc.), the skill should guide the agent through the session API (`start → plan → run → analyze → iterate → report`).

Add the following workflow section to the skill:

```markdown
## V3 Session Workflow

Use the ADEngine session API for the full anomaly detection lifecycle:

1. **Start:** `state = engine.start(data)` — profiles the data
2. **Plan:** `state = engine.plan(state)` — selects top-N detectors
3. **Run:** `state = engine.run(state)` — runs all detectors, computes consensus
4. **Analyze:** `state = engine.analyze(state)` — quality assessment, best detector
5. **Follow `state.next_action`:**
   - `'report_to_user'`: present `state.next_action['summary']` to the user
   - `'iterate'`: present the suggestion, ask if user wants to proceed
   - `'confirm_with_user'` with `proposed_change`: present suggestion, on approval call `engine.iterate(state, state.next_action['proposed_change'])`
   - `'confirm_with_user'` without `proposed_change` (error/retry): present reason, ask user what to try next
6. **On user feedback:** `state = engine.iterate(state, feedback)`
   - Structured: `{"action": "exclude", "detectors": ["IForest"]}`
   - Natural language: `"too many false positives"` (may need confirmation)
7. **Report:** `report = engine.report(state)` — generates final report

One-shot shortcut: `state = engine.investigate(data)` runs steps 1-4 automatically.
```

- [ ] **Step 2: Commit**

```bash
git add skills/od-expert/SKILL.md
git commit -m "docs: update od-expert skill to use V3 session API"
```

---

## Self-Review

**Spec coverage:**
- Section 4.1 (state machine): Task 1 (InvestigationState), all session methods implement transitions
- Section 4.2 (API): Tasks 2-6 implement all 7 session methods
- Section 4.3 (typed schemas): Task 1 (dataclass + enums), schemas enforced in Tasks 3-6
- Section 4.4 (behaviors): consensus in Task 3, quality in Task 4, iterate in Task 5, report wrapping in Task 6
- Section 5 (skill integration): Task 8 updates od-expert skill with V3 session workflow
- Section 6 (backward compat): all existing methods unchanged, `run()` avoids `detect()` conflict
- Section 7 (scope): all in-scope items covered, no out-of-scope items included

**Placeholder scan:** No TBD, TODO, or "similar to Task N" found. All code blocks complete.

**Type consistency:** `InvestigationState` used consistently. Method names match spec: `start`, `plan`, `run`, `analyze`, `iterate`, `report`, `investigate`. Schema field names match across tasks.
