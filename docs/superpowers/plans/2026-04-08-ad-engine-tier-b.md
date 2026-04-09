# ADEngine Tier B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the full anomaly detection lifecycle to ADEngine: execution, analysis, explainability, iteration suggestions, and report generation -- all pure Python, no MCP.

**Architecture:** Five new methods on the existing `ADEngine` class in `pyod/utils/ad_engine.py`. Each method takes and returns plain dicts/arrays so agents can chain them in a Python session. `run_detection()` replaces the existing `detect()` shortcut with a richer result dict. Analysis, explanation, and iteration are stateless functions over that result dict.

**Tech Stack:** Python 3.8+, numpy, scipy (for score statistics), PyOD BaseDetector API.

**Spec:** `docs/superpowers/specs/2026-04-07-pyod-expansion-design.md` (v6), Section 2.8 Tier B.

**Depends on:** Tier A (completed) -- `ADEngine` with `profile_data()`, `plan_detection()`, `build_detector()`, knowledge queries.

---

## File Structure

### Modified files

| File | Change |
|------|--------|
| `pyod/utils/ad_engine.py` | Add `run_detection()`, `analyze_results()`, `explain_findings()`, `suggest_next_step()`, `generate_report()` |
| `pyod/test/test_ad_engine.py` | Add test classes for all 5 new methods |
| `skills/od-expert/SKILL.md` | Update with full lifecycle guidance |
| `examples/ad_engine_example.py` | Extend with Tier B lifecycle demo |

---

## Task 1: `run_detection()`

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine.py`

- [ ] **Step 1: Write failing tests for run_detection**

Add to `pyod/test/test_ad_engine.py`:

```python
class TestRunDetection(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        self.rng = np.random.RandomState(42)
        self.X_train = self.rng.randn(200, 10)
        self.X_test = self.rng.randn(50, 10)
        profile = self.engine.profile_data(self.X_train)
        self.plan = self.engine.plan_detection(profile)

    def test_returns_required_keys(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        required = {'plan', 'scores_train', 'labels_train', 'threshold',
                    'n_anomalies', 'anomaly_ratio', 'detector',
                    'runtime_seconds', 'score_summary'}
        for key in required:
            assert key in result, f"Missing key '{key}'"

    def test_scores_shape(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        assert len(result['scores_train']) == 200
        assert len(result['labels_train']) == 200

    def test_with_test_data(self):
        result = self.engine.run_detection(
            self.X_train, self.plan, X_test=self.X_test)
        assert 'scores_test' in result
        assert 'labels_test' in result
        assert len(result['scores_test']) == 50

    def test_score_summary_has_stats(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        summary = result['score_summary']
        for key in ('mean', 'std', 'min', 'max', 'q25', 'q75'):
            assert key in summary, f"Missing stat '{key}'"

    def test_anomaly_ratio_is_fraction(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        assert 0.0 <= result['anomaly_ratio'] <= 1.0

    def test_runtime_is_positive(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        assert result['runtime_seconds'] >= 0.0

    def test_detector_is_fitted(self):
        result = self.engine.run_detection(self.X_train, self.plan)
        assert hasattr(result['detector'], 'decision_scores_')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestRunDetection -v`
Expected: FAIL with `AttributeError: 'ADEngine' object has no attribute 'run_detection'`

- [ ] **Step 3: Implement run_detection**

Add to `ADEngine` class in `pyod/utils/ad_engine.py`, after the `detect()` method. Also add `import time` at the top of the file.

```python
    def run_detection(self, X_train, plan, X_test=None):
        """Execute a detection plan.

        Parameters
        ----------
        X_train : array-like
            Training data.
        plan : dict (DetectionPlan)
            Output of plan_detection().
        X_test : array-like or None
            Optional test data.

        Returns
        -------
        result : dict
            Keys: 'plan', 'scores_train', 'labels_train', 'threshold',
            'n_anomalies', 'anomaly_ratio', 'detector', 'runtime_seconds',
            'score_summary'. If X_test: also 'scores_test', 'labels_test'.
        """
        import time
        start = time.time()

        clf = self.build_detector(plan)
        clf.fit(X_train)

        elapsed = time.time() - start

        scores = clf.decision_scores_
        labels = clf.labels_
        n_anomalies = int(labels.sum())

        result = {
            'plan': plan,
            'scores_train': scores,
            'labels_train': labels,
            'threshold': float(clf.threshold_),
            'n_anomalies': n_anomalies,
            'anomaly_ratio': n_anomalies / len(labels),
            'detector': clf,
            'runtime_seconds': elapsed,
            'score_summary': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'q25': float(np.percentile(scores, 25)),
                'q75': float(np.percentile(scores, 75)),
            },
        }

        if X_test is not None:
            result['scores_test'] = clf.decision_function(X_test)
            result['labels_test'] = clf.predict(X_test)

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestRunDetection -v`
Expected: All 7 PASS

- [ ] **Step 5: Propose commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine.py
git commit -m "feat: add run_detection() to ADEngine with score summary and timing"
```

---

## Task 2: `analyze_results()`

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine.py`

- [ ] **Step 1: Write failing tests**

Add to `pyod/test/test_ad_engine.py`:

```python
class TestAnalyzeResults(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(200, 10)
        profile = self.engine.profile_data(self.X_train)
        plan = self.engine.plan_detection(profile)
        self.result = self.engine.run_detection(self.X_train, plan)

    def test_returns_required_keys(self):
        analysis = self.engine.analyze_results(self.result)
        required = {'n_anomalies', 'anomaly_ratio', 'score_distribution',
                    'top_anomalies', 'summary'}
        for key in required:
            assert key in analysis, f"Missing key '{key}'"

    def test_top_anomalies_sorted_by_score(self):
        analysis = self.engine.analyze_results(self.result)
        top = analysis['top_anomalies']
        assert len(top) > 0
        scores = [a['score'] for a in top]
        assert scores == sorted(scores, reverse=True)

    def test_top_anomalies_have_index_and_score(self):
        analysis = self.engine.analyze_results(self.result)
        for entry in analysis['top_anomalies']:
            assert 'index' in entry
            assert 'score' in entry

    def test_score_distribution_has_stats(self):
        analysis = self.engine.analyze_results(self.result)
        dist = analysis['score_distribution']
        for key in ('mean', 'std', 'min', 'max', 'median', 'q25', 'q75'):
            assert key in dist

    def test_summary_is_string(self):
        analysis = self.engine.analyze_results(self.result)
        assert isinstance(analysis['summary'], str)
        assert len(analysis['summary']) > 0

    def test_with_feature_data(self):
        analysis = self.engine.analyze_results(self.result, X=self.X_train)
        # When X is provided, feature_importance may be included
        assert 'n_anomalies' in analysis

    def test_top_k_parameter(self):
        analysis = self.engine.analyze_results(self.result, top_k=3)
        assert len(analysis['top_anomalies']) <= 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestAnalyzeResults -v`
Expected: FAIL

- [ ] **Step 3: Implement analyze_results**

Add to `ADEngine` class:

```python
    def analyze_results(self, result, X=None, top_k=10):
        """Analyze detection results.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        X : array-like or None
            Original training data for feature-level analysis.
        top_k : int
            Number of top anomalies to return.

        Returns
        -------
        analysis : dict
        """
        scores = result['scores_train']
        labels = result['labels_train']
        n_anomalies = int(labels.sum())

        # Top anomalies sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_anomalies = [{'index': int(i), 'score': float(scores[i])}
                         for i in top_indices]

        # Score distribution
        score_dist = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores)),
            'q25': float(np.percentile(scores, 25)),
            'q75': float(np.percentile(scores, 75)),
        }

        # Narrative summary
        detector_name = result['plan'].get('detector_name', 'unknown')
        ratio = n_anomalies / len(labels) if len(labels) > 0 else 0
        summary = (
            "%d anomalies detected out of %d samples (%.1f%%) "
            "using %s. Scores range from %.4f to %.4f "
            "(mean=%.4f, std=%.4f). Threshold: %.4f."
            % (n_anomalies, len(labels), ratio * 100,
               detector_name,
               score_dist['min'], score_dist['max'],
               score_dist['mean'], score_dist['std'],
               result['threshold']))

        analysis = {
            'n_anomalies': n_anomalies,
            'anomaly_ratio': ratio,
            'score_distribution': score_dist,
            'top_anomalies': top_anomalies,
            'summary': summary,
        }

        # Feature importance (if X is provided and detector supports it)
        if X is not None:
            fi = self._compute_feature_importance(result, X)
            if fi is not None:
                analysis['feature_importance'] = fi

        return analysis

    @staticmethod
    def _compute_feature_importance(result, X):
        """Estimate per-feature contribution to anomaly scores.

        Uses a simple approach: correlation between each feature's
        absolute z-score and the anomaly score.
        """
        try:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim != 2:
                return None
            scores = result['scores_train']
            if len(scores) != X_arr.shape[0]:
                return None

            # Per-feature absolute z-score
            means = np.mean(X_arr, axis=0)
            stds = np.std(X_arr, axis=0)
            stds[stds == 0] = 1.0
            z_scores = np.abs((X_arr - means) / stds)

            # Correlation of each feature's z-score with anomaly score
            importances = []
            for j in range(X_arr.shape[1]):
                corr = np.corrcoef(z_scores[:, j], scores)[0, 1]
                importances.append(float(corr) if np.isfinite(corr) else 0.0)

            return importances
        except Exception:
            return None
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestAnalyzeResults -v`
Expected: All 7 PASS

- [ ] **Step 5: Propose commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine.py
git commit -m "feat: add analyze_results() to ADEngine with score stats and feature importance"
```

---

## Task 3: `explain_findings()`

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine.py`

- [ ] **Step 1: Write failing tests**

```python
class TestExplainFindings(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(200, 10)
        profile = self.engine.profile_data(self.X_train)
        plan = self.engine.plan_detection(profile)
        self.result = self.engine.run_detection(self.X_train, plan)

    def test_default_top_k(self):
        explanations = self.engine.explain_findings(self.result)
        assert len(explanations) == 5  # default top_k=5

    def test_custom_top_k(self):
        explanations = self.engine.explain_findings(self.result, top_k=3)
        assert len(explanations) == 3

    def test_specific_indices(self):
        explanations = self.engine.explain_findings(
            self.result, indices=[0, 5, 10])
        assert len(explanations) == 3
        assert explanations[0]['index'] == 0
        assert explanations[1]['index'] == 5

    def test_entry_has_required_fields(self):
        explanations = self.engine.explain_findings(self.result)
        for entry in explanations:
            assert 'index' in entry
            assert 'score' in entry
            assert 'percentile' in entry
            assert 'narrative' in entry

    def test_percentile_range(self):
        explanations = self.engine.explain_findings(self.result)
        for entry in explanations:
            assert 0.0 <= entry['percentile'] <= 100.0

    def test_narrative_is_string(self):
        explanations = self.engine.explain_findings(self.result)
        for entry in explanations:
            assert isinstance(entry['narrative'], str)
            assert len(entry['narrative']) > 0

    def test_with_feature_data(self):
        explanations = self.engine.explain_findings(
            self.result, X=self.X_train, top_k=2)
        for entry in explanations:
            assert 'contributing_features' in entry
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestExplainFindings -v`
Expected: FAIL

- [ ] **Step 3: Implement explain_findings**

```python
    def explain_findings(self, result, indices=None, top_k=5, X=None):
        """Explain why specific samples were flagged as anomalies.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        indices : list of int or None
            Specific sample indices. If None, explains top-k.
        top_k : int
            Number of top anomalies to explain if indices is None.
        X : array-like or None
            Original data for feature-level explanations.

        Returns
        -------
        explanations : list of dict
        """
        scores = result['scores_train']

        if indices is None:
            indices = list(np.argsort(scores)[::-1][:top_k])

        # Compute percentiles
        explanations = []
        for idx in indices:
            score = float(scores[idx])
            pctile = float(np.mean(scores <= score) * 100)
            label = 'anomaly' if score > result['threshold'] else 'normal'

            narrative = (
                "Sample %d has anomaly score %.4f (percentile: %.1f%%), "
                "classified as %s (threshold: %.4f)."
                % (idx, score, pctile, label, result['threshold']))

            entry = {
                'index': int(idx),
                'score': score,
                'percentile': pctile,
                'label': label,
                'narrative': narrative,
            }

            # Feature contributions if X provided
            if X is not None:
                contribs = self._feature_contributions(X, idx, scores)
                if contribs is not None:
                    entry['contributing_features'] = contribs

            explanations.append(entry)

        return explanations

    @staticmethod
    def _feature_contributions(X, idx, scores):
        """Compute per-feature z-score for a specific sample."""
        try:
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim != 2:
                return None
            means = np.mean(X_arr, axis=0)
            stds = np.std(X_arr, axis=0)
            stds[stds == 0] = 1.0
            z = np.abs((X_arr[idx] - means) / stds)
            # Return top-5 features by z-score
            top_feat = np.argsort(z)[::-1][:5]
            return [{'feature': int(f), 'z_score': float(z[f])}
                    for f in top_feat]
        except Exception:
            return None
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestExplainFindings -v`
Expected: All 7 PASS

- [ ] **Step 5: Propose commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine.py
git commit -m "feat: add explain_findings() to ADEngine with per-sample explanations"
```

---

## Task 4: `suggest_next_step()`

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine.py`

- [ ] **Step 1: Write failing tests**

```python
class TestSuggestNextStep(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 10)
        profile = self.engine.profile_data(X_train)
        plan = self.engine.plan_detection(profile)
        self.result = self.engine.run_detection(X_train, plan)
        self.analysis = self.engine.analyze_results(self.result)

    def test_returns_required_keys(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis)
        assert 'action' in suggestion
        assert 'reason' in suggestion

    def test_too_many_false_positives(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='too many false positives')
        assert suggestion['action'] in ('adjust_threshold',
                                         'try_alternative')

    def test_try_different_detector(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='try a different detector')
        assert suggestion['action'] == 'try_alternative'
        assert 'new_plan' in suggestion

    def test_no_feedback_suggests_done_or_alternative(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis)
        assert suggestion['action'] in ('done', 'try_alternative',
                                         'adjust_threshold')

    def test_new_plan_is_valid(self):
        suggestion = self.engine.suggest_next_step(
            self.result, self.analysis,
            feedback='try a different detector')
        if 'new_plan' in suggestion:
            plan = suggestion['new_plan']
            assert 'detector_name' in plan
            assert plan['detector_name'] != self.result['plan']['detector_name']
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestSuggestNextStep -v`
Expected: FAIL

- [ ] **Step 3: Implement suggest_next_step**

```python
    def suggest_next_step(self, result, analysis, feedback=None):
        """Suggest what to try next.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        analysis : dict
            Output of analyze_results().
        feedback : str or None
            User feedback like 'too many false positives'.

        Returns
        -------
        suggestion : dict
            Keys: 'action', 'reason', optionally 'new_plan'.
        """
        feedback_lower = (feedback or '').lower()
        current_detector = result['plan'].get('detector_name', '')
        ratio = analysis.get('anomaly_ratio', 0)

        # Parse feedback
        if 'false positive' in feedback_lower or 'too many' in feedback_lower:
            return {
                'action': 'adjust_threshold',
                'reason': 'High false positive rate reported. Try raising '
                          'the anomaly threshold (lower contamination).',
            }

        if ('different' in feedback_lower or 'another' in feedback_lower
                or 'try' in feedback_lower):
            new_plan = self._suggest_alternative(result)
            return {
                'action': 'try_alternative',
                'reason': 'Trying an alternative detector.',
                'new_plan': new_plan,
            }

        if 'false negative' in feedback_lower or 'missed' in feedback_lower:
            return {
                'action': 'adjust_threshold',
                'reason': 'Missed anomalies reported. Try lowering the '
                          'anomaly threshold (higher contamination).',
            }

        if 'ensemble' in feedback_lower:
            return {
                'action': 'try_alternative',
                'reason': 'Consider running multiple detectors and '
                          'combining scores.',
                'new_plan': self._suggest_alternative(result),
            }

        # No feedback: heuristic based on results
        if ratio > 0.3:
            return {
                'action': 'adjust_threshold',
                'reason': '%.0f%% flagged as anomalies, which is unusually '
                          'high. Consider raising the threshold.' % (ratio * 100),
            }
        if ratio == 0:
            new_plan = self._suggest_alternative(result)
            return {
                'action': 'try_alternative',
                'reason': 'No anomalies detected. Try a different detector.',
                'new_plan': new_plan,
            }

        return {
            'action': 'done',
            'reason': 'Results look reasonable (%.1f%% anomaly rate). '
                      'Review the top anomalies to validate.'
                      % (ratio * 100),
        }

    def _suggest_alternative(self, result):
        """Suggest an alternative detector different from the current one."""
        current = result['plan'].get('detector_name', '')
        profile = result.get('_profile')

        # Use alternatives from the original plan if available
        alternatives = result['plan'].get('alternatives', [])
        for alt in alternatives:
            if alt.get('detector_name') and alt['detector_name'] != current:
                return alt

        # Fallback: pick from a default list
        fallback_order = ['IForest', 'ECOD', 'KNN', 'LOF', 'HBOS',
                          'COPOD', 'CBLOF']
        for name in fallback_order:
            if name != current:
                algo = self.kb.get_algorithm(name)
                if algo and algo.get('status') == 'shipped':
                    return self._make_plan(
                        detector_name=name, params={},
                        reason='Alternative to %s' % current,
                        evidence=[], confidence=0.6)

        return self._make_plan(
            detector_name='IForest', params={},
            reason='Default fallback', evidence=[], confidence=0.5)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestSuggestNextStep -v`
Expected: All 5 PASS

- [ ] **Step 5: Propose commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine.py
git commit -m "feat: add suggest_next_step() to ADEngine for iterative refinement"
```

---

## Task 5: `generate_report()`

**Files:**
- Modify: `pyod/utils/ad_engine.py`
- Modify: `pyod/test/test_ad_engine.py`

- [ ] **Step 1: Write failing tests**

```python
class TestGenerateReport(unittest.TestCase):
    def setUp(self):
        self.engine = ADEngine()
        rng = np.random.RandomState(42)
        X_train = rng.randn(200, 10)
        profile = self.engine.profile_data(X_train)
        plan = self.engine.plan_detection(profile)
        self.result = self.engine.run_detection(X_train, plan)
        self.analysis = self.engine.analyze_results(self.result)

    def test_text_format(self):
        report = self.engine.generate_report(
            self.result, self.analysis, format='text')
        assert isinstance(report, str)
        assert 'Anomaly Detection Report' in report
        assert self.result['plan']['detector_name'] in report

    def test_json_format(self):
        import json
        report = self.engine.generate_report(
            self.result, self.analysis, format='json')
        parsed = json.loads(report)
        assert 'detector' in parsed
        assert 'n_anomalies' in parsed

    def test_report_contains_key_info(self):
        report = self.engine.generate_report(
            self.result, self.analysis, format='text')
        assert 'anomal' in report.lower()
        assert str(self.analysis['n_anomalies']) in report

    def test_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            self.engine.generate_report(
                self.result, self.analysis, format='pdf')
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestGenerateReport -v`
Expected: FAIL

- [ ] **Step 3: Implement generate_report**

```python
    def generate_report(self, result, analysis, format='text'):
        """Generate a summary report.

        Parameters
        ----------
        result : dict
            Output of run_detection().
        analysis : dict
            Output of analyze_results().
        format : str
            'text' (markdown) or 'json'.

        Returns
        -------
        report : str
        """
        import json as json_mod

        if format == 'json':
            report_dict = {
                'detector': result['plan'].get('detector_name', ''),
                'reason': result['plan'].get('reason', ''),
                'n_samples': len(result['scores_train']),
                'n_anomalies': analysis['n_anomalies'],
                'anomaly_ratio': analysis['anomaly_ratio'],
                'threshold': result['threshold'],
                'runtime_seconds': result.get('runtime_seconds', 0),
                'score_distribution': analysis['score_distribution'],
                'top_anomalies': analysis['top_anomalies'][:10],
            }
            return json_mod.dumps(report_dict, indent=2, default=str)

        if format == 'text':
            lines = []
            lines.append('# Anomaly Detection Report')
            lines.append('')
            det = result['plan'].get('detector_name', 'unknown')
            lines.append('## Configuration')
            lines.append('- **Detector:** %s' % det)
            lines.append('- **Reason:** %s' % result['plan'].get('reason', ''))
            lines.append('- **Samples:** %d' % len(result['scores_train']))
            lines.append('- **Runtime:** %.2fs'
                         % result.get('runtime_seconds', 0))
            lines.append('')
            lines.append('## Results')
            lines.append('- **Anomalies found:** %d (%.1f%%)'
                         % (analysis['n_anomalies'],
                            analysis['anomaly_ratio'] * 100))
            lines.append('- **Threshold:** %.4f' % result['threshold'])
            dist = analysis['score_distribution']
            lines.append('- **Score range:** %.4f to %.4f'
                         % (dist['min'], dist['max']))
            lines.append('- **Score mean/std:** %.4f / %.4f'
                         % (dist['mean'], dist['std']))
            lines.append('')
            lines.append('## Top Anomalies')
            lines.append('')
            lines.append('| Rank | Index | Score |')
            lines.append('|------|-------|-------|')
            for rank, entry in enumerate(analysis['top_anomalies'][:10], 1):
                lines.append('| %d | %d | %.4f |'
                             % (rank, entry['index'], entry['score']))
            lines.append('')
            return '\n'.join(lines)

        raise ValueError("Unknown report format: '%s'. "
                         "Use 'text' or 'json'." % format)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest pyod/test/test_ad_engine.py::TestGenerateReport -v`
Expected: All 4 PASS

- [ ] **Step 5: Propose commit**

```bash
git add pyod/utils/ad_engine.py pyod/test/test_ad_engine.py
git commit -m "feat: add generate_report() to ADEngine with text and JSON output"
```

---

## Task 6: Update skill and example

**Files:**
- Modify: `skills/od-expert/SKILL.md`
- Modify: `examples/ad_engine_example.py`

- [ ] **Step 1: Update the Claude Code skill**

Replace the contents of `skills/od-expert/SKILL.md` with:

```markdown
---
name: od-expert
description: Anomaly detection expert. Drives PyOD's ADEngine for the full detection lifecycle -- profiling, planning, execution, analysis, explanation, iteration, and reporting.
---

You are an anomaly detection expert backed by PyOD's ADEngine.

## When to activate
- User has data and wants anomaly detection
- User asks "which detector should I use?"
- User asks about PyOD algorithms or benchmarks
- User asks to compare detection methods
- User wants to analyze or explain anomaly detection results

## How to work
Import and call ADEngine directly in Python:

```python
from pyod.utils.ad_engine import ADEngine
engine = ADEngine()

# Full lifecycle
profile = engine.profile_data(X_train)
plan = engine.plan_detection(profile)
result = engine.run_detection(X_train, plan)
analysis = engine.analyze_results(result, X=X_train)
explanations = engine.explain_findings(result, X=X_train, top_k=5)
report = engine.generate_report(result, analysis)

# If user is unhappy with results:
suggestion = engine.suggest_next_step(result, analysis, feedback="too many false positives")
# Follow suggestion.action: 'adjust_threshold', 'try_alternative', or 'done'
```

For knowledge queries only (no execution), MCP tools are also available
if the MCP server is running: profile_data, plan_detection, build_detector,
list_detectors, explain_detector, compare_detectors, get_benchmarks.

## Lifecycle flow
profile_data -> plan_detection -> run_detection -> analyze_results
-> explain_findings -> (suggest_next_step if needed) -> generate_report
```

- [ ] **Step 2: Update the example**

Replace `examples/ad_engine_example.py` with a version that demonstrates the full lifecycle including Tier B methods. The example should show: one-shot detect, step-by-step lifecycle with run_detection/analyze/explain/report, and knowledge queries.

```python
"""ADEngine: Full anomaly detection lifecycle.

Demonstrates PyOD's ADEngine for automatic detector selection,
execution, analysis, explanation, and report generation.
"""
from pyod.utils.ad_engine import ADEngine
from pyod.utils.data import generate_data

# Generate sample data
X_train, X_test, y_train, y_test = generate_data(
    n_train=300, n_test=100, n_features=20, contamination=0.1)

# Initialize the engine
engine = ADEngine()

# === Full lifecycle ===
print("=" * 60)
print("FULL ANOMALY DETECTION LIFECYCLE")
print("=" * 60)

# 1. Profile
profile = engine.profile_data(X_train)
print("\n1. Data profile:", profile['data_type'],
      "(%d samples, %d features)" % (profile['n_samples'],
                                      profile['n_features']))

# 2. Plan
plan = engine.plan_detection(profile, priority='speed')
print("2. Plan:", plan['detector_name'], "-", plan['reason'])

# 3. Execute
result = engine.run_detection(X_train, plan, X_test=X_test)
print("3. Detection: %d anomalies (%.1f%%) in %.3fs"
      % (result['n_anomalies'], result['anomaly_ratio'] * 100,
         result['runtime_seconds']))

# 4. Analyze
analysis = engine.analyze_results(result, X=X_train)
print("4. Analysis:", analysis['summary'])

# 5. Explain
explanations = engine.explain_findings(result, X=X_train, top_k=3)
print("5. Top anomalies:")
for exp in explanations:
    print("   Sample %d: score=%.4f (%s)"
          % (exp['index'], exp['score'], exp['label']))

# 6. Suggest next step
suggestion = engine.suggest_next_step(result, analysis)
print("6. Suggestion:", suggestion['action'], "-", suggestion['reason'])

# 7. Report
report = engine.generate_report(result, analysis)
print("\n7. Report preview (first 500 chars):")
print(report[:500])

# === Knowledge queries ===
print("\n" + "=" * 60)
print("KNOWLEDGE QUERIES")
print("=" * 60)

print("\nAvailable text detectors:")
for d in engine.list_detectors(data_type='text'):
    print("  %s: %s" % (d['name'], d['full_name']))

print("\nECOD explained:")
info = engine.explain_detector('ECOD')
print("  %s" % info['full_name'])
print("  Best for: %s" % info['best_for'])

print("\nADBench top 5:")
bench = engine.get_benchmarks('ADBench')
print("  %s" % bench['ADBench']['rankings']['overall_top_5'])
```

- [ ] **Step 3: Run the example**

Run: `python examples/ad_engine_example.py`
Expected: Prints full lifecycle output without errors.

- [ ] **Step 4: Run full test suite**

Run: `python -m pytest pyod/test/test_ad_engine.py pyod/test/test_knowledge.py -v`
Expected: All tests PASS

- [ ] **Step 5: Propose commit**

```bash
git add skills/od-expert/SKILL.md examples/ad_engine_example.py
git commit -m "feat: update skill and example with full Tier B lifecycle"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] B1: `run_detection()` with scores, labels, timing, summary → Task 1
- [x] B2: `analyze_results()` with score distribution, top anomalies, feature importance, narrative → Task 2
- [x] B3: `explain_findings()` with per-sample explanations and contributing features → Task 3
- [x] B4: `suggest_next_step()` with feedback parsing and alternative suggestions → Task 4
- [x] B5: `generate_report()` with text and JSON output → Task 5
- [x] B6: Update skill with full lifecycle guidance → Task 6
- [x] B7: Tests + examples → integrated into Tasks 1-6

**Placeholder scan:** No TBD, TODO, or "implement later" found.

**Type consistency:** All methods operate on the same `result` dict structure returned by `run_detection()`. `analysis` dict from `analyze_results()` is consumed by `suggest_next_step()` and `generate_report()`. Method signatures consistent across tasks.

**Not in scope:** MCP lifecycle tools (Python-first decision), TimeSeriesOD (separate track).
