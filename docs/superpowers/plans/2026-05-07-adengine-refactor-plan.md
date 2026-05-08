# ADEngine Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the constant `quality['stability']` metric (closes #667), decompose `pyod/utils/ad_engine.py` (1,650 LOC) into private helper modules, and pay down latent tech debt. All without breaking the public API.

**Architecture:** 3 sequential PRs on `development`, each with its own branch. PR 1 fixes the bug in-place. PR 2 extracts helpers. PR 3 polishes the post-decomposition code. The bug-fix-first ordering closes #667 in the smallest possible diff and lets PR 2 move correct code rather than buggy code.

**Tech Stack:** Python 3.9+, pyod 3.x, numpy, scipy, pytest, miniforge `py312` env (default).

**Spec:** [`docs/superpowers/specs/2026-05-07-adengine-refactor-design.md`](../specs/2026-05-07-adengine-refactor-design.md) (committed at `9db1b59`).

**Approval rules:**

- The repo policy is **no `git commit` or `git push` without user approval**. The orchestrator (parent Claude Code session) must show each proposed commit and wait for explicit confirmation.
- Subagents work in their own worktree and **stage** changes (`git add`) but **do not commit**. The orchestrator commits after each task's user approval.
- The orchestrator never force-pushes or rewrites public history.

**Parallelization map:**

| Stage | Parallelism | Notes |
|-------|-------------|-------|
| PR 1 (Tasks 1.1-1.7) | Sequential | TDD flow: tests → fix → docstring → docs → CHANGES → commit |
| PR 2 (Tasks 2.1-2.4) | 4-way parallel inside one worktree, sequential commits | Each module extraction touches a different new file plus the same `ad_engine.py`. Parallel agents extract; orchestrator merges and commits one at a time |
| PR 3 (Tasks 3.1-3.6) | Mostly parallel | Type hints + magic numbers + NL parser + contamination dedup are independent. Validation and exception tightening touch overlapping sites — sequence them after the others |

**Branching:**

```
development (master eventually)
├── fix/adengine-stability-issue-667    (PR 1)
├── refactor/adengine-decomposition     (PR 2; branched from PR 1 after merge)
└── refactor/adengine-cleanup           (PR 3; branched from PR 2 after merge)
```

---

## PR 1: Stability Bug Fix (closes #667)

**Branch:** `fix/adengine-stability-issue-667` from `development` after the spec commit (`9db1b59`).

**Files touched:**
- `pyod/utils/ad_engine.py` (modify lines 1229-1283: `_compute_quality` body, docstring, explanation string)
- `pyod/test/test_ad_engine_v3.py` (add `TestStabilityMetric` class)
- 12 documentation files (full list in Task 1.6)
- `CHANGES.txt` (new entry)

### Task 1.1: Create the PR 1 branch

**Files:** None — branch creation only.

- [ ] **Step 1: Verify development branch is clean and current**

```bash
git status
git log -1 --oneline
```

Expected: Working tree clean except untracked `Review-*.md` files (which are gitignored). Latest commit is `9db1b59 docs: add ADEngine refactor design spec for issue #667`.

- [ ] **Step 2: Create and switch to the PR 1 branch**

```bash
git checkout -b fix/adengine-stability-issue-667
git status
```

Expected: `On branch fix/adengine-stability-issue-667. Your branch is up to date with no remote.`

### Task 1.2: Add the regression test class (TDD: red phase)

**Files:**
- Modify: `pyod/test/test_ad_engine_v3.py` (append a new class at the end)

- [ ] **Step 1: Read the current test file structure**

Open `pyod/test/test_ad_engine_v3.py`. Note the imports at the top (`numpy as np`, `from pyod.utils.ad_engine import ADEngine`, etc.) and the existing test class style (e.g., `TestSessionAnalyze`).

- [ ] **Step 2: Append the `TestStabilityMetric` class at the end of the file**

Paste verbatim from spec Section 3.9. The full class:

```python
class TestStabilityMetric:
    """Regression tests for the stability metric (issue #667)."""

    def _quality(self, scores, labels):
        """Call the quality function via whichever path is available.

        During PR 1 the helper module does not yet exist; we use the
        private method on ADEngine. After PR 2 the helper module is
        importable and we call it directly. This adapter keeps the
        same test class valid across the PR-1-to-PR-2 transition
        without violating PR 2's "tests pass without modification" rule.
        """
        results = [{'scores_train': scores, 'labels_train': labels}]
        consensus = {'agreement': 0.5}
        try:
            from pyod.utils._quality_metrics import compute_quality
        except ImportError:
            engine = ADEngine()
            return engine._compute_quality(
                scores, labels, results, consensus)
        return compute_quality(scores, labels, results, consensus)

    def test_stability_responds_to_data(self):
        """Large cutoff gaps score higher than nearly tied boundaries."""
        scores_clean = np.concatenate([
            np.full(100, 10.0),
            np.full(900, 0.0),
        ])
        labels_clean = np.zeros(1000, dtype=int)
        labels_clean[:100] = 1
        quality_clean = self._quality(scores_clean, labels_clean)

        scores_borderline = np.concatenate([
            np.linspace(2.0, 1.01, 100),
            np.array([1.009]),
            np.linspace(0.0, -1.0, 899),
        ])
        labels_borderline = np.zeros(1000, dtype=int)
        labels_borderline[:100] = 1
        quality_blurry = self._quality(scores_borderline, labels_borderline)

        assert quality_clean['stability'] == 1.0
        assert quality_blurry['stability'] < 0.01
        assert quality_clean['stability'] > quality_blurry['stability']

    def test_stability_is_deterministic(self):
        """Same input produces same output across calls."""
        rng = np.random.RandomState(42)
        scores = rng.randn(500)
        labels = (scores > np.quantile(scores, 0.9)).astype(int)
        q1 = self._quality(scores, labels)
        q2 = self._quality(scores, labels)
        assert q1['stability'] == q2['stability']
        assert q1['separation'] == q2['separation']

    def test_stability_zero_for_no_anomalies(self):
        """n_anomalies=0 returns stability=0.0."""
        scores = np.random.RandomState(0).randn(100)
        labels = np.zeros(100, dtype=int)
        assert self._quality(scores, labels)['stability'] == 0.0

    def test_stability_zero_for_all_anomalies(self):
        """n_anomalies=n returns stability=0.0."""
        scores = np.random.RandomState(0).randn(100)
        labels = np.ones(100, dtype=int)
        assert self._quality(scores, labels)['stability'] == 0.0

    def test_stability_zero_for_constant_scores(self):
        """All-equal scores produce stability=0.0 (std is 0)."""
        scores = np.full(100, 0.5)
        labels = np.zeros(100, dtype=int)
        labels[:10] = 1
        assert self._quality(scores, labels)['stability'] == 0.0

    def test_stability_zero_for_nonfinite_scores(self):
        """NaN or Inf in scores returns 0; never emit NaN."""
        labels = np.zeros(5, dtype=int)
        labels[:2] = 1
        for bad_value in (np.nan, np.inf, -np.inf):
            scores = np.array([5.0, 4.0, bad_value, 1.0, 0.0])
            q = self._quality(scores, labels)
            assert q['stability'] == 0.0
            assert np.isfinite(q['overall'])

    def test_stability_k_equals_one(self):
        """k=1 is well-defined: gap between highest score and second-highest."""
        scores = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
        labels = np.array([1, 0, 0, 0, 0])
        q = self._quality(scores, labels)
        expected = min(1.0, (5.0 - 3.0) / float(scores.std()))
        assert abs(q['stability'] - expected) < 1e-9

    def test_stability_k_equals_n_minus_one(self):
        """k=n-1 is well-defined: only one inlier."""
        scores = np.array([5.0, 4.0, 3.0, 2.0, 0.0])
        labels = np.array([1, 1, 1, 1, 0])
        q = self._quality(scores, labels)
        expected = min(1.0, (2.0 - 0.0) / float(scores.std()))
        assert abs(q['stability'] - expected) < 1e-9

    def test_stability_ties_at_boundary(self):
        """Tied scores at the cutoff produce gap=0 -> stability=0."""
        scores = np.array([5.0, 4.0, 3.0, 3.0, 1.0])
        labels = np.array([1, 1, 1, 0, 0])
        q = self._quality(scores, labels)
        assert q['stability'] == 0.0

    def test_stability_scale_invariant(self):
        """Multiplying scores by a positive constant leaves stability unchanged."""
        scores = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
        labels = np.array([1, 0, 0, 0, 0])
        q1 = self._quality(scores, labels)
        q2 = self._quality(scores * 1e6, labels)
        q3 = self._quality(scores * 1e-12, labels)
        assert abs(q1['stability'] - q2['stability']) < 1e-9
        assert abs(q1['stability'] - q3['stability']) < 1e-9

    def test_stability_in_unit_interval(self):
        """Stability is always in [0, 1] across random inputs."""
        rng = np.random.RandomState(7)
        for _ in range(50):
            n = rng.randint(20, 500)
            scores = rng.randn(n)
            k = rng.randint(1, n - 1)
            threshold = np.partition(scores, -k)[-k]
            labels = (scores >= threshold).astype(int)
            q = self._quality(scores, labels)
            assert 0.0 <= q['stability'] <= 1.0
```

- [ ] **Step 3: Run the new test class to confirm RED phase**

```bash
pytest pyod/test/test_ad_engine_v3.py::TestStabilityMetric -v
```

Expected: 11 tests, multiple FAIL. The exact failures depend on input but at minimum `test_stability_responds_to_data` (the buggy formula returns ≈0.817 for both fixtures, so `quality_clean['stability'] == 1.0` fails), `test_stability_zero_for_constant_scores` (buggy formula doesn't handle constant scores cleanly), `test_stability_zero_for_nonfinite_scores` (buggy formula propagates NaN), and `test_stability_scale_invariant` (buggy formula already invariant by accident, but assertions on specific values fail).

Some tests may pass under the bug (e.g., `test_stability_zero_for_no_anomalies` because the buggy code does check `if n_anomalies == 0`). That is fine — the failures we get are the contract being broken.

- [ ] **Step 4: Stage the test file (do not commit)**

```bash
git add pyod/test/test_ad_engine_v3.py
git diff --cached --stat
```

Expected output: `pyod/test/test_ad_engine_v3.py | NN +++++++++` showing only one file staged.

### Task 1.3: Apply the bug fix to `_compute_quality`

**Files:**
- Modify: `pyod/utils/ad_engine.py:1243-1264` (the `Stability` block inside `_compute_quality`)

- [ ] **Step 1: Read the current `_compute_quality` body to confirm line numbers**

```bash
sed -n '1229,1283p' pyod/utils/ad_engine.py
```

Expected: the docstring + body of `_compute_quality` ending at line 1283 with `}` of the return dict.

- [ ] **Step 2: Replace the stability block (lines 1243-1264) with the new formula**

Find the existing block:

```python
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
```

Replace with:

```python
        # Stability: standardized score gap at the rank-k cutoff.
        # Replaces the v1 Jaccard-of-nested-top-k formula which was
        # mathematically constant (issue #667).
        n_anomalies = int(labels.sum())
        n_samples = len(labels)
        if n_anomalies == 0 or n_anomalies >= n_samples:
            stability = 0.0
        elif not np.all(np.isfinite(scores)):
            # Non-finite scores: undefined gap; refuse to emit NaN.
            stability = 0.0
        else:
            sorted_scores = np.sort(scores)[::-1]
            gap = float(sorted_scores[n_anomalies - 1]
                        - sorted_scores[n_anomalies])
            std = float(scores.std())
            if std == 0.0:
                stability = 0.0
            else:
                stability = float(np.clip(gap / std, 0.0, 1.0))
```

- [ ] **Step 3: Run the regression tests; expect GREEN**

```bash
pytest pyod/test/test_ad_engine_v3.py::TestStabilityMetric -v
```

Expected: 11 tests, all PASS.

- [ ] **Step 4: Run the full ADEngine test suite; expect no new failures**

```bash
pytest pyod/test/test_ad_engine.py pyod/test/test_ad_engine_v3.py -v
```

Expected: All tests pass. If `test_quality_metrics` (in `TestSessionAnalyze`) asserts a specific verdict that changed because the new stability shifts `overall`, update **only the assertion that depends on the constant 0.817**, and document the change in CHANGES.txt (Task 1.7).

- [ ] **Step 5: Stage the change**

```bash
git add pyod/utils/ad_engine.py
git diff --cached --stat
```

### Task 1.4: Update the `_compute_quality` docstring

**Files:**
- Modify: `pyod/utils/ad_engine.py:1229-1242` (the docstring block)

- [ ] **Step 1: Replace the existing docstring**

Find the current docstring (`"""Compute quality metrics: separation, agreement, stability."""`).

Replace with the docstring from spec Section 3.5:

```python
    def _compute_quality(self, scores, labels, results, consensus):
        """Compute three diagnostic quality metrics for a detection run.

        Each metric diagnoses one independent failure mode and drives
        one branch of `ADEngine.iterate()`:

        - ``separation``: anomaly-vs-inlier mean score gap (global).
          Low value indicates the detector did not produce a usable signal.
        - ``agreement``: pairwise Spearman rank correlation across base
          detectors (cross-detector). Low value indicates detectors
          disagree on which samples are anomalous.
        - ``stability``: standardized score gap at the rank-k cutoff
          (local). Computed as
          ``(score[rank_k] - score[rank_k+1]) / scores.std()``, clipped
          to ``[0, 1]``. Low value indicates many tied scores near the
          threshold; the anomaly set is sensitive to the contamination
          value, and ``adjust_contamination`` is the suggested action.

        The ``stability`` key was historically defined (and
        mis-implemented) as the Jaccard index of nested top-k slices,
        which collapses to a constant. The formula was revised in pyod
        v3.3 (closes #667). The key name is retained for backwards
        compatibility with v3.2.x callers.

        Parameters
        ----------
        scores : np.ndarray, shape (n_samples,)
            Consensus or per-detector anomaly scores. Higher means more
            anomalous.
        labels : np.ndarray, shape (n_samples,)
            Binary labels (0 inlier, 1 anomaly).
        results : list of dict
            Per-detector results from `ADEngine.run()`. Reserved for
            callers that thread per-detector data through quality
            computation. Not directly read by this method.
        consensus : dict
            Consensus dict from `ADEngine.run()`. Provides the
            ``agreement`` field.

        Returns
        -------
        dict
            Keys: ``separation`` (float in [0, 1]), ``agreement`` (float
            in [0, 1]), ``stability`` (float in [0, 1]), ``overall``
            (float in [0, 1], mean of the three), ``verdict`` (one of
            ``'high'``, ``'medium'``, ``'low'``), ``explanation``
            (human-readable summary).
        """
```

- [ ] **Step 2: Run the test suite; expect GREEN**

```bash
pytest pyod/test/test_ad_engine.py pyod/test/test_ad_engine_v3.py -v
```

Expected: all pass.

- [ ] **Step 3: Stage**

```bash
git add pyod/utils/ad_engine.py
```

### Task 1.5: Update the `explanation` string format

**Files:**
- Modify: `pyod/utils/ad_engine.py` near line 1280 (the `explanation` field of the returned dict)

- [ ] **Step 1: Find the current explanation format**

```bash
grep -n "Separation=%" pyod/utils/ad_engine.py
```

Expected: one hit around line 1280 inside `_compute_quality`.

- [ ] **Step 2: Replace the format string**

Find:

```python
            'explanation': 'Separation=%.2f, agreement=%.2f, '
                           'stability=%.2f.' % (
                               separation, agreement, stability),
```

Replace with:

```python
            'explanation': 'separation={:.2f}, agreement={:.2f}, '
                           'stability={:.2f} (cutoff gap)'.format(
                               separation, agreement, stability),
```

The parenthetical "(cutoff gap)" flags the new semantic for human readers.

- [ ] **Step 3: Run the test suite**

```bash
pytest pyod/test/test_ad_engine.py pyod/test/test_ad_engine_v3.py -v
```

Expected: all pass. If any test asserts the exact explanation text, update it.

- [ ] **Step 4: Stage**

```bash
git add pyod/utils/ad_engine.py
```

### Task 1.6: Update documentation surfaces

**Files (each modified separately):**
- `docs/examples/adengine.rst:75`
- `docs/examples/agentic.rst:149`
- `pyod/skills/od_expert/SKILL.md:91`
- `pyod/skills/od_expert/references/graph.md:107`
- `pyod/skills/od_expert/references/tabular.md:100`
- `pyod/skills/od_expert/references/text_image.md:100, 109`
- `pyod/skills/od_expert/references/time_series.md:94, 107`
- `pyod/skills/od_expert/references/workflow.md:60, 88, 152, 160, 174`
- `pyod/skills/od_expert/references/pitfalls.md:83`
- `docs/superpowers/specs/2026-04-12-v3-agentic-design.md:524-532` (footnote update only)

These can be parallelized: each file is independent.

- [ ] **Step 1: Update `docs/examples/adengine.rst`**

Find line 75:

```rst
    print("Stability:",  q['stability'])       # 0.82
```

Replace with:

```rst
    print("Stability:",  q['stability'])       # cutoff-gap value (data-dependent)
```

The exact value depended on the buggy formula. Annotate as "data-dependent" since the new formula's value depends on score distribution.

- [ ] **Step 2: Update `docs/examples/agentic.rst`**

Find line 149:

```
    # state.quality['stability']               == 0.814
```

Replace with:

```
    # state.quality['stability']               (cutoff-gap value, data-dependent)
```

- [ ] **Step 3: Review `pyod/skills/od_expert/SKILL.md:91` (trigger threshold)**

Find:

```
4. **Quality assessment weak** — `state.quality.separation < 0.1` OR `state.quality.stability < 0.5`
```

Keep the threshold `0.5` as-is. Add a one-line comment in CHANGES.txt (Task 1.7) noting the threshold may need recalibration. Do not change the threshold here without empirical data.

- [ ] **Step 4: Update each od-expert reference file**

For each of `graph.md`, `tabular.md`, `text_image.md`, `time_series.md`, `workflow.md`, replace the hardcoded stability values with "cutoff-gap value, data-dependent" annotations. Use this find/replace style:

```
# state.quality: {agreement: 0.71, separation: 0.28, stability: 0.66}
```

becomes:

```
# state.quality: {agreement: 0.71, separation: 0.28, stability: 0.X}  # cutoff-gap, data-dependent
```

The exact wording can be tightened per file. The principle: do not commit to a fake number; flag the value as data-dependent.

For sentences like `workflow.md:160` ("...with agreement 0.68 and stability 0.82..."), restructure to "...with agreement 0.68 and a clean cutoff (stability near 1.0 on this synthetic example)..." or similar prose that does not commit to the bogus 0.82 number while preserving the narrative.

- [ ] **Step 5: Update `pyod/skills/od_expert/references/pitfalls.md:83`**

Find:

```
On n < 200, stability is naturally low because resampling has high variance. Mitigation: relax the trigger 4 threshold to 0.3 for small data.
```

Replace with:

```
On small n (< 200), stability is naturally low because the score distribution at the rank-k cutoff has high variance from one sample to the next, so the gap-to-std ratio is fragile. Mitigation: relax the trigger 4 threshold to 0.3 for small data.
```

- [ ] **Step 6: Update `docs/superpowers/specs/2026-04-12-v3-agentic-design.md:524-532` with a footnote**

The original spec defined the buggy formula. Do not rewrite — leave as historical record. Add a footnote at the top of section 4.4 step 3:

```
3. **Label stability** (`quality.stability`): Jaccard index of top-k anomaly sets when k varies by +/-20%.

> **Footnote (2026-05-07):** This formula was revised after issue #667 (Quentin Grimonprez, 2026-04-30) showed it is mathematically constant. The current source of truth is `docs/superpowers/specs/2026-05-07-adengine-refactor-design.md`. The text below is preserved as design history.
```

The rest of the section 4.4 step 3 text stays as-is.

- [ ] **Step 7: Run a sanity grep to confirm no stale "0.82" or "0.817" examples remain in live docs**

```bash
grep -rn "stability\b" docs/examples/ pyod/skills/ | grep -E "0\.8[12]|0\.61|0\.66|0\.71|0\.74|0\.814"
```

Expected: empty output. Any remaining hits are stale and need updating.

- [ ] **Step 8: Stage all updated docs**

```bash
git add docs/examples/adengine.rst docs/examples/agentic.rst \
        pyod/skills/od_expert/SKILL.md pyod/skills/od_expert/references/*.md \
        docs/superpowers/specs/2026-04-12-v3-agentic-design.md
git diff --cached --stat
```

Expected: ~10 files staged.

### Task 1.7: Add CHANGES.txt entry

**Files:**
- Modify: `CHANGES.txt` (top of file)

- [ ] **Step 1: Read the current CHANGES.txt structure**

```bash
head -20 CHANGES.txt
```

Note the version-section format and the latest version header.

- [ ] **Step 2: Add the entry under the next minor version (likely v3.3.0; confirm against current pyod/__init__.py `__version__`)**

```bash
grep -E "^__version__" pyod/__init__.py
```

If the current version is `3.2.1`, the next minor is `3.3.0`. Add a new section to CHANGES.txt above the v3.2.1 entry:

```
v<next>, <date>: -- Fixed: `quality['stability']` now measures cutoff
                    sharpness (standardized gap at the rank-k boundary)
                    instead of a constant. Cascade effect: `quality['overall']`
                    and `quality['verdict']` may shift on the same data
                    because the constant 0.817 no longer dominates the
                    average. Trigger threshold in od_expert/SKILL.md:91
                    (`stability < 0.5`) may need empirical recalibration.
                    Closes #667. Thanks to @Quentin62 for reporting.
```

Format the date and version per the repo's existing style.

- [ ] **Step 3: Stage**

```bash
git add CHANGES.txt
```

### Task 1.8: Final PR 1 commit and push

**Files:** All staged from Tasks 1.2-1.7.

- [ ] **Step 1: Show final diff for user review**

```bash
git diff --cached --stat
git diff --cached
```

Expected: roughly 13 files changed: `pyod/utils/ad_engine.py`, `pyod/test/test_ad_engine_v3.py`, ~10 doc files, `CHANGES.txt`.

- [ ] **Step 2: Pause for user approval before commit**

The orchestrator presents the diff and proposed commit message:

```
fix: ADEngine stability metric is now informative (closes #667)

Replaces the Jaccard-of-nested-top-k formula (mathematically constant
~0.817 for k>=5) with a standardized cutoff gap. The new formula is
local, deterministic, scale-invariant, and pairs cleanly with the
global `separation` and cross-detector `agreement` slots.

Cascade: `quality['overall']` and `quality['verdict']` may shift on
the same data. Trigger threshold in od_expert/SKILL.md:91 may need
empirical recalibration; deferred to a follow-up.

Tests: 11-test TestStabilityMetric class covering responds-to-data,
determinism, k=0/n/1/n-1, ties at boundary, scale invariance,
constant scores, non-finite scores, unit-interval invariant.

Spec: docs/superpowers/specs/2026-05-07-adengine-refactor-design.md

Closes #667
```

User confirms or edits the message.

- [ ] **Step 3: Commit (only after user approval)**

```bash
git commit -m "$(cat <<'EOF'
fix: ADEngine stability metric is now informative (closes #667)

[message body from Step 2]
EOF
)"
git log -1 --oneline
git status
```

- [ ] **Step 4: Push and open PR (only after user approval)**

```bash
git push -u origin fix/adengine-stability-issue-667
gh pr create --title "fix: ADEngine stability metric (closes #667)" \
             --body "$(cat <<'EOF'
## Summary
Closes #667. Replaces the constant `quality['stability']` with a
standardized cutoff gap. See spec for full rationale:
`docs/superpowers/specs/2026-05-07-adengine-refactor-design.md`.

## Test plan
- [x] All existing tests pass
- [x] New `TestStabilityMetric` class (11 tests) covers responds-to-data, determinism, edge cases, scale invariance, non-finite scores
- [x] Doc surfaces updated (12 files)
- [x] CHANGES.txt entry added

## BC
Importable namespace and method names unchanged. Numeric values in
`quality['stability']`, `overall`, `verdict`, `explanation`, and
generated reports DO change because the formula changed. See spec
Section 2 for the full BC posture table.
EOF
)"
```

---

## PR 2: Decomposition

**Branch:** `refactor/adengine-decomposition` from `development` after PR 1 merges.

**Files touched:**
- Create: `pyod/utils/_quality_metrics.py`
- Create: `pyod/utils/_kb_router.py`
- Create: `pyod/utils/_detector_factory.py`
- Create: `pyod/utils/_nl_feedback.py`
- Modify: `pyod/utils/ad_engine.py` (extract helpers, drop private methods, update imports)

**Parallelization:** Tasks 2.1, 2.2, 2.3, 2.4 can run as parallel subagents inside the same worktree. Each creates one new file and modifies different sections of `ad_engine.py`. The orchestrator merges and commits one task at a time after each subagent completes.

### Task 2.1: Create `_quality_metrics.py` and migrate quality helpers

**Files:**
- Create: `pyod/utils/_quality_metrics.py`
- Modify: `pyod/utils/ad_engine.py` (remove `_compute_quality`, `_select_best_detector`, `_compute_feature_importance`, `_feature_contributions`, inline consensus math; add imports and call sites)

- [ ] **Step 1: Create branch from current `development`**

```bash
git checkout development
git pull
git checkout -b refactor/adengine-decomposition
```

- [ ] **Step 2: Create `pyod/utils/_quality_metrics.py` with module docstring and license header matching repo convention**

Copy the file header from any existing `pyod/utils/*.py` file. Body skeleton:

```python
"""Quality metrics for ADEngine result analysis.

Pure helper functions extracted from `pyod.utils.ad_engine.ADEngine` in
2026-05 (issue #667 follow-up). Not part of the public API; the
leading underscore on the module name is the contract.
"""

import numpy as np
from scipy.stats import rankdata, spearmanr


def compute_quality(scores, labels, results, consensus):
    """[paste the docstring from spec Section 3.5]"""
    # [paste the body of the now-fixed _compute_quality, with self->no self]
    ...


def compute_consensus(successful_results):
    """Compute rank-normalized consensus from a list of successful detector results.

    Parameters
    ----------
    successful_results : list of dict
        Each dict has keys 'scores_train' (np.ndarray) and
        'labels_train' (np.ndarray).

    Returns
    -------
    dict
        Keys: 'scores' (np.ndarray, mean of rank-normed scores),
        'labels' (np.ndarray, majority-voted), 'n_detectors' (int),
        'agreement' (float, mean pairwise Spearman), 'disagreements'
        (list of int indices).
    """
    # [paste the consensus block from current ad_engine.py:994-1034]
    ...


def select_best_detector(results, consensus_scores):
    """[paste docstring from current _select_best_detector]"""
    # [paste body, no self]
    ...


def compute_feature_importance(result, X):
    """[paste docstring]"""
    # [paste body of static method]
    ...


def feature_contributions(X, idx, scores):
    """[paste docstring]"""
    # [paste body of static method]
    ...
```

The exact paste-source line numbers in `pyod/utils/ad_engine.py`:
- `_compute_quality` body: lines 1229-1283 (after PR 1 fix)
- `_select_best_detector`: lines 1178-1227
- `_compute_feature_importance`: lines 499-521
- `_feature_contributions`: lines 592-606
- Inline consensus math (becomes `compute_consensus`): lines 994-1034

When pasting, replace `self._make_plan(...)` and other `self.` references — none of these five functions need `self` per the migration table.

- [ ] **Step 3: Update `pyod/utils/ad_engine.py` to import from the new module**

Add at the top of `ad_engine.py` (with other imports):

```python
from pyod.utils._quality_metrics import (
    compute_quality,
    compute_consensus,
    select_best_detector,
    compute_feature_importance,
    feature_contributions,
)
```

- [ ] **Step 4: Replace the call sites in `ADEngine.analyze` and `ADEngine.run`**

In `analyze` (around line 1145), find:

```python
quality = self._compute_quality(scores, labels, results, consensus)
best_idx = self._select_best_detector(results, consensus_scores)
```

Replace with:

```python
quality = compute_quality(scores, labels, results, consensus)
best_idx = select_best_detector(results, consensus_scores)
```

In `run`, replace the inline consensus block (lines 994-1034) with a call to `compute_consensus(successful)`.

In `analyze_results` (around line 470), replace the call to `self._compute_feature_importance(...)` with `compute_feature_importance(...)`.

In `explain_findings` (around line 580), replace `self._feature_contributions(...)` with `feature_contributions(...)`.

- [ ] **Step 5: Delete the now-orphan private methods from `ADEngine`**

Remove from `pyod/utils/ad_engine.py`:
- `_compute_quality` (was lines 1229-1283)
- `_select_best_detector` (was lines 1178-1227)
- `_compute_feature_importance` (was lines 499-521)
- `_feature_contributions` (was lines 592-606)

- [ ] **Step 6: Run the full test suite**

```bash
pytest pyod/test/ -v
```

Expected: all tests pass with **zero modifications** to test files. The `TestStabilityMetric._quality()` adapter automatically switches to the new module path because `from pyod.utils._quality_metrics import compute_quality` now succeeds.

- [ ] **Step 7: Stage and pause for user approval**

```bash
git add pyod/utils/_quality_metrics.py pyod/utils/ad_engine.py
git diff --cached --stat
```

Orchestrator shows diff to user and asks for commit approval.

### Task 2.2: Create `_kb_router.py` and migrate rule engine + suggest_alternative

**Files:**
- Create: `pyod/utils/_kb_router.py`
- Modify: `pyod/utils/ad_engine.py`

(Steps follow the same pattern as Task 2.1.)

- [ ] **Step 1: Create `pyod/utils/_kb_router.py` skeleton**

```python
"""KB rule engine for ADEngine planning.

Extracts the `_evaluate_rules` / `_rule_matches` / `_eval_condition`
chain plus `_make_plan` and `_suggest_alternative` from
`pyod.utils.ad_engine.ADEngine`. Not part of the public API.
"""


def evaluate_rules(profile, priority, kb):
    """[paste docstring]"""
    # [paste from ad_engine _evaluate_rules; replace self.kb -> kb]
    ...


def rule_matches(rule, profile, priority):
    """..."""
    ...


def eval_condition(actual, op, value):
    """..."""
    ...


def make_plan(detector_name, params=None, preset=None, priority='balanced',
              alternatives=None, reason='', confidence=0.0):
    """..."""
    # [paste from ad_engine _make_plan; static method, no changes needed]
    ...


def suggest_alternative(result, kb, make_plan_fn):
    """[paste docstring from _suggest_alternative]"""
    # [paste body; replace self.kb -> kb, self._make_plan -> make_plan_fn]
    ...
```

Source lines in `ad_engine.py`:
- `_evaluate_rules`: 207-231
- `_rule_matches`: 233-250
- `_eval_condition`: 252-267
- `_make_plan`: 269-285
- `_suggest_alternative`: 733-756

- [ ] **Step 2: Update `pyod/utils/ad_engine.py` imports**

```python
from pyod.utils._kb_router import (
    evaluate_rules,
    rule_matches,
    eval_condition,
    make_plan,
    suggest_alternative,
)
```

- [ ] **Step 3: Replace call sites**

- `plan_detection` calls `self._evaluate_rules(profile, priority)` -> `evaluate_rules(profile, priority, self.kb)`.
- Anywhere `self._make_plan(...)` appears (e.g., in `plan_detection`, `_iterate_structured`), replace with `make_plan(...)`.
- `suggest_next_step` calls `self._suggest_alternative(result)` -> `suggest_alternative(result, self.kb, make_plan)`.

- [ ] **Step 4: Delete the orphaned private methods**

Remove `_evaluate_rules`, `_rule_matches`, `_eval_condition`, `_make_plan`, `_suggest_alternative` from `ADEngine`.

- [ ] **Step 5: Run tests**

```bash
pytest pyod/test/ -v
```

Expected: all pass.

- [ ] **Step 6: Stage and request approval**

### Task 2.3: Create `_detector_factory.py` and migrate constructor helpers

**Files:**
- Create: `pyod/utils/_detector_factory.py`
- Modify: `pyod/utils/ad_engine.py` (keep `build_detector` as thin wrapper)

- [ ] **Step 1: Create `pyod/utils/_detector_factory.py`**

```python
"""Detector construction from ADEngine plans.

Extracted from `pyod.utils.ad_engine.ADEngine` in 2026-05.
Not part of the public API.
"""


def build_detector_from_plan(plan, kb):
    """[paste docstring of public build_detector]"""
    # [paste body of public build_detector; replace self.kb -> kb]
    ...


def build_from_preset(detector_name, preset, extra_params):
    """..."""
    # [paste body of static _build_from_preset]
    ...
```

Source lines:
- `build_detector` (PUBLIC): 291-322
- `_build_from_preset`: 325-334

- [ ] **Step 2: Update `pyod/utils/ad_engine.py`**

Add import:

```python
from pyod.utils._detector_factory import build_detector_from_plan
```

Replace the body of `ADEngine.build_detector` (it stays as a public method) with:

```python
def build_detector(self, plan):
    """[exact original docstring, unchanged]"""
    return build_detector_from_plan(plan, self.kb)
```

Delete `_build_from_preset` (only called from inside the now-extracted `build_detector_from_plan`).

- [ ] **Step 3: Run tests; stage; request approval**

### Task 2.4: Create `_nl_feedback.py` and migrate iterate parsers

**Files:**
- Create: `pyod/utils/_nl_feedback.py`
- Modify: `pyod/utils/ad_engine.py`

- [ ] **Step 1: Create `pyod/utils/_nl_feedback.py`**

```python
"""ADEngine `iterate()` feedback parsing.

Extracts `_iterate_structured` and `_iterate_nl` from
`pyod.utils.ad_engine.ADEngine` in 2026-05.
"""

from pyod.utils.investigation import _make_history_entry


def apply_structured_feedback(state, feedback, kb, plan_detection_fn,
                              make_plan_fn):
    """[paste docstring from _iterate_structured]"""
    # [paste body; replace self.kb -> kb, self.plan_detection -> plan_detection_fn,
    #   self._make_plan -> make_plan_fn]
    ...


def apply_nl_feedback(state, feedback, kb, plan_detection_fn,
                      make_plan_fn):
    """[paste docstring from _iterate_nl]"""
    # [paste body; same substitutions]
    ...
```

Source lines:
- `_iterate_structured`: 1310-1391
- `_iterate_nl`: 1393-1454

- [ ] **Step 2: Update `pyod/utils/ad_engine.py`**

Add import:

```python
from pyod.utils._nl_feedback import (
    apply_structured_feedback,
    apply_nl_feedback,
)
```

Update `iterate()`:

```python
def iterate(self, state, feedback):
    """[unchanged docstring]"""
    self._require_phase(state, 'analyzed')
    if isinstance(feedback, dict):
        return apply_structured_feedback(
            state, feedback, self.kb, self.plan_detection, make_plan)
    return apply_nl_feedback(
        state, str(feedback), self.kb, self.plan_detection, make_plan)
```

Delete `_iterate_structured` and `_iterate_nl` from `ADEngine`.

- [ ] **Step 3: Run tests; stage; request approval**

### Task 2.5: PR 2 verification and final commit/push

- [ ] **Step 1: Confirm public API surface**

```bash
python -c "
from pyod.utils.ad_engine import ADEngine
import inspect
public = sorted([m for m in dir(ADEngine) if not m.startswith('_') and callable(getattr(ADEngine, m, None))])
print(f'Count: {len(public)}')
for m in public:
    print(f'  {m}')
"
```

Expected: 20 entries, exactly matching the spec Section 4.8 list.

- [ ] **Step 2: Confirm new module structure**

```bash
ls -la pyod/utils/_*.py
wc -l pyod/utils/ad_engine.py pyod/utils/_*.py
```

Expected: 4 new files, `ad_engine.py` ≈ 1,200 LOC.

- [ ] **Step 3: Run full test suite**

```bash
pytest pyod/test/ -v
```

Expected: all pass.

- [ ] **Step 4: Push and open PR (after user approval per Task 1.8 pattern)**

PR title: `refactor: decompose ADEngine into private helper modules`. PR body references the spec and notes "no behavior changes; pure structural extraction". Body should explicitly state that PR 1 (#667 fix) is the parent.

---

## PR 3: Tech Debt Cleanup

**Branch:** `refactor/adengine-cleanup` from `development` after PR 2 merges.

**Files touched:** All five modules from PR 2 plus tests.

**Parallelization:** Tasks 3.1, 3.2, 3.5, 3.6 are independent. Tasks 3.3 and 3.4 should sequence after the others because they touch the same NL feedback module. Recommended dispatch:
- Wave 1 (parallel): 3.1, 3.2, 3.5, 3.6
- Wave 2 (sequential): 3.3, 3.4

### Task 3.1: Add type hints to all five files

**Files:** `pyod/utils/ad_engine.py`, `_quality_metrics.py`, `_kb_router.py`, `_detector_factory.py`, `_nl_feedback.py`.

- [ ] **Step 1: Add `from __future__ import annotations` at the top of each file**

Below the module docstring, before other imports:

```python
"""Module docstring."""

from __future__ import annotations

import numpy as np
# ... rest of imports
```

- [ ] **Step 2: Define `QualityDict` TypedDict in `_quality_metrics.py`**

```python
from typing import TypedDict


class QualityDict(TypedDict):
    separation: float
    agreement: float
    stability: float
    overall: float
    verdict: str
    explanation: str
```

- [ ] **Step 3: Annotate every function in `_quality_metrics.py`**

Example for `compute_quality`:

```python
from numpy.typing import NDArray

def compute_quality(
    scores: NDArray[np.floating],
    labels: NDArray[np.integer],
    results: list[dict],
    consensus: dict,
) -> QualityDict:
    ...
```

Apply to all 5 functions. Be conservative: do not introduce `Protocol`, `Generic`, or other typing constructs that go beyond plain dict/list/ndarray.

- [ ] **Step 4: Annotate every function in `_kb_router.py`, `_detector_factory.py`, `_nl_feedback.py`**

Same conservative approach. Use `dict`, `list`, `Optional[X]` (or `X | None` under `__future__ annotations`), `bool`, `int`, `float`, `str` as needed.

- [ ] **Step 5: Annotate every public method on `ADEngine`**

20 public methods. Add return-type and parameter annotations. Use `np.ndarray` or `NDArray[np.floating]` for arrays where dtype matters.

- [ ] **Step 6: Run tests; type hints are evaluation-deferred so behavior is unchanged**

```bash
pytest pyod/test/ -v
```

- [ ] **Step 7: Stage**

### Task 3.2: Extract magic numbers to module constants

**Files:** `pyod/utils/_quality_metrics.py`, `pyod/utils/_nl_feedback.py`, `pyod/utils/ad_engine.py`.

- [ ] **Step 1: Add constants block at top of `_quality_metrics.py`**

After the imports:

```python
_VERDICT_HIGH_THRESHOLD: float = 0.7
"""Overall quality at or above this is reported as 'high'."""

_VERDICT_MEDIUM_THRESHOLD: float = 0.4
"""Overall quality at or above this (but below high) is 'medium'."""

_SINGLE_DETECTOR_AGREEMENT_FALLBACK: float = 0.5
"""Agreement returned when only one detector ran."""
```

Update call sites in `compute_quality`:

```python
if overall >= _VERDICT_HIGH_THRESHOLD:
    verdict = 'high'
elif overall >= _VERDICT_MEDIUM_THRESHOLD:
    verdict = 'medium'
else:
    verdict = 'low'
```

- [ ] **Step 2: Add constants block at top of `_nl_feedback.py`**

```python
_CONTAMINATION_INCREASE_FACTOR: float = 1.5
_CONTAMINATION_DECREASE_FACTOR: float = 0.5
_CONTAMINATION_MAX: float = 0.5
_CONTAMINATION_MIN: float = 0.01
_NL_HIGH_CONFIDENCE_THRESHOLD: float = 0.8
"""NL feedback parsed with confidence above this is auto-applied."""
```

Update call sites in `apply_nl_feedback` (the contamination math).

- [ ] **Step 3: Update `suggest_next_step` in `ad_engine.py`**

The three contamination expressions at lines 654, 680, 706 (pre-PR-2 line numbers; find the equivalent after decomposition) become helper calls. The helpers are introduced in Task 3.6.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Stage**

### Task 3.3: Add `validate_structured_feedback`

**Files:** `pyod/utils/_nl_feedback.py`, `pyod/test/test_ad_engine_v3.py`.

- [ ] **Step 1: Add validation function at top of `_nl_feedback.py`**

```python
_VALID_ACTIONS: frozenset[str] = frozenset({
    'adjust_contamination', 'exclude', 'include', 'rerun',
})

_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    'adjust_contamination': frozenset({'value'}),
    'exclude': frozenset({'detectors'}),
    'include': frozenset({'detectors'}),
    'rerun': frozenset(),
}


def validate_structured_feedback(feedback: dict) -> None:
    """Raise ValueError if feedback dict is malformed.

    Validates only structure and required fields. Semantic validation
    (e.g., whether a detector name exists in the KB) is the
    responsibility of `apply_structured_feedback`.

    Parameters
    ----------
    feedback : dict
        Structured feedback per design spec section 4.3.

    Raises
    ------
    ValueError
        If 'action' is missing, action is not in `_VALID_ACTIONS`, or
        required fields for the chosen action are missing.
    """
    if not isinstance(feedback, dict):
        raise ValueError(
            f'feedback must be a dict, got {type(feedback).__name__}')
    action = feedback.get('action')
    if action is None:
        raise ValueError(
            "feedback dict missing 'action' key; "
            f"valid actions: {sorted(_VALID_ACTIONS)}")
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"unknown action {action!r}; "
            f"valid actions: {sorted(_VALID_ACTIONS)}")
    missing = _REQUIRED_FIELDS[action] - feedback.keys()
    if missing:
        raise ValueError(
            f"action {action!r} requires fields {sorted(missing)}, "
            f"got fields {sorted(feedback.keys())}")
```

- [ ] **Step 2: Call `validate_structured_feedback` from `apply_structured_feedback`**

At the top of `apply_structured_feedback`:

```python
def apply_structured_feedback(state, feedback, kb, plan_detection_fn, make_plan_fn):
    validate_structured_feedback(feedback)
    # ... rest of body
```

- [ ] **Step 3: Add tests in `pyod/test/test_ad_engine_v3.py`**

```python
class TestValidateStructuredFeedback:
    def test_missing_action_raises(self):
        from pyod.utils._nl_feedback import validate_structured_feedback
        with pytest.raises(ValueError, match="missing 'action'"):
            validate_structured_feedback({})

    def test_unknown_action_raises(self):
        from pyod.utils._nl_feedback import validate_structured_feedback
        with pytest.raises(ValueError, match="unknown action"):
            validate_structured_feedback({'action': 'frobnicate'})

    def test_missing_required_field_raises(self):
        from pyod.utils._nl_feedback import validate_structured_feedback
        with pytest.raises(ValueError, match="requires fields"):
            validate_structured_feedback({'action': 'adjust_contamination'})

    def test_wrong_type_raises(self):
        from pyod.utils._nl_feedback import validate_structured_feedback
        with pytest.raises(ValueError, match="must be a dict"):
            validate_structured_feedback('rerun')

    def test_valid_input_does_not_raise(self):
        from pyod.utils._nl_feedback import validate_structured_feedback
        validate_structured_feedback({'action': 'rerun'})
        validate_structured_feedback({'action': 'adjust_contamination', 'value': 0.1})
```

- [ ] **Step 4: Run tests**

```bash
pytest pyod/test/test_ad_engine_v3.py::TestValidateStructuredFeedback -v
```

Expected: 5 pass.

- [ ] **Step 5: Run full suite to check no regressions**

```bash
pytest pyod/test/ -v
```

Expected: all pass. If any existing test passes a malformed dict expecting fallthrough, update that test.

- [ ] **Step 6: Stage**

### Task 3.4: Tighten exception handling and add WARNING logs

**Files:** `pyod/utils/_quality_metrics.py`, `pyod/utils/ad_engine.py`.

- [ ] **Step 1: Add module-level logger to each of the five files**

```python
import logging

logger = logging.getLogger(__name__)
```

- [ ] **Step 2: Tighten `compute_feature_importance` and `feature_contributions` in `_quality_metrics.py`**

Replace `except Exception:` with specific catches:

```python
def compute_feature_importance(result, X):
    try:
        # existing body
        ...
    except (AttributeError, ValueError, TypeError) as exc:
        logger.debug('compute_feature_importance: %s', exc)
        return None
```

Same pattern for `feature_contributions`.

- [ ] **Step 3: Add WARNING logs in `ADEngine.run()` and `ADEngine.analyze()`**

Find the per-detector exception handler in `run()`:

```python
except Exception as e:
    results.append({
        'detector_name': plan['detector_name'],
        'status': 'error',
        'error': str(e),
        'plan': plan,
    })
```

Replace with:

```python
except Exception as exc:
    logger.warning(
        'Detector %s raised %s during run(): %s',
        plan['detector_name'], type(exc).__name__, exc)
    results.append({
        'detector_name': plan['detector_name'],
        'status': 'error',
        'error': str(exc),
        'plan': plan,
    })
```

Same pattern for the broad except in `analyze()`.

- [ ] **Step 4: Add a caplog-based test**

```python
class TestExceptionLogging:
    def test_run_logs_warning_on_detector_error(self, caplog):
        # construct a state with one plan that will fail (e.g., bogus detector name)
        # call run, assert WARNING in caplog
        ...
```

- [ ] **Step 5: Run tests; stage**

### Task 3.5: Convert NL parser to declarative pattern table

**Files:** `pyod/utils/_nl_feedback.py`, `pyod/test/test_ad_engine_v3.py`.

- [ ] **Step 1: Replace the body of `apply_nl_feedback` with the new layout**

[Paste the post-PR-3 layout from spec Section 5.6 verbatim, including the `_NLPattern` dataclass, the four builder functions, the `_NL_PATTERNS` table, and `parse_nl_to_structured`.]

- [ ] **Step 2: Add tests for the pattern table**

```python
class TestNLPatternTable:
    def test_exclude_pattern_matches(self):
        from pyod.utils._nl_feedback import parse_nl_to_structured
        # set up a state with a result named 'IForest'
        state = ...
        proposed, conf = parse_nl_to_structured(state, "exclude IForest")
        assert proposed == {'action': 'exclude', 'detectors': ['IForest']}
        assert conf == 0.9

    def test_too_many_pattern_decreases_contamination(self):
        ...

    def test_missed_pattern_increases_contamination(self):
        ...

    def test_rerun_pattern_matches(self):
        ...

    def test_no_match_returns_low_confidence(self):
        ...

    def test_word_boundary_no_match_for_partial(self):
        # "withoutdoubt" should NOT match the without|exclude pattern
        ...
```

- [ ] **Step 3: Run tests; stage**

### Task 3.6: Extract contamination math helpers

**Files:** `pyod/utils/_quality_metrics.py` (or wherever `adjust_contamination_*` lands), `pyod/utils/ad_engine.py` (replace inline math in `suggest_next_step`).

- [ ] **Step 1: Add helper functions**

```python
# pyod/utils/_nl_feedback.py (or _quality_metrics.py per implementer's call)
def adjust_contamination_up(current: float) -> float:
    return min(current * _CONTAMINATION_INCREASE_FACTOR, _CONTAMINATION_MAX)


def adjust_contamination_down(current: float) -> float:
    return max(current * _CONTAMINATION_DECREASE_FACTOR, _CONTAMINATION_MIN)
```

- [ ] **Step 2: Replace the three sites in `suggest_next_step`**

In `pyod/utils/ad_engine.py`, find the three places `min(current * 1.5, 0.5)` and `max(current * 0.5, 0.01)` appear. Replace each with the appropriate helper call.

- [ ] **Step 3: Run tests; stage**

### Task 3.7: Final PR 3 commit and push

- [ ] **Step 1: Show full diff for user review**

- [ ] **Step 2: Pause for user approval**

Proposed commit message:

```
refactor: ADEngine type hints, validation, exception handling, NL parser

Tightens the post-decomposition codebase:
- Type hints across all five files (no mypy enforcement; documentation only)
- Magic numbers extracted to named constants
- validate_structured_feedback raises ValueError on malformed feedback dicts
- WARNING logs on caught detector exceptions in run() and analyze()
- NL parser switches from imperative if/elif chain to declarative pattern table
- Contamination math deduplicated to two helper functions

BC tightenings: malformed iterate(feedback) now raises (was: silent fallthrough);
NL regex moves from substring `in` to `re.search` with word boundaries.

Spec: docs/superpowers/specs/2026-05-07-adengine-refactor-design.md
Plan: docs/superpowers/plans/2026-05-07-adengine-refactor-plan.md
```

- [ ] **Step 3: Push and open PR**

---

## Self-Review Checklist (Plan)

Run this before handing off:

1. **Spec coverage:** Every section of the spec maps to a task here.
   - PR 1 sections (3.1-3.11): Tasks 1.1-1.8 ✓
   - PR 2 sections (4.1-4.9): Tasks 2.1-2.5 ✓
   - PR 3 sections (5.1-5.9): Tasks 3.1-3.7 ✓
2. **Placeholders:** No "TBD", "implement later", "similar to". Code blocks are concrete.
3. **Type consistency:** Function names match across tasks (`compute_quality`, not `_compute_quality` after PR 2).
4. **Approval gates:** Every commit step is gated by user approval per the project commit policy.
5. **Parallelization map:** PR 1 sequential, PR 2 four-way parallel, PR 3 has wave 1 (4 parallel) and wave 2 (2 sequential).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-07-adengine-refactor-plan.md`.

Two execution options:

1. **Subagent-Driven (recommended for parallelization).** Orchestrator dispatches a fresh subagent per task; uses parallel dispatch for PR 2 (Tasks 2.1-2.4) and PR 3 wave 1 (Tasks 3.1, 3.2, 3.5, 3.6); reviews between tasks; commits after user approval.
2. **Inline Execution.** Execute tasks in the current session sequentially, with checkpoints for user review.

Subagent-Driven is the right fit for this plan because the user explicitly asked for parallel agents.
