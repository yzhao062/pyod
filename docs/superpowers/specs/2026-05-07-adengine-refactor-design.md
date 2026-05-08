# ADEngine Stability Fix, Decomposition, and Tech Debt Cleanup

**Date:** 2026-05-07
**Status:** Draft v3 (Rounds 1-2 plan-review by Codex applied; see Section 10 for review history)
**Triggers:**
- Issue [#667](https://github.com/yzhao062/pyod/issues/667) (Quentin Grimonprez, 2026-04-30): the `stability` quality metric is mathematically constant.
- Latent debt accumulated in `pyod/utils/ad_engine.py` over the V3 build-out (2026-04 through 2026-05): 1,650 LOC, 20 public methods + 6 private, no type hints, scattered magic numbers, three methods over 100 LOC, fragile NL parsing, no schema validation on `iterate()` feedback, broad `except Exception` blocks without logging.

**Sequence:** 3 pull requests, in this order:

1. **PR 1**: Stability metric bug fix (closes #667). Targeted in-place fix in the current `pyod/utils/ad_engine.py`.
2. **PR 2**: Structural decomposition. Splits the (now-fixed) ADEngine into one main file plus four private helper modules. Pure refactor.
3. **PR 3**: Tech debt cleanup on the post-decomposition layout. Type hints, named constants, validation, exception tightening, NL parser declarative form, contamination math dedup.

The bug-fix-first ordering closes #667 in the smallest possible PR; PR 2 then moves correct code, not buggy code; PR 3's review is unblocked from any "did the move accidentally fix the bug" sub-questions.

**Backwards compatibility posture** (precise statement, replacing the loose "no public API changes" of draft v1):

| Property | Preserved across PR 1+2+3? |
|----------|---------------------------|
| `from pyod.utils.ad_engine import ADEngine` | Yes |
| Public method names (20 of them) | Yes |
| Public method parameter names and order | Yes |
| `quality` dict keys (separation, agreement, stability, overall, verdict, explanation) | Yes |
| `quality` dict numeric values | **No (PR 1 changes `stability`; cascade affects `overall` and `verdict`)** |
| `quality.explanation` text format | **No (PR 1 rewords)** |
| Generated report text content | **No (PR 1 changes any line that quotes the stability number)** |
| `state.next_action.reason` text | **No (PR 1 — when `iterate()` triggers from low quality, the reason string quotes stability)** |
| `state.history` entry text | **No (PR 1 — history entries on iteration include quality summary)** |
| `state.analysis['summary']` if it embeds quality numbers | **No (PR 1 — if the consolidated summary line includes stability, the value changes)** |
| `__annotations__` and `inspect.signature(...)` on public methods | **No (PR 3 adds type annotations)** |
| `logging` output | **No (PR 3 adds module loggers and emits WARNING on caught detector errors)** |
| Exception type for malformed `iterate(feedback)` dict | **No (PR 3 raises ValueError where current code silently produces `confirm_with_user`)** |
| NL feedback regex acceptance for borderline phrasings | **No (PR 3 switches substring `in` to `re.search` with word boundaries)** |
| Subclass-override surface for private helpers (`_compute_quality`, `_make_plan`, etc.) | **No (PR 2 drops these methods; the underscore-prefix contract permits this)** |

The spec uses "BC posture" for these eleven dimensions explicitly rather than the loose "no public API changes" wording that draft v1 used.

---

## 1. Background

### 1.1 Issue #667: Stability metric is mathematically constant

The `_compute_quality` method in `pyod/utils/ad_engine.py` (lines 1229-1283 at HEAD `a2ed7e5`) returns a `stability` value that does not depend on the model or data, only on `k` (the number of anomalies). The bug:

```python
# Current implementation (lines 1243-1264, paraphrased)
k = n_anomalies
k_low = max(1, int(k * 0.8))
k_high = min(n_samples, int(k * 1.2))
sorted_idx = np.argsort(scores)[::-1]
top_k = set(sorted_idx[:k].tolist())
top_low = set(sorted_idx[:k_low].tolist())     # nested in top_k
top_high = set(sorted_idx[:k_high].tolist())   # contains top_k
stability = 0.5 * (jaccard(top_k, top_low) + jaccard(top_k, top_high))
```

All three sets are slices of the same `sorted_idx` array, so by construction `top_low ⊂ top_k ⊂ top_high`. Plugging into Jaccard:

- `jaccard(top_k, top_low) = |top_low| / |top_k| = int(0.8 k) / k ≈ 0.8`
- `jaccard(top_k, top_high) = |top_k| / |top_high| = k / int(1.2 k) ≈ 0.833`

For any `k >= 5`, `stability ≈ 0.5 * (0.8 + 0.833) = 0.817`. The metric is constant. It carries no information about model behavior or data structure.

The issue is in the **specification**, not in the code's faithfulness to it. `docs/superpowers/specs/2026-04-12-v3-agentic-design.md` line 524-532 defined stability with this exact formula. The Codex review in section 9 (Round 1) of that earlier spec flagged the original phrasing as "vague" and asked for an exact formula; the response was to make it precise, but the precise formula was already mathematically degenerate.

### 1.2 Original design intent of the `stability` slot

The `quality` dict has three independent diagnostic slots, each driving one branch of `iterate()`:

| Slot | Diagnoses | Drives `iterate()` action |
|------|-----------|---------------------------|
| `separation` | Anomaly score mean vs inlier score mean (global) | "no signal, switch detector or features" |
| `agreement` | Spearman rank correlation across base detectors | "detectors disagree, exclude or replace" |
| `stability` | Whether the rank-k cutoff is supported by the data (local) | "k may be wrong, `adjust_contamination`" |

Without a stability slot, the `adjust_contamination` action in `StructuredFeedback` loses its automatic trigger condition. The slot is load-bearing for the V3 session control flow.

The fix preserves the slot and replaces the formula with one that actually measures what the original spec intended ("anomaly set is robust to contamination threshold").

### 1.3 Latent debt in `pyod/utils/ad_engine.py`

Beyond the bug, the file has the following issues:

| Issue | Location (line numbers at HEAD `a2ed7e5`) |
|-------|-------------------------------------------|
| Method `analyze` is 129 LOC | 1048-1176 |
| Method `run` is 115 LOC, with inline consensus math | 931-1046 |
| Method `_iterate_nl` is 61 LOC of imperative `if`/`elif` keyword matching | 1393-1454 |
| Method `suggest_next_step` is 122 LOC | 612-731 |
| Magic numbers: `0.7`, `0.4` (verdict thresholds) hardcoded | 1267-1271 |
| Magic numbers: `1.5`, `0.5` (contamination adjustment factors) duplicated in three places | 654, 680, 706 |
| No type hints on any method or function | entire file |
| Bare `except Exception` without logging or re-raise | 520, 605, 959, 1096 |
| `iterate(feedback=…)` accepts arbitrary dict shapes; unknown action falls through silently | 1310-1391 |
| Public method docstrings vary in completeness; some omit Returns section | scattered |
| `consensus` computation lives inline in `run()`, not reusable for tests | 994-1034 |

Each of these is small in isolation. Together they make the file hard to read, hard to test in isolation, and hard to extend.

### 1.4 User constraints

- Three PRs maximum.
- Importable namespace and method names preserved across the sequence; specific BC posture enumerated in the eleven-row table at the top.
- Each PR is independently reviewable and bisect-friendly.
- Bug fix ships first (PR 1) so issue #667 closes in the smallest possible diff.

---

## 2. PR Sequence Summary

| PR | Title | Behavior changes | Reviewable as |
|----|-------|------------------|---------------|
| 1 | `fix: ADEngine stability metric is now informative (closes #667)` | `quality['stability']` formula change with cascade effects on `overall`/`verdict`/`explanation`/report text | "Does the new formula match the design rationale, and does the test prove it responds to data?" |
| 2 | `refactor: decompose ADEngine into private helper modules` | None | "Does the test suite still pass exactly? Is the public surface byte-identical?" |
| 3 | `refactor: ADEngine type hints, validation, exception handling, NL parser` | Tightening only: (a) ValueError on malformed feedback dict; (b) WARNING logs on caught detector errors; (c) NL regex switches from substring `in` to `re.search` with word boundaries (a few borderline phrasings stop matching). | "Is each helper module production-ready and self-contained?" |

Bug fix lands in `pyod/utils/ad_engine.py` directly. Decomposition then moves the corrected code into helper modules. Tech debt cleanup operates on the post-decomposition layout.

---

## 3. PR 1: Stability Bug Fix

### 3.1 Goal

Replace the body of `_compute_quality` in `pyod/utils/ad_engine.py` (lines 1243-1264 at HEAD) with a formula that actually measures cutoff sharpness, and update all human-readable text that describes what `stability` means.

### 3.2 Root cause restated

The current Jaccard-of-nested-top-k formula always reduces to a constant ≈0.817 for `k >= 5` because the three sets are slices of the same sorted list and therefore nest by construction. The original spec used the wrong primitive: Jaccard on **nested** sets has no information content. To capture sensitivity, the comparison must involve some perturbation that breaks the nesting (added noise, resampled data) or it must use a different primitive altogether (a gap measure, a slope, a quantile).

### 3.3 New formula: standardized cutoff gap

```python
# pyod/utils/ad_engine.py, replacing lines 1243-1264 of the current _compute_quality
n_anomalies = int(labels.sum())
n_samples = len(scores)
if n_anomalies == 0 or n_anomalies >= n_samples:
    stability = 0.0
elif not np.all(np.isfinite(scores)):
    stability = 0.0  # non-finite scores: undefined gap; refuse to emit NaN
else:
    sorted_scores = np.sort(scores)[::-1]  # descending
    gap = float(sorted_scores[n_anomalies - 1] - sorted_scores[n_anomalies])
    std = float(scores.std())
    if std == 0.0:
        stability = 0.0  # constant scores: no meaningful gap to standardize
    else:
        stability = float(np.clip(gap / std, 0.0, 1.0))
```

`sorted_scores[n_anomalies - 1]` is the lowest-scored sample still in the anomaly set. `sorted_scores[n_anomalies]` is the highest-scored sample not in the anomaly set. Their difference is the "gap at the cutoff". Normalizing by `scores.std()` makes the result comparable across datasets with different score scales. Clipping to `[0, 1]` keeps the metric in the same range as `separation` and `agreement`, so `overall = mean(separation, agreement, stability)` remains in `[0, 1]`.

The explicit `if std == 0.0` check replaces draft v1's `+ _NUMERIC_EPSILON` denominator floor. The epsilon approach was not scale invariant: with `scores = [2e-12, 1e-12, 0]`, the epsilon dominated the actual std, producing `stability ≈ 0.01`, while the same rank geometry at scale `[2e-6, 1e-6, 0]` produced `stability = 1.0`. Verified by reproduction during plan-review (Round 1, Finding 3). The explicit zero check eliminates the scale dependency: any nonzero std uses the true gap-to-std ratio; std exactly zero is the only degenerate case.

The `np.isfinite(scores)` guard handles non-finite inputs (NaN, Inf). Verified during plan-review (Round 2, Finding 2): `[np.nan, 1.0, 0.0]` and `[np.inf, 1.0, 0.0]` both produced `stability = nan` under v2, breaking the docstring promise that stability is in `[0, 1]`. NaN propagated into `overall`, `verdict`, summaries, and `next_action`. The guard returns 0 for any non-finite score vector. NaN/Inf scores indicate a detector bug upstream; PR 3's WARNING-log policy will surface those (Section 5.4); PR 1 keeps the quality function side-effect-free and just refuses to emit NaN.

### 3.4 Why this formula

- **Local.** The gap is a property of the specific cutoff at rank `n_anomalies`. It directly answers "if the contamination value were slightly different, would the anomaly set shift?".
- **Deterministic.** No random sampling, no seed dependency. Same input gives same output.
- **Cheap.** `np.sort` is already done once for `argsort` upstream. The marginal cost is a single subtraction and a `std` call.
- **Independent from the other two slots.** `separation` is global (mean-vs-mean), `agreement` is cross-detector (Spearman across ranking). `stability` is a local geometric property of one detector's score distribution at one cutoff. Three orthogonal views.
- **Interpretable.** A user reading `stability=0.05` understands "the rank-k and rank-(k+1) samples differ by 0.05 standard deviations of the score distribution; small change in k flips one sample".
- **Scale invariant.** Any nonzero positive scale factor on `scores` leaves `stability` unchanged because both `gap` and `std` scale linearly. Constant scores are the only degenerate case and return 0.

### 3.5 Docstring update

The `_compute_quality` docstring is rewritten:

```python
def _compute_quality(self, scores, labels, results, consensus):
    """Compute three diagnostic quality metrics for a detection run.

    Each metric diagnoses one independent failure mode and drives one
    branch of `ADEngine.iterate()`:

    - ``separation``: anomaly-vs-inlier mean score gap (global).
      Low value indicates the detector did not produce a usable signal.
    - ``agreement``: pairwise Spearman rank correlation across base
      detectors (cross-detector). Low value indicates detectors disagree
      on which samples are anomalous.
    - ``stability``: standardized score gap at the rank-k cutoff (local).
      Computed as ``(score[rank_k] - score[rank_k+1]) / scores.std()``,
      clipped to ``[0, 1]``. Low value indicates many tied scores near
      the threshold; the anomaly set is sensitive to the contamination
      value, and ``adjust_contamination`` is the suggested action.

    The ``stability`` key was historically defined (and mis-implemented)
    as the Jaccard index of nested top-k slices, which collapses to a
    constant. The formula was revised in pyod v3.3 (closes #667). The
    key name is retained for backwards compatibility with v3.2.x callers.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples,)
        Consensus or per-detector anomaly scores. Higher means more anomalous.
    labels : np.ndarray, shape (n_samples,)
        Binary labels (0 inlier, 1 anomaly).
    results : list of dict
        Per-detector results from `ADEngine.run()`. Reserved for callers
        that thread per-detector data through quality computation. Not
        directly read by this method.
    consensus : dict
        Consensus dict from `ADEngine.run()`. Provides the ``agreement`` field.

    Returns
    -------
    dict
        Keys: ``separation`` (float in [0, 1]), ``agreement`` (float in
        [0, 1]), ``stability`` (float in [0, 1]), ``overall`` (float in
        [0, 1], mean of the three), ``verdict`` (one of ``'high'``,
        ``'medium'``, ``'low'``), ``explanation`` (human-readable summary).
    """
```

### 3.6 Explanation string update

The current `explanation` field reads:

```python
'Separation=%.2f, agreement=%.2f, stability=%.2f.' % (...)
```

Updated to flag the new semantic for human readers:

```python
'separation={:.2f}, agreement={:.2f}, stability={:.2f} (cutoff gap)'.format(...)
```

The parenthetical `(cutoff gap)` lets a reader who saw an old report ("stability=0.82") understand that the new report's stability is measured differently.

### 3.7 Key name preserved

The dictionary key `'stability'` is unchanged. Renaming to `'threshold_clarity'` or `'cutoff_sharpness'` would be a hard break for any v3.2.x user accessing `result.quality['stability']`. The name is kept; the docstring and explanation string carry the new meaning.

### 3.8 Doc surface updates

The new formula is the source of truth. Every place that documents the old formula or shows hardcoded stability values is updated. Verified by `grep -rn 'stability\b' docs/ pyod/skills/` during plan-review.

| File | Lines | Action |
|------|-------|--------|
| `pyod/utils/ad_engine.py` | 1229-1283 | Replace `_compute_quality` body and docstring per 3.3-3.5 |
| `docs/examples/adengine.rst` | 75 | Replace `# 0.82` example value with a value matching the new formula on the documented input, or annotate as "illustrative; actual value depends on data" |
| `docs/examples/agentic.rst` | 149 | Same treatment for `# 0.814` |
| `docs/examples/agentic.rst` | 244 | Key list mention; no number, no change needed |
| `pyod/skills/od_expert/SKILL.md` | 91 | Trigger condition `state.quality.stability < 0.5` may fire more often with the new formula. Section 5.5 of the design spec (`docs/superpowers/specs/2026-04-12-v3-agentic-design.md`) needs to be revisited. **Recalibration deferred to a follow-up issue** if the threshold proves wrong empirically; PR 1 keeps the threshold as-is and notes the risk in CHANGES.txt. |
| `pyod/skills/od_expert/references/graph.md` | 107 | Update `0.61` example value or annotate as illustrative |
| `pyod/skills/od_expert/references/tabular.md` | 100 | Update `0.66` example value or annotate as illustrative |
| `pyod/skills/od_expert/references/text_image.md` | 100, 109 | Update `0.71` example value (two mentions) or annotate as illustrative |
| `pyod/skills/od_expert/references/time_series.md` | 94, 107 | Update `0.74` example value (two mentions) or annotate as illustrative |
| `pyod/skills/od_expert/references/workflow.md` | 60, 88, 152, 160, 174 | Multiple mentions of stability values (0.82) and stability-based decision text. Replace numeric values, keep the decision-text patterns (which describe how an agent reasons about low stability). |
| `pyod/skills/od_expert/references/pitfalls.md` | 83 | Sentence "On n < 200, stability is naturally low because resampling has high variance." This was true under the old (buggy) formula in a hand-wavy sense; under the new formula it is also true but for a different reason (small n means rank-k boundary is more sensitive to single-sample noise). Update sentence to reflect new semantic. |
| `docs/superpowers/specs/2026-04-12-v3-agentic-design.md` | 524-532 | Rewrite section 4.4 step 3 (label stability) with the new formula. Add footnote: "Original v1 formulation in this section was Jaccard of nested top-k, which is mathematically constant; corrected in 2026-05 (issue #667)." |
| `examples/agentic_example.py` | (any output that prints `quality['stability']`) | Regenerate example output if the README or notebook embeds the printed values |
| `CHANGES.txt` | top | Add v3.3.0 entry: "Fixed: `quality['stability']` now measures cutoff sharpness (standardized gap at the rank-k boundary) instead of a constant. Cascade effect: `quality['overall']` and `quality['verdict']` may shift on the same data because the constant 0.817 no longer dominates the average. Closes #667. Thanks to @Quentin62 for reporting." |

The doc surface list covers **live** docs and skill surfaces. Plan-review Round 2 (Finding 3) flagged that historical plan files in `docs/superpowers/plans/` (e.g., `2026-04-12-v3-agentic-implementation.md:817`, `2026-04-13-v3.2-od-expert-deep-skill.md:1843, 1851, 2021`) still contain the old Jaccard formula and old hardcoded values. Those are **intentionally not updated**: plan files are point-in-time design artifacts and rewriting them would lose the historical record of how the design evolved. The 2026-05-07 spec (this document) is the current source of truth; the older plans are correctly superseded by it. PR 1 does not touch `docs/superpowers/plans/`.

### 3.9 Test plan

A new test class is added to `pyod/test/test_ad_engine_v3.py`:

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
            assert np.isfinite(q['overall'])  # cascade safety

    def test_stability_k_equals_one(self):
        """k=1 is well-defined: gap between highest score and second-highest."""
        scores = np.array([5.0, 3.0, 2.0, 1.0, 0.5])
        labels = np.array([1, 0, 0, 0, 0])
        q = self._quality(scores, labels)
        # gap = 5 - 3 = 2; std = scores.std()
        expected = min(1.0, (5.0 - 3.0) / float(scores.std()))
        assert abs(q['stability'] - expected) < 1e-9

    def test_stability_k_equals_n_minus_one(self):
        """k=n-1 is well-defined: only one inlier."""
        scores = np.array([5.0, 4.0, 3.0, 2.0, 0.0])
        labels = np.array([1, 1, 1, 1, 0])
        q = self._quality(scores, labels)
        # gap = scores[n-2 sorted] - scores[n-1 sorted] = 2 - 0 = 2
        expected = min(1.0, (2.0 - 0.0) / float(scores.std()))
        assert abs(q['stability'] - expected) < 1e-9

    def test_stability_ties_at_boundary(self):
        """Tied scores at the cutoff produce gap=0 -> stability=0."""
        scores = np.array([5.0, 4.0, 3.0, 3.0, 1.0])  # rank 3 and rank 4 tied at 3.0
        labels = np.array([1, 1, 1, 0, 0])  # k=3
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

The existing weak guard `0 <= q['stability'] <= 1` in `test_quality_metrics` continues to pass. The new tests add semantic guards against the bug returning.

### 3.10 Verification criteria for PR 1

1. New test class `TestStabilityMetric` is added (10 tests) and passes.
2. Existing test suite passes with no modification.
3. `_compute_quality` body matches Section 3.3.
4. Docstring matches Section 3.5.
5. Explanation string format matches Section 3.6.
6. All doc surfaces in the Section 3.8 table are updated.
7. CHANGES.txt entry references issue #667 by number.
8. PR description on GitHub uses `Closes #667` footer.

### 3.11 Risk register for PR 1

| Risk | Severity | Mitigation |
|------|----------|------------|
| New formula gives stability values that change `verdict` from `high` to `medium` (or vice versa) on data calibrated against the constant 0.817 | Medium | Verdict thresholds (`>= 0.7` high, `>= 0.4` medium) were chosen for an `overall` metric that included a constant 0.817 in its average. The new stability can be lower (smaller gap), which lowers `overall` and may shift verdicts. Document in CHANGES.txt as expected. The trigger condition in `od_expert/SKILL.md:91` (`stability < 0.5`) may also fire more often. Recalibration of the trigger threshold is deferred to a follow-up if proves wrong empirically. |
| Numeric edge case: `scores.std() = 0` (constant score vector) | Low | Explicit `if std == 0.0: stability = 0.0` branch, exercised by `test_stability_zero_for_constant_scores`. |
| Numeric edge case: `n_anomalies == n_samples` (all flagged) | Low | Explicit `if n_anomalies == 0 or n_anomalies >= n_samples` branch, exercised by `test_stability_zero_for_all_anomalies`. |
| Tiny-scale scores (`std ~ 1e-12`) producing nonsense values via epsilon dominance | Resolved in 3.3 | Plan-review Round 1 surfaced this. The `+ epsilon` denominator floor of draft v1 was replaced with explicit `if std == 0.0` check. Verified scale invariant for any nonzero std; zero std returns 0. Test: `test_stability_scale_invariant`. |
| Test fixture in draft v1 `test_stability_responds_to_data` did not actually produce a low-stability case | Resolved in 3.9 | Plan-review Round 1, Finding 1. Replaced fixture with constant 10.0/0.0 (clean) and `linspace(2.0, 1.01) + boundary_inlier(1.009) + linspace(0.0, -1.0)` (borderline) per Codex's rewrite. Reproduction confirmed `stability_clean = 1.0`, `stability_blurry < 0.01`. |

---

## 4. PR 2: Decomposition

### 4.1 Goal

Split `pyod/utils/ad_engine.py` into one main file (the `ADEngine` class) and four private helper modules. Behavior must be byte-identical to PR 1 output. The diff reads as "code moved", not "code rewritten".

### 4.2 Scope honesty: this is helper extraction, not full decomposition

Plan-review Round 1, Finding 2 surfaced that draft v1 over-claimed the LOC reduction. Helpers eligible for move (counted in the migration table below) total approximately 460 LOC. After the move, `pyod/utils/ad_engine.py` lands at approximately **1,200 LOC**, not the 600-700 LOC of draft v1. The reduction is 27%, which is meaningful but does not turn `ad_engine.py` into a thin facade.

A full decomposition (target 600-700 LOC) would require moving Tier A vs V3 session method overlap into a shared internal layer (`analyze_results` vs `analyze`, `generate_report` vs `report`). That work is **out of scope** here (Section 6) because it touches the public API surface and requires its own design pass.

The honest framing: PR 2 is helper extraction. It pays down the most expensive debt (14 LOC of buggy quality math, 60 LOC of inline consensus, 60+60 LOC of NL parsing, 50 LOC of rule engine) by giving each its own private module with single responsibility. It does not change the size category of `ad_engine.py`.

### 4.3 New file layout

```
pyod/utils/
├── ad_engine.py            # ADEngine class + dispatch + Tier A bodies     ~1,200 LOC
├── _quality_metrics.py     # quality math, consensus, best detector,
│                           #   feature importance, feature contributions   ~300 LOC
├── _kb_router.py           # KB rule engine + suggest_alternative          ~250 LOC
├── _detector_factory.py    # detector construction from plans              ~80 LOC
└── _nl_feedback.py         # iterate() feedback parsers                    ~200 LOC
```

The leading underscore on the four new module names follows existing PyOD convention for non-public modules. They will not appear in `pyod.utils.__all__` and will not be documented in the Sphinx API reference. External users continue to import only `from pyod.utils.ad_engine import ADEngine`.

### 4.4 Migration table

Each row is a method that moves out of `ADEngine`. The "Signature change" column shows the mechanical adjustment to make the method work as a module-level function.

| Source method (`ADEngine.<name>`) | Approx LOC | Target | New signature | Signature change rationale |
|----------------------------------|-----------:|--------|---------------|----------------------------|
| `_compute_quality(self, scores, labels, results, consensus)` (post PR 1) | 50 | `_quality_metrics.compute_quality` | `compute_quality(scores, labels, results, consensus)` | No `self` access in body; pure function. |
| `_select_best_detector(self, results, consensus_scores)` | 50 | `_quality_metrics.select_best_detector` | `select_best_detector(results, consensus_scores)` | No `self` access in body; pure function. |
| `_compute_feature_importance(result, X)` (already `@staticmethod`) | 25 | `_quality_metrics.compute_feature_importance` | `compute_feature_importance(result, X)` | Already static. Move as-is. |
| `_feature_contributions(X, idx, scores)` (already `@staticmethod`) | 15 | `_quality_metrics.feature_contributions` | `feature_contributions(X, idx, scores)` | Already static. Move as-is. |
| Inline consensus math in `run()` (lines 994-1034) | 40 | `_quality_metrics.compute_consensus` | `compute_consensus(successful_results) -> dict` | Extract the rank-norm + majority-vote + Spearman block into a helper that takes the list of successful detector results and returns the consensus dict. |
| `_evaluate_rules(self, profile, priority)` | 25 | `_kb_router.evaluate_rules` | `evaluate_rules(profile, priority, kb)` | `self.kb` becomes `kb` parameter. |
| `_rule_matches(self, rule, profile, priority)` | 20 | `_kb_router.rule_matches` | `rule_matches(rule, profile, priority)` | No `self` access. |
| `_eval_condition(actual, op, value)` (already `@staticmethod`) | 17 | `_kb_router.eval_condition` | `eval_condition(actual, op, value)` | Move as-is. |
| `_make_plan(detector_name, params, preset, priority, alternatives)` (already `@staticmethod`) | 20 | `_kb_router.make_plan` | `make_plan(...)` | Move as-is. |
| `_suggest_alternative(self, result)` | 24 | `_kb_router.suggest_alternative` | `suggest_alternative(result, kb, make_plan_fn)` | `self.kb` and `self._make_plan` become explicit parameters; same dependency profile as `_evaluate_rules`. **Added in v2 of this spec per Round 1 Finding 2.** |
| `build_detector(self, plan)` **(public)** | 30 | `_detector_factory.build_detector_from_plan` | `build_detector_from_plan(plan, kb)` | `self.kb` becomes `kb` parameter. **Public method on `ADEngine` is preserved as a thin wrapper** (see 4.5). |
| `_build_from_preset(detector_name, preset, extra_params)` (already `@staticmethod`) | 10 | `_detector_factory.build_from_preset` | `build_from_preset(...)` | Move as-is. |
| `_iterate_structured(self, state, feedback)` | 80 | `_nl_feedback.apply_structured_feedback` | `apply_structured_feedback(state, feedback, kb, plan_detection_fn, make_plan_fn)` | `self.kb`, `self.plan_detection`, `self._make_plan` become explicit parameters. |
| `_iterate_nl(self, state, feedback)` | 60 | `_nl_feedback.apply_nl_feedback` | `apply_nl_feedback(state, feedback, kb, plan_detection_fn, make_plan_fn)` | Same as above; the function delegates to `apply_structured_feedback` for high-confidence parses. |

Total moved: approximately 466 LOC.

`_require_phase`, `_sniff_data_type`, `_looks_like_image_paths` stay on `ADEngine`:

- `_require_phase` checks `state.phase` and is tightly coupled to the V3 session lifecycle on the class. Moving it would mean every session method imports a one-line guard from a helper module.
- `_sniff_data_type` is a 19-line classifier (PyG Data → graph, dict → multimodal, list of strings → text or image) called only from `profile_data`. It does **not** read `self.kb` (verified by source inspection during plan-review Round 2, Finding 4); the only `self` access is `self._looks_like_image_paths(...)`.
- `_looks_like_image_paths` is an 8-line static method called only by `_sniff_data_type`.

The combined cluster (data-type sniffing) is 27 LOC and used only inside `profile_data`. Moving them into one of the four new modules would stretch that module's scope; a fifth helper module for two short functions would be over-modularization. The honest reason to keep them on the class is "small, single call site, no clear receiving module" rather than the v2 wording that incorrectly cited a `self.kb` dependency.

### 4.5 Public API preservation

`ADEngine.build_detector(self, plan)` is the only public method whose body moves. It is preserved on the class as a thin wrapper:

```python
# pyod/utils/ad_engine.py (after PR 2)
def build_detector(self, plan):
    """[exact original docstring, unchanged]"""
    from pyod.utils._detector_factory import build_detector_from_plan
    return build_detector_from_plan(plan, self.kb)
```

The other 20 public methods stay on `ADEngine` and call the new helper functions internally. Example for `analyze`:

```python
# Before PR 2 (after PR 1)
quality = self._compute_quality(scores, labels, results, consensus)
best_idx = self._select_best_detector(results, consensus_scores)

# After PR 2
from pyod.utils._quality_metrics import compute_quality, select_best_detector
quality = compute_quality(scores, labels, results, consensus)
best_idx = select_best_detector(results, consensus_scores)
```

The import lives at the top of the file (not inside the method), once per module.

### 4.6 Private method handling

Private helpers that move (those beginning with `_`) are dropped from `ADEngine` entirely. No thin wrappers are kept on the class for `_compute_quality`, `_evaluate_rules`, etc. The leading underscore is the contract: external callers should not be importing or overriding these. Codebase survey confirmed no external usage; the only callers are inside `ADEngine`.

This is a BC posture change for code that monkey-patches or subclasses ADEngine to override private helpers. The eleven-row table at the top of this spec lists this as a known surface change. It is consistent with PyOD's existing convention.

### 4.7 Docstring polish for moved code

When a method moves to a new module, its docstring is reviewed and updated for the new location: the leading line summarizes what the function does (no `self` reference), the Parameters and Returns sections match the new signature, and existing weak docstrings are tightened where the move makes the function more visible. This polish is bundled into PR 2 because it lands naturally with the move; doing it in PR 3 would mean re-touching the same files.

The polish is bounded:
- Every moved function gets a `Parameters` and `Returns` section in NumPy style.
- Every moved function's leading line is one short sentence.
- No NEW docstrings on functions that did not have them; existing weak docstrings are kept unless trivially fixable.
- Docstring polish for non-moved methods on `ADEngine` is **not** part of PR 2; that lands in PR 3 if at all.

### 4.8 Verification criteria for PR 2

PR 2 is accepted if all of the following hold:

1. The full test suite (`pytest pyod/test/`) passes with **zero modifications to any test file**.
2. `git diff master..HEAD pyod/utils/ad_engine.py` shows a net deletion of approximately 460 LOC; final file size ≈ 1,200 LOC.
3. `git diff master..HEAD --stat` shows four new files under `pyod/utils/` with `_` prefix.
4. The public surface, verified by `inspect.getmembers(ADEngine, callable)` for non-underscore names, matches the v3.2.x reference list of 20 entries: analyze, analyze_results, build_detector, compare_detectors, detect, explain_detector, explain_findings, generate_report, get_benchmarks, investigate, iterate, list_detectors, plan, plan_detection, profile_data, report, run, run_detection, start, suggest_next_step.
5. `pyod/skills/od_expert/SKILL.md`, the design specs, and the examples are unchanged. (PR 2 does not touch documentation.)
6. Each moved function has a `Parameters` and `Returns` section per Section 4.7.

### 4.9 Risk register for PR 2

| Risk | Severity | Mitigation |
|------|----------|------------|
| Helper that took `self` accesses something other than `self.kb` (caught late) | Medium | Audit each candidate before moving: grep the body for `self.` and list every attribute it reads. The migration table in 4.4 shows the audit result; only `self.kb`, `self.plan_detection`, and `self._make_plan` turn up. `_suggest_alternative` was added in v2 because Round 1 Finding 2 caught it as having the same profile. |
| Circular import between `ad_engine.py` and a helper module | Low | Helper modules import nothing from `ad_engine.py`. ADEngine imports from helpers. One-way dependency. |
| Moving a method changes `__qualname__`, breaking `isinstance` or pickle | Low | No code in PyOD or in the test suite relies on `__qualname__` of these methods. ADEngine state is small (`self.kb`, `self.knowledge_dir`); pickling ADEngine is not part of any test. |
| Removed private methods break user code that monkey-patches `ADEngine._compute_quality` etc. | Low (BC posture) | Underscore-prefix is the contract: such code was relying on internals at its own risk. Documented in CHANGES.txt; no compat shim. |

---

## 5. PR 3: Tech Debt Cleanup (Tightened)

### 5.1 Goal

After PR 1's fix and PR 2's split, finish the remaining cleanup: type hints, named constants, exception handling, validation, NL parser declarative form, contamination math dedup. Each item is small in isolation; bundling them after the structural base is in place keeps the diffs scoped to single modules.

Round 1 plan-review (Finding 6) flagged that draft v1's PR 3 was too broad. v2 narrows it: docstring polish moved into PR 2 (so it lands with the move); `_types.py` introduction dropped (TypedDicts go inline); `mypy --strict` enforcement gate dropped (type hints land without an enforcement requirement). Six items remain.

### 5.2 Type hints

Scope: every public and private function in all five files (`ad_engine.py`, `_quality_metrics.py`, `_kb_router.py`, `_detector_factory.py`, `_nl_feedback.py`).

Style:

- Add `from __future__ import annotations` at the top of each file. This enables PEP 604 union syntax (`X | None`) on Python 3.9 by deferring evaluation, and lets us forward-reference types without quoting.
- Use built-in generics (`list[int]`, `dict[str, float]`) directly. With `from __future__ import annotations` they are evaluation-deferred and safe.
- Use `numpy.typing.NDArray[np.floating]` for score arrays where dtype is meaningful; `np.ndarray` for label arrays where dtype is bool/int. Be specific where it helps the reader.
- TypedDict definitions for `quality`, `consensus`, and `analysis` dict shapes go inline at the top of the relevant helper module (e.g., `QualityDict` in `_quality_metrics.py`). Cross-module imports use `from pyod.utils._quality_metrics import QualityDict` where needed. **No separate `_types.py` module.** Round 1 Finding 6 surfaced that introducing `_types.py` adds a sixth file, contradicting the verification criteria scope.

Example signature, for `compute_quality` (after PR 2):

```python
from __future__ import annotations
from typing import TypedDict
import numpy as np
from numpy.typing import NDArray

class QualityDict(TypedDict):
    separation: float
    agreement: float
    stability: float
    overall: float
    verdict: str
    explanation: str

def compute_quality(
    scores: NDArray[np.floating],
    labels: NDArray[np.integer],
    results: list[dict],
    consensus: dict,
) -> QualityDict:
    ...
```

Public method signatures on `ADEngine` get the same treatment. `Optional[dict]` (or the equivalent `dict | None` under future annotations) is used for parameters that default to `None`.

**No `mypy --strict` enforcement gate.** Type hints land in v3.3 as documentation and IDE help; mypy enforcement is deferred to a future effort because PyOD does not currently use mypy and adding strict checks across the project is a separate scope question.

**Conservative annotations only.** Without enforcement, type hints are documentation. PR 3 should resist type-driven rewrites (e.g., introducing `Protocol` classes or `Generic[T]` parametrization where a plain `dict` annotation suffices, or rewriting function bodies to satisfy a stricter type than the runtime requires). The goal is reader help, not type system completeness. Plan-review Round 2 scope position (b) raised this; we adopt the cautious stance.

### 5.3 Magic number extraction

A new section at the top of each module gathers numeric constants:

```python
# pyod/utils/_quality_metrics.py
_VERDICT_HIGH_THRESHOLD: float = 0.7
"""Overall quality at or above this is reported as 'high'."""

_VERDICT_MEDIUM_THRESHOLD: float = 0.4
"""Overall quality at or above this (but below high) is 'medium'."""

_SINGLE_DETECTOR_AGREEMENT_FALLBACK: float = 0.5
"""Agreement returned when only one detector ran (no basis for agreement)."""
```

```python
# pyod/utils/_nl_feedback.py
_CONTAMINATION_INCREASE_FACTOR: float = 1.5
_CONTAMINATION_DECREASE_FACTOR: float = 0.5
_CONTAMINATION_MAX: float = 0.5
_CONTAMINATION_MIN: float = 0.01
_NL_HIGH_CONFIDENCE_THRESHOLD: float = 0.8
"""NL feedback parsed with confidence above this is auto-applied."""
```

All call sites are updated to use the constants. Underscored at module level: not part of any public API.

### 5.4 Exception handling

Three sites in the original code use `except Exception` without logging. After PR 2 they live in `_quality_metrics.py` and `ad_engine.py`. PR 3 tightens each:

| Site | Original | New |
|------|----------|-----|
| `_quality_metrics.compute_feature_importance` | `except Exception: return None` | Catch only `(AttributeError, ValueError, TypeError)`. Log via `logger.debug(...)`. Return `None` only for these expected cases. Let other exceptions propagate. |
| `_quality_metrics.feature_contributions` | Same pattern | Same fix |
| `ad_engine.run` (line 959 in pre-PR-2) | `except Exception as e:` | Catching `Exception` is appropriate (a detector can raise anything), but **log at WARNING level** with detector name and the exception. The `error` field in the per-detector result dict is preserved as-is for backwards compatibility. |
| `ad_engine.analyze` (line 1096 in pre-PR-2) | `except Exception:` | Same as `run`: keep broad catch (per-detector failure isolation), add WARNING log. |

A module-level logger is added to each of the five files:

```python
import logging
logger = logging.getLogger(__name__)
```

PyOD does not currently configure a root logger; users see logs only if they configure their own handler. This is the expected behavior; we do not add a default handler.

### 5.5 Feedback dict schema validation

Currently `apply_structured_feedback` (post-PR-2) accepts arbitrary dicts and falls through to `confirm_with_user` for unknown actions, masking typos and bugs. PR 3 adds explicit validation:

```python
# pyod/utils/_nl_feedback.py
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

    Validates only structure and required fields. Does not validate
    semantic content (e.g., whether a detector name actually exists in
    the KB; that is the responsibility of `apply_structured_feedback`).
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

`apply_structured_feedback` calls `validate_structured_feedback` as its first step. This **is a behavior change** (previously: silent fallthrough; now: ValueError). The eleven-row BC posture table at the top documents this.

### 5.6 NL parser declarative form

After PR 2, `_iterate_nl` lives at `_nl_feedback.apply_nl_feedback`. The function shape is preserved. PR 3 only replaces its **internals**: the `if`/`elif` chain becomes a pattern table dispatch, and a new private helper `parse_nl_to_structured` extracts the parse step.

Layout after PR 3:

```python
def apply_nl_feedback(state, feedback, kb, plan_detection_fn, make_plan_fn):
    """[docstring]"""
    proposed, confidence = parse_nl_to_structured(state, feedback)
    if confidence >= _NL_HIGH_CONFIDENCE_THRESHOLD:
        return apply_structured_feedback(
            state, proposed, kb, plan_detection_fn, make_plan_fn)
    state.next_action = {
        'action': 'confirm_with_user',
        'reason': f'Interpreted "{feedback}" as: {proposed.get("action", "?")} '
                  f'(confidence={confidence:.1f}).',
        'suggestion': f'Proposed: {proposed}. Proceed?',
        'proposed_change': proposed,
    }
    state.history.append(_make_history_entry(
        state.phase, 'iterate_nl', state.iteration,
        f'NL feedback: "{feedback}" -> confidence={confidence:.1f}'))
    return state
```

The pattern table and the helper:

```python
import re
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class _NLPattern:
    pattern: re.Pattern[str]
    confidence: float
    builder: Callable[[object, str], dict]

def _build_exclude_action(state, feedback_lower):
    for r in state.results:
        name = r.get('detector_name', '')
        if name and name.lower() in feedback_lower:
            return {'action': 'exclude', 'detectors': [name]}
    return {'action': 'exclude', 'detectors': []}

def _build_decrease_contamination(state, _feedback_lower):
    current = (state.plans[0].get('params', {}).get('contamination', 0.1)
               if state.plans else 0.1)
    return {
        'action': 'adjust_contamination',
        'value': max(current * _CONTAMINATION_DECREASE_FACTOR, _CONTAMINATION_MIN),
    }

def _build_increase_contamination(state, _feedback_lower):
    current = (state.plans[0].get('params', {}).get('contamination', 0.1)
               if state.plans else 0.1)
    return {
        'action': 'adjust_contamination',
        'value': min(current * _CONTAMINATION_INCREASE_FACTOR, _CONTAMINATION_MAX),
    }

def _build_rerun(_state, _feedback_lower):
    return {'action': 'rerun'}

_NL_PATTERNS: list[_NLPattern] = [
    _NLPattern(re.compile(r'\b(without|exclude)\b'), 0.9, _build_exclude_action),
    _NLPattern(re.compile(r'\b(false positive|too many)\b'), 0.7, _build_decrease_contamination),
    _NLPattern(re.compile(r'\b(missed|false negative)\b'), 0.7, _build_increase_contamination),
    _NLPattern(re.compile(r'\b(rerun|again)\b'), 0.9, _build_rerun),
]

def parse_nl_to_structured(state, feedback: str) -> tuple[dict, float]:
    """Match feedback against the pattern table; return (proposed, confidence)."""
    feedback_lower = feedback.lower()
    for entry in _NL_PATTERNS:
        if entry.pattern.search(feedback_lower):
            proposed = entry.builder(state, feedback_lower)
            if (entry.builder is _build_exclude_action
                    and not proposed['detectors']):
                return proposed, 0.3
            return proposed, entry.confidence
    return {'action': 'rerun'}, 0.0
```

Adding a new pattern in the future is one `_NLPattern` entry. Each builder is testable in isolation without spinning up `ADEngine`.

The contamination factors and bounds reference the constants from 5.3, so the same numbers are not repeated in three places.

### 5.7 Contamination math deduplication

Three sites in `suggest_next_step` (lines 654, 680, 706 of pre-PR-2 code) compute `min(c * 1.5, 0.5)` or `max(c * 0.5, 0.01)`. PR 3 replaces them with helper calls:

```python
# pyod/utils/_quality_metrics.py
def adjust_contamination_up(current: float) -> float:
    return min(current * _CONTAMINATION_INCREASE_FACTOR, _CONTAMINATION_MAX)

def adjust_contamination_down(current: float) -> float:
    return max(current * _CONTAMINATION_DECREASE_FACTOR, _CONTAMINATION_MIN)
```

All call sites use the helpers. The numeric values appear once. (The constants live in `_nl_feedback.py` per Section 5.3; the helpers re-import them or are co-located in the same module — implementation choice during PR 3.)

### 5.8 Verification criteria for PR 3

1. All five files have `from __future__ import annotations` at top.
2. All functions in the five files have type-hinted signatures.
3. `validate_structured_feedback` raises `ValueError` for: missing action, unknown action, missing required fields, wrong type. Tests cover all four cases.
4. NL pattern table has tests for each pattern matching its expected phrasings AND rejecting unrelated phrasings.
5. WARNING log fires once per caught detector exception in `run()` and `analyze()`. Test captures via `caplog`.
6. Contamination helper functions exist and are called from all three former in-line sites in `suggest_next_step`.
7. CHANGES.txt entry mentions the three behavior tightenings (validation raises, WARNING logs added, NL regex word-boundary).
8. **No `mypy --strict` enforcement is added.** Type hints are documentation-only in this PR.
9. No new file beyond the five from PR 2. (No `_types.py`.)

### 5.9 Risk register for PR 3

| Risk | Severity | Mitigation |
|------|----------|------------|
| `validate_structured_feedback` raises on dicts that previously fell through. Some downstream user code may rely on the silent fallthrough. | Medium | The fallthrough was a bug (typos and malformed input were masked). Document in CHANGES.txt. The error message lists valid actions and required fields, so users get an actionable diagnostic. |
| `from __future__ import annotations` interacts with runtime-introspecting libraries | Low | None of the touched files use such patterns. |
| Pattern table changes regex matching semantics for some borderline NL inputs | Medium | The original used substring `in`; the new uses `re.search` with `\b` boundaries. Add tests for inputs that worked before to verify they still work (e.g., "exclude IForest" still matches the exclude pattern). Inputs like "withoutdoubt" that incidentally matched the substring-`without` will no longer match; not a regression because the original match was incidental. |
| Adding type hints reveals incompatible types via runtime errors | Low | Type hints are evaluation-deferred under `from __future__ import annotations`; they do not affect runtime behavior. |

---

## 6. Out of Scope

These items are real but not in this sequence. Each is its own future PR.

- **KB lazy loading.** `ADEngine.__init__` reads the entire knowledge base eagerly. Move to lazy `@property kb` on first access.
- **Tier A / V3 deduplication.** `analyze_results` (Tier A) and `analyze` (V3) overlap in logic; same for `generate_report` and `report`. Either deduplicate the implementation or document the divergence as deliberate. Touches public API; needs its own design pass. **This is the deferred item that conflicts with a "full decomposition" target; Section 4.2 acknowledges PR 2 is helper extraction precisely because this dedup is deferred.**
- **Cross-cutting error-handling style.** Some methods raise on bad input, others return error dicts with `'status': 'error'`. Standardize the policy across the file. PR 3 only touches the four sites with bare `except Exception`; it does not standardize the wider policy.
- **`mypy --strict` enforcement.** PR 3 adds type hints but does not gate CI on mypy. Adding mypy checks across the project is its own scope.
- **Splitting V3 session state machine into its own class.** Currently `ADEngine` mixes Tier A and V3 methods. A future v4 could split. Defers a v4 API design discussion.
- **Replacing the rule engine with a typed schema.** The `_kb_router` rule engine evaluates string conditions at runtime; a typed schema (pydantic, dataclasses) would catch malformed rules at load time. Orthogonal.
- **Recalibrating the trigger threshold in `od_expert/SKILL.md:91`.** The new stability formula will produce a different distribution of values from the old constant 0.817, so the existing trigger `stability < 0.5` may fire more often (or less often) than intended. Plan-review Round 2 (scope position c) flagged that this is user-visible because stability drives the `adjust_contamination` branch of `iterate()`. PR 1 includes a CHANGES.txt note that the threshold may need recalibration based on observed distributions, and references this section. Empirical recalibration (run the new formula on the documented example datasets, observe distribution, decide whether 0.5 is the right threshold) is deferred to a follow-up issue. The deferral is intentional: rushing a threshold change without evidence is worse than letting the trigger fire imperfectly until calibration data is available.

---

## 7. Combined Acceptance Criteria

The full sequence is accepted when:

1. PR 1 merged: issue #667 closed, new stability formula in place, regression tests added (11 tests covering responds-to-data, determinism, k=0, k=n, k=1, k=n-1, ties at boundary, scale invariance, constant scores, non-finite scores, unit interval), all live doc surfaces in Section 3.8 updated, CHANGES.txt entry references #667.
2. PR 2 merged: structural decomposition complete, four new private modules, `pyod/utils/ad_engine.py` ≈ 1,200 LOC, all tests pass unmodified, importable surface byte-identical, moved functions have NumPy-style docstrings.
3. PR 3 merged: type hints across all five files (no enforcement gate), magic numbers extracted, contamination helpers in place, NL parser declarative, validation raises on malformed feedback, WARNING logs on caught detector exceptions.

After all three merge:
- `pyod/utils/ad_engine.py` ≈ 1,200 LOC (down from 1,650, a 27% reduction).
- Four helper modules are private (`_` prefix), each with a single responsibility, each independently testable.
- `quality['stability']` is informative and reproducible.
- Type hints, named constants, and structural validation are baseline.

---

## 8. Test Plan Summary

| PR | New tests | Existing tests touched |
|----|-----------|------------------------|
| PR 1 | `TestStabilityMetric` class in `pyod/test/test_ad_engine_v3.py` (10 tests, see 3.9) | `test_quality_metrics` continues to pass; `verdict` may shift on test data calibrated against constant 0.817; if any existing test asserts a specific verdict value, it is updated as part of PR 1 (and the change is noted in CHANGES.txt) |
| PR 2 | None (pure structural) | None — they must pass unmodified |
| PR 3 | Test class for `validate_structured_feedback` (4 tests), test class for NL pattern matching (one test per pattern + edge cases for word-boundary regex behavior, ~8 tests), `caplog`-based test for WARNING logs in `run()`/`analyze()` | Tests for `_iterate_structured` may need updates if any passes a malformed dict and expects fallthrough (survey did not find any; re-verified in PR 3) |

---

## 9. References

- Issue [#667](https://github.com/yzhao062/pyod/issues/667): Quentin Grimonprez, "stability measure in ADEngine is not informative" (2026-04-30).
- `docs/superpowers/specs/2026-04-07-pyod-expansion-design.md`: master architecture spec for the PyOD V3 lineage.
- `docs/superpowers/specs/2026-04-12-v3-agentic-design.md` section 4.4: original (incorrect) stability formula.
- `docs/superpowers/plans/2026-04-08-ad-engine-tier-b.md`: original Tier B implementation plan that defined the quality metrics.
- `pyod/skills/od_expert/SKILL.md`: live skill that references ADEngine quality metrics in the workflow walkthrough.
- `Review-Codex.md` (Round 1, deleted post-merge): plan-review feedback that produced this v2.

---

## 10. Open Questions and Review History

### Round 2 plan-review (Codex, 2026-05-07)

4 new findings + 2 prior findings re-opened. All addressed in v3.

**New findings:**

| # | Severity | Finding | Outcome |
|---|----------|---------|---------|
| 1 | High | PR 1 tests call `engine._compute_quality(...)` but PR 2 drops the private method while forbidding test edits | **Fixed in v3** — `_quality()` helper in `TestStabilityMetric` uses Codex's verbatim `try ImportError` rewrite to call either the private method (during PR 1) or `pyod.utils._quality_metrics.compute_quality` (after PR 2), keeping the test class unmodified across the transition (Section 3.9) |
| 2 | Medium | Non-finite scores (NaN, Inf) produce `stability = nan`, violating `[0, 1]` docstring; verified by reproduction | **Fixed in v3** — added `np.all(np.isfinite(scores))` guard returning 0 for non-finite input; added `test_stability_zero_for_nonfinite_scores` covering NaN, Inf, -Inf (Section 3.3, 3.9) |
| 3 | Medium | Section 3.8 "global grep" claim false; `docs/superpowers/plans/` files contain old formulas and values | **Fixed in v3** — narrowed claim to "live docs and skill surfaces"; explicit note that historical plan files are intentionally not updated (Section 3.8) |
| 4 | Medium | Public method count is 20, not 21 (verified by introspection); `_sniff_data_type` does not access `self.kb` (verified by source read) | **Fixed in v3** — corrected count to 20 throughout (BC posture table, Section 1.3, Section 4.8 verification); rewrote `_sniff_data_type` retention rationale to match the actual source (small, single call site, no clear receiving module) |

**Re-opened prior findings:**

| # | Original status | Re-opened reason | Outcome |
|---|----------------|------------------|---------|
| Round 1 #4 | Resolved (BC table) | BC table missed `state.next_action.reason`, `state.history` entries, `state.analysis['summary']` as user-visible workflow surfaces affected by quality value changes; also repeated the 21-method count error | **Fixed in v3** — added three rows to BC posture table covering workflow surfaces; corrected method count |
| Round 1 #5 | Resolved (doc table) | Doc surface table not closed under stated `docs/ pyod/skills/` grep because `docs/superpowers/plans/` was outside scope but the claim implied global coverage | **Fixed in v3 (combined with new Finding 3)** — claim narrowed to "live docs and skill surfaces"; archived plans explicitly out of scope |

**Codex scope position (Round 2):**

- (a) PR-1-only case got stronger after the bug-fix-first reordering. **User has chosen 3-PR sequence; not actionable in this revision.**
- (b) v3 PR 3 cuts are right; cautioned that type hints without enforcement are documentation only and PR 3 should keep them conservative (avoid type-driven rewrites). **Acknowledged in v3 (Section 5.2).**
- (c) Trigger threshold deserves at least characterization. **Addressed in v3 (Section 6 deferred-items entry expanded with explicit CHANGES.txt note plan).**

### Round 1 plan-review (Codex, 2026-05-07)

7 findings. Outcomes:

| # | Severity | Finding | Outcome |
|---|----------|---------|---------|
| 1 | High | Test fixture in `test_stability_responds_to_data` produces `stability=1.0`, not `<=0.5` as asserted (verified by reproduction: `gap=0.374`, `std=0.316`) | **Fixed in v2** — applied Codex's rewrite verbatim (Section 3.9) |
| 2 | Medium | Migration table missing `_suggest_alternative`; LOC target of 600-700 not supported by helper sizes | **Fixed in v2** — added `_suggest_alternative` row to migration table; revised target to ≈1,200 LOC with explicit "helper extraction, not full decomposition" framing (Section 4.2, 4.4) |
| 3 | Medium | Stability formula not scale invariant for tiny std (verified: `[2e-12, 1e-12, 0]` gives 0.0099 vs `[2e-6, 1e-6, 0]` gives 1.0) | **Fixed in v2** — replaced `+ epsilon` denominator with explicit `if std == 0.0` check; added scale-invariance test and edge-case tests for k=1, k=n-1, ties at boundary (Section 3.3, 3.9) |
| 4 | Medium | "No public API changes" claim too strong; misses dict values, annotations, log output, error behavior, subclass-override surface | **Fixed in v2** — replaced loose claim with eleven-row "BC posture" table at top (top of doc) enumerating each preserved vs changed dimension |
| 5 | Medium | Wrong changelog filename (`CHANGELOG.rst` doesn't exist; `CHANGES.txt` does); 12 missed doc surfaces with hardcoded stability values | **Fixed in v2** — Section 3.8 now lists every surface with file path, line numbers, and action; verified by global grep |
| 6 | Medium | PR 3 too broad; internal inconsistency (`_types.py` vs five-file scope) | **Fixed in v2 via Option A** — moved docstring polish into PR 2 (Section 4.7); dropped `_types.py`; dropped `mypy --strict` gate |
| 7 | Medium | Scope position: ship bug fix first, structural decomposition delays user-visible repair | **Accepted in v2** — PR ordering swapped: PR 1 is now the bug fix, PR 2 is decomposition, PR 3 is tech debt |

No findings remain open or deferred. All seven were addressed in v2.

### Open questions

None. All scope decisions were made interactively with the user before this v2 was drafted.
