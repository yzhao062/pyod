# Extended pitfalls library

This file extends the top-10 critical pitfalls in SKILL.md with 20 more, organized by phase. Each pitfall has:

- **Phase** — where it surfaces (preprocessing, detection, analysis, reporting, iteration)
- **Severity** — HIGH (silently wrong results), MEDIUM (degraded quality), LOW (inefficient)
- **Mitigation** — what to do instead

The agent loads this file when it suspects a specific category of issue (e.g., the user reports "the result looks weird" → load this file and walk the relevant phase).

## Phase 1: Preprocessing

### P1. Categorical columns left as strings (HIGH)

PyOD detectors expect numeric input. String columns crash `engine.profile_data` or get silently dropped. Mitigation: detect string columns via `df.dtypes`, prompt user for encoding (label vs one-hot), apply before `engine.start`.

<!-- Source: standard data-cleaning practice; surfaces in nearly every real tabular OD case -->

### P2. Datetime columns treated as features (MEDIUM)

A datetime column has very high cardinality and very large numeric range. Distance-based detectors will give it overwhelming weight. Mitigation: extract calendar features (hour, day of week, month) and drop the raw timestamp before tabular detection. Or treat as time series.

### P3. NaN values left in (HIGH)

Most PyOD detectors do not handle NaN. They either crash or silently skip rows. Mitigation: impute (median for numerical, mode for categorical) or drop NaN-containing rows before `engine.start`. Document the choice. NeurIPS 2024 introduced a missing-value-aware variant ("Unsupervised Anomaly Detection in the Presence of Missing Values"); until that lands in PyOD, imputation is the practical default.

<!-- Source: NeurIPS 2024 unsupervised AD with missing values -->

### P4. Outlier-driven scaling (MEDIUM)

`StandardScaler` uses mean and std, both of which are sensitive to extreme outliers. Scaling on data with strong outliers shifts the entire distribution. Mitigation: use `RobustScaler` (median + IQR) when the data is expected to have outliers — which is always the case for OD.

### P5. Class-imbalanced data treated as unsupervised (LOW)

If labels are available and the positive class is very rare (< 1%), supervised detectors like `XGBOD` or `DevNet` outperform unsupervised by a wide margin. Mitigation: ask for labels (Trigger 5), use supervised mode.

<!-- Source: ADBench 2022; SPADE Google blog 2024-05 -->

## Phase 2: Detection

### D1. Single-iteration on novel data (MEDIUM)

The first detector run on unfamiliar data is rarely the best one. Mitigation: always run the top-3 from `engine.plan` before reporting.

### D2. n_neighbors default for LOF on small data (MEDIUM)

`LOF`'s default `n_neighbors=20` is wrong for n < 100. Mitigation: scale n_neighbors to ~min(20, n // 5).

### D3. AutoEncoder without validation split (HIGH)

`AutoEncoder` needs a held-out validation set to detect over-fitting. Without it, training can converge to a degenerate solution. Mitigation: ADEngine handles this if you use `engine.start` → `engine.plan` → `engine.run`. Do NOT bypass.

### D4. ECOD on data with strong feature correlations (MEDIUM)

`ECOD` assumes feature independence. On highly correlated features (e.g., raw pixel data, embeddings), it underperforms. Mitigation: use `COPOD` instead, or PCA-reduce first.

### D5. XGBOD without hyperparameter tuning (LOW)

`XGBOD`'s defaults are not tuned for OD specifically. Mitigation: pass a parameter dict from the KB's `default_params` field, or use ADEngine's planner which knows the right defaults.

### D6. SUOD without enough cores (LOW)

`SUOD` parallelizes ensembles. Running with `n_jobs=1` defeats the purpose. Mitigation: set `n_jobs=-1` or use ADEngine's automatic selection.

### D7. DeepSVDD on data with feature scale mismatch (HIGH)

`DeepSVDD`'s center initialization is sensitive to feature scales. Mitigation: standardize before training.

### D8. Foundation-model TSAD treated as default (MEDIUM)

Recent benchmarks show foundation-model time-series detectors (the published ones, not yet in PyOD) frequently tie with one-line baselines like moving-window variance. The lesson for PyOD: prefer simple statistical TSAD methods (`SpectralResidual`, `MatrixProfile`) before reaching for transformers like `AnomalyTransformer` or `LSTMAD`.

<!-- Source: TSB-AD NeurIPS 2024; "When Foundation Models are One-Liners" OpenReview 2025 -->

## Phase 3: Analysis

### A1. Reading consensus when only one detector ran (MEDIUM)

`state.quality.agreement` is undefined for n=1 detectors. Mitigation: always run >= 2 detectors. If only one was requested, set agreement to None and skip Trigger 3.

### A2. Stability low on small datasets (LOW false alarm)

On n < 200, stability is naturally low because resampling has high variance. Mitigation: relax the trigger 4 threshold to 0.3 for small data.

### A3. Separation interpretation depends on detector type (MEDIUM)

Distance-based detectors (`LOF`, `KNN`) and density-based (`HBOS`, `COPOD`) produce score distributions with different shapes. Comparing separation across detector families is misleading. Mitigation: only compare separation within the same family, or use the rank-normalized consensus from `state.consensus['scores']`.

### A4. Detector disagreement is the norm (MEDIUM)

A 2026 KDnuggets case study ran 5 classic OD methods on the wine dataset and found they disagreed on 96% of flagged points. Detector disagreement is the rule, not the exception. Trust only points flagged by ≥ 3 of 5 detectors when reporting from a wide ensemble. ADEngine's consensus mechanism implements this.

<!-- Source: KDnuggets 2026-03 "5 outlier detection methods... disagreed on 96%" -->

## Phase 4: Reporting

### R1. Reporting raw decision_function scores (HIGH)

Raw scores are not interpretable. A user reading "score = 1.43" has no context. Mitigation: report percentile rank, label, or a calibrated probability via `predict_proba`.

### R2. Forgetting the contamination assumption in the report (MEDIUM)

If the agent silently used contamination=0.1 when the true rate was 0.02, the user gets 5x too many false positives. Always include the contamination value in the "what I assumed" section.

### R3. Reporting > 100 anomalies in one list (LOW)

Long lists are unreadable. Mitigation: report top-10 by score, summarize the rest as a count + score histogram.

### R4. Missing the "next action" recommendation (LOW)

A bare report leaves the user wondering what to do next. Mitigation: always end with "want me to drill into specific points / iterate / try a different detector?"

## Phase 5: Iteration

### I1. Iterating without changing detectors (HIGH)

`engine.iterate` with the same detector set produces the same result. Mitigation: when iterating, swap at least one detector. The planner does this automatically when called inside `engine.iterate`.

### I2. Endless iteration on noise (HIGH)

Iterating on a dataset with no real anomalies converges to picking arbitrary points as outliers. Trigger 11 (deadlock detection) catches this after 2 rounds.

### I3. User feedback that contradicts itself (MEDIUM)

If the user says "exclude detector A" then "include detector A" in successive turns, log the contradiction and ask which they prefer. Do not silently follow the latest input.
