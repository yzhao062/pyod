# od-expert Workflow Reference

This file is loaded on demand by the agent (per the same on-demand model documented in SKILL.md). Load it before the first detection run, then keep it in working memory for the rest of the session. It contains:

1. The autonomous loop pattern
2. The 11 adaptive escalation triggers with example phrasings
3. The cardio canonical worked example (narrative + transcript)
4. Result interpretation patterns

For modality-specific decision tables and worked snippets, see the per-modality reference files.

## The autonomous loop

```
1. Receive user request → understand intent (data + goal)
2. Profile: engine.start(X)
   ├── Walk modality decision tree from SKILL.md
   ├── Check escalation triggers 1, 2, 5, 7, 9
   └── If any trigger fires → escalate (see Trigger Phrasing below)
3. Plan: engine.plan(state)
   ├── Inspect state.plans — list of plan dicts, each with
   │      'detector_name', 'confidence', 'reason', 'evidence', 'alternatives'
   ├── Check escalation trigger 6 (heavy detector + small data)
   └── If trigger fires → escalate
4. Run: engine.run(state)
5. Analyze: engine.analyze(state)
   ├── Check escalation triggers 3, 4, 10
   └── If trigger fires → escalate or hedge in report
6. Decide: state.next_action['action']
   ├── 'report_to_user' → go to step 7
   ├── 'iterate' → engine.iterate(state, suggested_feedback) → loop to step 5
   │              (check trigger 11 — deadlock after 2 iterations)
   └── 'confirm_with_user' → present state.next_action['suggestion']
                             (only set for iterate/confirm branches)
7. Report: engine.report(state)
   ├── Translate state.consensus['scores'] and state.consensus['labels']
   │      into user-facing language (see Result Interpretation below)
   ├── Always include "what I assumed and why" section
   └── Cite the detectors used and their consensus quality
```

## Trigger phrasings

When a trigger fires, the agent pauses and asks the user. These phrasings are intentionally minimal — short, specific, and include a default the user can confirm with one word.

### Trigger 1: Modality ambiguity

> "I see your data has a timestamp column AND feature columns. I can either treat the timestamps as ordinal features (tabular detection) or as the temporal axis (time series detection). The two give different results. Which fits your use case? **Default: tabular** — say 'time series' to switch."

### Trigger 2: Contamination uncertainty

> "Anomaly detection needs an estimate of how many points are unusual (the contamination rate). The default is 10%, but that's often wrong by a wide margin. If you have a sense of the actual rate (from domain knowledge or a labeled sample), tell me. Otherwise I'll run with the default and show you the flagged fraction in `state.analysis['consensus_analysis']['anomaly_ratio']` — you can adjust from there via `engine.iterate(state, {'action': 'adjust_contamination', 'value': <rate>})`."

### Trigger 3: Detector disagreement post-run

> "The top-3 detectors I ran ([A], [B], [C]) disagree on which points are anomalies (consensus = [X], where 1.0 is full agreement). I can iterate with a different set of detectors, or report what I have with a clear caveat. **Recommended: iterate** — say 'report' to skip iteration."

### Trigger 4: Quality assessment weak

> "The quality of the result is low: separation = [X], stability = [Y]. The detection is essentially noise at these levels. Two options: (1) iterate with a different detector mix, or (2) accept that this dataset may not have clear anomalies. I'll iterate unless you say 'report'."

### Trigger 5: Labels mentioned but not provided

> "You mentioned [some known cases / ground truth labels / etc.]. The `ADEngine` session API is unsupervised in v3.2.x. For labelled data the better path is to drop out of the session and call a supervised detector like `XGBOD` directly via its classic `fit(X, y)` / `predict(X)` interface (needs `pyod[xgboost]`). Supervised detection typically outperforms unsupervised by 10-20 AUC points on a labelled cohort. Should I finish the unsupervised run and recommend the switch, or pause here so you can set up the supervised path?"

### Trigger 6: Heavy detector + small data

> "The detector [name] (deep learning) needs > 1000 rows to train reliably. Your data has [N] rows. Recommended: switch to `ECOD` / `IForest` / `HBOS` which works well on small data. Say 'force [name]' if you want to proceed anyway."

### Trigger 7: Missing optional extra

> "Graph detection requires the optional `pyod[graph]` extra. Run `pip install pyod[graph]` and re-run, or I can use a non-graph detector on your node features only (loses edge information)."

### Trigger 8: High-stakes domain hint

> "You mentioned [medical / fraud / security / safety]. For high-stakes domains, I'll always validate with a second detector and include a confidence caveat in the final report. Should I also flag any specific failure modes you care about?"

### Trigger 9: Cross-modality ambiguity

> "Your data has both numerical columns and text columns. I can either (1) handle as multimodal (use `EmbeddingOD` on the text + `LOF` on the numerics, then combine), or (2) pick the dominant modality. Which fits your use case?"

### Trigger 10: Result feels too confident

> "The detectors all agree on the same [X]% of points with very high consensus (> 90%). That can mean a real signal — or that the dataset has trivial separation that doesn't reflect production reality. Can you sanity check the top 5 flagged points to confirm they're actually anomalous in your domain?"

### Trigger 11: Iteration loop deadlock

> "I've iterated twice with no quality improvement. The result is still [X separation, Y stability]. I can keep going with a fresh detector mix, or settle on what we have and report. Recommended: report. Say 'iterate' to keep going."

## The cardio canonical worked example

This is the canonical end-to-end example. Use it as the template for any tabular investigation.

The numeric values in this example are taken from `examples/agentic_demo.html` (the checked-in interactive demo). Re-verify against that file if the demo has been regenerated since this was written.

### Setup

User provides the UCI Cardiotocography dataset: 1,831 fetal heart rate recordings with 21 clinical features. The user is not a fetal monitoring expert; they want to know which recordings look unusual and why.

### Turn 1: Profile and plan

> User: I have a cardiotocography dataset with 1,831 fetal heart rate recordings and 21 clinical features. Can you find unusual recordings?

Agent does:

```python
import numpy as np
import pandas as pd
from pyod.utils.ad_engine import ADEngine

np.random.seed(42)  # reproducible across the reference numbers below
df = pd.read_csv("examples/data/cardio.csv")
X = df.values[:, :-1]  # exclude the trailing label column; use it only for offline validation
engine = ADEngine()
state = engine.start(X)
```

Agent inspects `state.profile`:

- `state.profile['data_type']`: tabular ✓
- `state.profile['n_samples']`: 1831 ✓ (no Trigger 6 — well above the 1000-row floor for DL)
- `state.profile['n_features']`: 21 ✓
- `state.profile['dimensionality_class']`: low (21 features)
- `state.profile['has_nan']`: False
- No timestamps → no Trigger 1
- Numerical columns only → no Trigger 5, 9
- Contamination rate: unknown at this point — the profiler does not estimate it. Will be observed post-run via `state.analysis['consensus_analysis']['anomaly_ratio']` and adjusted if it looks wrong.

No triggers fire. Proceed.

```python
state = engine.plan(state)
# [p['detector_name'] for p in state.plans[:3]] == ['IForest', 'ECOD', 'KNN']
```

These are the right starters for tabular, mid-size data — matches the master decision tree in SKILL.md.

Agent reports to user:

> "Profiled the data: 1,831 rows × 21 features. Selected three detectors: `IForest`, `ECOD`, `KNN`. Running now."

### Turn 2: Run and analyze

```python
state = engine.run(state)
state = engine.analyze(state)
```

Agent inspects `state.quality`:

- `state.quality['agreement']`: 0.68 ✓ (no Trigger 3 — above the 0.4 floor)
- `state.quality['stability']`: 0.82 ✓ (no Trigger 4)
- `state.quality['overall']`: 0.83 (`state.quality['verdict']` = 'high')
- `state.analysis['best_detector']`: IForest (highest Spearman correlation with consensus)

No triggers fire. Move toward report.

Agent reports to user:

> "Detection complete. The three detectors converged 3/3 with agreement 0.68 and stability 0.82 — overall quality 0.83 (HIGH). `IForest` had the highest correlation with the consensus. Found 172 cases flagged out of 1,831 at the default contamination rate of 0.1."

### Turn 3: Report

```python
report = engine.report(state)
```

Agent translates the report into user-facing language using the result interpretation patterns below.

> "Here's what I found:
>
> - **172 of 1,831 recordings (9.4%) flagged as unusual.** The agent ran a consensus of 3 detectors (`IForest`, `ECOD`, `KNN`) using ADEngine's default contamination of 0.1. If that rate looks wrong for the clinical context, re-run via `engine.iterate(state, {"action": "adjust_contamination", "value": <rate>})`.
> - **Top case**: row #1656 (analyzed via `IForest`, the detector with the highest Spearman correlation with the consensus). Drill into `state.analysis['consensus_analysis']['top_anomalies']` for the ranked index list, then call `engine.explain_findings(state.results[best_idx])` for per-feature z-scores.
> - **Confidence**: high. Overall quality 0.83 (HIGH). Agreement across detectors is 0.68 and stability is 0.82. For reference only (not seen by the agent during detection): validation against the cardio ground truth gave precision 49.4% (85/172) at recall 48.3% (85/176).
>
> **What I assumed**:
> - Data is tabular and unlabeled
> - Contamination: 0.1 (ADEngine default for this one-pass run; revisit only if the observed anomaly ratio looks wrong for the domain)
> - Detectors: `IForest`, `ECOD`, `KNN` — selected by ADEngine's planner for mid-size tabular data
> - Best detector: `IForest` (highest Spearman correlation with the consensus of all three)
>
> Want me to drill into specific recordings, or iterate with a different detector mix?"

## Result interpretation patterns

When translating `state.consensus['scores']`, `state.consensus['labels']`, and `state.quality` into user-facing language:

### High agreement + high separation → confident report

Phrasing: "Found N anomalies. The detectors agreed well (consensus X) and the separation is strong (Y). Confidence: high."

### High agreement + low separation → calibration warning

Phrasing: "The detectors agree on the flagged points, but the gap between flagged and normal is small (separation Y < 0.2). The result is internally consistent but may reflect a dataset with no strong outliers. Confidence: medium-low."

### Low agreement + any separation → hedged report or iterate

Phrasing: "The detectors disagree (consensus X < 0.4). Iterating with a different mix is recommended. If reporting now: confidence is low."

### Single-detector consensus (one detector clearly best)

Phrasing: "`ECOD` performed substantially better than `IForest` and `LOF` on this dataset (separation 0.34 vs 0.12 and 0.18). Reporting `ECOD`'s flagged points as the primary result."

### Result with labels (supervised mode via XGBOD)

Phrasing: "With your provided labels, `XGBOD` achieved AUC X. Top-5 misclassified false negatives: ... Top-5 false positives: ..."

## Always include "what I assumed and why"

Every report ends with a short transparency section. Format:

```markdown
**What I assumed**:
- Data type: <type> (auto-detected from <heuristic>)
- Contamination: <value> (<source: estimated / domain-supplied / default>)
- Detectors: <list> (selected by <reason>)
- Primary detector: <name> (chosen because <metric>)
```

If any of these assumptions look wrong to the user, they say so and we iterate. Without this section, the user has no way to sanity check the agent's choices.
