# PyOD V3: Agentic Anomaly Detection Design

**Date:** 2026-04-12
**Status:** Draft (v6 -- Round 5 review fixes)

---

## 1. Vision

PyOD V3 thesis: **any AI agent can do expert-level anomaly detection through PyOD, without knowing OD.**

The goal is not multi-modal coverage (already shipped). The goal is making the anomaly detection workflow **agentic** — ADEngine guides any agent (LLM or application) through the expert OD workflow step by step, so the agent never has to know which algorithm to pick, how to interpret scores, or when to iterate.

**What "agentic" means concretely:**
- Understand the data modality (tabular, TS, graph, text, image)
- Choose the right algorithm(s) based on benchmark evidence
- Run detection, compare multiple detectors, assess result quality
- Discuss results with users, explain findings
- Iterate based on feedback (too many false positives, missed anomalies, try something different)
- Generate reports

**What it does NOT mean:**
- No changes to Layer 1 (BaseDetector models, direct `fit`/`predict` API)
- No mandatory dependency on LLMs or agents
- No persistent state across Python sessions (stateless library)

---

## 2. Architecture: Three Layers

```
Layer 3:  Skill (od-expert)               ← agent conversation layer
Layer 2:  ADEngine (workflow + intelligence) ← agentic orchestration
Layer 1:  BaseDetector models (fit/predict)  ← direct Python API (unchanged)
```

Each layer is independently useful:
- **Layer 1** users: `from pyod.models.iforest import IForest; clf.fit(X)` — no changes, no new requirements
- **Layer 2** users: `engine = ADEngine(); result = engine.investigate(data)` — intelligent orchestration
- **Layer 3** users: agent follows od-expert skill, which calls ADEngine — full conversational workflow

Intelligence lives in **Layer 2** (portable across agents). Conversation flow lives in **Layer 3** (agent-specific). Layer 1 is untouched.

---

## 3. Current State (ADEngine Tier A + B)

ADEngine today has lifecycle methods, but they are **disconnected building blocks**. An agent must know which to call, in what order, and how to interpret results:

```python
profile = engine.profile_data(data)           # agent decides what to do
plan = engine.plan_detection(profile)          # returns 1 detector
result = engine.run_detection(data, plan)      # agent must run this
analysis = engine.analyze_results(result, X=data)  # agent decides if good
# ... agent parses analysis, decides whether to iterate, how, etc.
```

**Problems:**
1. Routes to 1 detector — experts run 2-3 and compare
2. No quality assessment — agent cannot tell if results are trustworthy
3. No guided iteration — `suggest_next_step` returns text, not an executable action
4. No workflow enforcement — agent can call methods in any order, skip steps
5. No session context — each call is stateless, agent must carry all context

---

## 4. V3 Design: Workflow Engine

### 4.1 Session-Based State Machine

ADEngine manages an **investigation session** that tracks the workflow state:

```
START → PROFILED → PLANNED → DETECTED → ANALYZED → [ITERATE → PLANNED | DONE]
```

Each step returns the updated `InvestigationState` with a typed `next_action` that tells the agent what to do next. The agent (or skill) follows `next_action` without needing OD knowledge.

**Decision (Codex finding #1):** The session execution method is named `run(state)`, not `detect(state)`. The existing `detect(X_train, ...)` one-shot method is unchanged. No naming conflict.

### 4.2 New API: Investigation Session

```python
class ADEngine:
    # --- Existing methods (unchanged, still work independently) ---
    def profile_data(self, X, data_type=None): ...
    def plan_detection(self, profile, ...): ...
    def run_detection(self, X_train, plan, ...): ...
    def analyze_results(self, result, ...): ...
    # ... all existing methods preserved ...

    # --- V3: Session-based workflow ---

    def start(self, X, data_type=None):
        """Start an investigation session.

        Profiles the data and returns an InvestigationState with
        the profile and recommended next action.

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

    def plan(self, state, priority='balanced', constraints=None):
        """Plan detection for the investigation.

        Selects top-N candidate detectors based on profile and
        benchmark evidence. Updates state with detection plan.

        Parameters
        ----------
        state : InvestigationState
        priority : str
            'speed', 'accuracy', or 'balanced'.
        constraints : dict or None
            {'exclude_detectors': [...], 'max_detectors': int}

        Returns
        -------
        state : InvestigationState
        """

    def run(self, state):
        """Run detection with all planned detectors.

        Calls ``run_detection()`` for each plan in ``state.plans``.
        Collects per-detector results, computes consensus scores
        via rank normalization, and measures detector agreement.
        If a detector errors, it is recorded in ``state.results``
        with ``status='error'`` and excluded from consensus.

        Parameters
        ----------
        state : InvestigationState
            Must be in phase ``'planned'``.

        Returns
        -------
        state : InvestigationState
            Phase set to ``'detected'``.
        """

    def analyze(self, state):
        """Analyze detection results.

        Assesses result quality (score distribution, detector
        agreement, contamination stability), identifies top
        anomalies, computes feature importance, and generates
        a human-readable summary. Updates state with analysis.

        Parameters
        ----------
        state : InvestigationState

        Returns
        -------
        state : InvestigationState
        """

    def iterate(self, state, feedback):
        """Iterate based on user/agent feedback.

        **Structured feedback (primary, executes immediately):**
        - ``{"action": "adjust_contamination", "value": 0.05}``
        - ``{"action": "exclude", "detectors": ["IForest"]}``
        - ``{"action": "include", "detectors": ["ECOD"]}``
        - ``{"action": "rerun"}``  (same plan, different random seed)

        **Natural-language feedback (best-effort, needs confirmation):**
        - ``"too many false positives"``
        - ``"try without IForest"``
        When NL feedback is provided, the engine parses it into a
        proposed structured action with a confidence score. If
        confidence < 0.8, ``next_action`` is set to
        ``'confirm_with_user'`` with the proposed change for the
        agent to confirm before executing.

        Parameters
        ----------
        state : InvestigationState
            Must be in phase ``'analyzed'``.
        feedback : str or dict
            Structured dict (executed immediately) or NL string
            (parsed to proposed action, may need confirmation).

        Returns
        -------
        state : InvestigationState
            Phase reset to ``'planned'`` if action is clear,
            or kept at ``'analyzed'`` with ``next_action =
            'confirm_with_user'`` if ambiguous.
        """

    def report(self, state, format='text'):
        """Generate a report for the investigation.

        Parameters
        ----------
        state : InvestigationState
        format : str
            'text' or 'json'.

        Returns
        -------
        report : str or dict
        """

    def investigate(self, X, data_type=None, priority='balanced'):
        """One-shot expert investigation (convenience).

        Runs the full workflow: start → plan → run → analyze.
        Returns an InvestigationState ready for user review,
        iteration, or reporting.

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
```

### 4.3 InvestigationState

A typed dataclass with closed enums and defined schemas.

**Phase enum (closed):**

```python
PHASES = ('profiled', 'planned', 'detected', 'analyzed')
```

**ActionType enum (closed):**

```python
ACTION_TYPES = (
    'plan',              # engine recommends planning next
    'run',               # engine recommends running detection
    'analyze',           # engine recommends analyzing results
    'report_to_user',    # results ready for user review
    'confirm_with_user', # engine needs user confirmation before acting
    'iterate',           # engine suggests trying a different approach
    'done',              # investigation complete
)
```

**State dataclass:**

```python
@dataclass
class InvestigationState:
    # --- Workflow tracking ---
    phase: str                    # one of PHASES
    iteration: int                # 0 = first run, increments on iterate()
    history: list                 # list of HistoryEntry dicts (see schema)

    # --- Data context ---
    data: object                  # reference to input data (not copied)
    profile: dict                 # output of profile_data()

    # --- Detection ---
    plans: list                   # list of DetectionPlan dicts (top-N)
    results: list                 # list of DetectorResult dicts (see schema)
    consensus: dict or None       # ConsensusResult dict (see schema)

    # --- Analysis ---
    analysis: dict or None        # InvestigationAnalysis dict (see schema)
    quality: dict or None         # QualityAssessment dict (see schema)

    # --- Workflow guidance ---
    next_action: dict             # NextAction dict (see schema)
```

**Typed schemas:**

```python
# HistoryEntry: one per workflow step
HistoryEntry = {
    'phase': str,           # phase after this step
    'action': str,          # what was done ('start', 'plan', 'run', ...)
    'iteration': int,       # iteration number
    'timestamp': float,     # time.time()
    'detail': str,          # human-readable summary
}

# DetectorResult: one per detector in run()
# Superset of run_detection() output — raw result stored verbatim
# plus status/error fields for the session wrapper.
DetectorResult = {
    'detector_name': str,
    'status': str,          # 'success' | 'error' | 'skipped'
    'error': str or None,   # error message if status='error'
    # --- Fields from run_detection() (present when status='success') ---
    'plan': dict,           # the DetectionPlan used
    'scores_train': np.ndarray,  # (n_samples,)
    'labels_train': np.ndarray,  # (n_samples,)
    'threshold': float,
    'n_anomalies': int,
    'anomaly_ratio': float,
    'detector': object,     # fitted BaseDetector
    'runtime_seconds': float,
    'score_summary': dict,  # mean, std, min, max, q25, q75
}

# ConsensusResult: aggregated across successful detectors
ConsensusResult = {
    'scores': np.ndarray,       # (n_samples,) rank-normalized mean
    'labels': np.ndarray,       # (n_samples,) majority-vote labels
    'n_detectors': int,         # number of successful detectors
    'agreement': float,         # mean pairwise Spearman correlation [0,1]
    'disagreements': list,      # sample indices where detectors disagree
}

# ConsensusAnalysis: lightweight summary (NOT analyze_results() output)
ConsensusAnalysis = {
    'n_anomalies': int,
    'anomaly_ratio': float,
    'score_distribution': dict,     # mean, std, min, max, median, q25, q75
    'top_anomalies': list,          # top-k by consensus score
    'summary': str,                 # generated narrative
}

# InvestigationAnalysis: output of analyze()
InvestigationAnalysis = {
    'consensus_analysis': ConsensusAnalysis,
    'per_detector_analysis': list,  # positionally aligned with state.results;
                                    # None for error/skipped entries,
                                    # analyze_results() output for successful
    'best_detector': str,           # name of best detector
    'best_detector_index': int,     # index into state.results (always a
                                    # successful entry)
    'summary': str,                 # human-readable summary
}
# best_detector selection (deterministic fallback chain):
# 1. Highest finite Spearman correlation with consensus scores
# 2. If tied: highest plan confidence (from routing)
# 3. If still tied: fastest successful detector (lowest runtime)
# 4. If all correlations are NaN (constant scores): first successful detector
# Single-detector case: best_detector_index = that detector's index

# QualityAssessment
QualityAssessment = {
    'separation': float,    # score separation ratio [0, 1] (see 4.4)
    'agreement': float,     # detector agreement [0, 1], N/A → 0.5
    'stability': float,     # label stability [0, 1] (see 4.4)
    'overall': float,       # mean(separation, agreement, stability)
    'verdict': str,         # 'high' | 'medium' | 'low'
    'explanation': str,     # human-readable quality summary
}

# StructuredFeedback: typed actions for iterate()
# Each action has a closed set of required fields.
StructuredFeedback = {
    'action': str,  # one of:
    # 'adjust_contamination' — requires 'value': float
    # 'exclude'             — requires 'detectors': list[str]
    # 'include'             — requires 'detectors': list[str]
    # 'rerun'               — no extra fields
}

# NextAction: closed action type with typed payload
# 'action' and 'reason' are always required.
# Additional fields are per action type (R=required, O=optional):
#
# action='plan':              (no extra fields)
# action='run':               O 'adjustment': str  (present after iterate)
# action='analyze':           (no extra fields)
# action='report_to_user':    R 'summary': str
#                             R 'confidence': float
# action='confirm_with_user': O 'suggestion': str    (present for change confirmation)
#                             O 'proposed_change': StructuredFeedback (present for change confirmation)
#                             (when used for error/retry, only 'reason' is present)
# action='iterate':           R 'suggestion': str
# action='done':              (no extra fields)
NextAction = {
    'action': str,          # one of ACTION_TYPES
    'reason': str,          # always present
}
```

**Worked example: state after `plan()`**

```python
state.phase = 'planned'
state.iteration = 0
state.profile = {'data_type': 'tabular', 'n_samples': 1000, 'n_features': 20, ...}
state.plans = [
    {'detector_name': 'IForest', 'params': {}, 'confidence': 0.85, ...},
    {'detector_name': 'ECOD', 'params': {}, 'confidence': 0.8, ...},
    {'detector_name': 'KNN', 'params': {}, 'confidence': 0.75, ...},
]
state.results = []
state.consensus = None
state.next_action = {
    'action': 'run',
    'reason': 'Top 3 detectors selected: IForest (0.85), ECOD (0.80), KNN (0.75). Ready to run.',
}
```

**Worked example: state after `analyze()`**

```python
state.phase = 'analyzed'
state.iteration = 0
state.quality = {
    'separation': 0.82,
    'agreement': 0.91,
    'stability': 0.88,
    'overall': 0.87,
    'verdict': 'high',
    'explanation': 'Strong score separation, high detector agreement (Spearman 0.91), stable labels under contamination perturbation.',
}
state.next_action = {
    'action': 'report_to_user',
    'reason': 'High-quality results (0.87). 3 detectors agree on 95 of 100 anomaly labels.',
    'summary': '100 anomalies detected (10%) with high confidence. Top anomalies at indices [42, 87, 156, ...].',
    'confidence': 0.87,
}
```

### 4.4 Key Behaviors

**Multi-detector comparison (in `run()`):**

`run(state)` wraps the existing `run_detection(X, plan)` method, called once per plan in `state.plans`.

How it works:
1. For each plan in `state.plans`, call `self.run_detection(state.data, plan)`.
2. Collect results into `state.results` as `DetectorResult` dicts.
3. If a detector raises an exception, record `status='error'` with the error message and continue.
4. After all detectors finish, compute consensus from successful results.

**Consensus preconditions:**
- All successful detectors must produce scores of the same length (`n_samples`). This is guaranteed because they all fit on `state.data`.
- Scores are rank-normalized per detector (rank / n_samples) before averaging, so different score scales are comparable.
- Labels are majority-voted across detectors.

**Consensus computation:**
```
rank_scores[i] = rankdata(scores_i) / n_samples    # per detector
consensus_scores = mean(rank_scores, axis=0)        # across detectors
consensus_labels = (vote_count > n_detectors / 2).astype(int)
```

**Fallback for single detector:** consensus = that detector's raw scores and labels, agreement = 0.5.

**Fallback for all detectors erroring:** `state.results` is all errors, `state.consensus = None`, `next_action = 'confirm_with_user'` with explanation.

**How `plan(state)` wraps `plan_detection()`:**
- Calls `self.plan_detection(state.profile, priority, constraints)` to get primary plan with up to 2 alternatives.
- Extracts primary + alternatives into `state.plans` (up to 3 detectors).
- v1 limit: `max_detectors` is capped at 3 because `plan_detection()` returns at most 1 primary + 2 alternatives. Higher values are silently capped. Future versions may extend `plan_detection()` to support larger candidate sets.

**How `report(state)` wraps `generate_report()`:**
- Selects `best_idx = state.analysis['best_detector_index']`.
- Selects `best_result = state.results[best_idx]` (raw `run_detection()` output, fully compatible with `generate_report()`).
- Selects `best_analysis = state.analysis['per_detector_analysis'][best_idx]` (raw `analyze_results()` output for that detector).
- Calls `self.generate_report(best_result, best_analysis, format)` for the main report body. This is a direct wrapper with no contract mismatch — both inputs are exactly what the existing helpers produce.
- Prepends a session-level section with: consensus summary, detector agreement score, quality verdict, and disagreement highlights. This section is generated by `report()` itself, not by `generate_report()`.
- If format='json', constructs the best-detector section directly from `best_result` and `best_analysis` (bypasses `generate_report(format='json')` which returns a string). Returns a native Python dict:
  ```python
  {
      'session': {
          'consensus': state.consensus,
          'quality': state.quality,
          'comparison': {detector agreement, disagreements},
      },
      'best_detector': {
          'name': best_result['detector_name'],
          'scores': best_result['scores_train'].tolist(),
          'labels': best_result['labels_train'].tolist(),
          'threshold': best_result['threshold'],
          'analysis': best_analysis,
      },
  }
  ```
- If format='text', calls `self.generate_report(best_result, best_analysis, format='text')` for the main body (returns a string) and prepends the session-level section.

**How `analyze()` constructs `state.analysis`:**
- `consensus_analysis` is a `ConsensusAnalysis` dict (see schema in 4.3), built directly by `analyze()` from the consensus scores and labels. It is NOT produced by calling `analyze_results()` (since consensus has no `plan` or `threshold`).
- `per_detector_analysis` is a list positionally aligned with `state.results`. For each entry:
  - If `status='success'`: calls `self.analyze_results(result, X=state.data)` and stores the output. Fully compatible with `generate_report()`.
  - If `status='error'` or `'skipped'`: stores `None`.
- This alignment means `state.analysis['per_detector_analysis'][i]` always corresponds to `state.results[i]`, regardless of error/skip entries.
- **All-detectors-error path:** If every detector in `state.results` has `status='error'`, then `state.analysis = None`. In this case, `state.quality` is set to all-zeros with verdict `'low'`, and `state.next_action = {'action': 'confirm_with_user', 'reason': 'All detectors failed...'}`. Calling `report(state)` when `state.analysis is None` raises `ValueError("No successful detectors to report on. Use iterate() to adjust the plan.")`.

**Result quality assessment (in `analyze()`):**

Three metrics, each normalized to [0, 1]. No new dependencies beyond numpy and scipy (both already required by PyOD).

1. **Score separation** (`quality.separation`): ratio of mean anomaly score to mean inlier score, clipped to [0, 1].
   ```
   anomaly_mean = mean(scores[labels == 1])
   inlier_mean  = mean(scores[labels == 0])
   separation = clip(anomaly_mean / (inlier_mean + 1e-10) - 1, 0, 1)
   ```
   Values near 1.0 indicate anomalies have much higher scores (good). Near 0.0 means indistinguishable (bad).

2. **Detector agreement** (`quality.agreement`): mean pairwise Spearman rank correlation across detectors, clipped to [0, 1].
   ```
   from scipy.stats import spearmanr
   correlations = []
   for i in range(n_detectors):
       for j in range(i+1, n_detectors):
           rho, _ = spearmanr(scores_i, scores_j)
           correlations.append(max(0, rho))
   agreement = mean(correlations)
   ```
   For single-detector runs: agreement = 0.5 (neutral, neither high nor low confidence).

3. **Label stability** (`quality.stability`): Jaccard index of top-k anomaly sets when k varies by +/-20%.
   ```
   k = n_anomalies  # from contamination
   k_low  = max(1, int(k * 0.8))
   k_high = min(n_samples, int(k * 1.2))
   top_k     = set(argsort(scores)[-k:])
   top_k_low = set(argsort(scores)[-k_low:])
   top_k_high = set(argsort(scores)[-k_high:])
   stability = 0.5 * (jaccard(top_k, top_k_low) + jaccard(top_k, top_k_high))
   ```
   Uses the consensus scores. No re-fitting needed — just checks if the anomaly set is robust to the contamination threshold.

4. **Overall** (`quality.overall`): `mean(separation, agreement, stability)`.

5. **Verdict**: `'high'` if overall >= 0.7, `'medium'` if >= 0.4, `'low'` otherwise. Explanation string constructed from the three component values.

**Edge-case fallbacks:**
- If all consensus labels are 0 or all are 1 (empty anomaly/inlier set): `separation = 0.0` (no separation detected).
- If `spearmanr()` returns NaN (constant score vector): that pair contributes 0.0 to the agreement mean. If all pairs are NaN: `agreement = 0.0`.
- If `k = 0` (no anomalies found): `stability = 0.0`.
- For single-detector runs: `agreement = 0.5` (neutral — no basis for agreement or disagreement).
- If `state.consensus is None` (all detectors errored): all metrics = 0.0, verdict = `'low'`, `next_action = {'action': 'confirm_with_user', 'reason': 'All detectors failed. Check data format or try different detector family.'}`.
- `overall = mean(separation, agreement, stability)` — no values are excluded; edge-case fallbacks above ensure all three are always defined floats.

**Intelligent `next_action`:**
- After `analyze()` with high confidence: `{'action': 'report_to_user', 'summary': '...', 'confidence': 0.87}`
- After `analyze()` with low confidence: `{'action': 'iterate', 'reason': 'Detectors disagree (agreement=0.3); consider different algorithm family', 'suggestion': 'Exclude lowest-agreement detector and re-run'}`
- After `iterate()` with structured feedback: `{'action': 'run', 'reason': 'Plan adjusted: excluded IForest, added ECOD', 'adjustment': 'Excluded IForest, added ECOD'}`
- After `iterate()` with ambiguous NL: `{'action': 'confirm_with_user', 'reason': 'Interpreted "too many" as lower contamination', 'suggestion': 'Lower contamination from 0.1 to 0.05?', 'proposed_change': {...}}`

**Actionable `iterate()`:**

Two modes:

*Structured feedback (primary):* dict with a closed set of actions:
- `{"action": "adjust_contamination", "value": 0.05}` → updates contamination in all plans, re-runs
- `{"action": "exclude", "detectors": ["IForest"]}` → removes from plans, re-plans if needed
- `{"action": "include", "detectors": ["ECOD"]}` → adds to plans
- `{"action": "rerun"}` → same plans, fresh fit

These execute immediately and set `next_action` to `'run'`.

*NL feedback (best-effort):* string parsed via keyword matching:
- "too many false positives" → proposes `adjust_contamination` with lower value
- "try without X" → proposes `exclude`
- "missed anomalies" → proposes `adjust_contamination` with higher value

NL parsing assigns a confidence score. If confidence >= 0.8, executes immediately. If < 0.8, sets `next_action` to `'confirm_with_user'` with:
```python
next_action = {
    'action': 'confirm_with_user',
    'reason': 'Interpreted "too many false positives" as lower contamination.',
    'suggestion': 'Lower contamination from 0.1 to 0.05?',
    'proposed_change': {'action': 'adjust_contamination', 'value': 0.05},
}
```
The agent presents this to the user. On confirmation, the agent calls `iterate(state, proposed_change)` with the structured dict.

All iterations are logged in `state.history`. The engine tracks which detector-parameter combinations have been tried and avoids repeating them.

---

## 5. Skill Integration (od-expert)

The od-expert skill uses the session API to guide the conversation:

```
User: "Find anomalies in this sensor data"
Skill: state = engine.start(data)
       state = engine.plan(state)        # next_action='run'
       state = engine.run(state)         # next_action='analyze'
       state = engine.analyze(state)     # next_action='report_to_user'
Skill: presents state.next_action['summary'] to user

User: "Too many false positives"
Skill: state = engine.iterate(state, "too many false positives")
       # NL confidence=0.6 < 0.8, so next_action='confirm_with_user'
Skill: "I interpreted that as: lower contamination from 0.1 to 0.05. Proceed?"

User: "Yes"
Skill: state = engine.iterate(state, state.next_action['proposed_change'])
       # Structured dict, executes immediately → next_action='run'
       state = engine.run(state)
       state = engine.analyze(state)
Skill: presents updated results

User: "Good, give me the report"
Skill: report = engine.report(state)
```

The skill does not need OD knowledge. It follows `state.next_action` and translates between user and engine. When `next_action` is `'confirm_with_user'`, the skill presents the proposed change and waits for user approval before executing.

**Skill responsibilities:**
- Translate user intent to `iterate()` feedback
- Present results in human-readable format
- Decide when to show code vs just results
- Handle data loading (file paths, formats)

**Engine responsibilities:**
- All OD domain knowledge
- Workflow state tracking
- Algorithm selection, comparison, quality assessment
- Iteration logic (what to change based on feedback)

---

## 6. Backward Compatibility

**No breaking changes. All additions are strictly additive.**

- All existing ADEngine methods remain unchanged with identical signatures: `profile_data`, `plan_detection`, `run_detection`, `analyze_results`, `explain_findings`, `suggest_next_step`, `generate_report`, `detect`, `list_detectors`, `explain_detector`, `compare_detectors`, `get_benchmarks`.
- New session methods are additive: `start`, `plan`, `run`, `analyze`, `iterate`, `report`, `investigate`.
- The session execution method is named `run(state)` to avoid conflict with the existing `detect(X_train, ...)` one-shot method. Both coexist on the same class.
- `InvestigationState` is a new class in a new file (`pyod/utils/investigation.py`), no conflicts.
- Layer 1 (BaseDetector models) is untouched.

---

## 7. Scope and Non-Goals

**In scope for V3:**
- `InvestigationState` typed dataclass with closed enums and schemas
- Session workflow methods: `start`, `plan`, `run`, `analyze`, `iterate`, `report`, `investigate`
- Multi-detector comparison with rank-normalized consensus scoring
- Result quality assessment (separation, agreement, stability — exact formulas defined)
- Actionable iteration: structured-first with NL best-effort + confirmation
- Updated od-expert skill
- Tests and documentation

**Not in scope:**
- MCP server changes (Python-first; MCP can wrap session API later)
- Persistent cross-session memory (library stays stateless within Python session)
- New detectors (all shipped)
- Changes to BaseDetector or any detector classes
- AutoML / hyperparameter search
- UI or visualization

---

## 8. Implementation Feasibility

All new code lives in `pyod/utils/ad_engine.py` (session methods, ~400-500 lines) plus a new `pyod/utils/investigation.py` for `InvestigationState` (~50-100 lines). The od-expert skill update is documentation and workflow instructions (~100 lines).

| Component | Effort | Lines | Dependencies |
|-----------|--------|-------|-------------|
| `InvestigationState` dataclass + enums | Low | ~100 | None (new file `pyod/utils/investigation.py`) |
| Session methods (start, plan, run, analyze, iterate, report) | Medium | ~400 | Existing ADEngine methods |
| Multi-detector comparison + consensus | Medium | ~100 | Existing `run_detection`, scipy.stats.rankdata, scipy.stats.spearmanr |
| Result quality assessment (3 metrics) | Medium | ~80 | numpy, scipy.stats (both already required) |
| `investigate()` convenience | Low | ~20 | Session methods |
| od-expert skill update | Low | ~100 | Skill markdown |
| Tests | Medium | ~200 | pytest |
| Documentation | Low | ~50 | README, CHANGES |

Total: ~1000 lines of new code, all additive.

---

## 9. Codex Review Resolution (Round 1)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | Blocker: session `detect(state)` conflicts with existing `detect(X_train, ...)` | **Resolved** | Renamed to `run(state)`. Existing `detect()` unchanged. No naming conflict. |
| 2 | Blocker: quality assessment under-defined (dip test not in scipy, stability vague) | **Resolved** | Replaced with 3 exact metrics: score separation (ratio), detector agreement (Spearman), label stability (Jaccard). All use numpy/scipy already required. No new dependencies. |
| 3 | High: `iterate()` overcommits on NL feedback | **Resolved** | Structured dict feedback is primary (executes immediately). NL feedback is best-effort with confidence score; if < 0.8, returns `'confirm_with_user'` action for agent to present to user. |
| 4 | Medium: `InvestigationState` schema is open-ended | **Resolved** | Added closed `PHASES` and `ACTION_TYPES` enums. Defined typed schemas for `HistoryEntry`, `DetectorResult`, `ConsensusResult`, `QualityAssessment`, `NextAction`. Added 2 worked examples (after `plan()` and after `analyze()`). |
| 5 | Medium: multi-detector flow underspecified against existing helpers | **Resolved** | Defined how `plan()` wraps `plan_detection()`, how `run()` wraps `run_detection()` per-plan with error handling, how `report()` wraps `generate_report()`. Defined consensus preconditions (same n_samples, rank normalization) and fallback for single/error cases. |

## 10. Codex Review Resolution (Round 2)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | Blocker: `next_action` protocol inconsistent — stale `detect`/`iterate` values outside enum | **Resolved** | Added `'iterate'` to `ACTION_TYPES` enum. Replaced all stale `detect` references with `run`. Fixed `investigate()` docstring. Updated all `next_action` examples to use only enum values. |
| 2 | High: `DetectorResult` mismatches `run_detection()` schema; `state.analysis` untyped | **Resolved** | `DetectorResult` is now a superset of `run_detection()` output (stores raw result verbatim). Added `InvestigationAnalysis` typed schema with `consensus_analysis`, `per_detector_analysis`, `best_detector` (selected by Spearman correlation with consensus). Updated `report()` to use typed `best_detector_index`. |
| 3 | High: quality metrics undefined for edge cases | **Resolved** | Added explicit fallback values: empty labels → separation=0.0, NaN Spearman → 0.0, k=0 → stability=0.0, single detector → agreement=0.5, all errors → all metrics=0.0 with `confirm_with_user`. All three metrics always produce defined floats. |
| 4 | Medium: `max_detectors` overstated vs `plan_detection()` | **Resolved** | Capped at 3 in v1 (matches `plan_detection()` output: 1 primary + 2 alternatives). Documented as v1 limit, silently caps higher values. |

## 11. Codex Review Resolution (Round 3)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | Medium: `best_detector` no tie-break or degenerate fallback | **Resolved** | Added deterministic fallback chain: (1) highest finite Spearman, (2) highest plan confidence, (3) fastest runtime, (4) first successful if all NaN. Single-detector case explicit. |
| 2 | Medium: `proposed_change` untyped, `suggestion` field scope unclear | **Resolved** | Added typed `StructuredFeedback` schema with closed action names and required fields. `NextAction` payload documented per action type: `suggestion` valid for `iterate` and `confirm_with_user`, `proposed_change` is a `StructuredFeedback`. |
| 3 | High (still open from R2): `report()` calls `analyze_results()` on consensus but consensus lacks `plan`/`threshold` | **Resolved** | `report()` now uses `per_detector_analysis[best_idx]` (fully compatible with `generate_report()`). `consensus_analysis` is a lightweight summary built by `analyze()` directly, not by calling `analyze_results()`. Session-level consensus info rendered in a separate section prepended to the report. |

## 12. Codex Review Resolution (Round 4)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | High: `per_detector_analysis` index misaligns with `state.results` when detectors error | **Resolved** | `per_detector_analysis` is now positionally aligned with `state.results`. Error/skipped entries get `None`. `best_detector_index` always points to a successful entry in both lists. |
| 2 | Medium: NextAction payload fields not marked required vs optional per action type | **Resolved** | Each action type now documents R (required) vs O (optional) fields. `confirm_with_user` has optional `suggestion`/`proposed_change` (present for change confirmation, absent for error/retry). |
| 3 | Medium: `consensus_analysis` schema comment still said `analyze_results() output` | **Resolved** | Defined typed `ConsensusAnalysis` schema. `InvestigationAnalysis` references it by name. Behavior section confirms it is built directly by `analyze()`, not via `analyze_results()`. |

## 13. Codex Review Resolution (Round 5)

| # | Finding | Status | Resolution |
|---|---------|--------|------------|
| 1 | High: `generate_report(format='json')` returns JSON string, not dict — wrapper contract broken | **Resolved** | JSON path now bypasses `generate_report()` and constructs a native Python dict directly from `best_result` and `best_analysis`. Text path still wraps `generate_report(format='text')`. |
| 2 | Medium: all-detectors-error leaves `state.analysis` undefined | **Resolved** | `state.analysis = None` when all detectors error. `state.quality` set to all-zeros with verdict `'low'`. `report(state)` raises `ValueError` when `state.analysis is None`. |
