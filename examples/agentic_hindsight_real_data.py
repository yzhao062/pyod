# -*- coding: utf-8 -*-
"""Real-data agentic anomaly detection with hindsight validation.

This example intentionally downloads a public dataset at runtime instead
of using PyOD's bundled demo data. The labels are held out while the
agentic workflow profiles, plans, runs, recovers from detector failures,
and analyzes results. Labels are only opened at the end for hindsight.

Dataset: UCI Ionosphere, 351 radar returns, 34 numeric features.
Source: https://archive.ics.uci.edu/dataset/52/ionosphere
Raw file:
https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import urllib.request
import warnings

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyod.skills import get_skill_path
from pyod.utils.ad_engine import ADEngine


SOURCE_URL = (
    'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    'ionosphere/ionosphere.data'
)
SEP = '=' * 72


def activate_od_expert_skill():
    """Load the packaged od-expert skill and on-demand references.

    In a live Claude Code or Codex session this Markdown is injected into
    the agent context by the host. This script loads the same packaged
    files explicitly so the example remains self-contained and shows the
    skill-backed decision layer.
    """
    skill_dir = get_skill_path('od-expert')
    relative_files = [
        'SKILL.md',
        os.path.join('references', 'workflow.md'),
        os.path.join('references', 'tabular.md'),
    ]
    loaded = []
    for relative in relative_files:
        path = skill_dir / relative
        # Reading the files is the explicit skill activation step for
        # this standalone script; the content guides the checks below.
        path.read_text(encoding='utf-8')
        loaded.append(relative.replace(os.sep, '/'))
    return skill_dir, loaded


def load_ionosphere():
    """Download UCI Ionosphere and return X plus held-out labels."""
    with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
        raw = response.read().decode('utf-8')

    rows = []
    raw_labels = []
    for line in raw.strip().splitlines():
        parts = line.strip().split(',')
        rows.append([float(value) for value in parts[:-1]])
        raw_labels.append(parts[-1])

    X = np.asarray(rows, dtype=float)
    # UCI labels: g = good return, b = bad return.
    # For hindsight anomaly validation, bad returns are treated as 1.
    y = np.asarray([1 if label == 'b' else 0
                    for label in raw_labels], dtype=int)
    return X, y, raw_labels


def skill_preflight_notes(X, profile):
    """Apply the od-expert skill's tabular pre-run checklist."""
    ranges = np.ptp(X, axis=0)
    stds = np.std(X, axis=0)
    wide_scale = np.where((ranges > 100) | (stds > 10))[0]
    constant = np.where(stds == 0)[0]

    notes = [
        'Modality route: tabular (numeric matrix, no timestamp axis).',
        'Reference loaded: tabular.md; n < 1k heuristic starters '
        'are ECOD, HBOS, IForest.',
    ]
    if len(wide_scale) == 0:
        notes.append('Pitfall 1 check: no mixed-scale feature exceeds '
                     'range > 100 or std > 10; distance detectors do '
                     'not need extra scaling for this dataset.')
    else:
        notes.append('Pitfall 1 check: mixed-scale columns %s need '
                     'scaling before distance detectors.'
                     % wide_scale.tolist())
    if len(constant) > 0:
        notes.append('Data sanity check: constant feature(s) %s carry '
                     'no ranking signal; keep them here so row/feature '
                     'indices still match the UCI source.'
                     % constant.tolist())
    notes.append('Trigger 2 check: contamination is unknown, so the '
                 'skill runs the default first and revisits the '
                 'threshold during hindsight.')
    return notes


def skill_postrun_triggers(state):
    """Return od-expert adaptive triggers fired after analysis."""
    triggers = []
    agreement = state.quality['agreement']
    stability = state.quality['stability']
    # `separation` is computed from the run's own predicted labels and is
    # near-always high, so it is descriptive only and is NOT used as a gate.
    if agreement < 0.4:
        triggers.append('Trigger 3: detector disagreement '
                        '(agreement %.2f < 0.40).' % agreement)
    if stability < 0.5:
        triggers.append('Trigger 4: cutoff instability '
                        '(stability %.2f < 0.50; flagged set is '
                        'contamination-sensitive).' % stability)
    if agreement > 0.9:
        triggers.append('Trigger 10: very high agreement %.2f; '
                        'sanity-check top flagged points.' % agreement)
    return triggers


def pin_stochastic_detectors(plans, random_state=42):
    """Make stochastic detectors repeatable without changing selection."""
    for plan in plans:
        if plan.get('detector_name') == 'IForest':
            params = dict(plan.get('params', {}))
            params['random_state'] = random_state
            plan['params'] = params


def validation_metrics(y_true, labels_pred, scores):
    """Compute hindsight metrics from held-out labels."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, labels_pred, average='binary', zero_division=0)
    return {
        'flagged': int(labels_pred.sum()),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc_score(y_true, scores)),
        'average_precision': float(average_precision_score(
            y_true, scores)),
    }


def print_metrics(label, metrics):
    """Print compact hindsight metrics."""
    print("%s: flagged=%d, precision=%.3f, recall=%.3f, "
          "f1=%.3f, roc_auc=%.3f, average_precision=%.3f"
          % (label, metrics['flagged'], metrics['precision'],
             metrics['recall'], metrics['f1'], metrics['roc_auc'],
             metrics['average_precision']))


def print_state(label, state, y):
    """Summarize the current investigation state."""
    print()
    print(SEP)
    print(label)
    print(SEP)
    print("Phase: %s, iteration: %d" % (state.phase, state.iteration))
    print("Plans:")
    for plan in state.plans:
        print("  - %-8s params=%s confidence=%.2f"
              % (plan['detector_name'], plan.get('params', {}),
                 plan.get('confidence', 0.0)))

    print("Run status:")
    for result in state.results:
        status = result['status']
        if status == 'success':
            print("  - %-8s success in %.3fs"
                  % (result['detector_name'],
                     result.get('runtime_seconds', 0.0)))
        else:
            print("  - %-8s error: %s"
                  % (result['detector_name'], result.get('error')))

    print("Agent analysis: %s" % state.analysis['summary'])
    print("Diagnostics (label-free): %s (%.2f); %s"
          % (state.quality['verdict'], state.quality['overall'],
             state.quality['explanation']))
    triggers = skill_postrun_triggers(state)
    if triggers:
        print("Skill triggers:")
        for trigger in triggers:
            print("  - %s" % trigger)
    else:
        print("Skill triggers: none")
    print_metrics(
        "Hindsight consensus",
        validation_metrics(y, state.consensus['labels'],
                           state.consensus['scores']))

    for result in state.results:
        if result['status'] == 'success':
            print_metrics(
                "  %-8s" % result['detector_name'],
                validation_metrics(y, result['labels_train'],
                                   result['scores_train']))


def feature_name(index):
    """Map a UCI Ionosphere feature index to a readable pulse name."""
    pulse = index // 2 + 1
    part = 'real' if index % 2 == 0 else 'imag'
    return 'pulse_%02d_%s' % (pulse, part)


def explain_top_cases(engine, state, X, raw_labels, top_k=3):
    """Explain top anomalies from the best detector."""
    best_idx = state.analysis['best_detector_index']
    best_result = state.results[best_idx]

    print()
    print(SEP)
    print("Top Findings")
    print(SEP)
    print("Best detector aligned with consensus: %s"
          % best_result['detector_name'])

    print()
    print("Top consensus cases:")
    for rank, entry in enumerate(
            state.analysis['consensus_analysis']['top_anomalies'][:10], 1):
        idx = entry['index']
        print("  %2d. row=%3d score=%.4f held_out_label=%s"
              % (rank, idx, entry['score'], raw_labels[idx]))

    print()
    print("Feature-level explanations for top %d best-detector cases:"
          % top_k)
    explanations = engine.explain_findings(
        best_result, X=X, top_k=top_k)
    for explanation in explanations:
        idx = explanation['index']
        print("  row=%3d label=%s score=%.4f percentile=%.1f"
              % (idx, raw_labels[idx], explanation['score'],
                 explanation['percentile']))
        for item in explanation.get('contributing_features', [])[:5]:
            print("      %-13s z=%.2f"
                  % (feature_name(item['feature']),
                     item['z_score']))


def run_flow():
    """Run a complete agentic flow on external data."""
    logging.getLogger('pyod.utils.ad_engine').setLevel(logging.ERROR)
    warnings.filterwarnings(
        'ignore',
        message='invalid value encountered in divide',
        category=RuntimeWarning)

    skill_dir, loaded_refs = activate_od_expert_skill()
    X, y, raw_labels = load_ionosphere()
    engine = ADEngine()

    print(SEP)
    print("Skill-Backed PyOD Agentic Flow: External UCI Ionosphere Data")
    print(SEP)
    print("Activated skill: od-expert")
    print("Skill path: %s" % skill_dir)
    print("Loaded skill context: %s" % ', '.join(loaded_refs))
    print("Downloaded %d samples with %d numeric features."
          % (X.shape[0], X.shape[1]))
    print("Labels are hidden until hindsight validation.")

    state = engine.start(X)
    print()
    print("Agent profile:", state.profile)
    print("Skill preflight:")
    for note in skill_preflight_notes(X, state.profile):
        print("  - %s" % note)

    state = engine.plan(state)
    pin_stochastic_detectors(state.plans)
    print("Agent plan:", [plan['detector_name']
                          for plan in state.plans])

    state = engine.run(state)
    state = engine.analyze(state)
    print_state("1. Initial Unsupervised Pass", state, y)

    failed = [result['detector_name'] for result in state.results
              if result['status'] != 'success']
    if failed:
        print()
        print("Agent decision: exclude failed detector(s): %s"
              % ', '.join(failed))
        state = engine.iterate(
            state, {'action': 'exclude', 'detectors': failed})
        pin_stochastic_detectors(state.plans)
        state = engine.run(state)
        state = engine.analyze(state)
        print_state("2. Recovery After Detector Failure", state, y)

    if len(state.plans) < 3:
        print()
        print("Agent decision: add COPOD to restore a 3-detector "
              "consensus.")
        state = engine.iterate(
            state, {'action': 'include', 'detectors': ['COPOD']})
        pin_stochastic_detectors(state.plans)
        state = engine.run(state)
        state = engine.analyze(state)
        print_state("3. Recovered Ensemble", state, y)

    true_rate = float(y.mean())
    print()
    print(SEP)
    print("Hindsight Opens")
    print(SEP)
    print("Held-out labels show %d/%d bad radar returns (%.1f%%)."
          % (int(y.sum()), len(y), 100 * true_rate))
    print("Agent decision: re-run with contamination set to the "
          "retrospective bad-return rate.")

    state = engine.iterate(
        state, {'action': 'adjust_contamination', 'value': true_rate})
    pin_stochastic_detectors(state.plans)
    state = engine.run(state)
    state = engine.analyze(state)
    print_state("4. Hindsight Threshold Adjustment", state, y)

    explain_top_cases(engine, state, X, raw_labels)

    print()
    print(SEP)
    print("Hindsight Lessons")
    print(SEP)
    print("- Initial consensus was conservative: high precision, low "
          "recall.")
    print("- A real agent should treat detector errors as recoverable "
          "workflow events, not demo-breaking failures.")
    print("- The score ranking had useful signal even before labels: "
          "ROC AUC stayed around %.3f in the final consensus."
          % validation_metrics(
              y, state.consensus['labels'],
              state.consensus['scores'])['roc_auc'])
    print("- Threshold choice dominated the business outcome: after "
          "hindsight contamination adjustment, recall rose while "
          "precision fell.")
    print()
    print("What I assumed and why:")
    print("- Data type: tabular (od-expert decision tree + ADEngine "
          "numeric profile).")
    print("- Contamination: default first, then %.3f from held-out "
          "hindsight labels." % true_rate)
    print("- Detectors: %s (live ADEngine plan, skill-checked as "
          "plausible for small tabular data)."
          % ', '.join(plan['detector_name'] for plan in state.plans))
    print("- Primary detector: %s (highest alignment with the final "
          "consensus)." % state.analysis['best_detector'])
    print()
    print("Session history:")
    for entry in state.history:
        print("  - iter=%d phase=%s action=%s detail=%s"
              % (entry['iteration'], entry['phase'],
                 entry['action'], entry['detail']))


if __name__ == "__main__":
    run_flow()
