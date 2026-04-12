# -*- coding: utf-8 -*-
"""Example: Agent-driven anomaly detection with iteration (Layer 3).

Demonstrates the full agentic workflow on a real medical dataset
(cardiotocography — fetal heart rate monitoring). An AI agent
investigates anomalies, responds to user feedback, iterates,
and produces a final report.

Dataset: UCI Cardiotocography (1,831 recordings, 21 clinical
features, ~9.6% pathological cases). Shipped with PyOD at
examples/data/cardio.csv.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))

import numpy as np
from pyod.utils.ad_engine import ADEngine

if __name__ == "__main__":
    # Load real cardiotocography dataset
    data_path = os.path.join(
        os.path.dirname(__file__), 'data', 'cardio.csv')
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    X = data[:, :-1]  # 1831 samples, 21 features
    y = data[:, -1].astype(int)  # ground truth (for validation only)

    engine = ADEngine()

    # --- Turn 1: User asks for anomaly detection ---
    print("=" * 60)
    print("User: I have a cardiotocography dataset with %d fetal"
          % X.shape[0])
    print("      heart rate recordings and %d clinical features."
          % X.shape[1])
    print("      Find abnormal cases that might indicate fetal")
    print("      distress.")
    print("=" * 60)
    print()

    state = engine.investigate(X)
    n_det = state.consensus['n_detectors']
    n_anom = int(state.consensus['labels'].sum())
    agreement = state.consensus['agreement']
    verdict = state.quality['verdict']
    detectors = [r['detector_name'] for r in state.results
                 if r['status'] == 'success']

    print("Agent: I profiled your data: %d samples, %d features "
          "(tabular)." % (state.profile['n_samples'],
                          state.profile['n_features']))
    print("       Running %d detectors: %s..."
          % (n_det, ', '.join(detectors)))
    print()
    print("       %d anomalies detected (%.1f%%)."
          % (n_anom, 100.0 * n_anom / X.shape[0]))
    print("       Detector agreement: %.2f (Spearman)."
          % agreement)
    print("       Result quality: %s (%.2f)."
          % (verdict, state.quality['overall']))
    print()

    # Show top anomalies
    top5 = state.analysis['consensus_analysis']['top_anomalies'][:5]
    print("       Top anomalous recordings:")
    for a in top5:
        print("         Case #%d: consensus score %.3f"
              % (a['index'], a['score']))
    print()

    # --- Turn 2: User says too many ---
    print("=" * 60)
    print("User: %d is too many for clinical review. Show only"
          % n_anom)
    print("      the top 3%% most critical cases.")
    print("=" * 60)
    print()

    state = engine.iterate(
        state, {"action": "adjust_contamination", "value": 0.03})
    state = engine.run(state)
    state = engine.analyze(state)
    n_anom2 = int(state.consensus['labels'].sum())

    print("Agent: Adjusted threshold to top 3%%.")
    print("       %d critical cases flagged for review." % n_anom2)
    print("       Quality: %s (%.2f)."
          % (state.quality['verdict'], state.quality['overall']))
    print()

    # --- Turn 3: User asks what features drive the anomalies ---
    print("=" * 60)
    print("User: What clinical features are driving these?")
    print("=" * 60)
    print()

    best_idx = state.analysis['best_detector_index']
    best_result = state.results[best_idx]
    explanations = engine.explain_findings(
        best_result, X=X, top_k=3)

    print("Agent: Top 3 anomalous cases from %s:"
          % best_result['detector_name'])
    print()
    for exp in explanations:
        idx = exp['index']
        print("  Case #%d (score %.3f, %s):"
              % (idx, exp['score'], exp['label']))
        if 'contributing_features' in exp:
            for cf in exp['contributing_features'][:3]:
                print("    - Feature %d: z-score %.1f (value: %.2f)"
                      % (cf['feature'], cf['z_score'],
                         X[idx, cf['feature']]))
        print()

    # --- Turn 4: User asks for report ---
    print("=" * 60)
    print("User: Good analysis. Generate the report.")
    print("=" * 60)
    print()

    report = engine.report(state, format='text')
    print("Agent: Here is the investigation report:")
    print()
    print(report)

    # Validation against ground truth (not part of the agentic
    # workflow — shown here for demonstration purposes only)
    print()
    print("--- Validation (ground truth) ---")
    flagged = state.consensus['labels'].astype(bool)
    true_pos = int((flagged & y.astype(bool)).sum())
    print("True positives in flagged set: %d / %d (%.0f%%)"
          % (true_pos, int(flagged.sum()),
             100 * true_pos / max(int(flagged.sum()), 1)))
