# -*- coding: utf-8 -*-
"""Example: Agent-driven anomaly detection with PyOD 3 (Layer 3).

Demonstrates what makes PyOD distinctive in the agentic workflow:
    - 60+ detectors across 5 data modalities
    - Benchmark-backed detector selection (ADBench, TSB-AD, BOND)
    - Multi-detector consensus with per-detector scores
    - Result quality assessment
    - Multi-modal: same API for tabular, time series, graph, text, image

Dataset: UCI Cardiotocography (1,831 recordings, 21 clinical
features). Shipped with PyOD at examples/data/cardio.csv.
"""
# Author: Yue Zhao <yzhao062@gmail.com>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from pyod.utils.ad_engine import ADEngine

SEP = "=" * 64


def show_detectors_with_benchmark(engine, data_type='tabular'):
    """Show which detectors PyOD's knowledge base ranks for this data."""
    detectors = engine.list_detectors(data_type=data_type)
    return detectors


if __name__ == "__main__":
    # Pin randomness so the demo conversation is reproducible.
    np.random.seed(42)

    # Load real cardiotocography dataset
    data_path = os.path.join(
        os.path.dirname(__file__), 'data', 'cardio.csv')
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    X = data[:, :-1]  # 1831 samples, 21 features
    y = data[:, -1].astype(int)  # ground truth (for validation)

    engine = ADEngine()

    # --- Turn 1: User asks for anomaly detection ---
    print(SEP)
    print("User: I have a cardiotocography dataset with %d fetal"
          % X.shape[0])
    print("      heart rate recordings, %d clinical features."
          % X.shape[1])
    print("      Find abnormal cases.")
    print(SEP)
    print()

    state = engine.investigate(X)
    successful = [r for r in state.results if r['status'] == 'success']
    n_total = len(engine.list_detectors())

    print("Agent: Profiled as tabular data. PyOD has %d+ detectors;"
          % n_total)
    print("       based on ADBench benchmark (NeurIPS 2022, 57")
    print("       datasets), I selected the top 3 for your profile:")
    print()
    for r in successful:
        name = r['detector_name']
        info = engine.explain_detector(name)
        short_desc = info.get('full_name', name)
        print("       - %-8s  %s" % (name, short_desc))
    print()

    # Show consensus + per-detector scores
    n_anom = int(state.consensus['labels'].sum())
    agreement = state.consensus['agreement']
    print("       Running all 3 in parallel...")
    print("       Consensus: %d anomalies (%.1f%%), agreement %.2f."
          % (n_anom, 100.0 * n_anom / X.shape[0], agreement))
    print("       Result quality: %s (%.2f)."
          % (state.quality['verdict'], state.quality['overall']))
    print()

    # Show top anomaly with per-detector scores
    top1 = state.analysis['consensus_analysis']['top_anomalies'][0]
    top_idx = top1['index']
    print("       Top case #%d (each detector agrees):" % top_idx)
    for r in successful:
        name = r['detector_name']
        score = float(r['scores_train'][top_idx])
        threshold = r['threshold']
        max_score = float(r['scores_train'].max())
        pctile = 100.0 * (r['scores_train'] <= score).mean()
        print("         %-8s  score %7.4f  "
              "(threshold %.4f, %.0fth percentile)"
              % (name, score, threshold, pctile))
    print()

    # --- Turn 2: User says too many ---
    print(SEP)
    print("User: %d is too many for clinical review. Show only"
          % n_anom)
    print("      the top 3%%.")
    print(SEP)
    print()

    state = engine.iterate(
        state, {"action": "adjust_contamination", "value": 0.03})
    state = engine.run(state)
    state = engine.analyze(state)
    n_anom2 = int(state.consensus['labels'].sum())

    print("Agent: Re-running all 3 detectors with contamination=0.03.")
    print("       Each detector recomputes its threshold; consensus")
    print("       is majority vote on labels.")
    print()
    print("       %d cases flagged (%.1f%%). Quality: %s (%.2f)."
          % (n_anom2, 100.0 * n_anom2 / X.shape[0],
             state.quality['verdict'], state.quality['overall']))
    print()

    # --- Turn 3: User asks what features drive it ---
    print(SEP)
    print("User: What clinical features are driving these?")
    print(SEP)
    print()

    best_idx = state.analysis['best_detector_index']
    best_result = state.results[best_idx]
    best_name = best_result['detector_name']
    explanations = engine.explain_findings(
        best_result, X=X, top_k=1)

    print("Agent: Analyzing top case #%d via %s (best agreement "
          "with consensus):" % (top_idx, best_name))
    print()
    if explanations and 'contributing_features' in explanations[0]:
        print("       Feature contributions (z-score deviation):")
        for cf in explanations[0]['contributing_features'][:5]:
            print("         feature_%-2d  z=%.2f  (value: %.2f)"
                  % (cf['feature'], cf['z_score'],
                     X[explanations[0]['index'], cf['feature']]))
    print()
    print("       Cardiotocography dataset features include")
    print("       fetal heart rate variability and deceleration")
    print("       patterns; high z-scores in these suggest")
    print("       possible fetal distress per FIGO guidelines.")
    print()

    # --- Turn 4: Multi-modal, user asks about time series ---
    print(SEP)
    print("User: What if I had continuous time-series of these")
    print("      metrics instead of snapshots?")
    print(SEP)
    print()

    ts_detectors = engine.list_detectors(data_type='time_series')
    print("Agent: Same API, different detectors. PyOD has %d"
          % len(ts_detectors))
    print("       time-series detectors. TSB-AD benchmark")
    print("       (NeurIPS 2024, 1,070 datasets) ranks these top:")
    print()
    for d in ts_detectors[:4]:
        name = d.get('name', '?')
        full_name = d.get('full_name', name)
        print("       - %-20s  %s" % (name, full_name))
    print()
    print("       Just call engine.investigate(ts_data) and I'll")
    print("       run them. Same workflow, same session state.")
    print()

    # --- Turn 5: Final report ---
    print(SEP)
    print("User: Good analysis. Generate the report.")
    print(SEP)
    print()

    report = engine.report(state, format='text')
    print("Agent:")
    print(report)

    # --- Validation (not part of workflow) ---
    print()
    print("-" * 64)
    print("Validation against ground truth (not shown to agent):")
    flagged = state.consensus['labels'].astype(bool)
    true_pos = int((flagged & y.astype(bool)).sum())
    precision = true_pos / max(int(flagged.sum()), 1)
    recall = true_pos / max(int(y.sum()), 1)
    print("  Precision: %d/%d = %.1f%%"
          % (true_pos, int(flagged.sum()), 100 * precision))
    print("  Recall:    %d/%d = %.1f%%"
          % (true_pos, int(y.sum()), 100 * recall))
