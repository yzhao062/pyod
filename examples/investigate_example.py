# -*- coding: utf-8 -*-
"""Example: One-shot anomaly investigation with ADEngine (Layer 2).

ADEngine profiles your data, selects the best detectors based on
benchmark evidence, runs them, computes consensus scores, and
assesses result quality, all in one call.

This is the recommended approach when you do not know which
detector to use. For manual detector selection, see
knn_example.py or ecod_example.py (Layer 1). For agent-driven
workflows with user feedback, see agentic_example.py (Layer 3).

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

if __name__ == "__main__":
    # Pin randomness so the demo output is reproducible.
    np.random.seed(42)

    # Load UCI Cardiotocography: 1,831 recordings, 21 clinical
    # features, ground-truth labels in the last column (1 = abnormal).
    data_path = os.path.join(
        os.path.dirname(__file__), 'data', 'cardio.csv')
    data = np.genfromtxt(data_path, delimiter=',', skip_header=1)
    X = data[:, :-1]
    y_true = data[:, -1].astype(int)

    # One call does everything: profile, plan, run top-3, analyze.
    engine = ADEngine()
    state = engine.investigate(X)

    # Results summary
    print("=== Investigation Summary ===")
    print(state.analysis['summary'])
    print()

    # Quality assessment
    q = state.quality
    print("=== Result Quality ===")
    print("  Verdict:             %s (%.2f)" % (q['verdict'], q['overall']))
    print("  Score separation:    %.2f" % q['separation'])
    print("  Detector agreement:  %.2f" % q['agreement'])
    print("  Label stability:     %.2f" % q['stability'])
    print()

    # Consensus across detectors
    c = state.consensus
    print("=== Detector Consensus ===")
    print("  Detectors used:      %d" % c['n_detectors'])
    print("  Agreement (Spearman): %.2f" % c['agreement'])
    print("  Disagreement on %d samples" % len(c['disagreements']))
    print()

    # Top anomalies (consensus ranking)
    print("=== Top 5 Anomalies (consensus) ===")
    for a in state.analysis['consensus_analysis']['top_anomalies'][:5]:
        true_label = y_true[a['index']]
        tag = "abnormal" if true_label == 1 else "normal"
        print("  Index %4d, consensus score %.4f  (ground truth: %s)" % (
            a['index'], a['score'], tag))
    print()

    # Best single detector
    print("=== Best Detector ===")
    print("  %s (index %d)" % (
        state.analysis['best_detector'],
        state.analysis['best_detector_index']))

    # Next action recommendation
    print()
    print("=== Next Action ===")
    print("  %s: %s" % (
        state.next_action['action'],
        state.next_action['reason']))
