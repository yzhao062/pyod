"""Quality metrics for ADEngine result analysis.

Pure helper functions extracted from `pyod.utils.ad_engine.ADEngine` in
2026-05 (issue #667 follow-up). Not part of the public API; the
leading underscore on the module name is the contract.
"""

import numpy as np
from scipy.stats import rankdata, spearmanr


def compute_quality(scores, labels, results, consensus):
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
    n_anomalies = int(labels.sum())
    n_samples = len(labels)
    # Non-finite scores poison both separation (mean) and stability
    # (sort + std). Short-circuit both to refuse to emit NaN.
    nonfinite_scores = not np.all(np.isfinite(scores))

    # Separation
    if (nonfinite_scores or n_anomalies == 0
            or n_anomalies == n_samples):
        separation = 0.0
    else:
        anomaly_mean = float(np.mean(scores[labels == 1]))
        inlier_mean = float(np.mean(scores[labels == 0]))
        separation = float(np.clip(
            anomaly_mean / (inlier_mean + 1e-10) - 1, 0, 1))

    # Agreement (from consensus)
    agreement = float(consensus.get('agreement', 0.5))

    # Stability: standardized score gap at the rank-k cutoff.
    # Replaces the v1 Jaccard-of-nested-top-k formula which was
    # mathematically constant (issue #667).
    if (n_anomalies == 0 or n_anomalies >= n_samples
            or nonfinite_scores):
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

    overall = float(np.mean([separation, agreement, stability]))
    if overall >= 0.7:
        verdict = 'high'
    elif overall >= 0.4:
        verdict = 'medium'
    else:
        verdict = 'low'

    return {
        'separation': separation,
        'agreement': agreement,
        'stability': stability,
        'overall': overall,
        'verdict': verdict,
        'explanation': 'separation={:.2f}, agreement={:.2f}, '
                       'stability={:.2f} (cutoff gap)'.format(
                           separation, agreement, stability),
    }


def select_best_detector(results, consensus_scores):
    """Select best detector via Spearman with consensus.

    Fallback chain (per spec):

    1. Highest finite Spearman correlation against consensus.
    2. If tied: highest plan confidence.
    3. If still tied: fastest runtime.
    4. If ALL correlations are non-finite: first successful detector.

    Parameters
    ----------
    results : list of dict
        Per-detector result dicts from `ADEngine.run()`. Each dict has
        keys ``'status'``, ``'scores_train'``, ``'detector_name'``,
        ``'plan'`` (with ``'confidence'``), and ``'runtime_seconds'``.
        Only entries with ``status == 'success'`` are considered.
    consensus_scores : np.ndarray, shape (n_samples,)
        Consensus anomaly scores from `compute_consensus`.

    Returns
    -------
    int
        Index into `results` of the best-aligned successful detector.
        When only one detector succeeds, returns its index. When all
        Spearman correlations are non-finite, returns the index of the
        first successful detector.
    """
    successful = [
        (i, r) for i, r in enumerate(results)
        if r['status'] == 'success']
    if len(successful) == 1:
        return successful[0][0]

    # Compute Spearman for each successful detector
    rhos = []
    for i, r in successful:
        rho, _ = spearmanr(r['scores_train'], consensus_scores)
        rhos.append(float(rho) if np.isfinite(rho) else None)

    # If ALL NaN: return first successful (spec rule 4)
    if all(rho is None for rho in rhos):
        return successful[0][0]

    # Find best by finite Spearman, then tie-break
    best_j = 0  # index into successful list
    best_rho = -1.0
    for j, (i, r) in enumerate(successful):
        rho = rhos[j]
        if rho is None:
            continue
        if rho > best_rho:
            best_rho = rho
            best_j = j
        elif rho == best_rho:
            # Tie-break: plan confidence
            curr_conf = r.get('plan', {}).get('confidence', 0)
            prev_conf = successful[best_j][1].get(
                'plan', {}).get('confidence', 0)
            if curr_conf > prev_conf:
                best_j = j
            elif curr_conf == prev_conf:
                # Tie-break: fastest
                if r.get('runtime_seconds', 999) < successful[
                        best_j][1].get('runtime_seconds', 999):
                    best_j = j
    return successful[best_j][0]


def compute_feature_importance(result, X):
    """Estimate per-feature contribution to anomaly scores.

    For each feature column, computes the Pearson correlation between
    the column's absolute z-scores and the detector's anomaly scores.
    Higher absolute correlation indicates the feature drives the
    detector's ranking. Failures (non-2D `X`, length mismatch, or any
    exception during computation) return ``None`` rather than raise.

    Parameters
    ----------
    result : dict
        Detector result dict. Must contain ``'scores_train'`` (an
        ndarray of length ``n_samples``).
    X : array-like, shape (n_samples, n_features)
        Training data the detector was fit on.

    Returns
    -------
    list of float or None
        One importance per feature, in column order. Each value is in
        [-1, 1]; non-finite correlations are coerced to 0.0. Returns
        ``None`` if `X` is not 2D, if length of scores does not match
        `X.shape[0]`, or if any error occurs during computation.
    """
    try:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            return None
        scores = result['scores_train']
        if len(scores) != X_arr.shape[0]:
            return None

        means = np.mean(X_arr, axis=0)
        stds = np.std(X_arr, axis=0)
        stds[stds == 0] = 1.0
        z_scores = np.abs((X_arr - means) / stds)

        importances = []
        for j in range(X_arr.shape[1]):
            corr = np.corrcoef(z_scores[:, j], scores)[0, 1]
            importances.append(float(corr) if np.isfinite(corr) else 0.0)

        return importances
    except Exception:
        return None


def feature_contributions(X, idx, scores):
    """Compute per-feature z-score for a specific sample.

    Returns the top-5 features by absolute z-score for the row at
    `idx`. Used to explain why a single sample looks anomalous. The
    `scores` argument is part of the call signature for API symmetry
    with `compute_feature_importance` but is not currently read.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    idx : int
        Row index of the sample to explain.
    scores : np.ndarray, shape (n_samples,)
        Anomaly scores (currently unused; reserved for future weighting).

    Returns
    -------
    list of dict or None
        Up to five entries, each with keys ``'feature'`` (int column
        index) and ``'z_score'`` (float, absolute z-score), sorted by
        descending z-score. Returns ``None`` if `X` is not 2D or any
        error occurs during computation.
    """
    try:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            return None
        means = np.mean(X_arr, axis=0)
        stds = np.std(X_arr, axis=0)
        stds[stds == 0] = 1.0
        z = np.abs((X_arr[idx] - means) / stds)
        top_feat = np.argsort(z)[::-1][:5]
        return [{'feature': int(f), 'z_score': float(z[f])}
                for f in top_feat]
    except Exception:
        return None


def compute_consensus(successful_results):
    """Compute consensus from successful detector results.

    Rank-normalizes scores per detector via ``rankdata``, averages to
    get consensus scores, takes a majority vote on labels, computes
    pairwise Spearman correlation for the agreement metric, and
    flags indices where detectors disagree.

    Parameters
    ----------
    successful_results : list of dict
        Successful detector result dicts. Each must contain
        ``'scores_train'`` and ``'labels_train'`` ndarrays.

    Returns
    -------
    dict or None
        Returns ``None`` when ``successful_results`` is empty. With
        exactly one successful result, returns a single-detector
        consensus with ``agreement=0.5``. Otherwise returns a dict
        with keys ``'scores'``, ``'labels'``, ``'n_detectors'``,
        ``'agreement'``, and ``'disagreements'``.
    """
    successful = successful_results

    if len(successful) == 0:
        return None

    if len(successful) == 1:
        r = successful[0]
        return {
            'scores': r['scores_train'],
            'labels': r['labels_train'],
            'n_detectors': 1,
            'agreement': 0.5,
            'disagreements': [],
        }

    n_samples = len(successful[0]['scores_train'])
    # Rank-normalize scores per detector
    rank_scores = np.array([
        rankdata(r['scores_train']) / n_samples
        for r in successful
    ])
    consensus_scores = np.mean(rank_scores, axis=0)

    # Majority-vote labels
    all_labels = np.array([
        r['labels_train'] for r in successful])
    vote_count = np.sum(all_labels, axis=0)
    consensus_labels = (
        vote_count > len(successful) / 2).astype(int)

    # Pairwise Spearman agreement
    correlations = []
    for i in range(len(successful)):
        for j in range(i + 1, len(successful)):
            rho, _ = spearmanr(
                successful[i]['scores_train'],
                successful[j]['scores_train'])
            correlations.append(
                max(0.0, rho) if np.isfinite(rho) else 0.0)
    agreement = float(np.mean(correlations)) if correlations else 0.5

    # Disagreements: indices where detectors disagree
    disagreements = []
    for idx in range(n_samples):
        votes = all_labels[:, idx]
        if not (votes.all() or not votes.any()):
            disagreements.append(int(idx))

    return {
        'scores': consensus_scores,
        'labels': consensus_labels,
        'n_detectors': len(successful),
        'agreement': agreement,
        'disagreements': disagreements,
    }
