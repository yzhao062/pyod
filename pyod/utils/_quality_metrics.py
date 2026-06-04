"""Quality metrics for ADEngine result analysis.

Pure helper functions extracted from `pyod.utils.ad_engine.ADEngine` in
2026-05 (issue #667 follow-up). Not part of the public API; the
leading underscore on the module name is the contract.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import numpy as np
from scipy.stats import rankdata, spearmanr

logger = logging.getLogger(__name__)


_VERDICT_HIGH_THRESHOLD: float = 0.7
"""Overall quality at or above this is reported as 'high'."""

_VERDICT_MEDIUM_THRESHOLD: float = 0.4
"""Overall quality at or above this (but below high) is 'medium'."""

_SINGLE_DETECTOR_AGREEMENT_FALLBACK: float = 0.5
"""Agreement returned when only one detector ran (no basis for agreement)."""


class QualityDict(TypedDict):
    separation: float
    agreement: float
    stability: float
    overall: float
    verdict: str
    explanation: str


def compute_quality(
    scores: np.ndarray,
    labels: np.ndarray,
    results: list[dict],
    consensus: dict,
) -> QualityDict:
    """Compute three diagnostic quality metrics for a detection run.

    Each metric diagnoses one independent failure mode and drives
    one branch of `ADEngine.iterate()`:

    - ``separation``: relative mean score gap between the samples
      flagged by the run and the rest, computed from the run's OWN
      predicted labels. In ADEngine consensus, labels come from
      detector votes while scores are rank-averaged, so this stays
      descriptive and circular: it is not independent correctness
      evidence. Treat it as descriptive, not as a label-free quality
      signal.
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
    if overall >= _VERDICT_HIGH_THRESHOLD:
        verdict = 'high'
    elif overall >= _VERDICT_MEDIUM_THRESHOLD:
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


def select_best_detector(
    results: list[dict],
    consensus_scores: np.ndarray,
) -> int:
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


def compute_feature_importance(
    result: dict,
    X: np.ndarray | None,
) -> list | None:
    """Estimate per-feature contribution to anomaly scores.

    For each feature column, computes the Pearson correlation between
    the column's absolute z-scores and the detector's anomaly scores.
    Higher absolute correlation indicates the feature drives the
    detector's ranking. Failures (non-2D `X` or length mismatch)
    return ``None``. Validation/conversion errors caught from the
    numerical pipeline (``AttributeError``, ``ValueError``,
    ``TypeError``) are logged at DEBUG level and also return
    ``None``; other unexpected exceptions propagate.

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
        `X.shape[0]`, or if a caught
        ``(AttributeError, ValueError, TypeError)`` is raised during
        computation.
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
    except (AttributeError, ValueError, TypeError) as exc:
        logger.debug('compute_feature_importance: %s', exc)
        return None


def feature_contributions(
    X: np.ndarray,
    idx: int,
    scores: np.ndarray,
    feature_names: list[str] | None = None,
) -> list | None:
    """Compute per-feature z-score, value, and direction for a sample.

    Returns the top-5 features by absolute z-score for the row at
    `idx`. Each entry includes the column index, a feature name, the
    raw value at this row, the column mean, the absolute z-score, and
    the direction (`'high'` if the value is at or above the mean,
    `'low'` otherwise). Used to explain why a single sample looks
    anomalous. The `scores` argument is part of the call signature for
    API symmetry with `compute_feature_importance` but is not currently
    read.

    Validation/conversion errors caught from the numerical pipeline
    (``AttributeError``, ``ValueError``, ``TypeError``) are logged
    at DEBUG level and return ``None``; other unexpected exceptions
    (for example ``IndexError`` when `idx` is out of range) propagate.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training data.
    idx : int
        Row index of the sample to explain.
    scores : np.ndarray, shape (n_samples,)
        Anomaly scores (currently unused; reserved for future weighting).
    feature_names : list of str or None
        Optional feature labels in column order. When provided, each
        entry's ``'name'`` is the corresponding label; otherwise the
        name defaults to ``f'feature_{column_index}'``.

    Returns
    -------
    list of dict or None
        Up to five entries, sorted by descending absolute z-score.
        Each entry has keys: ``'feature'`` (int column index),
        ``'name'`` (str), ``'value'`` (float, raw value at this row),
        ``'mean'`` (float, column mean), ``'z_score'`` (float,
        absolute z-score), ``'direction'`` (`'high'` or `'low'`).
        Returns ``None`` if `X` is not 2D or if a caught
        ``(AttributeError, ValueError, TypeError)`` is raised during
        computation.
    """
    try:
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            return None
        means = np.mean(X_arr, axis=0)
        stds = np.std(X_arr, axis=0)
        stds_safe = np.where(stds == 0, 1.0, stds)
        signed_z = (X_arr[idx] - means) / stds_safe
        abs_z = np.abs(signed_z)
        top_feat = np.argsort(abs_z)[::-1][:5]
        results = []
        for f in top_feat:
            f_int = int(f)
            if (feature_names is not None
                    and f_int < len(feature_names)):
                name = feature_names[f_int]
            else:
                name = f'feature_{f_int}'
            results.append({
                'feature': f_int,
                'name': name,
                'value': float(X_arr[idx, f_int]),
                'mean': float(means[f_int]),
                'z_score': float(abs_z[f_int]),
                'direction': (
                    'high' if signed_z[f_int] >= 0 else 'low'),
            })
        return results
    except (AttributeError, ValueError, TypeError) as exc:
        logger.debug('feature_contributions: %s', exc)
        return None


def label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Compute label-based validation metrics for one detector or
    consensus.

    Returns binary precision / recall / F1 (sklearn `'binary'` average,
    `zero_division=0`), ROC AUC, and average precision. ROC AUC and AP
    require both classes in `y_true`; if only one class is present,
    those fields come back as ``None`` rather than raising.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth binary labels (0 = inlier, 1 = anomaly).
    y_pred : array-like, shape (n_samples,)
        Predicted binary labels.
    scores : array-like, shape (n_samples,)
        Continuous anomaly scores (used for ROC AUC and AP).

    Returns
    -------
    dict
        Keys: ``precision``, ``recall``, ``f1`` (floats);
        ``roc_auc`` (float or None); ``average_precision`` (float or
        None); ``n_flagged`` (int, count of `y_pred` == 1);
        ``n_true_positive`` (int, count where both `y_pred` == 1 and
        `y_true` == 1).
    """
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    scores_arr = np.asarray(scores)

    p, r, f, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average='binary', zero_division=0)

    if len(np.unique(y_true_arr)) > 1:
        roc = float(roc_auc_score(y_true_arr, scores_arr))
        ap = float(average_precision_score(y_true_arr, scores_arr))
    else:
        roc = None
        ap = None

    return {
        'precision': float(p),
        'recall': float(r),
        'f1': float(f),
        'roc_auc': roc,
        'average_precision': ap,
        'n_flagged': int(y_pred_arr.sum()),
        'n_true_positive': int(
            ((y_pred_arr == 1) & (y_true_arr == 1)).sum()),
    }


def compute_consensus(
    successful_results: list[dict],
) -> dict | None:
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
            'agreement': _SINGLE_DETECTOR_AGREEMENT_FALLBACK,
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
    agreement = (
        float(np.mean(correlations)) if correlations
        else _SINGLE_DETECTOR_AGREEMENT_FALLBACK
    )

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
