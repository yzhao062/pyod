#!/usr/bin/env python3
"""Regenerate soft-knowledge KB scores from raw benchmark matrices.

Phase A step 2 of the KB redesign (see PLAN-kb-redesign.md). Consumes the
verified matrices under ``_raw/`` plus ``_raw/benchmark_aliases.json`` and
emits ``_raw/kb_scores.json``: per-(modality, canonical detector, metric)
calibrated scores with bootstrap uncertainty and coverage/failure/missing
counts.

Method (within-modality ordinal calibration):
  1. For each (benchmark, metric, dataset), rank the competing detectors by
     their native score and convert to a tie-aware percentile in [0, 1]
     where 1 is best. This removes per-dataset scale and the cross-benchmark
     metric heterogeneity (AUC-ROC / AUC-PR / AP / VUS-PR) at the dataset
     level. Failed detectors (BOND OOM/TLE) take worst rank 0.0; when more
     than one detector fails on a dataset they all share 0.0.
  2. Aggregate per (canonical, modality, metric, embedding): kb_score is the
     mean percentile; uncertainty is a nonparametric bootstrap CI over
     datasets; coverage_n / failure_n / missing_n are tracked separately and
     eligible_n = coverage_n + failure_n + missing_n.
  3. Text methods are kept as (embedding, detector) pairs; a detector-only
     rollup pools percentiles across embeddings.
  4. A within-modality rollup pools every per-dataset percentile across all
     metrics; its kb_score is that pooled-sample mean (so the bootstrap CI is
     drawn from the same distribution as the point estimate), NOT a mean of
     the per-metric kb_scores.

Empty-cell semantics are per-benchmark: BOND empties are OOM/TLE failures
(worst rank, counted as failures); every other benchmark's empties are
"not evaluated" (missing, excluded from that dataset's ranking). This is the
only place the failure-vs-missing distinction is applied, so it is explicit
and auditable.

Scores are strictly within modality. The script never compares a graph
score to a tabular score. External subsequence-windowed variants (TSB-AD
``Sub-*``) are scored under their own key, never merged into the shipped
detector. Detectors absent from every benchmark are emitted in
``unbenchmarked`` so the merge step can mark them kb_score=null.

Each (canonical, modality, metric, embedding) bootstrap CI is seeded from a
stable hash of its key, so CIs do not churn when an unrelated alias is added
or reordered; only the affected key's CI changes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
RAW = HERE / "_raw"
ALIASES_PATH = RAW / "benchmark_aliases.json"
ALGS_PATH = HERE / "algorithms.json"
OUT_PATH = RAW / "kb_scores.json"

BOOTSTRAP_B = 2000
RNG_SEED = 42

# Detector axis orientation per benchmark CSV.
ORIENTATION = {
    "ADBench": "col",
    "NLP-ADBench": "row",
    "BOND": "row",
    "TSB-AD-uni": "col",
    "TSB-AD-multi": "col",
}

# Per-benchmark empty-cell meaning. Default is "missing".
EMPTY_MEANS = {"BOND": "failure"}


def file_to_metric(filename: str) -> str:
    n = filename.lower()
    if "aucroc" in n or "auroc" in n:
        return "ROC"
    if "aucpr" in n or "auprc" in n:
        return "PR"
    if "vus-pr" in n or "vus_pr" in n:
        return "VUS-PR"
    if "bond_ap" in n or "_ap" in n:
        return "AP"
    return "metric"


def _parse_cell(cell: str):
    cell = (cell or "").strip()
    if cell == "":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _score_key(raw: str, rec: dict) -> str:
    """Aggregation key for a raw detector name.

    External subsequence-windowed variants (TSB-AD ``Sub-*``) are distinct
    methods, not the plain pyod detector, so they score under their own key
    rather than being pooled into the shipped ``canonical``.
    """
    if rec.get("score_key"):
        return rec["score_key"]
    if rec.get("kind") == "subsequence_variant":
        return raw.replace("-", "_")
    return rec["canonical"]


def _key_seed(key: tuple) -> int:
    """Stable per-key bootstrap seed (independent of run and insertion order)."""
    s = f"{RNG_SEED}|" + "|".join(str(x) for x in key)
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def load_matrix(path: Path, orientation: str, detector_names: set[str]):
    """Return {dataset: {raw_detector_name: float | None}} for known detectors."""
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    body = rows[1:]
    out: dict[str, dict[str, float | None]] = {}
    if orientation == "col":
        # col 0 = dataset label; detector columns are those in detector_names.
        det_cols = [(i, name) for i, name in enumerate(header)
                    if i > 0 and name in detector_names]
        for r in body:
            if not r or not r[0].strip():
                continue
            dataset = r[0].strip()
            out[dataset] = {name: _parse_cell(r[i]) for i, name in det_cols
                            if i < len(r)}
    else:  # "row": col 0 = detector name; remaining columns are datasets.
        datasets = [c.strip() for c in header[1:]]
        for ds in datasets:
            out[ds] = {}
        for r in body:
            if not r or r[0].strip() not in detector_names:
                continue
            raw = r[0].strip()
            for j, ds in enumerate(datasets, start=1):
                if j < len(r):
                    out[ds][raw] = _parse_cell(r[j])
    return out


def percentiles_for_dataset(row: dict[str, float | None], empty_means: str):
    """Tie-aware percentile in [0,1] (1=best) per attempted detector.

    Failed detectors (BOND OOM/TLE) take worst rank 0.0; observed detectors
    are ranked above the whole failure block. Returns
    (percentiles: {raw: float}, failed: set[raw], missing: set[raw]).
    """
    observed_names, values, failed_names = [], [], []
    missing = set()
    for raw, val in row.items():
        if val is not None:
            observed_names.append(raw)
            values.append(val)
        elif empty_means == "failure":
            failed_names.append(raw)
        else:
            missing.add(raw)

    total_n = len(observed_names) + len(failed_names)
    pcts: dict[str, float] = {}
    if total_n == 1:
        if observed_names:
            pcts[observed_names[0]] = 1.0
        else:
            pcts[failed_names[0]] = 0.0
    elif total_n > 1:
        for name in failed_names:
            pcts[name] = 0.0
        if observed_names:
            ranks = rankdata(values, method="average")  # ascending: best -> highest
            failure_offset = len(failed_names)
            for name, rk in zip(observed_names, ranks):
                pcts[name] = float((rk + failure_offset - 1.0) / (total_n - 1.0))
    return pcts, set(failed_names), missing


def _bootstrap_ci(samples: list[float], rng: np.random.Generator):
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return None, None
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    idx = rng.integers(0, arr.size, size=(BOOTSTRAP_B, arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _summ(samples: list[float], failure_n: int, missing_n: int,
          source: str, key: tuple) -> dict:
    arr = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(_key_seed(key))
    lo, hi = _bootstrap_ci(samples, rng)
    coverage_n = int(arr.size) - failure_n
    return {
        "kb_score": round(float(arr.mean()), 4),
        "ci_low": None if lo is None else round(lo, 4),
        "ci_high": None if hi is None else round(hi, 4),
        "coverage_n": coverage_n,
        "eligible_n": coverage_n + failure_n + missing_n,
        "failure_n": failure_n,
        "missing_n": missing_n,
        "effective_n": int(arr.size),
        "source_benchmark": source,
    }


def _sort_key(key: tuple):
    return tuple("" if x is None else str(x) for x in key)


def main() -> int:
    aliases = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    algs = json.loads(ALGS_PATH.read_text(encoding="utf-8"))
    canonical_keys = set(algs.keys())

    # raw percentile samples keyed by (score_key, modality, metric, embedding)
    samples: dict[tuple, list[float]] = {}
    failures: dict[tuple, int] = {}
    missings: dict[tuple, int] = {}
    sources: dict[tuple, str] = {}
    benchmarks = [b for b in aliases if not b.startswith("_")]
    scored_canon_by_modality: dict[str, set[str]] = {}
    skipped = []

    for bench in benchmarks:
        spec = aliases[bench]
        if spec.get("pending_extraction"):
            skipped.append(bench)
            continue
        modality = spec["modality"]
        det_map = spec["detectors"]
        det_names = set(det_map.keys())
        orientation = ORIENTATION[bench]
        empty_means = EMPTY_MEANS.get(bench, "missing")
        scored_canon_by_modality.setdefault(modality, set())

        for fname in spec["files"]:
            metric = file_to_metric(fname)
            if spec.get("subtype"):  # keep TSB-AD univariate/multivariate distinct
                metric = f"{metric}/{spec['subtype']}"
            matrix = load_matrix(RAW / fname, orientation, det_names)
            for dataset, row in matrix.items():
                pcts, failed, missing = percentiles_for_dataset(row, empty_means)
                for raw, pct in pcts.items():
                    rec = det_map[raw]
                    canon = _score_key(raw, rec)
                    emb = rec.get("embedding")
                    key = (canon, modality, metric, emb)
                    samples.setdefault(key, []).append(pct)
                    sources[key] = bench
                    if raw in failed:
                        failures[key] = failures.get(key, 0) + 1
                    if rec.get("status") == "shipped":
                        scored_canon_by_modality[modality].add(canon)
                for raw in missing:
                    rec = det_map[raw]
                    key = (_score_key(raw, rec), modality, metric, rec.get("embedding"))
                    missings[key] = missings.get(key, 0) + 1

    # Build nested scores: scores[canonical][modality]. Iterate in sorted key
    # order so the emitted JSON is deterministic; CIs are key-seeded so they
    # do not depend on iteration order regardless.
    scores: dict[str, dict] = {}
    for key in sorted(samples, key=_sort_key):
        canon, modality, metric, emb = key
        samp = samples[key]
        node = scores.setdefault(canon, {}).setdefault(modality, {"metrics": {}})
        m = node["metrics"].setdefault(metric, {"by_embedding": {}, "pooled": []})
        m["pooled"].extend(samp)
        if emb is not None:
            m["by_embedding"][emb] = _summ(
                samp, failures.get(key, 0), missings.get(key, 0),
                sources[key], key)

    # Collapse pooled samples into per-metric summaries + within-modality rollup.
    for canon in sorted(scores):
        by_mod = scores[canon]
        for modality in sorted(by_mod):
            node = by_mod[modality]
            pooled_all, fail_all, miss_all = [], 0, 0
            src = None
            for metric in sorted(node["metrics"]):
                m = node["metrics"][metric]
                pooled = m.pop("pooled")
                # failure/missing summed across embeddings for this (canon, modality, metric)
                f = sum(failures.get((canon, modality, metric, emb), 0)
                        for emb in {None, *m["by_embedding"].keys()})
                mi = sum(missings.get((canon, modality, metric, emb), 0)
                         for emb in {None, *m["by_embedding"].keys()})
                src = sources.get((canon, modality, metric, None)) or \
                    next((sources[(canon, modality, metric, e)]
                          for e in m["by_embedding"]), None)
                summ = _summ(pooled, f, mi, src,
                             (canon, modality, metric, "__pooled__"))
                if not m["by_embedding"]:
                    m.pop("by_embedding")
                node["metrics"][metric] = {**summ, **(
                    {"by_embedding": m["by_embedding"]} if "by_embedding" in m else {})}
                pooled_all.extend(pooled)
                fail_all += f
                miss_all += mi
            # rollup kb_score is the pooled-sample mean (CI consistent with it),
            # NOT a mean of the per-metric kb_scores. source_benchmark lists
            # every benchmark the rollup pooled (e.g. TSB-AD-multi+TSB-AD-uni),
            # not just the last metric's source.
            roll_sources = "+".join(sorted(
                {node["metrics"][mk]["source_benchmark"] for mk in node["metrics"]}))
            roll = _summ(pooled_all, fail_all, miss_all, roll_sources,
                         (canon, modality, "__rollup__"))
            roll["metric_scopes"] = sorted(node["metrics"].keys())
            node["rollup"] = roll

    unbenchmarked = sorted(k for k in canonical_keys if k not in scores)

    out = {
        "_meta": {
            "generated_by": "regenerate_kb.py (Phase A step 2)",
            "bootstrap_b": BOOTSTRAP_B,
            "seed": RNG_SEED,
            "empty_means": {**{b: "missing" for b in benchmarks}, **EMPTY_MEANS},
            "metric_note": "kb_score is the mean within-dataset tie-aware percentile (1=best), strictly within modality; never compare across modalities.",
            "rollup_note": "A modality rollup pools every per-dataset percentile across metrics; its kb_score is that pooled-sample mean (CI drawn from the same distribution), not a mean of per-metric kb_scores.",
            "pending_sources": skipped,
            "n_scored_canonical": len(scores),
            "n_unbenchmarked": len(unbenchmarked),
        },
        "scores": scores,
        "unbenchmarked": unbenchmarked,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Console summary.
    print(f"wrote {OUT_PATH}")
    print(f"scored canonical detectors: {len(scores)}  |  unbenchmarked: {len(unbenchmarked)}")
    if skipped:
        print(f"skipped (pending extraction): {skipped}")
    for modality in sorted({m for v in scores.values() for m in v}):
        rows = []
        for canon, by_mod in scores.items():
            if modality in by_mod:
                r = by_mod[modality]["rollup"]
                rows.append((r["kb_score"], canon, r["effective_n"], r["failure_n"]))
        rows.sort(reverse=True)
        print(f"\n[{modality}] top by rollup kb_score (score, detector, eff_n, fail_n):")
        for sc, canon, en, fn in rows[:8]:
            print(f"   {sc:.3f}  {canon:20s} n={en} fail={fn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
