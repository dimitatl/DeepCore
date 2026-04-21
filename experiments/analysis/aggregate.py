"""Aggregate results across seeds and conditions."""
import os
import json
import glob
import numpy as np


def load_runs(root):
    """Walk `root` and return list of dicts {dir, manifest, arrays_path}."""
    runs = []
    for path in glob.glob(os.path.join(root, "**", "manifest.json"), recursive=True):
        with open(path) as f:
            manifest = json.load(f)
        run_dir = os.path.dirname(path)
        arrays_path = os.path.join(run_dir, "arrays.npz")
        runs.append({
            "dir": run_dir,
            "manifest": manifest,
            "arrays_path": arrays_path if os.path.exists(arrays_path) else None,
        })
    return runs


def _get(obj, dotted, default=None):
    cur = obj
    for k in dotted.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def aggregate(runs, group_by, metrics):
    """
    runs: list from load_runs
    group_by: list of dotted-keys into manifest (e.g. ['config.selection', 'config.label_noise.ratio'])
    metrics: list of dotted-keys (e.g. ['performance.accuracy', 'calibration.ece'])

    Returns list of dicts with group values + mean/std/n for each metric.
    """
    groups = {}
    for r in runs:
        key = tuple(_get(r["manifest"], k) for k in group_by)
        groups.setdefault(key, []).append(r)

    rows = []
    for key, group_runs in groups.items():
        row = {gb: val for gb, val in zip(group_by, key)}
        row["n_seeds"] = len(group_runs)
        for m in metrics:
            vals = [_get(r["manifest"], m) for r in group_runs]
            vals = [v for v in vals if v is not None]
            if vals:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                row[f"{m}_n"] = len(vals)
            else:
                row[f"{m}_mean"] = None
                row[f"{m}_std"] = None
                row[f"{m}_n"] = 0
        rows.append(row)
    return rows


def compare_methods_ttest(runs, metric, group_by_method="config.selection", reference="Uniform",
                          condition_keys=None):
    """Welch's t-test for each (method, condition) vs (reference, same condition)."""
    from scipy import stats

    condition_keys = condition_keys or []

    buckets = {}
    for r in runs:
        method = _get(r["manifest"], group_by_method)
        cond = tuple(_get(r["manifest"], k) for k in condition_keys)
        val = _get(r["manifest"], metric)
        if val is None:
            continue
        buckets.setdefault((cond, method), []).append(val)

    rows = []
    conditions = sorted(set(c for (c, _) in buckets.keys()))
    methods = sorted(set(m for (_, m) in buckets.keys() if m != reference))
    for cond in conditions:
        ref_vals = buckets.get((cond, reference), [])
        if not ref_vals:
            continue
        for m in methods:
            vals = buckets.get((cond, m), [])
            if not vals:
                continue
            t, p = stats.ttest_ind(vals, ref_vals, equal_var=False)
            rows.append({
                "condition": dict(zip(condition_keys, cond)),
                "method": m,
                "reference": reference,
                "metric": metric,
                "mean_method": float(np.mean(vals)),
                "mean_reference": float(np.mean(ref_vals)),
                "delta": float(np.mean(vals) - np.mean(ref_vals)),
                "t_stat": float(t),
                "p_value": float(p),
                "n_method": len(vals),
                "n_reference": len(ref_vals),
            })
    return rows
