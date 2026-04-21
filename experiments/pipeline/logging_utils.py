"""Structured logging: JSON manifest + numpy arrays per run."""
import json
import os
import numpy as np


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 64:
            return {"__ndarray__": True, "shape": list(obj.shape), "dtype": str(obj.dtype), "summary": "stored in .npz"}
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_run(run_dir, manifest, arrays):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump(_to_jsonable(manifest), f, indent=2, default=str)
    if arrays:
        np.savez_compressed(os.path.join(run_dir, "arrays.npz"),
                            **{k: np.asarray(v) for k, v in arrays.items()})


def load_run(run_dir):
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    arrays_path = os.path.join(run_dir, "arrays.npz")
    arrays = dict(np.load(arrays_path)) if os.path.exists(arrays_path) else {}
    return manifest, arrays
