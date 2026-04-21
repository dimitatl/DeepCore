"""Predictive uncertainty: entropy, confidence, and per-sample summaries."""
import numpy as np


def predictive_entropy(probs, eps=1e-12):
    probs = np.clip(np.asarray(probs), eps, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def max_confidence(probs):
    return np.asarray(probs).max(axis=1)


def uncertainty_metrics(probs):
    """Summary statistics over per-sample entropy and confidence."""
    probs = np.asarray(probs)
    ent = predictive_entropy(probs)
    conf = max_confidence(probs)
    return {
        "entropy_mean": float(ent.mean()),
        "entropy_std": float(ent.std()),
        "entropy_median": float(np.median(ent)),
        "confidence_mean": float(conf.mean()),
        "confidence_std": float(conf.std()),
        "entropy_per_sample": ent,
        "confidence_per_sample": conf,
    }
