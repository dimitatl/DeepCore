"""Clean-vs-noisy separation: can a score (entropy, loss, confidence) detect noisy samples?

Returns AUROC for using each score to detect noisy samples, plus confidence/entropy gaps.
"""
import numpy as np


def _auroc(scores, labels):
    """Compute AUROC with scores where higher=positive class.

    labels: bool/int 0/1 arrays; 1 = positive (noisy).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U: fraction of (pos, neg) pairs where pos > neg, ties=0.5
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Handle ties by averaging ranks
    # (Simple approach — adequate for continuous scores)
    n_pos = len(pos)
    n_neg = len(neg)
    sum_ranks_pos = ranks[labels == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def separation_metrics(per_sample_entropy, per_sample_loss, per_sample_confidence,
                       is_noisy):
    """All arrays over the training set (or whatever subset was scored).

    Returns AUROC for detecting noisy samples via each score, and summary gaps.
    """
    ent = np.asarray(per_sample_entropy)
    loss = np.asarray(per_sample_loss)
    conf = np.asarray(per_sample_confidence)
    noisy = np.asarray(is_noisy).astype(bool)

    out = {
        "auroc_entropy": _auroc(ent, noisy),
        "auroc_loss": _auroc(loss, noisy),
        "auroc_neg_confidence": _auroc(-conf, noisy),  # low confidence -> high score for noisy
        "n_noisy": int(noisy.sum()),
        "n_clean": int((~noisy).sum()),
    }

    if noisy.any() and (~noisy).any():
        out["entropy_clean_mean"] = float(ent[~noisy].mean())
        out["entropy_noisy_mean"] = float(ent[noisy].mean())
        out["entropy_gap"] = float(ent[noisy].mean() - ent[~noisy].mean())
        out["confidence_clean_mean"] = float(conf[~noisy].mean())
        out["confidence_noisy_mean"] = float(conf[noisy].mean())
        out["confidence_gap"] = float(conf[~noisy].mean() - conf[noisy].mean())
        out["loss_clean_mean"] = float(loss[~noisy].mean())
        out["loss_noisy_mean"] = float(loss[noisy].mean())
        out["loss_gap"] = float(loss[noisy].mean() - loss[~noisy].mean())
    return out
