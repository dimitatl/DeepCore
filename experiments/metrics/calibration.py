"""Expected / Maximum Calibration Error and Brier score."""
import numpy as np


def calibration_metrics(probs, y_true, n_bins=15):
    """
    probs: (N, K) probability matrix
    y_true: (N,) integer labels
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    N = len(y_true)
    reliability = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        if mask.sum() == 0:
            reliability.append({"bin_low": float(lo), "bin_high": float(hi),
                                "avg_conf": None, "avg_acc": None, "count": 0})
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(accuracies[mask].mean())
        gap = abs(avg_conf - avg_acc)
        ece += (mask.sum() / N) * gap
        mce = max(mce, gap)
        reliability.append({"bin_low": float(lo), "bin_high": float(hi),
                            "avg_conf": avg_conf, "avg_acc": avg_acc,
                            "count": int(mask.sum())})

    # Brier (multi-class): mean squared error between one-hot and probs
    K = probs.shape[1]
    one_hot = np.eye(K)[y_true]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))

    return {"ece": float(ece), "mce": float(mce), "brier": brier, "reliability": reliability}
