"""Apply controlled class imbalance to a dataset."""
import numpy as np
import torch
from torch import tensor, long


def apply_imbalance(dst, imbalance_ratio, majority_class=0, seed=0):
    """Keep full majority class; downsample minority classes to imbalance_ratio * n_majority.

    imbalance_ratio=1.0 -> no imbalance. imbalance_ratio=0.1 -> minorities are 10% of majority size.
    Mutates and returns dst.
    """
    if imbalance_ratio <= 0 or imbalance_ratio > 1.0:
        raise ValueError("imbalance_ratio must be in (0, 1]")

    rng = np.random.RandomState(seed)
    targets = dst.targets.numpy() if isinstance(dst.targets, torch.Tensor) else np.array(dst.targets)
    classes = sorted(set(int(t) for t in targets))

    majority_idx = np.where(targets == majority_class)[0]
    n_majority = len(majority_idx)
    target_minority = max(1, int(round(imbalance_ratio * n_majority)))

    keep = [majority_idx]
    for c in classes:
        if c == majority_class:
            continue
        c_idx = np.where(targets == c)[0]
        n_keep = min(len(c_idx), target_minority)
        keep.append(rng.choice(c_idx, size=n_keep, replace=False))

    keep = np.sort(np.concatenate(keep))
    dst.data = dst.data[keep]
    dst.targets = tensor([int(targets[i]) for i in keep], dtype=long)
    return dst


def class_distribution(targets):
    t = targets.numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    classes, counts = np.unique(t, return_counts=True)
    return {int(c): int(n) for c, n in zip(classes, counts)}
