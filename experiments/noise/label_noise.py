"""Label noise: uniform flipping and class-dependent flipping.

For binary tasks, uniform and class-dependent are distinct only if asymmetric flip
rates are used per class.
"""
import numpy as np
import torch
from torch import tensor, long


def build_label_noise(targets, noise_ratio, num_classes, noise_type="uniform",
                      class_flip_matrix=None, seed=0):
    """Return dict with keys {'new_targets', 'is_noisy', 'transition'}.

    noise_type:
        'uniform'         : each sample independently flipped with prob=noise_ratio
                             to a uniformly random *other* class.
        'class_dependent' : per-class flip rates; use `class_flip_matrix` if given,
                             otherwise build an asymmetric pairwise flip where class c
                             flips to (c+1) % num_classes with prob=noise_ratio.
        'pair_flip'       : alias for class_dependent with pairwise flip.
    """
    rng = np.random.RandomState(seed)
    t = targets.numpy().copy() if isinstance(targets, torch.Tensor) else np.asarray(targets).copy()
    n = len(t)
    is_noisy = np.zeros(n, dtype=bool)
    new_t = t.copy()

    if noise_type == "uniform":
        transition = np.full((num_classes, num_classes), noise_ratio / max(1, num_classes - 1))
        np.fill_diagonal(transition, 1.0 - noise_ratio)
        flip_mask = rng.rand(n) < noise_ratio
        for i in np.where(flip_mask)[0]:
            candidates = [c for c in range(num_classes) if c != t[i]]
            new_t[i] = rng.choice(candidates)
            is_noisy[i] = True

    elif noise_type in ("class_dependent", "pair_flip"):
        if class_flip_matrix is None:
            # Pair flip: c -> (c+1) % K with prob noise_ratio
            T = np.eye(num_classes) * (1.0 - noise_ratio)
            for c in range(num_classes):
                T[c, (c + 1) % num_classes] = noise_ratio
        else:
            T = np.asarray(class_flip_matrix)
            assert T.shape == (num_classes, num_classes)
        transition = T
        for i in range(n):
            probs = T[int(t[i])]
            sampled = rng.choice(num_classes, p=probs / probs.sum())
            new_t[i] = sampled
            if sampled != t[i]:
                is_noisy[i] = True

    else:
        raise ValueError(f"Unknown label noise_type: {noise_type}")

    return {
        "new_targets": tensor(new_t.astype(np.int64), dtype=long),
        "is_noisy": is_noisy,
        "transition": transition,
        "noise_type": noise_type,
        "noise_ratio": noise_ratio,
    }
