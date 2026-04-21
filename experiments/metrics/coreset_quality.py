"""Coreset quality metrics: noisy fraction, class distribution, minority retention."""
import numpy as np


def coreset_quality_metrics(selected_indices, is_noisy, targets_clean, num_classes,
                            is_label_noisy=None, is_feature_noisy=None,
                            full_class_dist=None):
    """
    selected_indices: indices into the full training set
    is_noisy: bool array over full training set (label | feature)
    targets_clean: clean targets (before label noise) over full training set
    """
    selected_indices = np.asarray(selected_indices)
    is_noisy = np.asarray(is_noisy)
    t_clean = targets_clean.numpy() if hasattr(targets_clean, "numpy") else np.asarray(targets_clean)

    n_sel = len(selected_indices)
    n_full = len(is_noisy)

    sel_noisy = is_noisy[selected_indices]
    n_noisy_selected = int(sel_noisy.sum())
    frac_noisy_selected = n_noisy_selected / max(1, n_sel)

    n_noisy_total = int(is_noisy.sum())
    noisy_coverage = n_noisy_selected / max(1, n_noisy_total)  # fraction of all noisy samples captured

    # Selection bias: P(selected | noisy) vs P(selected | clean)
    p_sel_given_noisy = n_noisy_selected / max(1, n_noisy_total)
    n_clean_total = n_full - n_noisy_total
    n_clean_selected = n_sel - n_noisy_selected
    p_sel_given_clean = n_clean_selected / max(1, n_clean_total)

    # Class distribution of coreset (using clean labels)
    sel_targets = t_clean[selected_indices]
    classes, counts = np.unique(sel_targets, return_counts=True)
    coreset_dist = {int(c): int(n) for c, n in zip(classes, counts)}
    for c in range(num_classes):
        coreset_dist.setdefault(c, 0)

    # Minority retention rate
    full_dist = full_class_dist
    if full_dist is None:
        classes_f, counts_f = np.unique(t_clean, return_counts=True)
        full_dist = {int(c): int(n) for c, n in zip(classes_f, counts_f)}
    minority_class = min(full_dist, key=full_dist.get)
    minority_retention = coreset_dist[minority_class] / max(1, full_dist[minority_class])

    out = {
        "n_selected": n_sel,
        "frac_noisy_selected": float(frac_noisy_selected),
        "n_noisy_selected": n_noisy_selected,
        "n_noisy_total": n_noisy_total,
        "noisy_coverage": float(noisy_coverage),
        "p_selected_given_noisy": float(p_sel_given_noisy),
        "p_selected_given_clean": float(p_sel_given_clean),
        "selection_bias_ratio": float(p_sel_given_noisy / max(1e-12, p_sel_given_clean)),
        "coreset_class_distribution": coreset_dist,
        "full_class_distribution": full_dist,
        "minority_class": int(minority_class),
        "minority_retention": float(minority_retention),
    }

    if is_label_noisy is not None:
        lbl_sel = np.asarray(is_label_noisy)[selected_indices]
        out["frac_label_noisy_selected"] = float(lbl_sel.mean())
    if is_feature_noisy is not None:
        feat_sel = np.asarray(is_feature_noisy)[selected_indices]
        out["frac_feature_noisy_selected"] = float(feat_sel.mean())

    return out
