"""Plotting utilities. Imports matplotlib lazily so cluster runs don't require a display."""
import os
import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_accuracy_vs_noise(rows, method_key="config.selection",
                           noise_key="config.label_noise.ratio",
                           metric="performance.accuracy", out_path=None, title=None):
    plt = _mpl()
    by_method = {}
    for r in rows:
        by_method.setdefault(r[method_key], []).append(r)
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in sorted(by_method.items(), key=lambda kv: str(kv[0])):
        group = sorted(group, key=lambda r: r[noise_key] or 0)
        xs = [r[noise_key] for r in group]
        means = [r[f"{metric}_mean"] for r in group]
        stds = [r[f"{metric}_std"] or 0 for r in group]
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=str(method))
    ax.set_xlabel(noise_key)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {noise_key}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig


def plot_accuracy_vs_fraction(rows, method_key="config.selection",
                              frac_key="config.fraction",
                              metric="performance.accuracy", out_path=None, title=None):
    return plot_accuracy_vs_noise(rows, method_key=method_key, noise_key=frac_key,
                                  metric=metric, out_path=out_path,
                                  title=title or f"{metric} vs coreset fraction")


def plot_reliability_diagram(reliability, out_path=None, title=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 5))
    confs = [b["avg_conf"] for b in reliability if b["avg_conf"] is not None]
    accs = [b["avg_acc"] for b in reliability if b["avg_acc"] is not None]
    counts = [b["count"] for b in reliability if b["avg_conf"] is not None]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    ax.scatter(confs, accs, s=[max(10, c / 5) for c in counts], alpha=0.7, label="bins")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title or "Reliability diagram")
    ax.legend()
    ax.grid(alpha=0.3)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig


def plot_entropy_histogram(entropy, is_noisy, out_path=None, title=None, bins=40):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7, 5))
    e = np.asarray(entropy)
    noisy = np.asarray(is_noisy).astype(bool)
    ax.hist(e[~noisy], bins=bins, alpha=0.5, label=f"clean (n={(~noisy).sum()})", density=True)
    ax.hist(e[noisy], bins=bins, alpha=0.5, label=f"noisy (n={noisy.sum()})", density=True)
    ax.set_xlabel("entropy")
    ax.set_ylabel("density")
    ax.set_title(title or "Entropy: clean vs noisy training samples")
    ax.legend()
    ax.grid(alpha=0.3)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig


def plot_class_distribution(coreset_dist, full_dist, out_path=None, title=None):
    plt = _mpl()
    classes = sorted(set(coreset_dist.keys()) | set(full_dist.keys()))
    coreset = [coreset_dist.get(int(c), coreset_dist.get(str(c), 0)) for c in classes]
    full = [full_dist.get(int(c), full_dist.get(str(c), 0)) for c in classes]
    # normalize to fractions for comparison
    coreset_frac = np.array(coreset) / max(1, sum(coreset))
    full_frac = np.array(full) / max(1, sum(full))
    x = np.arange(len(classes))
    w = 0.4
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, full_frac, width=w, label="full")
    ax.bar(x + w / 2, coreset_frac, width=w, label="coreset")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_ylabel("class fraction")
    ax.set_title(title or "Class distribution: coreset vs full")
    ax.legend()
    ax.grid(alpha=0.3)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig
