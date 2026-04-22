"""End-to-end experiment runner.

A single experiment = (dataset config) x (noise config) x (imbalance config)
                     x (selection method) x (model) x (coreset fraction) x (seed).

Produces a directory with:
    manifest.json   -- config + all scalar metrics
    arrays.npz      -- per-sample arrays (predictions, probs, entropies, is_noisy, selected)

Grid execution order: for each (dataset config, seed), the dataset is built ONCE and all
(selection, fraction) combinations are evaluated before moving to the next dataset config.
"""
import copy
import itertools
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..data import build_binary_cifar10, apply_imbalance, NoisyDataset
from ..data.imbalance import class_distribution
from ..noise import build_label_noise, build_feature_noise
from ..metrics import coreset_quality_metrics
from .args import build_args
from .selection import select_coreset
from .training import build_network, train_model, _augment_cifar
from .evaluation import evaluate_test, evaluate_train_separation
from .logging_utils import save_run
from utils import WeightedSubset

# Axes that define the dataset — everything else is a method/model axis.
_DATASET_AXES = frozenset({
    "imbalance.ratio", "imbalance.majority_class",
    "label_noise.ratio", "label_noise.type", "label_noise.flip_matrix",
    "feature_noise.ratio", "feature_noise.sigma",
    "class_ids", "data_path",
})


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataset(config, seed):
    """Build the (possibly imbalanced, possibly noisy) training set + test set."""
    data_path = config.get("data_path", "data")
    class_ids = tuple(config.get("class_ids", [3, 5]))
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = build_binary_cifar10(
        data_path, class_ids=class_ids)

    imbalance = config.get("imbalance", {})
    imbalance_ratio = imbalance.get("ratio", 1.0)
    if imbalance_ratio < 1.0:
        majority = imbalance.get("majority_class", 0)
        dst_train = apply_imbalance(dst_train, imbalance_ratio, majority_class=majority, seed=seed)

    # Save clean (imbalanced) distribution before noise
    full_class_dist = class_distribution(dst_train.targets)

    # Label noise
    label_cfg = config.get("label_noise", {})
    label_noise = None
    if label_cfg.get("ratio", 0.0) > 0:
        label_noise = build_label_noise(
            dst_train.targets,
            noise_ratio=label_cfg["ratio"],
            num_classes=num_classes,
            noise_type=label_cfg.get("type", "uniform"),
            class_flip_matrix=label_cfg.get("flip_matrix"),
            seed=seed + 1001,
        )

    # Feature noise -- operates on post-transform tensor shape (C, H, W)
    feature_cfg = config.get("feature_noise", {})
    feature_noise = None
    if feature_cfg.get("ratio", 0.0) > 0:
        sample_shape = (channel, im_size[0], im_size[1])
        feature_noise = build_feature_noise(
            n_samples=len(dst_train),
            sample_shape=sample_shape,
            noise_ratio=feature_cfg["ratio"],
            sigma=feature_cfg.get("sigma", 0.1),
            seed=seed + 2002,
        )

    noisy_train = NoisyDataset(dst_train, label_noise=label_noise, feature_noise=feature_noise)
    return {
        "channel": channel,
        "im_size": im_size,
        "num_classes": num_classes,
        "class_names": class_names,
        "dst_train": noisy_train,
        "dst_test": dst_test,
        "full_class_dist": full_class_dist,
        # Store original transform so run_experiment can reset before re-augmenting.
        "base_transform": noisy_train.transform,
    }


def run_experiment(config: Dict[str, Any], output_dir: str, verbose: bool = True,
                   prebuilt_data: Optional[Dict] = None):
    """Run one experimental condition for one seed and write out results.

    If `prebuilt_data` is provided the dataset build step is skipped.  The
    caller must ensure the config's dataset axes match what was used to build
    it.  The dataset's transform is reset to its original state before
    augmentation is applied so that multiple experiments can safely share one
    dataset object.
    """
    t0 = time.time()
    seed = int(config["seed"])
    _set_seed(seed)

    pin_memory = torch.cuda.is_available()

    data = prebuilt_data if prebuilt_data is not None else _build_dataset(config, seed)
    dst_train = data["dst_train"]
    dst_test = data["dst_test"]
    num_classes = data["num_classes"]
    channel = data["channel"]
    im_size = data["im_size"]

    # Reset to base transform before augmenting — required when dataset is reused.
    if prebuilt_data is not None:
        dst_train.transform = data["base_transform"]

    args = build_args({
        "dataset": "CIFAR10_BINARY",
        "model": config["model"],
        "selection": config["selection"],
        "fraction": float(config["fraction"]),
        "seed": seed,
        "epochs": int(config.get("epochs", 50)),
        "selection_epochs": int(config.get("selection_epochs", 10)),
        "batch": int(config.get("batch", 128)),
        "train_batch": int(config.get("batch", 128)),
        "selection_batch": int(config.get("batch", 128)),
        "lr": float(config.get("lr", 0.1)),
        "workers": int(config.get("workers", 2)),
        "test_interval": int(config.get("test_interval", 0)),
        "balance": bool(config.get("balance", True)),
        "print_freq": int(config.get("print_freq", 100)),
    })
    args.channel = channel
    args.im_size = im_size
    args.num_classes = num_classes
    args.class_names = data["class_names"]

    # --- Selection -----------------------------------------------------------
    sel_kwargs = config.get("selection_kwargs", {})
    if config["selection"] == "Uncertainty" and "selection_method" in config:
        sel_kwargs["selection_method"] = config["selection_method"]
    if config["selection"] == "Submodular" and "submodular_function" in config:
        sel_kwargs["function"] = config["submodular_function"]
    sel_kwargs.setdefault("balance", args.balance)

    selection_start = time.time()
    subset = select_coreset(config["selection"], dst_train, args, args.fraction, seed,
                            selection_kwargs=sel_kwargs)
    selection_time = time.time() - selection_start
    indices = np.asarray(subset["indices"])
    if_weighted = "weights" in subset

    # --- Coreset quality metrics --------------------------------------------
    quality = coreset_quality_metrics(
        selected_indices=indices,
        is_noisy=dst_train.is_noisy,
        targets_clean=dst_train.clean_targets,
        num_classes=num_classes,
        is_label_noisy=dst_train.is_label_noisy,
        is_feature_noisy=dst_train.is_feature_noisy,
        full_class_dist=data["full_class_dist"],
    )

    # --- Prepare training data ----------------------------------------------
    _augment_cifar(dst_train, im_size)
    if if_weighted:
        dst_subset = WeightedSubset(dst_train, indices, subset["weights"])
    else:
        dst_subset = torch.utils.data.Subset(dst_train, indices)

    workers = args.workers
    train_loader = torch.utils.data.DataLoader(
        dst_subset, batch_size=args.train_batch, shuffle=True,
        num_workers=workers, pin_memory=pin_memory, drop_last=False,
        persistent_workers=(workers > 0))
    test_loader = torch.utils.data.DataLoader(
        dst_test, batch_size=args.train_batch, shuffle=False,
        num_workers=workers, pin_memory=pin_memory,
        persistent_workers=(workers > 0))

    # --- Train ---------------------------------------------------------------
    network = build_network(args.model, channel, num_classes, im_size, args.device)
    train_start = time.time()
    network, history = train_model(network, train_loader, test_loader, args,
                                   num_epochs=args.epochs, verbose=verbose)
    train_time = time.time() - train_start

    # --- Evaluate on test ----------------------------------------------------
    test_eval = evaluate_test(network, test_loader, num_classes, args.device)

    # --- Separation: score entire training set (deterministic loader) --------
    eval_transform = _strip_augmentation(dst_train)
    try:
        full_train_loader = torch.utils.data.DataLoader(
            dst_train, batch_size=args.train_batch, shuffle=False,
            num_workers=workers, pin_memory=pin_memory,
            persistent_workers=(workers > 0))
        sep_eval = evaluate_train_separation(network, full_train_loader, dst_train, args.device)
    finally:
        dst_train.transform = eval_transform  # restore augmented transform

    # --- Assemble manifest ---------------------------------------------------
    manifest = {
        "config": config,
        "seed": seed,
        "num_classes": num_classes,
        "coreset_quality": quality,
        "performance": test_eval["performance"],
        "calibration": {k: v for k, v in test_eval["calibration"].items() if k != "reliability"},
        "reliability_diagram": test_eval["calibration"]["reliability"],
        "test_uncertainty": test_eval["uncertainty"],
        "separation": sep_eval["separation"],
        "timing": {
            "selection_sec": selection_time,
            "train_sec": train_time,
            "total_sec": time.time() - t0,
        },
        "history": history,
    }
    arrays = {
        "selected_indices": indices,
        "test_probs": test_eval["probs"],
        "test_preds": test_eval["predictions"],
        "test_targets": test_eval["targets"],
        "test_entropy": test_eval["entropy_per_sample"],
        "test_confidence": test_eval["confidence_per_sample"],
        "train_entropy": sep_eval["train_entropy"],
        "train_loss": sep_eval["train_loss"],
        "train_confidence": sep_eval["train_confidence"],
        "is_label_noisy": sep_eval["is_label_noisy"],
        "is_feature_noisy": sep_eval["is_feature_noisy"],
        "clean_targets": dst_train.clean_targets.numpy(),
        "noisy_targets": dst_train.targets.numpy() if hasattr(dst_train.targets, "numpy") else np.asarray(dst_train.targets),
    }

    save_run(output_dir, manifest, arrays)
    if verbose:
        print(f"[runner] saved to {output_dir} (acc={test_eval['performance']['accuracy']:.4f}, "
              f"bal_acc={test_eval['performance']['balanced_accuracy']:.4f}, "
              f"ece={test_eval['calibration']['ece']:.4f})")
    return manifest


def _strip_augmentation(dst_train):
    """Strip augmentation transforms for deterministic eval; return current transform for restore."""
    from torchvision import transforms
    t = dst_train.transform
    if isinstance(t, transforms.Compose):
        base = t.transforms[-1]
        dst_train.transform = base
    return t


# Axes that become irrelevant when a "parent" axis takes a particular value.
REDUNDANT_AXES = [
    {"trigger_key": "label_noise.ratio", "trigger_value": 0.0,
     "collapse": {"label_noise.type": "uniform",
                  "label_noise.flip_matrix": None}},
    {"trigger_key": "feature_noise.ratio", "trigger_value": 0.0,
     "collapse": {"feature_noise.sigma": 0.0}},
    {"trigger_key": "imbalance.ratio", "trigger_value": 1.0,
     "collapse": {"imbalance.majority_class": 0}},
]


def _get_nested(cfg, dotted_key, default=None):
    cur = cfg
    for p in dotted_key.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _set_nested(cfg, dotted_key, value):
    parts = dotted_key.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _canonicalize(combo_cfg, active_keys):
    """Collapse redundant axes in-place; return the set of axes that remain meaningful."""
    effective = set(active_keys)
    for rule in REDUNDANT_AXES:
        tk = rule["trigger_key"]
        if _get_nested(combo_cfg, tk) == rule["trigger_value"]:
            for dep_key, canon_val in rule["collapse"].items():
                if dep_key in effective:
                    _set_nested(combo_cfg, dep_key, canon_val)
                    effective.discard(dep_key)
    return effective


def _combo_name(keys, values):
    parts = []
    for k, v in zip(keys, values):
        short = k.split(".")[-1]
        parts.append(f"{short}={v}")
    return "__".join(parts)


def _dataset_key(cfg):
    """Return a hashable key representing the dataset-defining portion of a config."""
    return (
        tuple(cfg.get("class_ids", [3, 5])),
        cfg.get("data_path", "data"),
        _get_nested(cfg, "imbalance.ratio", 1.0),
        _get_nested(cfg, "imbalance.majority_class", 0),
        _get_nested(cfg, "label_noise.ratio", 0.0),
        _get_nested(cfg, "label_noise.type", "uniform"),
        _get_nested(cfg, "label_noise.flip_matrix", None),
        _get_nested(cfg, "feature_noise.ratio", 0.0),
        _get_nested(cfg, "feature_noise.sigma", 0.1),
    )


def run_grid(grid_config: Dict[str, Any], output_root: str, verbose: bool = True):
    """Expand a grid config and run all experiments.

    Iteration order: for each unique (dataset config, seed) pair, the dataset
    is built once and all (selection, fraction) combinations are evaluated
    before moving on.  This avoids redundant dataset rebuilds and keeps the
    GPU pipeline warm across method comparisons on the same data.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    base = grid_config.get("base", {})
    grid = grid_config.get("grid", {})
    seeds = grid_config.get("seeds", [0])

    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    combos = list(itertools.product(*value_lists)) if keys else [()]

    seen_combo_names = set()
    unique_combos = []
    skipped = 0
    for combo in combos:
        combo_cfg = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            _set_nested(combo_cfg, k, v)
        effective_keys = _canonicalize(combo_cfg, keys)
        name_keys = [k for k in keys if k in effective_keys]
        name_values = [_get_nested(combo_cfg, k) for k in name_keys]
        combo_name = _combo_name(name_keys, name_values) or "default"
        if combo_name in seen_combo_names:
            skipped += 1
            continue
        seen_combo_names.add(combo_name)
        unique_combos.append((combo_cfg, combo_name))

    if verbose:
        print(f"[runner] grid: {len(combos)} raw combos -> {len(unique_combos)} unique "
              f"({skipped} collapsed as redundant); {len(seeds)} seeds each")

    # Group combos by their dataset-defining config so we build the dataset once
    # per (dataset config, seed) and run all methods before moving on.
    dataset_groups: Dict[tuple, List] = defaultdict(list)
    for combo_cfg, combo_name in unique_combos:
        dk = _dataset_key(combo_cfg)
        dataset_groups[dk].append((combo_cfg, combo_name))

    if verbose:
        print(f"[runner] {len(dataset_groups)} unique dataset configs "
              f"x {len(seeds)} seeds = {len(dataset_groups) * len(seeds)} dataset builds")

    results = []
    os.makedirs(output_root, exist_ok=True)

    for dk, combos_in_group in dataset_groups.items():
        for seed in seeds:
            # Check if every run in this (dataset, seed) group is already done.
            pending = []
            for combo_cfg, combo_name in combos_in_group:
                run_dir = os.path.join(output_root, combo_name, f"seed{seed}")
                if os.path.exists(os.path.join(run_dir, "manifest.json")) and \
                        not grid_config.get("overwrite", False):
                    if verbose:
                        print(f"[runner] skipping existing: {run_dir}")
                else:
                    pending.append((combo_cfg, combo_name, run_dir))

            if not pending:
                continue

            # Build dataset once for all methods in this group.
            ref_cfg = copy.deepcopy(combos_in_group[0][0])
            ref_cfg["seed"] = int(seed)
            if verbose:
                dk_str = f"imb={_get_nested(ref_cfg, 'imbalance.ratio', 1.0)}, " \
                         f"noise={_get_nested(ref_cfg, 'label_noise.ratio', 0.0)}" \
                         f"({_get_nested(ref_cfg, 'label_noise.type', 'uniform')}), " \
                         f"seed={seed}"
                print(f"[runner] building dataset: {dk_str} "
                      f"({len(pending)} method/fraction combos)")
            prebuilt_data = _build_dataset(ref_cfg, int(seed))

            for combo_cfg, combo_name, run_dir in pending:
                cfg = copy.deepcopy(combo_cfg)
                cfg["seed"] = int(seed)
                manifest = run_experiment(cfg, run_dir, verbose=verbose,
                                          prebuilt_data=prebuilt_data)
                results.append((run_dir, manifest))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results
