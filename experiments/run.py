"""CLI entry point.

Usage:
    # Single experiment from YAML (one condition, one seed)
    python -m experiments.run --config experiments/config/single.yaml \
                              --output results/single_run

    # Grid experiment (cartesian product across methods/noise/fraction/seeds)
    python -m experiments.run --grid experiments/config/grid_full.yaml \
                              --output results/grid_full

    # Analyze results already on disk
    python -m experiments.run --analyze results/grid_full --output results/grid_full/analysis
"""
import argparse
import os
import sys

# Ensure project root is importable when run from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from experiments.pipeline import run_experiment, run_grid


def _cmd_single(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    os.makedirs(args.output, exist_ok=True)
    run_experiment(config, args.output, verbose=True)


def _cmd_grid(args):
    with open(args.grid) as f:
        grid_cfg = yaml.safe_load(f)
    run_grid(grid_cfg, args.output, verbose=True)


def _cmd_analyze(args):
    from experiments.analysis import load_runs, aggregate, compare_methods_ttest
    from experiments.analysis.plots import (
        plot_accuracy_vs_noise,
        plot_accuracy_vs_fraction,
    )
    import json

    runs = load_runs(args.analyze)
    if not runs:
        print(f"no runs found under {args.analyze}")
        return

    os.makedirs(args.output, exist_ok=True)

    metrics = [
        "performance.accuracy",
        "performance.balanced_accuracy",
        "performance.macro_f1",
        "calibration.ece",
        "calibration.mce",
        "calibration.brier",
        "coreset_quality.frac_noisy_selected",
        "coreset_quality.selection_bias_ratio",
        "coreset_quality.minority_retention",
        "separation.auroc_entropy",
        "separation.auroc_loss",
        "test_uncertainty.entropy_mean",
    ]

    # Aggregate across (method, label_noise ratio, feature_noise ratio, imbalance ratio, fraction)
    group_by = [
        "config.selection",
        "config.label_noise.ratio",
        "config.feature_noise.ratio",
        "config.imbalance.ratio",
        "config.fraction",
    ]
    rows = aggregate(runs, group_by, metrics)
    with open(os.path.join(args.output, "aggregate.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"[analyze] wrote aggregate.json with {len(rows)} groups")

    # T-tests vs Uniform baseline
    ttest_rows = compare_methods_ttest(
        runs,
        metric="performance.accuracy",
        group_by_method="config.selection",
        reference="Uniform",
        condition_keys=["config.label_noise.ratio", "config.feature_noise.ratio",
                        "config.imbalance.ratio", "config.fraction"],
    )
    with open(os.path.join(args.output, "ttest_vs_uniform.json"), "w") as f:
        json.dump(ttest_rows, f, indent=2, default=str)

    # Basic plots
    plot_accuracy_vs_noise(rows, out_path=os.path.join(args.output, "acc_vs_label_noise.png"))
    plot_accuracy_vs_fraction(rows, out_path=os.path.join(args.output, "acc_vs_fraction.png"))
    print(f"[analyze] plots saved to {args.output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, help="YAML for a single experiment")
    p.add_argument("--grid", type=str, help="YAML describing a grid of experiments")
    p.add_argument("--analyze", type=str, help="Path to a results root to aggregate + plot")
    p.add_argument("--output", type=str, required=True, help="Output directory")
    args = p.parse_args()

    if args.config:
        _cmd_single(args)
    elif args.grid:
        _cmd_grid(args)
    elif args.analyze:
        _cmd_analyze(args)
    else:
        p.error("specify one of --config, --grid, --analyze")


if __name__ == "__main__":
    main()
