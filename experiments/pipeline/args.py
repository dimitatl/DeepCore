"""Build an argparse.Namespace-like object that DeepCore methods expect."""
from types import SimpleNamespace
import torch


DEFAULTS = dict(
    dataset="CIFAR10",
    model="ResNet18",
    selection="Uniform",
    num_exp=1,
    epochs=50,
    data_path="data",
    gpu=None,
    print_freq=50,
    fraction=0.1,
    seed=0,
    workers=2,
    # Optimizer (training)
    optimizer="SGD",
    lr=0.1,
    min_lr=1e-4,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    scheduler="CosineAnnealingLR",
    gamma=0.5,
    step_size=50,
    # Training batches
    batch=128,
    train_batch=128,
    selection_batch=128,
    test_interval=1,
    test_fraction=1.0,
    # Selection
    selection_epochs=10,
    selection_momentum=0.9,
    selection_weight_decay=5e-4,
    selection_optimizer="SGD",
    selection_nesterov=True,
    selection_lr=0.1,
    selection_test_interval=0,
    selection_test_fraction=1.0,
    balance=True,
    submodular="GraphCut",
    submodular_greedy="LazyGreedy",
    uncertainty="Entropy",
    save_path="",
    resume="",
)


def build_args(config_overrides=None):
    ns = SimpleNamespace(**DEFAULTS)
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(ns, k, v)
    ns.device = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(ns, "train_batch", None) is None:
        ns.train_batch = ns.batch
    if getattr(ns, "selection_batch", None) is None:
        ns.selection_batch = ns.batch
    return ns
