"""Run a trained model over a dataset and compute all metrics."""
import numpy as np
import torch
import torch.nn.functional as F

from ..metrics import (
    performance_metrics,
    calibration_metrics,
    uncertainty_metrics,
    separation_metrics,
)


@torch.no_grad()
def collect_predictions(network, loader, device):
    """Return (logits, probs, targets) as numpy arrays."""
    network.eval()
    all_logits = []
    all_targets = []
    for input_, target in loader:
        input_ = input_.to(device)
        out = network(input_)
        all_logits.append(out.detach().cpu().numpy())
        all_targets.append(np.asarray(target))
    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    probs = _softmax(logits)
    return logits, probs, targets


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def per_sample_loss(probs, targets, eps=1e-12):
    probs = np.clip(probs, eps, 1.0)
    return -np.log(probs[np.arange(len(targets)), targets])


def evaluate_test(network, test_loader, num_classes, device):
    logits, probs, targets = collect_predictions(network, test_loader, device)
    preds = probs.argmax(axis=1)
    perf = performance_metrics(targets, preds, num_classes)
    calib = calibration_metrics(probs, targets)
    unc = uncertainty_metrics(probs)
    return {
        "performance": perf,
        "calibration": calib,
        "uncertainty": {k: v for k, v in unc.items() if not k.endswith("per_sample")},
        "predictions": preds,
        "probs": probs,
        "targets": targets,
        "entropy_per_sample": unc["entropy_per_sample"],
        "confidence_per_sample": unc["confidence_per_sample"],
    }


def evaluate_train_separation(network, train_loader, noisy_dataset, device):
    """Score every training sample and test whether scores separate clean/noisy.

    train_loader must iterate the noisy training set in deterministic order.
    """
    logits, probs, targets = collect_predictions(network, train_loader, device)
    losses = per_sample_loss(probs, targets)
    from ..metrics.uncertainty import predictive_entropy, max_confidence
    ent = predictive_entropy(probs)
    conf = max_confidence(probs)

    is_noisy = noisy_dataset.is_noisy
    sep = separation_metrics(ent, losses, conf, is_noisy)
    return {
        "separation": sep,
        "train_entropy": ent,
        "train_loss": losses,
        "train_confidence": conf,
        "is_label_noisy": noisy_dataset.is_label_noisy,
        "is_feature_noisy": noisy_dataset.is_feature_noisy,
    }
