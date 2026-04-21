"""Build a binary (or k-class) CIFAR-10 subset compatible with DeepCore methods."""
from copy import deepcopy
import numpy as np
import torch
from torch import tensor, long
from torchvision import datasets, transforms


CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def _filter_classes(dst, class_ids):
    targets = dst.targets if isinstance(dst.targets, list) else dst.targets.tolist()
    keep = [i for i, t in enumerate(targets) if t in class_ids]
    class_map = {c: i for i, c in enumerate(class_ids)}

    dst.data = dst.data[keep]
    new_targets = [class_map[targets[i]] for i in keep]
    dst.targets = tensor(new_targets, dtype=long)
    dst.classes = [dst.classes[c] for c in class_ids]
    return dst


def build_binary_cifar10(data_path, class_ids=(3, 5)):
    """Return a DeepCore-compatible binary CIFAR-10.

    Returns the same tuple signature as deepcore.datasets.CIFAR10:
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
    """
    channel = 3
    im_size = (32, 32)
    mean = CIFAR10_MEAN
    std = CIFAR10_STD

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)

    dst_train = _filter_classes(dst_train, list(class_ids))
    dst_test = _filter_classes(dst_test, list(class_ids))

    num_classes = len(class_ids)
    class_names = dst_train.classes
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
