"""Dataset wrapper that injects label and/or feature noise while tracking which samples are noisy.

Exposes `.targets`, `.classes`, `.transform` so DeepCore coreset methods (which do
`dataset.targets == c`, `len(dataset.classes)`, etc.) work unchanged.

After instantiation, `is_label_noisy` / `is_feature_noisy` are boolean arrays indicating
which samples had noise applied. `clean_targets` retains the original labels.
"""
import copy
import numpy as np
import torch
from torch import tensor, long


class NoisyDataset(torch.utils.data.Dataset):
    def __init__(self, base, label_noise=None, feature_noise=None):
        """
        base: a dataset with .data (ndarray), .targets (tensor/list), .classes, .transform
        label_noise: dict from experiments.noise.label_noise.build_label_noise or None
            keys: 'new_targets' (tensor), 'is_noisy' (bool array)
        feature_noise: dict from experiments.noise.feature_noise.build_feature_noise or None
            keys: 'noise' (ndarray same shape as .data, float), 'is_noisy' (bool array)
        """
        self.base = base
        self.transform = getattr(base, "transform", None)
        self.classes = getattr(base, "classes", None)

        base_targets = base.targets
        if not isinstance(base_targets, torch.Tensor):
            base_targets = torch.tensor(list(base_targets), dtype=long)
        self.clean_targets = base_targets.clone()

        n = len(base)

        if label_noise is not None:
            self.targets = label_noise["new_targets"].clone()
            self.is_label_noisy = np.asarray(label_noise["is_noisy"], dtype=bool)
        else:
            self.targets = base_targets.clone()
            self.is_label_noisy = np.zeros(n, dtype=bool)

        if feature_noise is not None:
            self._feature_noise = feature_noise["noise"].astype(np.float32)
            self.is_feature_noisy = np.asarray(feature_noise["is_noisy"], dtype=bool)
        else:
            self._feature_noise = None
            self.is_feature_noisy = np.zeros(n, dtype=bool)

    @property
    def data(self):
        return self.base.data

    @property
    def is_noisy(self):
        return self.is_label_noisy | self.is_feature_noisy

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, _ = self.base[idx]
        if self._feature_noise is not None and self.is_feature_noisy[idx]:
            noise = torch.from_numpy(self._feature_noise[idx])
            if noise.shape == img.shape:
                img = img + noise
            else:
                # Accommodate channel-first tensor vs HWC noise
                if noise.dim() == 3 and noise.shape[-1] == img.shape[0]:
                    noise = noise.permute(2, 0, 1)
                img = img + noise
        target = int(self.targets[idx].item()) if isinstance(self.targets, torch.Tensor) else int(self.targets[idx])
        return img, target
