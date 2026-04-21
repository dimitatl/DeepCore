"""Feature noise: additive Gaussian noise applied at dataset __getitem__ time."""
import numpy as np


def build_feature_noise(n_samples, sample_shape, noise_ratio, sigma=0.1, seed=0):
    """Return dict {'noise', 'is_noisy', 'sigma'}.

    `noise_ratio` is the fraction of samples that receive any noise (rest stay clean).
    `sigma` controls the magnitude (std of Gaussian applied to normalized tensor).

    `noise` is an array of shape (n_samples, *sample_shape) of float32; zeros for
    non-selected samples. Applied at __getitem__ time on the tensor after normalization.
    """
    rng = np.random.RandomState(seed)
    is_noisy = rng.rand(n_samples) < noise_ratio

    noise = np.zeros((n_samples, *sample_shape), dtype=np.float32)
    if is_noisy.any():
        idx = np.where(is_noisy)[0]
        noise[idx] = rng.normal(loc=0.0, scale=sigma, size=(len(idx), *sample_shape)).astype(np.float32)

    return {"noise": noise, "is_noisy": is_noisy, "sigma": sigma, "noise_ratio": noise_ratio}
