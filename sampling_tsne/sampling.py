import numpy as np

from .config import SamplingMethod
from .sampling_strategies import (
    uniform_sample,
    poisson_disk_sample,
    geosketch_sample,
    scsampler_sample,
    sketchkh_sample,
    farthest_point_sample,
)


def sample_indices(
    X: np.ndarray, # X_reduced
    method: SamplingMethod,
    sample_rate: float,
    seed: int,
    affinities=None,
) -> np.ndarray:
    n = X.shape[0]
    sample_size = max(2, int(n * sample_rate))

    if method == SamplingMethod.FULL:
        return np.arange(n) # baseline
    if method == SamplingMethod.UNIFORM:
        return uniform_sample(n, sample_size, seed)
    if method == SamplingMethod.POISSON_DISK:
        return poisson_disk_sample(X, sample_size, affinities.P) # pass p_matrix
    if method == SamplingMethod.GEOSKETCH:
        return geosketch_sample(X, sample_size, seed)
    if method == SamplingMethod.SCSAMPLER:
        return scsampler_sample(X, sample_size, seed)
    if method == SamplingMethod.SKETCHKH:
        return sketchkh_sample(X, sample_size, seed)
    if method == SamplingMethod.FARTHEST_POINT:
        return farthest_point_sample(X, sample_size, seed)
    raise ValueError(f"Unsupported sampling method: {method}")
