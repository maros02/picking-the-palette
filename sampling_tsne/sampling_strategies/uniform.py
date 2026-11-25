import numpy as np


def uniform_sample(n: int, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    idx.sort()
    return idx