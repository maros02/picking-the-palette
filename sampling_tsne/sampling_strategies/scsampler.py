import numpy as np
from scsampler import scsampler


def scsampler_sample(X, sample_size, seed):
    X = np.asarray(X, dtype=np.float64)
    X = X / X.std() # rescale to avoid (1/d)^200 overflow bug
    idx = scsampler(
        data=np.ascontiguousarray(X),
        n_obs=sample_size,
        random_state=seed,
        copy=False,
        random_split=16
    )
    return np.sort(np.asarray(idx, dtype=int))