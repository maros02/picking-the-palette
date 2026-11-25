import numpy as np
from sketchKH import kernel_herding_main


def sketchkh_sample(X, n_samples, seed):
    X = np.asarray(X, dtype=np.float64)
    X = X / X.std()
    sample_set_ind = np.arange(X.shape[0])
    idx = kernel_herding_main(
        sample_set_ind=sample_set_ind,
        X=X,
        frequency_seed=seed,
        num_subsamples=n_samples,
    )
    return np.sort(np.asarray(idx, dtype=int))