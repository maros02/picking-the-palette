from geosketch import gs
import numpy as np


def geosketch_sample(X, n_samples, seed):
    return np.array(gs(X, n_samples, seed=seed))