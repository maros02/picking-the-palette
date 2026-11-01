import numpy as np
import tqdm


def poisson_disk_sample(X, sample_size, p_matrix):
    n = X.shape[0]
    if sample_size <= 0 or sample_size > n:
        raise ValueError(f"sample_size must be in [1, {n}], got {sample_size}")
    if p_matrix is None:
        raise ValueError("poisson-disk sampling requires p_matrix.")

    point_weights = np.array(np.sum(p_matrix, axis=0))[0]
    removed = set()

    for _ in tqdm.tqdm(range(n - int(sample_size))):
        sample = int(np.argmax(point_weights)) # sample point to be removed
        point_weights -= p_matrix[sample].toarray()[0]
        point_weights[sample] = -np.inf # set weight to -inf to avoid re-selection
        removed.add(sample)

    kept = np.array(sorted(set(range(n)) - removed), dtype=int)
    return kept