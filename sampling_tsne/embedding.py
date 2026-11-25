from time import perf_counter

import numpy as np
import scipy.sparse as sp
from openTSNE import TSNEEmbedding, affinity
from openTSNE.affinity import joint_probabilities_nn
from openTSNE.nearest_neighbors import Annoy
from scipy.sparse import csr_matrix

from .dijkstra import dijkstra


def distance_matrix_and_annoy(
    X: np.ndarray,
    num_neighbors: int,
    n_jobs: int,
    random_state: int,
    epsilon: float | bool = False,
    return_neighbors: bool = False,
):
    """Modified method from Skrodzki et al., computes the distance matrix and annoy index."""

    annoy = make_annoy(X, num_neighbors, n_jobs, random_state)
    neighbors, distances = annoy.build()
    if epsilon:
        distances[distances == 0.0] = epsilon

    # convert to CSR matrix
    row_indices = np.repeat(
        np.arange(X.shape[0]),
        num_neighbors
    )
    data_distance_matrix = csr_matrix(
        (distances.flatten(), (row_indices, neighbors.flatten())),
        shape=(X.shape[0], X.shape[0])
    )
    if return_neighbors: return data_distance_matrix, annoy, neighbors, distances
    return data_distance_matrix, annoy


def make_annoy(X, num_neighbors, n_jobs, random_state) -> Annoy:
    return Annoy(
        data=X,
        k=num_neighbors,
        metric="euclidean",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=False,
    )


def compute_affinities(
    annoy,
    perplexity: float,
    n_jobs: int,
    random_state: int):
    return affinity.PerplexityBasedNN(
        knn_index=annoy,
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=False,
    )


def fit_tsne(
    init: np.ndarray,
    n_jobs: int,
    random_state: int,
    early_iter: int,
    late_iter: int,
    affinities) -> np.ndarray:
    """Create the t-SNE embedding with FIt-SNE.

    Args:
        init (np.ndarray): initialized embedding (PCA)
        n_jobs (int): number of threads
        random_state (int): random seed
        early_iter (int): number of early iterations
        late_iter (int): number of late iterations
        affinities: affinity matrix

    Returns:
        A t-SNE embedding!
    """

    # create embedding
    emb = TSNEEmbedding(
        init,
        affinities,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
        random_state=random_state
    )

    # optimization
    emb = emb.optimize(n_iter=early_iter, exaggeration=12, momentum=0.5) # early
    emb = emb.optimize(n_iter=late_iter, exaggeration=1, momentum=0.8) # late

    return np.asarray(emb)

# --- Below: Dijkstra on Distances by F. Markov (2025) ---

EPSILON = 1e-12

class PrecomputedAffinities(affinity.Affinities):

    def __init__(self, P, normalize=True):
        if not isinstance(P, sp.csr_matrix):
            P = sp.csr_matrix(P)
        if normalize:
            P /= np.sum(P)
        self.P = P

    def to_new(self, data, return_distances=True):
        raise RuntimeError("Precomputed affinity matrices cannot be queried.")


def reconstruct_dijkstra_distances(
    distance_matrix,
    sample_idx: np.ndarray,
    sample_perplexity: float,
    k = None,
) -> sp.csr_matrix:

    distance_matrix = distance_matrix.tocsr()
    if k is None: k = max(1, int(3 * sample_perplexity)) + 1

    t0 = perf_counter()
    dijkstra_distances = dijkstra(distance_matrix, directed=False, indices=sample_idx, sample_indices=sample_idx, k=k)
    print(f"  dijkstra: {perf_counter() - t0:.2f}s")
    dijkstra_distances = dijkstra_distances[:, sample_idx]

    # keep only finite, positive entries
    mask = np.isfinite(dijkstra_distances) & (dijkstra_distances > EPSILON)
    rows, cols = np.where(mask)
    values = dijkstra_distances[rows, cols]
    result = sp.csr_matrix((values, (rows, cols)), shape=(len(sample_idx), len(sample_idx)))

    # symmetrize
    result = result.maximum(result.T)
    result.setdiag(0)
    result.eliminate_zeros()

    return result # dijkstra distances


def sample_graph_from_full_distances(
    distance_matrix,
    sample_idx: np.ndarray,
    sample_perplexity: float,
) -> PrecomputedAffinities:

    sample_distances = reconstruct_dijkstra_distances(distance_matrix, sample_idx, sample_perplexity) # dijkstra distances
    n = sample_distances.shape[0] # n of sampled points
    max_k = max(np.diff(sample_distances.indptr)) # max num of nonzero neighbors in any row of the sparse matrix

    knn_indices = np.full((n, max_k), fill_value=0, dtype=int) # sample indices
    knn_dists = np.full((n, max_k), fill_value=np.inf, dtype=float) # corresponding Dijkstra distances

    # process each sample point (row by row)
    for i in range(n):
        row_start = sample_distances.indptr[i]
        row_end = sample_distances.indptr[i + 1]
        indices = sample_distances.indices[row_start:row_end]
        data = sample_distances.data[row_start:row_end]

        sorted_idx = np.argsort(data)
        indices = indices[sorted_idx]
        data = data[sorted_idx]

        # sort neighbors by ascending shortest-path dist
        k_i = len(indices)
        num_neighbors = min(k_i, max_k)
        knn_indices[i, :num_neighbors] = indices[:num_neighbors]
        knn_dists[i, :num_neighbors] = data[:num_neighbors]

    # construct p matrix and affinities
    P = joint_probabilities_nn(knn_indices, knn_dists, [sample_perplexity], True)
    sample_affinities = PrecomputedAffinities(P, normalize=True)

    return sample_affinities


def sample_graph_from_full_affinities(
    full_P: sp.spmatrix, # full P matrix
    sample_idx: np.ndarray,
    sample_perplexity: float,
) -> PrecomputedAffinities:

    reversed_affinities = full_P.copy()
    reversed_affinities.data = 1.0 / (reversed_affinities.data + EPSILON)

    sample_distances = reconstruct_dijkstra_distances(reversed_affinities, sample_idx, sample_perplexity) # reconstruct distances
    sample_affinities_matrix = sample_distances.copy()
    sample_affinities_matrix.data = 1.0 / (sample_affinities_matrix.data + EPSILON)

    return PrecomputedAffinities(sample_affinities_matrix, normalize=True)