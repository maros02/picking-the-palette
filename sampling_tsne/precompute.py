"""
This file handles all the precomputation logic!
the *actual* calculations are computed in embedding.py
"""

import hashlib
import pickle
from pathlib import Path
from time import perf_counter
import numpy as np

from .config import AffinityMode, Config, SamplingMethod
from .embedding import (
    PrecomputedAffinities,
    compute_affinities,
    distance_matrix_and_annoy,
    make_annoy,
)


def load_or_compute_high_dimensional_distance_matrix_and_affinities(X_r, config):
    """Modified method from Skrodzki et al. to load/compute the distance matrix and affinities of the data."""

    distance_matrix = None
    affinities = None
    setup_time_distances = 0.0
    setup_time_affinities = 0.0

    dist_computed = False
    aff_computed = False

    require_distance_matrix = needs_full_distance_matrix(config)
    require_affinities = needs_full_affinities(config)

    if require_distance_matrix or require_affinities:

        cache_prefix = generate_cache_prefix(
            dataset=config.dataset,
            data=X_r,
            num_neighbors=config.num_neighbors,
            perplexity=config.perplexity,
            random_state=config.random_state,
        )

        distance_matrix_path, affinities_path = generate_precomputed_directories(cache_prefix)

        if require_distance_matrix:
            distance_matrix = load_distance_matrix(distance_matrix_path)
            if distance_matrix is None: # case: there is no precomputed distance matrix
                dist_computed = True
                t0 = perf_counter()
                distance_matrix = precompute_distance_matrix(X_r, config, distance_matrix_path)
                setup_time_distances = perf_counter() - t0

        if require_affinities:
            affinities = load_affinities(affinities_path)
            if affinities is None:
                aff_computed = True
                t0 = perf_counter()
                affinities = precompute_affinities(X_r, config, affinities_path)
                setup_time_affinities = perf_counter() - t0

        print(f"[precompute] precomputing missing values took dist={setup_time_distances:.2f}s affinity={setup_time_affinities:.2f}s.")

    else:
        print("[precompute] no loading or precomputation needed - skipping!")

    return distance_matrix, affinities, setup_time_distances, setup_time_affinities, dist_computed, aff_computed


def precompute_distance_matrix(X, config, distance_matrix_path):
    distance_matrix, annoy, neighbors, distances = distance_matrix_and_annoy(
        X=X,
        num_neighbors=config.num_neighbors,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        epsilon=config.epsilon,
        return_neighbors=True,
    )
    save_pickle(distance_matrix_path, distance_matrix)

    return distance_matrix


def precompute_affinities(X, config, affinities_path):
    annoy = make_annoy(X, config.num_neighbors, config.n_jobs, config.random_state)
    affinities = compute_affinities(
        annoy=annoy,
        perplexity=config.perplexity,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )

    save_pickle(affinities_path, affinities.P) # save p matrix
    affinities = PrecomputedAffinities(affinities.P, normalize=False)  # normalize return type

    return affinities


def load_distance_matrix(path: Path):
    if not path.is_file(): return None

    print("Loading high-dimensional distance matrix from saved pickle.")

    with open(path, "rb") as distance_matrix_file:
        distance_matrix = pickle.load(distance_matrix_file)

    return distance_matrix


def load_affinities(path: Path):
    if not path.is_file(): return None

    print("Loading high-dimensional affinities from saved pickle.")
    with open(path, "rb") as affinities_file:
        affinities = pickle.load(affinities_file)

    return PrecomputedAffinities(
        affinities.P if hasattr(affinities, "P") else affinities,
        normalize=False,
    )


# --- helpers below
def generate_precomputed_directories(cache_prefix) -> tuple[Path, Path]:
    precomputed_directory = Path(__file__).resolve().parents[1] / "precomputed"
    precomputed_directory.mkdir(parents=True, exist_ok=True)
    distance_matrix_path = precomputed_directory / f"{cache_prefix}_high_dimensional_distance_matrix.pkl"
    affinities_path = precomputed_directory / f"{cache_prefix}_affinities.pkl"

    return distance_matrix_path, affinities_path


def needs_full_distance_matrix(config: Config) -> bool:
    return config.sample_affinity_mode == AffinityMode.DIJKSTRA_DISTANCES


def needs_full_affinities(config: Config) -> bool:
    return config.sampling_method == SamplingMethod.POISSON_DISK or config.sample_affinity_mode == AffinityMode.DIJKSTRA_AFFINITIES


def generate_cache_prefix(dataset: str, data: np.ndarray, num_neighbors: int, perplexity: float, random_state: int):
    cache_prefix = (
        f"{dataset}"
        f"_perplexity_{perplexity}"
        f"_neighbors_{num_neighbors}"
        f"_seed_{random_state}"
        f"_{hash_array(data)}" # extra hash
    )
    return cache_prefix


def hash_array(data: np.ndarray) -> str:
    return hashlib.blake2b(data.tobytes(), digest_size=8).hexdigest()


def save_pickle(path: Path, value) -> None:
    with open(path, "wb") as output_file: pickle.dump(value, output_file)
