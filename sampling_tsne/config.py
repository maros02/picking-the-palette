from dataclasses import dataclass
from enum import Enum


class SamplingMethod(str, Enum):
    FULL = "full" # no sampling: embed all points (the baseline)
    UNIFORM = "uniform"
    POISSON_DISK = "poisson_disk"
    GEOSKETCH = "geosketch"
    SCSAMPLER = "scsampler"
    SKETCHKH = "sketchkh"
    FARTHEST_POINT = "farthest_point"


class AffinityMode(str, Enum):
    NONE = "none"
    DIJKSTRA_DISTANCES = "distances"
    DIJKSTRA_AFFINITIES = "affinities"


@dataclass
class Config:

    dataset: str = "mnist"
    output_dir: str = "outputs"
    random_state: int = 42 # default rnd state
    n_jobs: int = 8

    rare_class_threshold: float = 0.01 # <1% = rare class
    knn_k: int = 10 # for AC and metrics
    pca_pre_dims: int = 50
    early_iter: int = 250
    late_iter: int = 750

    # config space C = (r, P, S, A)
    sample_rate: float = None
    perplexity: float = None
    num_neighbors: int = None
    epsilon: float | bool = False
    sample_perplexity: float | None = None
    sampling_method: SamplingMethod = SamplingMethod.UNIFORM
    sample_affinity_mode: AffinityMode = AffinityMode.NONE