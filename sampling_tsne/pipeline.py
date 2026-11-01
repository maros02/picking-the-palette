import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import Config, AffinityMode, SamplingMethod
from .data import load_dataset, pca_reduce
from .embedding import (
    compute_affinities,
    fit_tsne,
    make_annoy,
    sample_graph_from_full_affinities,
    sample_graph_from_full_distances,
)
from .metrics import (
    class_coverage,
    hausdorff_distance,
    identify_rare_classes,
    rare_class_purity,
    avg_rare_class_recall,
    rare_class_coverage,
    save_metrics,
    cpd_spearman,
    knn_class_overlap,
    knn_preservation,
    neighborhood_hit,
    cluster_freq_eval,
)
from .precompute import load_or_compute_high_dimensional_distance_matrix_and_affinities
from .sampling import sample_indices


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def run(config: Config):

    out_dir, arrays_dir = generate_directories(config)

    # init conf vars
    config.num_neighbors = int(config.perplexity * 3)
    config.sample_perplexity = max(1, int(config.perplexity * config.sample_rate))

    print(f"1. loading {config.dataset} dataset")
    X, y = load_dataset(config.dataset, seed=config.random_state)

    print(f"2. PCA reduce to {config.pca_pre_dims} dimensions")
    X_reduced = pca_reduce(X, config.pca_pre_dims, config.random_state)

    print(f"3. precompute full distance matrix or affinities")
    distance_matrix, affinities, t_setup_dist, t_setup_affinity, dist_computed, affinity_computed = load_or_compute_high_dimensional_distance_matrix_and_affinities(
        X_r=X_reduced,
        config=config,
    )

    print(f"4. sampling data ({config.sampling_method.value})")
    t0 = perf_counter() # sampling time
    sample_idx = sample_indices(
        X=X_reduced,
        method=config.sampling_method,
        sample_rate=config.sample_rate,
        seed=config.random_state,
        affinities=affinities,
    )
    X_sample = X_reduced[sample_idx] # sampled datapoints
    y_sample = y[sample_idx] # sampled labels
    t_sampling = perf_counter() - t0

    print("5. PCA init")
    init_pca = pca_reduce(X_reduced, 2, config.random_state)
    init_pca_sample = init_pca[sample_idx]

    print("6. checking sampled perplexity")
    sample_perplexity = check_sample_perplexity(config.sample_perplexity, sample_idx)

    print("7. sample affinity construction")
    t0 = perf_counter()
    if config.sample_affinity_mode == AffinityMode.NONE:
        print("computing sample distance matrix & affinities (standard)...")
        sample_annoy = make_annoy(
            X_sample,
            num_neighbors=int(sample_perplexity * 3), # sample NN ~ 3 * sample perp
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )
        sample_affinities = compute_affinities(
            annoy=sample_annoy,
            perplexity=sample_perplexity,
            n_jobs=config.n_jobs,
            random_state=config.random_state,
        )
    elif config.sample_affinity_mode == AffinityMode.DIJKSTRA_DISTANCES:
        print("computing sample affinities via Dijkstra on distances...")
        sample_affinities = sample_graph_from_full_distances(
            distance_matrix, sample_idx, sample_perplexity)
    elif config.sample_affinity_mode == AffinityMode.DIJKSTRA_AFFINITIES: # Unused
        print("computing sample affinities via Dijkstra on affinities...")
        # pass the p_matrix from full affinities
        sample_affinities = sample_graph_from_full_affinities(
            affinities.P, sample_idx, sample_perplexity)
    else:
        raise ValueError(f"unknown affinity mode: {config.sample_affinity_mode}")
    t_sample_affinity = perf_counter() - t0

    print("8. embed sample")
    t0 = perf_counter()
    emb_sample = fit_tsne(
        init=init_pca_sample,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        early_iter=config.early_iter,
        late_iter=config.late_iter,
        affinities=sample_affinities
    )
    t_optim_sample = perf_counter() - t0

    sample_time = t_sampling + t_sample_affinity + t_optim_sample
    print(
        f"sample pipeline: {sample_time:.2f}s "
        f"(sample={t_sampling:.2f}s, "
        f"affinity={t_sample_affinity:.2f}s, "
        f"optim={t_optim_sample:.2f}s)"
    )

    print("9. save snapshots")
    # store sample arrays
    np.save(arrays_dir / "sample_idx.npy", sample_idx)
    np.save(arrays_dir / "sample_embedding.npy", emb_sample)
    np.save(arrays_dir / "X_sample.npy", X_sample)
    np.save(arrays_dir / "y_sample.npy", y_sample)

    # dataset
    np.save(arrays_dir / "X_reduced.npy", X_reduced)
    np.save(arrays_dir / "y.npy", y)

    # metadata + timings
    run_data = {
        "dataset": config.dataset,
        "seed": config.random_state,
        "sampling_method": config.sampling_method.value,
        "sample_rate": config.sample_rate,
        "perplexity": config.perplexity,
        "sample_perplexity": sample_perplexity, # effective (post-clamp) value used
        "sample_perplexity_input": config.sample_perplexity, # pre-clamp perplexity input
        "affinity_mode": config.sample_affinity_mode.value,
        "sample_emb_time_s": sample_time,
        "setup_time_dist_s": t_setup_dist,
        "setup_time_affinity_s": t_setup_affinity,
        "setup_dist_computed": dist_computed, # distance matrix computed this run (cache miss)
        "setup_affinity_computed": affinity_computed, # affinities computed this run (cache miss)
        "t_sampling_s": t_sampling,
        "t_sample_affinity_s": t_sample_affinity,
        "t_optim_sample_s": t_optim_sample,
    }

    print("STEP 10: Computing metrics")
    knn_k = min(config.knn_k, emb_sample.shape[0] - 1)
    compute_and_save_metrics(
        out_dir=out_dir,
        run_data=run_data,
        X_reduced=X_reduced,
        X_sample=X_sample,
        y_reduced=y,
        y_sample=y_sample,
        sample_idx=sample_idx,
        emb_sample=emb_sample,
        knn_k=knn_k,
        rare_class_threshold=config.rare_class_threshold,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )


def compute_and_save_metrics(
    out_dir: Path,
    run_data: dict,
    X_reduced,
    X_sample,
    y_reduced,
    y_sample,
    sample_idx,
    emb_sample,
    knn_k,
    rare_class_threshold,
    n_jobs,
    random_state,
):

    out_dir = Path(out_dir)
    arrays_dir = out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    # construct the metrics dict
    print("[METRICS] Computing metrics...")
    rare_classes = identify_rare_classes(y_reduced, rare_class_threshold)
    metrics = {
        # run data (Speedster raw) + hyperparameters (Speedster)
        **run_data,
        "knn_k": knn_k,
        "n_jobs": n_jobs,
        "rare_class_threshold": rare_class_threshold, # default 1%
        "n_rare_classes": int(len(rare_classes)),

        # high-dim sample vs sample embedding
        "sample_knn": knn_preservation(X_sample, emb_sample, knn_k), #k=10
        "sample_knc": knn_class_overlap(X_sample, emb_sample, y_sample),
        "sample_cpd": cpd_spearman(X_sample, emb_sample, random_state),

        # high-dim full data vs sampled data
        "class_coverage": class_coverage(y_reduced, y_sample),
        "hausdorff_coverage": hausdorff_distance(X_reduced, sample_idx, random_state),
        "rare_class_recall": avg_rare_class_recall(y_reduced, y_sample, rare_classes),
        "rare_class_coverage": rare_class_coverage(y_sample, rare_classes),

        # cluster frequencies
        "cluster_freq_10": cluster_freq_eval(X_reduced, sample_idx, 10, random_state),
        "cluster_freq_30": cluster_freq_eval(X_reduced, sample_idx, 30, random_state),
        "cluster_freq_50": cluster_freq_eval(X_reduced, sample_idx, 50, random_state),

        # embedding quality
        "rare_class_purity": rare_class_purity(emb_sample, y_sample, rare_classes, knn_k), # are rare points clustered together in the sample?
        "neighborhood_hit": neighborhood_hit(emb_sample, y_sample, knn_k), # do nearby points share labels?
    }

    print("[METRICS] Saving metrics...")
    save_metrics(out_dir / "metrics.json", metrics)

    return metrics


# --- helpers below ---


def generate_directories(config :Config):
    run_name = (
        f"{config.dataset}"
        f"_{config.sampling_method.value}"
        f"_r{config.sample_rate:.2f}"
        f"_p{config.perplexity:g}"
        f"_{str(config.sample_affinity_mode)}"
    )

    out_dir = Path(config.output_dir) / run_name
    arrays_dir = out_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    return out_dir, arrays_dir


def check_sample_perplexity(input_sample_perplexity, sample_idx):
    # clamp sample perp if needed
    sample_perplexity = min(input_sample_perplexity, max(1, (len(sample_idx) - 1) // 3))
    if sample_perplexity < input_sample_perplexity:
        print(f"warning: clamped sample perplexity {input_sample_perplexity} to {sample_perplexity}")

    return int(sample_perplexity)