from pathlib import Path
import pandas as pd
import numpy as np


def extract_times(df: pd.DataFrame):
    cols = ["dataset", "perplexity", "seed", "affinity_mode",] # config basis
    time_cols = ["setup_time_dist_s", "setup_time_affinity_s"] # time cols we wanna extract

    times = df.loc[
        df["setup_dist_computed"] | df["setup_affinity_computed"],
        cols + ["setup_time_dist_s", "setup_time_affinity_s"]
    ].copy()

    times[time_cols] = times[time_cols].replace(0, np.nan)

    return times.groupby(["dataset", "perplexity"])[time_cols].mean().round(3).reset_index()


def aggregate_runs(df: pd.DataFrame):
    config_cols = ["dataset", "sampling_method", "sample_rate", "perplexity", "sample_perplexity", "sample_perplexity_input", "affinity_mode", "knn_k", "n_jobs"]
    metric_cols = [
        # run info
        "n_rare_classes",
        "rare_class_threshold",
        # timings (Speedster)
        "sample_emb_time_s",
        "t_sampling_s",
        "t_sample_affinity_s",
        "t_optim_sample_s",
        # sample HD -> sample emb
        "sample_knn",
        "sample_knc",
        "sample_cpd",
        # full HD -> sample HD
        "class_coverage",
        "hausdorff_coverage",
        "rare_class_recall",
        "rare_class_coverage",
        "cluster_freq_10",
        "cluster_freq_30",
        "cluster_freq_50",
        # sample emb
        "rare_class_purity",
        "neighborhood_hit",
        # sample emb vs full emb (added post-hoc)
        "procrustes_vs_full",
        "knn_vs_full",
        "knc_vs_full",
        "cpd_vs_full",
    ]

    present = [c for c in metric_cols if c in df.columns]
    missing = [c for c in metric_cols if c not in df.columns]
    if missing: print(f"[create_tables] skipping absent metric columns: {missing}")

    aggregated = df.groupby(config_cols)[present].agg(["mean", "std", "var", "median"])
    aggregated.columns = [metric if stat == "mean" else f"{metric}_{stat}" for metric, stat in aggregated.columns]
    return aggregated.reset_index()

def split_by_dataset(df, output_dir):
    for dataset, group in df.groupby("dataset"):
        group.to_csv(Path(output_dir) / f"aggregated_{dataset}.csv", index=False)