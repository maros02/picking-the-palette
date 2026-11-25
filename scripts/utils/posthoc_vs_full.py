import json
import numpy as np
import pandas as pd
from pathlib import Path
from sampling_tsne.metrics import (
    cpd_spearman,
    knn_class_overlap,
    knn_preservation,
    procrustes_distance,
    save_metrics,
)


# sample embedding vs the full (baseline) embedding metrics
vs_full_cols = [
    "procrustes_vs_full",
    "knn_vs_full",
    "knc_vs_full",
    "cpd_vs_full",
]


def attach_vs_full(df) -> pd.DataFrame:
    """
    Compute the vs_full metrics for all runs (using the full runs as reference)
    Then, append them to the dataframe

    :param df: input dataframe
    :return: the same dataframe with vs_full metrics appended
    """

    if "sampling_method" not in df.columns: return df

    is_full = df["sampling_method"] == "full"
    baseline = (df[is_full].set_index(["dataset", "perplexity", "seed"]))["run_dir"].to_dict()

    # NaN when no matching baseline
    for col in vs_full_cols:
        if col not in df.columns: df[col] = np.nan

    not_full = df["sampling_method"] != "full"
    missing = df[vs_full_cols].isna().any(axis=1)
    to_compute = df[not_full & missing]

    print(f"computing vs_full metrics for {len(to_compute)} runs")

    for i, row in to_compute.iterrows():

        key = (row["dataset"], row["perplexity"], row["seed"])

        run_dir = Path(row["run_dir"])
        full_dir = baseline.get(key)
        if full_dir is None: raise ValueError(f"no full baseline for {key}")

        emb = np.load(run_dir / "arrays" / "sample_embedding.npy")
        idx = np.load(run_dir / "arrays" / "sample_idx.npy")
        full = np.load(Path(full_dir) / "arrays" / "sample_embedding.npy")[idx]
        y = np.load(run_dir / "arrays" / "y_sample.npy")

        k = int(row["knn_k"])
        seed = int(row["seed"])

        metrics = {
            "procrustes_vs_full": procrustes_distance(full.copy(), emb.copy()),
            "knn_vs_full": knn_preservation(full, emb, k),
            "knc_vs_full": knn_class_overlap(full, emb, y),
            "cpd_vs_full": cpd_spearman(full, emb, seed),
        }
        metrics = {c: (None if v is None else float(v)) for c, v in metrics.items()}
        for col, value in metrics.items(): df.at[i, col] = value

        cache_vs_full(run_dir, metrics) # so the next run skips this one

    return df


def cache_vs_full(run_dir: Path, full_metrics: dict) -> None:
    """
    Append the vs_full metrics to the run's metrics.json,
    so future runs skip it

    :param run_dir: directory with metrics.json
    :param full_metrics: dictionary of full metrics
    """
    path = run_dir / "metrics.json"
    if not path.is_file(): raise FileNotFoundError(f"no metrics.json file found for")
    run_data = json.loads(path.read_text(encoding="ascii"))
    run_data.update(full_metrics)
    save_metrics(path, run_data)