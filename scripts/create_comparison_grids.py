import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sampling_tsne.metrics import identify_rare_classes


def plot_embedding_grids(df: pd.DataFrame, output_dir: Path, s=2):
    """
    sample embedding grid
    (p=50, none), (p=50, affinity correction), (p=n/100, none)
    row = sample rate
    column = sampling method
    """

    df = df[df["sampling_method"] != "full"] # exclude full runs
    groups = df.groupby(["dataset", "affinity_mode", "perplexity", "seed"])

    for (dataset, affinity_mode, perplexity, seed), subset in groups:
        # note: subset is the df with only the rows that match the key
        sample_rates = sorted(subset["sample_rate"].unique())
        sampling_methods = sorted(subset["sampling_method"].unique())

        cells = subset.set_index(["sample_rate", "sampling_method"])["run_dir"].to_dict()
        is_rare = subset["n_rare_classes"].max() > 0

        tag = f"{dataset}_{affinity_mode}_p{perplexity}_seed{seed}"
        title = f"{dataset} (affinity={affinity_mode}, perp={perplexity}, seed={seed})"

        for rare in (False, True):
            if rare and not is_rare: continue
            render_grid(
                cells,
                rows=sample_rates,
                cols=sampling_methods,
                row_label="sampling rate",
                col_label="sampling method",
                title="", #f"{title}{' (rare highlighted)' if rare else ''}",
                path=output_dir / f"grid_{tag}{'_rare' if rare else ''}.png",
                s=s,
                rare=rare,
            )


def plot_baseline_grids(df: pd.DataFrame, output_dir: Path, s=2):
    """
    full embedding grid
    row = target (full) perplexity (p=50, p=n/100)
    column = seed
    """

    df = df[df["sampling_method"] == "full"]
    if df.empty: return

    for dataset, subset in df.groupby("dataset"):
        perplexities = sorted(subset["perplexity"].unique())
        seeds = sorted(subset["seed"].unique())
        cells = subset.set_index(["perplexity", "seed"])["run_dir"].to_dict()
        is_rare = subset["n_rare_classes"].max() > 0

        for rare in (False, True):
            if rare and not is_rare: continue
            render_grid(
                cells,
                rows=perplexities,
                cols=seeds,
                row_label="perplexity",
                col_label="seed",
                title="", #f"{dataset} (full){' (rare highlighted)' if rare else ''}",
                path = output_dir / f"grid_{dataset}_full{'_rare' if rare else ''}.png",
                s=s,
                rare=rare,
            )


def render_grid(cells, rows, cols, row_label, col_label, title, path, s: float = 2, rare: bool = False):
    if not cells: return
    n_rows = len(rows)
    n_cols = len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.8), squeeze=False)

    for i, row_value in enumerate(rows):
        for j, col_value in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])

            run_dir = cells.get((row_value, col_value))
            emb_path = Path(run_dir) / "arrays" / "sample_embedding.npy" if run_dir is not None else None
            if emb_path is None or not emb_path.exists():
                ax.text(0.5, 0.5, "missing!", ha="center", va="center", fontsize=9)
            else:
                run_dir = Path(run_dir)
                emb = np.load(emb_path)
                labels = np.load(run_dir / "arrays" / "y_sample.npy")
                meta = json.loads((run_dir / "metrics.json").read_text())

                if rare: # plot all points as gray, then make it colorful~~~!
                    y_full = np.load(run_dir / "arrays" / "y.npy")
                    rare_classes = identify_rare_classes(y_full, meta["rare_class_threshold"])
                    ax.scatter(emb[:, 0], emb[:, 1], c="lightgrey", s=s, linewidths=0)
                    is_rare = np.isin(labels, rare_classes)
                    ax.scatter(emb[is_rare, 0], emb[is_rare, 1], c=labels[is_rare], cmap="tab20", s=s * 3,linewidths=0)
                else:
                    ax.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=s, alpha=0.5, linewidths=0) # standard

                perp = meta.get("sample_perplexity") # effective perp
                #ax.set_xlabel(f"n: {emb.shape[0]}, perp: {perp:g}", fontsize=12)

            if i == 0: ax.set_title(str(col_value), fontsize=16) # (sampling strategies)
            if j == 0: ax.set_ylabel(f"{row_label}={row_value}", fontsize=16) # (sampling rates)

    fig.suptitle(title, fontsize=24)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)