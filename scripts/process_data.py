from pathlib import Path
import pandas as pd
from scripts.utils.load_data import load_runs
from scripts.utils.posthoc_vs_full import attach_vs_full
from scripts.utils.create_tables import extract_times, aggregate_runs
from create_comparison_grids import plot_embedding_grids, plot_baseline_grids


def main(run_name: str, baseline_name: str = "baselines"):

    # 1. fetch all data
    outputs = Path(__file__).resolve().parent.parent / "outputs"
    root = outputs / run_name
    baseline_root = outputs / baseline_name

    output_dir = Path(__file__).resolve().parent.parent / "processed" / run_name
    grid_dir = output_dir / "embedding_grids"
    for d in (grid_dir, output_dir): d.mkdir(parents=True, exist_ok=True)

    # 2. save all data
    df = load_runs(root)
    if baseline_root.exists() and baseline_root != root:
        df = pd.concat([df, load_runs(baseline_root)], ignore_index=True)
    df = attach_vs_full(df) # post-hoc
    df.to_csv(output_dir / "all_data.csv", index=False)
    print("saved all runs")

    # 3. plot embeddings (visualizations!)
    print("plotting embedding grids")
    plot_embedding_grids(df, grid_dir, s=6)
    plot_baseline_grids(df, grid_dir, s=6)

    # 4. extract setup time
    setup_times = extract_times(df.copy())
    setup_times.to_csv(output_dir / "setup_times.csv", index=False)

    # 5. aggregate runs by seed (and drop the time cols we don't need anymore)
    time_cols = ["setup_time_dist_s", "setup_time_affinity_s", "setup_dist_computed", "setup_affinity_computed"]
    df = df.drop(time_cols, axis=1, errors="ignore")
    agg = aggregate_runs(df)
    agg.to_csv(output_dir / "aggregated_runs.csv", index=False)


if __name__ == "__main__":
    main("final-run")