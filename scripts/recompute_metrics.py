import json
import numpy as np
from pathlib import Path
from sampling_tsne.pipeline import compute_and_save_metrics


def metrics_from_dir(run_dir):
    # recompute metrics from snapshots (click!)
    run_dir = Path(run_dir)
    run_data = json.loads((run_dir / "metrics.json").read_text(encoding="ascii"))

    a = run_dir / "arrays"
    return compute_and_save_metrics(
        out_dir=run_dir,
        run_data=run_data,
        X_reduced=np.load(a / "X_reduced.npy"),
        X_sample=np.load(a / "X_sample.npy"),
        y_reduced=np.load(a / "y.npy"),
        y_sample=np.load(a / "y_sample.npy"),
        sample_idx=np.load(a / "sample_idx.npy"),
        emb_sample=np.load(a / "sample_embedding.npy"),
         knn_k=run_data["knn_k"],
        rare_class_threshold=run_data["rare_class_threshold"],
        n_jobs=run_data["n_jobs"],
        random_state=run_data["seed"]
    )


def main(experiment: str):
    """
    recompute metrics for a given experiment.
    e.g. python recompute_metrics.py <experiment_name>
    """
    root = Path(__file__).resolve().parent.parent / "outputs" / experiment
    runs = sorted(p.parent for p in root.glob("seed_*/fullperp_*/*/metrics.json"))
    print(f"recomputing metrics for {len(runs)} runs in {experiment}")

    for i, run in enumerate(runs):
        print(f"[{i}/{len(runs)}] {run.relative_to(root)}")
        metrics_from_dir(run)


if __name__ == "__main__":
    main("final-run-recomputed")