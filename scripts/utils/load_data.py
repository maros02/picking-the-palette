import json
from pathlib import Path
import pandas as pd


def config_key(row):
    method = row.get("sampling_method", "?")
    affinity = row.get("affinity_mode", "?")
    return f"{method}+{affinity}"


def load_runs(directory: Path) -> pd.DataFrame:
    # seed_X > fullperp_Y >  <dataset>_<sampling-method>_r<sample-rate>_p<full-perplexity>_AffinityMode.<DIJKSTRA or NONE>
    rows = []
    for seed_directory in sorted(directory.glob("seed_*")):
        seed = seed_directory.name.split("_", 1)[1]

        # extract metrics
        for run_directory in sorted(seed_directory.glob("fullperp_*/*")):
            if not run_directory.is_dir(): continue
            metric_path = run_directory / "metrics.json"
            if not metric_path.is_file(): continue
            data = json.loads(metric_path.read_text())
            data["seed"] = seed
            data["run"] = run_directory.name
            data["run_dir"] = run_directory
            rows.append(data)

    df = pd.DataFrame(rows)
    df["configuration"] = df.apply(config_key, axis=1)
    return df