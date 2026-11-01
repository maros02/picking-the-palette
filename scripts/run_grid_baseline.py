from itertools import product
from pathlib import Path
from sampling_tsne import AffinityMode, Config, SamplingMethod, run


# --- run config! (baseline: no sampler / rate / affinity axes) ---
run_name = "baselines"
seeds = [12, 22, 32, 42, 52]

datasets = ["mnist","pbmc","c_elegans"]
dataset_n = {"mnist": 70000,"c_elegans": 89701,"pbmc": 65877}

default_perplexities = [50]
full_perplexity_grid = {
    dataset: sorted(set(default_perplexities + [max(1, n // 100)])) #n/100
    for dataset, n in dataset_n.items()}


def build_config(dataset: str, seed: int, full_perp: int) -> Config:
    output_root = Path(__file__).resolve().parents[1] / "outputs" / run_name / f"seed_{seed}" / f"fullperp_{full_perp}"

    return Config(
        dataset=dataset,
        output_dir=str(output_root),
        random_state=seed,
        sampling_method=SamplingMethod.FULL, # no sampling: embed all points
        sample_rate=1.0,
        sample_affinity_mode=AffinityMode.NONE,
        perplexity=full_perp,
        sample_perplexity=full_perp, # full perplexity (not scaled)
    )


if __name__ == "__main__":
    measurements = [(seed, dataset, full_perp)
                    for seed, dataset in product(seeds, datasets)
                    for full_perp in full_perplexity_grid[dataset]]

    n_total = len(measurements)
    print(f"run_name={run_name}, seeds={seeds}, "
          f"{n_total} baseline configs "
          f"({len(seeds)}x{len(datasets)}x dataset_perplexities)")

    for i, (seed, dataset, full_perp) in enumerate(measurements, 1):
        print(f"\n[{i}/{n_total}] seed={seed} dataset={dataset} "
              f"method=full full_perp={full_perp}")
        cfg = build_config(dataset, seed, full_perp)
        run(cfg)