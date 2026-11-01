from itertools import product
from pathlib import Path
from sampling_tsne import AffinityMode, Config, SamplingMethod, run


# --- run config! ---
run_name = "final-run"
seeds = [12, 22, 32, 42, 52]

datasets = ["mnist","pbmc","c_elegans"]
dataset_n = {"mnist": 70000,"c_elegans": 89701,"pbmc": 65877}

sample_rates = [0.01, 0.05, 0.1]

sampling_methods = [
    SamplingMethod.UNIFORM,
    SamplingMethod.GEOSKETCH,
    SamplingMethod.SCSAMPLER,
    SamplingMethod.POISSON_DISK,
    SamplingMethod.SKETCHKH,
]

affinity_modes = [
    AffinityMode.NONE,
    AffinityMode.DIJKSTRA_DISTANCES,
]

default_perplexities = [50]
full_perplexity_grid = {
    dataset: sorted(set(default_perplexities + [max(1, n // 100)])) # n/100
    for dataset, n in dataset_n.items()}


def build_config(dataset: str, sample_rate: float, sampling_method: SamplingMethod, affinity_mode: AffinityMode, seed: int,full_perp: int) -> Config:
    sample_perplexity = max(1, int(full_perp * sample_rate))
    output_root = Path(__file__).resolve().parents[1] / "outputs" / run_name / f"seed_{seed}" / f"fullperp_{full_perp}"

    return Config(
        dataset=dataset,
        output_dir=str(output_root),
        random_state=seed,
        sampling_method=sampling_method,
        sample_rate=sample_rate,
        sample_affinity_mode=affinity_mode,
        perplexity=full_perp,
        sample_perplexity=sample_perplexity,
    )


if __name__ == "__main__":
    measurements = [(seed, dataset, rate, method, affinity, full_perp)
        for seed, dataset, rate, method, affinity in
        product(seeds, datasets, sample_rates, sampling_methods, affinity_modes)
        for full_perp in full_perplexity_grid[dataset]
        if not (affinity == AffinityMode.DIJKSTRA_DISTANCES and full_perp != 50)
    ]

    n_total = len(measurements)
    print(f"run_name={run_name}, seeds={seeds}, "
        f"{n_total} configs "
        f"({len(seeds)}x{len(datasets)}x{len(sample_rates)}x"
        f"{len(sampling_methods)}x{len(affinity_modes)}x dataset_perplexities)")

    for i, (seed, dataset, rate, method, affinity, full_perp) in enumerate(measurements, 1):
        sample_perp = max(1, int(full_perp * rate))
        print(f"\n[{i}/{n_total}] seed={seed} dataset={dataset} "
            f"method={method.value} rate={rate} affinity={affinity.value} "
            f"full_perp={full_perp} sample_perp={sample_perp}")
        cfg = build_config(dataset, rate, method, affinity, seed, full_perp)
        run(cfg)