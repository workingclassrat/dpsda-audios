"""
Sweep sigma/H to trace out a quality-vs-epsilon curve

1. try several values of t, with sigma = t*sqrt(2) and H = 2*t for each
2. run PE once per t
3. plot the resulting privacy cost (epsilon) against a quality metric (mean nearest-neighbor distance, or full FAD if there are enough samples) of the resulting synthetic population vs. the real private data

Usage:
    python epsilon_vs_quality.py --class_name dog_bark --t_values 0.5 1 2 5
"""

__version__ = "v1"

import copy
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import PEConfig
from dataset import load_urbansound8k, train_test_split_by_fold, subsample_per_class
from evaluate import mean_nn_distance
from pe_core import private_evolution
from privacy import compute_epsilon


def run_sweep(class_name: str, t_values: list, delta: float = None, use_fad: bool = False) -> tuple:
    """
    For each t in t_values: sets sigma=t*sqrt(2), H=2*t, runs Algorithm 1 once for `class_name` at that setting, and records the resulting (epsilon, quality) point
    """
    base_cfg = PEConfig()
    if delta is None:
        delta = base_cfg.delta

    all_data = load_urbansound8k(base_cfg.urbansound8k_root)
    private_data, test_data = train_test_split_by_fold(all_data, test_folds=(10,))
    private_data = subsample_per_class(
        private_data, base_cfg.max_private_samples_per_class, seed=base_cfg.subsample_seed
    )
    if class_name not in private_data:
        raise ValueError(f"'{class_name}' not found. Available: {sorted(private_data.keys())}")
    real_paths = private_data[class_name]
    eval_paths = test_data[class_name]         # held-out fold 10 used only for scoring

    results = []
    for t in t_values:
        cfg = copy.deepcopy(base_cfg)
        cfg.sigma = t * (2 ** 0.5)
        cfg.H = 2 * t
        cfg.output_dir = os.path.join(base_cfg.output_dir, f"_sweep_t{t}")

        print(f"\n=== t={t}  (sigma={cfg.sigma:.3f}, H={cfg.H:.3f}, T={cfg.T}) ===")
        class_out_dir = os.path.join(cfg.output_dir, class_name)
        synthetic_paths = private_evolution(real_paths, cfg, class_out_dir)

        epsilon = compute_epsilon(sigma=cfg.sigma, T=cfg.T, delta=delta)

        if use_fad:
            from evaluate import compute_fad
            quality = compute_fad(eval_paths, synthetic_paths)
            metric_name = "FAD"
        else:
            stats = mean_nn_distance(eval_paths, synthetic_paths)
            quality = stats["mean"]
            metric_name = "mean nearest-neighbor distance"

        print(f"  -> epsilon={epsilon:.4f}, {metric_name}={quality:.4f}")
        results.append({
            "t": t, "sigma": cfg.sigma, "H": cfg.H, "T": cfg.T,
            "epsilon": epsilon, "quality": quality, "metric_name": metric_name,
        })

    return results, (results[0]["metric_name"] if results else None)


def plot_results(results: list, metric_name: str, class_name: str, output_path: str) -> None:
    if not results:
        raise ValueError("No results to plot -- t_values was empty?")

    ordered = sorted(results, key=lambda r: r["epsilon"])
    epsilons = [r["epsilon"] for r in ordered]
    qualities = [r["quality"] for r in ordered]

    plt.figure(figsize=(6, 4.5))
    plt.plot(epsilons, qualities, marker="o")
    for r in ordered:
        plt.annotate(f"t={r['t']}", (r["epsilon"], r["quality"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Epsilon (privacy budget)")
    plt.ylabel(f"{metric_name}  (lower = closer to real data)")
    plt.title(f"{metric_name} vs. Epsilon -- class '{class_name}'\n(audio equivalent of the paper's Fig. 4)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep sigma/H (via t) to trace a quality-vs-epsilon curve for one class."
    )
    parser.add_argument("--class_name", required=True)
    parser.add_argument(
        "--t_values", type=float, nargs="+", default=[0.5, 1.0, 2.0, 5.0],
        help="Values of t to try; sigma=t*sqrt(2), H=2*t for each (paper's Appendix J pattern)."
    )
    parser.add_argument("--delta", type=float, default=None,
                        help="override cfg.delta (default: use the value in config.py)")
    parser.add_argument("--fad", action="store_true",
                         help="use full FAD instead of mean NN distance (only trust this with large sample counts)")
    parser.add_argument("--output", default="epsilon_vs_quality.png")
    parser.add_argument("--results_json", default="epsilon_vs_quality_results.json")
    parser.add_argument("--against", choices=["test", "private"], default="test",
                        help="test = held-out fold 10 (does not touch S_priv); "
                        "private = the manifest's private clips (NON-PRIVATE, debugging only)"
    )
    args = parser.parse_args()

    results, metric_name = run_sweep(args.class_name, args.t_values, args.delta, args.fad)

    with open(args.results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw (epsilon, quality) results written to {args.results_json}")

    plot_results(results, metric_name, args.class_name, args.output)
