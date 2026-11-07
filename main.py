__version__ = "v4"

import json
import os

import audioldm_apis
import config
import dataset
import embedding
import pe_core
import privacy
from config import PEConfig
from dataset import load_urbansound8k, train_test_split_by_fold, subsample_per_class
from pe_core import run_conditional_pe
from privacy import compute_epsilon


def main():
    cfg = PEConfig()

    # 1. Load the private dataset, split into a PE-input set and a held-out test set
    all_data = load_urbansound8k(cfg.urbansound8k_root)
    private_data, test_data = train_test_split_by_fold(all_data, test_folds=(10,))

    # 1a. Restrict to a subset of classes, per cfg.test_classes
    if cfg.test_classes is not None:
        missing = [c for c in cfg.test_classes if c not in private_data]
        if missing:
            raise ValueError(
                f"cfg.test_classes contains class(es) not found in the dataset: {missing}. "
                f"Available classes: {sorted(private_data.keys())}"
            )
        private_data = {c: private_data[c] for c in cfg.test_classes}
        print(f"Restricting to {len(private_data)} class(es) for this run: {list(private_data.keys())}")

    # 1b. Shrink the private set for testing, per cfg.max_private_samples_per_class.
    private_data = subsample_per_class(
        private_data, cfg.max_private_samples_per_class, seed=cfg.subsample_seed
    )
    for class_name, paths in private_data.items():
        print(f"  using {len(paths)} private samples for class '{class_name}'")

    # 1c. Write out exactly which private files were used, per class
    os.makedirs(cfg.output_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.output_dir, "private_samples_used.json")
    with open(manifest_path, "w") as f:
        json.dump(private_data, f, indent=2)
    print(f"  exact private file paths used written to: {manifest_path}")

    # 2. Run Algorithm 3 (Private Evolution, once per class).
    synthetic_data = run_conditional_pe(private_data, cfg)

    # 3. Report the (epsilon, delta)-DP guarantee this run satisfies.
    epsilon = compute_epsilon(sigma=cfg.sigma, T=cfg.T, delta=PEConfig.delta)
    print(f"\nThis run of PE satisfies ({epsilon:.3f}, {PEConfig.delta})-DP.")
    print(f"Synthetic dataset written to: {cfg.output_dir}")
    for class_name, paths in synthetic_data.items():
        print(f"  {class_name}: {len(paths)} synthetic clips")


if __name__ == "__main__":
    main()
