"""
1. mean_nn_distance() for small sample sizes (audio equivalent of the distribution of embedding-space distances between generated audios and their nearest private audios)

2. compute_fad(): the full Frechet Audio Distance (audio's version of FID)
"""

__version__ = "v2"

import glob
import json
import os

import numpy as np
from scipy import linalg

from config import PEConfig
from embedding import embed_paths
from dataset import load_urbansound8k, train_test_split_by_fold


def mean_nn_distance(real_paths: list, generated_paths: list) -> dict:
    """
    For each generated sample, find its distance (in Phi's embedding space) to its NEAREST real sample
    Returns summary statistics across all generated samples.

    Track this across iterations for the same class: it should trend downward as PE converges
    """
    real_embeddings = embed_paths(real_paths)            # (n_real, 2048)
    gen_embeddings = embed_paths(generated_paths)        # (n_gen, 2048)

    nn_distances = []
    for gen_emb in gen_embeddings:
        dists = np.linalg.norm(real_embeddings - gen_emb[None, :], axis=1)
        nn_distances.append(float(dists.min()))

    nn_distances = np.array(nn_distances)
    return {
        "mean": float(nn_distances.mean()),
        "median": float(np.median(nn_distances)),
        "min": float(nn_distances.min()),
        "max": float(nn_distances.max()),
        "per_sample": nn_distances.tolist(),
    }


def compute_fad(real_paths: list, generated_paths: list) -> float:
    real_embeddings = embed_paths(real_paths)
    gen_embeddings = embed_paths(generated_paths)

    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = gen_embeddings.mean(axis=0), np.cov(gen_embeddings, rowvar=False)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
    return fad


def _load_manifest_real_paths(class_name: str, output_dir: str, manifest_path: str = None) -> list:
    manifest_path = manifest_path or os.path.join(output_dir, "private_samples_used.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest[class_name]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare one PE iteration's synthetic samples to the real private samples for a class."
    )
    parser.add_argument("--class_name", required=True, help="e.g. dog_bark")
    parser.add_argument("--iter", type=int, required=True, help="which iteration's folder to compare, e.g. 0, 1, 2")
    parser.add_argument("--output_dir", default=PEConfig().output_dir)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--fad", action="store_true", help="also compute full FAD (only trust this at large sample sizes)")
    parser.add_argument(
        "--against", choices=["test", "private"], default="test",
        help="test = held-out fold 10 (does not touch S_priv); private = the manifest's private clips (NON-PRIVATE, debugging only)"
    )
    args = parser.parse_args()

    if args.against == "test":
        cfg = PEConfig()
        all_data = load_urbansound8k(cfg.urbansound8k_root)
        _, test_data = train_test_split_by_fold(all_data, test_folds=(10,))
        real_paths = test_data[args.class_name]
        print(f"Scoring against held-out fold 10 ({len(real_paths)} clips)")
    else:
        if args.against == "test":
            cfg = PEConfig()
            all_data = load_urbansound8k(cfg.urbansound8k_root)
            _, test_data = train_test_split_by_fold(all_data, test_folds=(10,))
            real_paths = test_data[args.class_name]
            print(f"Scoring against held-out fold 10 ({len(real_paths)} clips)")
        else:
            real_paths = _load_manifest_real_paths(args.class_name, args.output_dir, args.manifest)
            print("Scoring against PRIVATE clips (NON-PRIVATE result, debugging only)")    iter_dir = os.path.join(args.output_dir, args.class_name, f"iter_{args.iter}")
    generated_paths = sorted(glob.glob(os.path.join(iter_dir, "*.wav")))
    if not generated_paths:
        raise FileNotFoundError(f"No .wav files found directly in {iter_dir}")

    print(f"Comparing {len(real_paths)} real vs {len(generated_paths)} generated ({args.class_name}, iter {args.iter})")
    print()

    stats = mean_nn_distance(real_paths, generated_paths)
    print("Mean nearest-neighbor distance (embedding space, meaningful at any sample size):")
    print(f"  mean={stats['mean']:.3f}  median={stats['median']:.3f}  min={stats['min']:.3f}  max={stats['max']:.3f}")
    print(f"  per-sample: {[round(d, 2) for d in stats['per_sample']]}")

    if args.fad:
        n_min = min(len(real_paths), len(generated_paths))
        if n_min < 50:
            print()
            print(f"  WARNING: only {n_min} samples on the smaller side and FAD below is very likely NOT statistically meaningful")
        score = compute_fad(real_paths, generated_paths)
        print(f"\nFAD = {score:.3f}")
