"""
Algorithm 2 (DP_NN_HISTOGRAM), Algorithm 1 (Private Evolution), and Algorithm 3 (per-class wrapper) translated to audio
"""

__version__ = "v3"

import os
import shutil

import numpy as np

from audioldm_apis import random_api, variation_api
from config import PEConfig
from embedding import embed_paths


def dp_nn_histogram(private_paths: list, population_paths: list, cfg: PEConfig) -> np.ndarray:
    """
    Algorithm 2: DP Nearest Neighbors Histogram.

    private_paths:    Spriv for this class (UrbanSound8K clips)
    population_paths: S, the current generated population (St-1 in Alg. 1)

    Returns a length-len(population_paths) array: histogram[i] is the (DP-noised, thresholded) count of how many private samples had population_paths[i] as their nearest neighbor in embedding space
    """
    # lookahead: instead of scoring each candidate z by Phi(z) directly, score it by the AVERAGE embedding of k variations of z
    lookahead_dir = os.path.join(cfg.output_dir, "_lookahead_tmp")
    population_embeddings = np.zeros((len(population_paths), 2048), dtype=np.float32)
    for i, z_path in enumerate(population_paths):
        variant_paths = [
            variation_api(z_path, transfer_strength=lookahead_strength, cfg=cfg, save_dir=lookahead_dir)
            for _ in range(cfg.lookahead_k)
        ]
        variant_embeddings = embed_paths(variant_paths)      # (k, 2048)
        population_embeddings[i] = variant_embeddings.mean(axis=0)  # Eq. in Sec 4, "Lookahead"
        shutil.rmtree(lookahead_dir, ignore_errors=True)

    # nearest-neighbor voting (Alg. 2, lines 1-4)
    private_embeddings = embed_paths(private_paths)          # (Npriv, 2048)
    histogram = np.zeros(len(population_paths), dtype=np.float64)
    for x_priv_emb in private_embeddings:
        # d(x, z) = || Phi(x) - Phi(z) ||_2   (Eq. 1)
        dists = np.linalg.norm(population_embeddings - x_priv_emb[None, :], axis=1)
        nearest_idx = int(np.argmin(dists))
        histogram[nearest_idx] += 1                          # this private sample "votes" for z_i

    noisy = histogram + np.random.normal(0.0, cfg.sigma, size=histogram.shape)  # add DP noise (Alg. 2, line 5)
    thresholded = np.maximum(noisy - cfg.H, 0.0)

    # print raw vote counts and vote counts after noise is added for debugging
    n_survived = int((histogram > 0).sum())
    print(f"    [dp_nn_histogram] raw (pre-DP) vote counts: {raw_histogram.astype(int).tolist()}")
    print(f"    [dp_nn_histogram] bins surviving noise+threshold (H={cfg.H}): {n_survived}/{len(histogram)}")
    if n_survived <= 1 and len(private_paths) < 50:
        print(f"    [dp_nn_histogram] WARNING: with only {len(private_paths)} private samples, "
              f"sigma={cfg.sigma:.2f} and H={cfg.H} likely drown out real signal -- selection this "
              f"iteration is probably noise-dominated, not reflecting genuine nearest-neighbor structure. "
              f"This is expected at small test scale, not a bug.")

    return histogram


def private_evolution(private_paths: list, cfg: PEConfig, class_out_dir: str) -> list:
    """
    Algorithm 1: Private Evolution, for ONE class's private samples.

    Returns the final population's file paths (this class's slice of Ssyn)
    """
    os.makedirs(class_out_dir, exist_ok=True)

    population = random_api(cfg.N_syn_per_class, cfg, save_dir=os.path.join(class_out_dir, "iter_0"))    # S0 <- RANDOM_API(Nsyn)

    for t in range(1, cfg.T + 1):
        histogram = dp_nn_histogram(private_paths, population, cfg)    # histogram_t <- DP_NN_HISTOGRAM(Spriv, St-1, sigma, H)

        # Pt <- histogram_t / sum(histogram_t)
        total = histogram.sum()
        if total <= 0:
            # Every bin got thresholded to zero (this can happen in very early iterations when the population is still far from the private data)
            probs = np.ones(len(population)) / len(population)
        else:
            probs = histogram / total

        #  S't <- draw Nsyn samples with replacement from Pt
        parent_indices = np.random.choice(len(population), size=cfg.N_syn_per_class, p=probs)
        parents = [population[i] for i in parent_indices]

        # linearly decay the SDEdit noise level across iterations
        strength = cfg.transfer_strength_start + (
            (cfg.transfer_strength_end - cfg.transfer_strength_start) * (t - 1) / max(cfg.T - 1, 1)
        )
        histogram = dp_nn_histogram(private_paths, population, cfg, variation_strength=strength)
        
        iter_dir = os.path.join(class_out_dir, f"iter_{t}")
        population = [
            variation_api(p, transfer_strength=strength, cfg=cfg, save_dir=iter_dir)
            for p in parents
        ]
        print(f"  iteration {t}/{cfg.T} done (transfer_strength={strength:.3f})")

    return population


def run_conditional_pe(class_to_private_paths: dict, cfg: PEConfig) -> dict:
    """
    Algorithm 3: run Algorithm 1 independently for every class in C

    Returns {class_name: [list of synthetic .wav paths]} (Ssyn)
    """
    synthetic = {}
    for class_name, private_paths in class_to_private_paths.items():
        print(f"Running PE for class '{class_name}' ({len(private_paths)} private samples)...")
        class_out_dir = os.path.join(cfg.output_dir, class_name)
        synthetic[class_name] = private_evolution(private_paths, cfg, class_out_dir)
    return synthetic
