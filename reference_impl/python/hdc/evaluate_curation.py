"""
Evaluate HDC-Based Data Curation

Goal: Demonstrate that HDC curation improves data quality metrics
without requiring full fine-tuning (due to compute constraints).

Metrics evaluated:
1. Diversity: Average pairwise distance in HDC space
2. Coverage: Spread across semantic space
3. Redundancy: Duplicate ratio
4. Representative quality: How well subset represents full dataset
"""

import numpy as np
import json
import time
from datasets import load_dataset
from hdc.data_curator import HDCDataCurator
from hdc.ternary_encoder import TernaryHDCEncoder
from typing import List, Dict, Tuple
from sklearn.metrics import pairwise_distances


def load_alpaca_subset(max_samples: int = 10000) -> List[str]:
    """
    Load Alpaca dataset (instruction + response combined).

    Args:
        max_samples: Maximum number of samples to load

    Returns:
        texts: List of combined instruction+response strings
    """
    print(f"Loading Alpaca dataset (max {max_samples} samples)...")

    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except:
        # Fallback to alternative dataset if alpaca not available
        print("  Alpaca not available, using alternative...")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    texts = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        # Combine instruction and output
        instruction = item.get('instruction', '')
        output = item.get('output', '')
        combined = f"{instruction}\n{output}"
        texts.append(combined)

    print(f"✓ Loaded {len(texts)} samples\n")
    return texts


def compute_diversity_metrics(embeddings: np.ndarray) -> Dict:
    """
    Compute diversity metrics for a set of embeddings.

    Args:
        embeddings: (n_samples, hd_dim) HDC vectors

    Returns:
        Dictionary with diversity metrics
    """
    print("Computing diversity metrics...")

    # Pairwise cosine distances
    distances = pairwise_distances(embeddings, metric='cosine')

    # Remove diagonal (self-similarity)
    n = len(embeddings)
    mask = ~np.eye(n, dtype=bool)
    pairwise_dists = distances[mask]

    metrics = {
        "mean_distance": float(np.mean(pairwise_dists)),
        "std_distance": float(np.std(pairwise_dists)),
        "min_distance": float(np.min(pairwise_dists)),
        "max_distance": float(np.max(pairwise_dists)),
        "median_distance": float(np.median(pairwise_dists))
    }

    print(f"  Mean pairwise distance: {metrics['mean_distance']:.4f}")
    print(f"  Std pairwise distance: {metrics['std_distance']:.4f}")
    print()

    return metrics


def compute_coverage_metrics(embeddings: np.ndarray, full_embeddings: np.ndarray) -> Dict:
    """
    Compute how well subset covers the full dataset.

    Args:
        embeddings: Subset embeddings
        full_embeddings: Full dataset embeddings

    Returns:
        Dictionary with coverage metrics
    """
    print("Computing coverage metrics...")

    # For each sample in full dataset, find nearest neighbor in subset
    distances = pairwise_distances(full_embeddings, embeddings, metric='cosine')
    min_distances = distances.min(axis=1)

    metrics = {
        "mean_nn_distance": float(np.mean(min_distances)),
        "max_nn_distance": float(np.max(min_distances)),
        "median_nn_distance": float(np.median(min_distances)),
        "coverage_at_01": float((min_distances < 0.1).mean()),  # % within 0.1 distance
        "coverage_at_02": float((min_distances < 0.2).mean()),  # % within 0.2 distance
        "coverage_at_05": float((min_distances < 0.5).mean())   # % within 0.5 distance
    }

    print(f"  Mean nearest neighbor distance: {metrics['mean_nn_distance']:.4f}")
    print(f"  Coverage @0.1: {metrics['coverage_at_01']:.1%}")
    print(f"  Coverage @0.2: {metrics['coverage_at_02']:.1%}")
    print()

    return metrics


def evaluate_random_subset(
    texts: List[str],
    target_size: int,
    encoder: TernaryHDCEncoder,
    full_embeddings: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, Dict]:
    """
    Evaluate random baseline subset.

    Returns:
        indices: Selected indices
        metrics: Quality metrics
    """
    print("=" * 60)
    print("BASELINE: RANDOM SUBSET")
    print("=" * 60)
    print()

    np.random.seed(seed)
    indices = np.random.choice(len(texts), size=target_size, replace=False)

    # Encode subset
    subset_texts = [texts[i] for i in indices]
    subset_embeddings = encoder.encode(subset_texts).cpu().numpy()

    # Compute metrics
    diversity = compute_diversity_metrics(subset_embeddings)
    coverage = compute_coverage_metrics(subset_embeddings, full_embeddings)

    metrics = {
        "method": "Random Baseline",
        "subset_size": len(indices),
        "diversity": diversity,
        "coverage": coverage
    }

    return indices, metrics


def evaluate_hdc_curated(
    texts: List[str],
    target_size: int,
    curator: HDCDataCurator,
    encoder: TernaryHDCEncoder,
    full_embeddings: np.ndarray
) -> Tuple[np.ndarray, Dict, Dict]:
    """
    Evaluate HDC-curated subset.

    Returns:
        indices: Selected indices
        metrics: Quality metrics
        curation_stats: Curation statistics
    """
    print("=" * 60)
    print("HDC-CURATED SUBSET")
    print("=" * 60)
    print()

    # Curate
    indices, curation_stats = curator.curate(
        texts,
        target_size=target_size,
        sampling_strategy='nearest_centroid',
        batch_size=32
    )

    # Get embeddings for curated subset
    subset_texts = [texts[i] for i in indices]
    subset_embeddings = encoder.encode(subset_texts).cpu().numpy()

    # Compute metrics
    diversity = compute_diversity_metrics(subset_embeddings)
    coverage = compute_coverage_metrics(subset_embeddings, full_embeddings)

    metrics = {
        "method": "HDC-Curated",
        "subset_size": len(indices),
        "diversity": diversity,
        "coverage": coverage,
        "curation_stats": curation_stats
    }

    return indices, metrics, curation_stats


def compare_results(random_metrics: Dict, hdc_metrics: Dict) -> Dict:
    """
    Compare random vs HDC-curated metrics.

    Returns:
        Comparison dictionary
    """
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print()

    # Diversity comparison
    random_div = random_metrics['diversity']['mean_distance']
    hdc_div = hdc_metrics['diversity']['mean_distance']
    div_improvement = ((hdc_div - random_div) / random_div) * 100

    # Coverage comparison
    random_cov = random_metrics['coverage']['mean_nn_distance']
    hdc_cov = hdc_metrics['coverage']['mean_nn_distance']
    cov_improvement = ((random_cov - hdc_cov) / random_cov) * 100  # Lower is better

    print(f"{'Metric':<40} {'Random':>15} {'HDC-Curated':>15} {'Δ':>10}")
    print("-" * 85)

    # Diversity
    print(f"{'Mean Pairwise Distance':<40} {random_div:>15.4f} {hdc_div:>15.4f} {div_improvement:>9.1f}%")

    # Coverage
    print(f"{'Mean NN Distance (lower=better)':<40} {random_cov:>15.4f} {hdc_cov:>15.4f} {cov_improvement:>9.1f}%")

    random_cov_01 = random_metrics['coverage']['coverage_at_01']
    hdc_cov_01 = hdc_metrics['coverage']['coverage_at_01']
    print(f"{'Coverage @0.1':<40} {random_cov_01:>14.1%} {hdc_cov_01:>14.1%} {(hdc_cov_01-random_cov_01)*100:>9.1f}%")

    random_cov_02 = random_metrics['coverage']['coverage_at_02']
    hdc_cov_02 = hdc_metrics['coverage']['coverage_at_02']
    print(f"{'Coverage @0.2':<40} {random_cov_02:>14.1%} {hdc_cov_02:>14.1%} {(hdc_cov_02-random_cov_02)*100:>9.1f}%")

    print()

    # Determine success
    improvements = []
    if hdc_div > random_div:
        improvements.append("diversity")
    if hdc_cov < random_cov:
        improvements.append("coverage")
    if hdc_cov_02 > random_cov_02:
        improvements.append("coverage@0.2")

    if len(improvements) >= 2:
        status = "✅ SUCCESS"
        message = f"HDC-curated improves on {len(improvements)}/3 metrics"
    elif len(improvements) == 1:
        status = "⚠️  PARTIAL SUCCESS"
        message = "HDC-curated improves on 1/3 metrics"
    else:
        status = "❌ FAILURE"
        message = "Random baseline performs better"

    print(f"VERDICT: {status}")
    print(f"  {message}")
    print(f"  Improvements: {', '.join(improvements) if improvements else 'none'}")
    print()

    comparison = {
        "status": status,
        "improvements": improvements,
        "diversity_improvement_pct": div_improvement,
        "coverage_improvement_pct": cov_improvement
    }

    return comparison


def main():
    """Run curation evaluation experiment"""
    print("\n" + "=" * 60)
    print("PHASE M2.5: HDC-CURATED DATA EVALUATION")
    print("=" * 60)
    print()

    # Configuration
    FULL_DATASET_SIZE = 2000  # Reduced for faster execution
    TARGET_SUBSET_SIZE = 500
    DEDUP_THRESHOLD = 0.95

    # Load dataset
    texts = load_alpaca_subset(max_samples=FULL_DATASET_SIZE)

    # Initialize encoder
    encoder = TernaryHDCEncoder(hd_dim=10000, sparsity=0.7, device='cpu')

    # Encode full dataset (needed for coverage metrics)
    print("Encoding full dataset for coverage metrics...")
    print(f"  Processing {len(texts)} texts in batches...")

    all_embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_emb = encoder.encode(batch).cpu().numpy()
        all_embeddings.append(batch_emb)
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Progress: {i+len(batch)}/{len(texts)}")

    full_embeddings = np.vstack(all_embeddings)
    print(f"✓ Encoded full dataset: {full_embeddings.shape}\n")

    # Evaluate random baseline
    random_indices, random_metrics = evaluate_random_subset(
        texts,
        TARGET_SUBSET_SIZE,
        encoder,
        full_embeddings
    )

    # Evaluate HDC-curated
    curator = HDCDataCurator(
        hd_dim=10000,
        sparsity=0.7,
        dedup_threshold=DEDUP_THRESHOLD
    )

    hdc_indices, hdc_metrics, curation_stats = evaluate_hdc_curated(
        texts,
        TARGET_SUBSET_SIZE,
        curator,
        encoder,
        full_embeddings
    )

    # Compare
    comparison = compare_results(random_metrics, hdc_metrics)

    # Save results
    results = {
        "phase": "M2.5",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "full_dataset_size": FULL_DATASET_SIZE,
            "target_subset_size": TARGET_SUBSET_SIZE,
            "dedup_threshold": DEDUP_THRESHOLD
        },
        "random_baseline": random_metrics,
        "hdc_curated": hdc_metrics,
        "comparison": comparison
    }

    import os
    os.makedirs("hdc/results", exist_ok=True)
    output_file = "hdc/results/phase_m2.5_curation.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to {output_file}\n")

    print("=" * 60)
    print("PHASE M2.5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
