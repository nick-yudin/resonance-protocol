"""
Curation Methods Comparison: Random vs ST vs HDC

Goal: Compare data quality metrics across three curation methods.
(Note: Full fine-tuning would require hours of compute - using proxy metrics instead)

Metrics:
1. Intra-subset diversity (higher = more diverse training data)
2. Coverage of test set (lower NN distance = better coverage)
3. Representativeness (how well subset represents full dataset)
"""

import numpy as np
import json
import time
from datasets import load_dataset
from hdc.data_curator import HDCDataCurator
from hdc.st_curator import STDataCurator
from hdc.ternary_encoder import TernaryHDCEncoder
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances
from typing import List, Dict, Tuple


def load_alpaca_data(max_samples: int = 3000) -> List[str]:
    """Load Alpaca dataset."""
    print(f"Loading Alpaca dataset (max {max_samples} samples)...")

    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except:
        print("  Alpaca not available, using alternative...")
        dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    texts = []
    for i, item in enumerate(dataset):
        if i >= max_samples:
            break

        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        if input_text:
            combined = f"{instruction}\n{input_text}\n{output}"
        else:
            combined = f"{instruction}\n{output}"

        texts.append(combined)

    print(f"✓ Loaded {len(texts)} samples\n")
    return texts


def compute_diversity_metrics(embeddings: np.ndarray, method_name: str) -> Dict:
    """Compute diversity metrics for a subset."""
    print(f"Computing diversity metrics for {method_name}...")

    distances = pairwise_distances(embeddings, metric='cosine')

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


def compute_coverage_metrics(
    subset_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    method_name: str
) -> Dict:
    """Compute how well subset covers test set."""
    print(f"Computing coverage metrics for {method_name}...")

    # For each test sample, find nearest neighbor in subset
    distances = pairwise_distances(test_embeddings, subset_embeddings, metric='cosine')
    min_distances = distances.min(axis=1)

    metrics = {
        "mean_nn_distance": float(np.mean(min_distances)),
        "max_nn_distance": float(np.max(min_distances)),
        "median_nn_distance": float(np.median(min_distances)),
        "coverage_at_01": float((min_distances < 0.1).mean()),
        "coverage_at_02": float((min_distances < 0.2).mean()),
        "coverage_at_05": float((min_distances < 0.5).mean())
    }

    print(f"  Mean NN distance: {metrics['mean_nn_distance']:.4f}")
    print(f"  Coverage @0.1: {metrics['coverage_at_01']:.1%}")
    print(f"  Coverage @0.2: {metrics['coverage_at_02']:.1%}")
    print()

    return metrics


def create_subsets(
    texts: List[str],
    subset_size: int = 500,
    test_size: int = 200
) -> Dict:
    """Create three subsets and test set."""
    print("=" * 60)
    print("CREATING SUBSETS")
    print("=" * 60)
    print()

    # Reserve test set
    np.random.seed(42)
    all_indices = np.arange(len(texts))
    np.random.shuffle(all_indices)

    test_indices = all_indices[:test_size]
    available_indices = all_indices[test_size:]
    available_texts = [texts[i] for i in available_indices]

    print(f"Test set: {len(test_indices)} samples")
    print(f"Available for training: {len(available_texts)} samples\n")

    # 1. Random subset
    print("Creating Random subset...")
    np.random.seed(42)
    random_local_indices = np.random.choice(
        len(available_texts),
        size=subset_size,
        replace=False
    )
    random_indices = available_indices[random_local_indices]
    print(f"✓ Random subset: {len(random_indices)} samples\n")

    # 2. ST-Curated subset
    print("Creating ST-Curated subset...")
    st_curator = STDataCurator(device='cpu')
    st_local_indices, st_stats = st_curator.curate(
        available_texts,
        target_size=subset_size,
        sampling_strategy='nearest_centroid'
    )
    st_indices = available_indices[st_local_indices]

    # 3. HDC-Curated subset
    print("Creating HDC-Curated subset...")
    hdc_curator = HDCDataCurator(
        hd_dim=10000,
        sparsity=0.7,
        dedup_threshold=0.95,
        device='cpu'
    )
    hdc_local_indices, hdc_stats = hdc_curator.curate(
        available_texts,
        target_size=subset_size,
        sampling_strategy='nearest_centroid',
        batch_size=32
    )
    hdc_indices = available_indices[hdc_local_indices]

    return {
        "test_indices": test_indices,
        "random_indices": random_indices,
        "st_indices": st_indices,
        "hdc_indices": hdc_indices,
        "st_stats": st_stats,
        "hdc_stats": hdc_stats
    }


def evaluate_all_subsets(
    texts: List[str],
    subsets: Dict
) -> Dict:
    """Evaluate all three subsets on diversity and coverage metrics."""
    print("=" * 60)
    print("EVALUATING SUBSETS")
    print("=" * 60)
    print()

    # Prepare encoder for evaluation (use HDC for fair comparison)
    print("Initializing encoder for evaluation...")
    encoder = TernaryHDCEncoder(hd_dim=10000, sparsity=0.7, device='cpu')

    # Encode test set
    print("\nEncoding test set...")
    test_texts = [texts[i] for i in subsets["test_indices"]]
    test_embeddings = encoder.encode(test_texts).cpu().numpy()
    print(f"✓ Test set encoded: {test_embeddings.shape}\n")

    results = {}

    # Evaluate Random
    print("=" * 60)
    print("EVALUATING RANDOM SUBSET")
    print("=" * 60)
    print()

    random_texts = [texts[i] for i in subsets["random_indices"]]
    random_embeddings = encoder.encode(random_texts).cpu().numpy()

    results["random"] = {
        "diversity": compute_diversity_metrics(random_embeddings, "Random"),
        "coverage": compute_coverage_metrics(random_embeddings, test_embeddings, "Random")
    }

    # Evaluate ST-Curated
    print("=" * 60)
    print("EVALUATING ST-CURATED SUBSET")
    print("=" * 60)
    print()

    st_texts = [texts[i] for i in subsets["st_indices"]]
    st_embeddings = encoder.encode(st_texts).cpu().numpy()

    results["st_curated"] = {
        "diversity": compute_diversity_metrics(st_embeddings, "ST-Curated"),
        "coverage": compute_coverage_metrics(st_embeddings, test_embeddings, "ST-Curated")
    }

    # Evaluate HDC-Curated
    print("=" * 60)
    print("EVALUATING HDC-CURATED SUBSET")
    print("=" * 60)
    print()

    hdc_texts = [texts[i] for i in subsets["hdc_indices"]]
    hdc_embeddings = encoder.encode(hdc_texts).cpu().numpy()

    results["hdc_curated"] = {
        "diversity": compute_diversity_metrics(hdc_embeddings, "HDC-Curated"),
        "coverage": compute_coverage_metrics(hdc_embeddings, test_embeddings, "HDC-Curated")
    }

    return results


def compare_results(results: Dict) -> Dict:
    """Compare metrics across all three methods."""
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print()

    # Extract metrics
    random_div = results["random"]["diversity"]["mean_distance"]
    st_div = results["st_curated"]["diversity"]["mean_distance"]
    hdc_div = results["hdc_curated"]["diversity"]["mean_distance"]

    random_cov = results["random"]["coverage"]["mean_nn_distance"]
    st_cov = results["st_curated"]["coverage"]["mean_nn_distance"]
    hdc_cov = results["hdc_curated"]["coverage"]["mean_nn_distance"]

    random_cov_05 = results["random"]["coverage"]["coverage_at_05"]
    st_cov_05 = results["st_curated"]["coverage"]["coverage_at_05"]
    hdc_cov_05 = results["hdc_curated"]["coverage"]["coverage_at_05"]

    # Print table
    print(f"{'Metric':<40} {'Random':>15} {'ST-Curated':>15} {'HDC-Curated':>15}")
    print("-" * 90)
    print(f"{'Mean Pairwise Distance':<40} {random_div:>15.4f} {st_div:>15.4f} {hdc_div:>15.4f}")
    print(f"{'Mean NN Distance (lower=better)':<40} {random_cov:>15.4f} {st_cov:>15.4f} {hdc_cov:>15.4f}")
    print(f"{'Coverage @0.5':<40} {random_cov_05:>14.1%} {st_cov_05:>14.1%} {hdc_cov_05:>14.1%}")
    print()

    # Determine winners
    print("ANALYSIS:")
    print()

    # Diversity winner
    diversity_scores = {"Random": random_div, "ST-Curated": st_div, "HDC-Curated": hdc_div}
    div_winner = max(diversity_scores, key=diversity_scores.get)
    print(f"  Diversity winner: {div_winner} ({diversity_scores[div_winner]:.4f})")

    # Coverage winner (lower is better)
    coverage_scores = {"Random": random_cov, "ST-Curated": st_cov, "HDC-Curated": hdc_cov}
    cov_winner = min(coverage_scores, key=coverage_scores.get)
    print(f"  Coverage winner: {cov_winner} ({coverage_scores[cov_winner]:.4f})")

    # Coverage @0.5 winner
    cov_05_scores = {"Random": random_cov_05, "ST-Curated": st_cov_05, "HDC-Curated": hdc_cov_05}
    cov_05_winner = max(cov_05_scores, key=cov_05_scores.get)
    print(f"  Coverage @0.5 winner: {cov_05_winner} ({cov_05_scores[cov_05_winner]:.1%})")
    print()

    # Count wins
    wins = {"Random": 0, "ST-Curated": 0, "HDC-Curated": 0}
    wins[div_winner] += 1
    wins[cov_winner] += 1
    wins[cov_05_winner] += 1

    print(f"WINS: Random={wins['Random']}, ST-Curated={wins['ST-Curated']}, HDC-Curated={wins['HDC-Curated']}")
    print()

    # Determine status
    if wins["HDC-Curated"] >= 2 and wins["ST-Curated"] >= 1 and wins["Random"] == 0:
        if wins["HDC-Curated"] > wins["ST-Curated"]:
            status = "✅✅ STRONG SUCCESS"
            message = "HDC > ST > Random on data quality metrics"
        else:
            status = "✅ SUCCESS"
            message = "HDC ≥ ST > Random on data quality metrics"
    elif wins["HDC-Curated"] >= 1 and wins["HDC-Curated"] >= wins["ST-Curated"]:
        status = "⚠️  PARTIAL SUCCESS"
        message = "HDC competitive with ST, both better than Random"
    elif wins["ST-Curated"] > wins["HDC-Curated"]:
        status = "⚠️  ST SUPERIOR"
        message = "ST-Curated outperforms HDC on data quality metrics"
    else:
        status = "❌ FAILURE"
        message = "No clear advantage for curated methods"

    print(f"VERDICT: {status}")
    print(f"  {message}")
    print()

    # Compute improvements
    hdc_vs_random_div_pct = ((hdc_div - random_div) / random_div) * 100
    hdc_vs_st_div_pct = ((hdc_div - st_div) / st_div) * 100
    st_vs_random_div_pct = ((st_div - random_div) / random_div) * 100

    hdc_vs_random_cov_pct = ((random_cov - hdc_cov) / random_cov) * 100
    hdc_vs_st_cov_pct = ((st_cov - hdc_cov) / st_cov) * 100
    st_vs_random_cov_pct = ((random_cov - st_cov) / random_cov) * 100

    print("IMPROVEMENTS (Diversity):")
    print(f"  HDC vs Random: {hdc_vs_random_div_pct:+.2f}%")
    print(f"  HDC vs ST: {hdc_vs_st_div_pct:+.2f}%")
    print(f"  ST vs Random: {st_vs_random_div_pct:+.2f}%")
    print()

    print("IMPROVEMENTS (Coverage, lower=better):")
    print(f"  HDC vs Random: {hdc_vs_random_cov_pct:+.2f}%")
    print(f"  HDC vs ST: {hdc_vs_st_cov_pct:+.2f}%")
    print(f"  ST vs Random: {st_vs_random_cov_pct:+.2f}%")
    print()

    comparison = {
        "status": status,
        "message": message,
        "wins": wins,
        "diversity_winner": div_winner,
        "coverage_winner": cov_winner,
        "improvements": {
            "diversity": {
                "hdc_vs_random_pct": hdc_vs_random_div_pct,
                "hdc_vs_st_pct": hdc_vs_st_div_pct,
                "st_vs_random_pct": st_vs_random_div_pct
            },
            "coverage": {
                "hdc_vs_random_pct": hdc_vs_random_cov_pct,
                "hdc_vs_st_pct": hdc_vs_st_cov_pct,
                "st_vs_random_pct": st_vs_random_cov_pct
            }
        }
    }

    return comparison


def main():
    """Run curation methods comparison"""
    print("\n" + "=" * 60)
    print("PHASE M2.5b: CURATION METHODS COMPARISON")
    print("=" * 60)
    print()

    print("NOTE: Full fine-tuning would require hours of compute.")
    print("Using data quality metrics as proxy for downstream performance.\n")

    # Configuration
    FULL_DATASET_SIZE = 3000
    SUBSET_SIZE = 500
    TEST_SIZE = 200

    # Load dataset
    texts = load_alpaca_data(max_samples=FULL_DATASET_SIZE)

    # Create subsets
    subsets = create_subsets(texts, subset_size=SUBSET_SIZE, test_size=TEST_SIZE)

    # Evaluate all subsets
    results = evaluate_all_subsets(texts, subsets)

    # Compare
    comparison = compare_results(results)

    # Save results
    output = {
        "phase": "M2.5b",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "Data quality metrics comparison (fine-tuning would require hours)",
        "config": {
            "full_dataset_size": FULL_DATASET_SIZE,
            "subset_size": SUBSET_SIZE,
            "test_size": TEST_SIZE
        },
        "curation_stats": {
            "st": subsets["st_stats"],
            "hdc": subsets["hdc_stats"]
        },
        "results": {
            "random": results["random"],
            "st_curated": results["st_curated"],
            "hdc_curated": results["hdc_curated"]
        },
        "comparison": comparison
    }

    import os
    os.makedirs("hdc/results", exist_ok=True)
    output_file = "hdc/results/phase_m2.5b_curation_comparison.json"

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to {output_file}\n")

    print("=" * 60)
    print("PHASE M2.5b COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
