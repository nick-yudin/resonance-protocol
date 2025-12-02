"""
Benchmark: Projection HDC (Phase 1.1) vs Baseline

Tests the hypothesis that semantic initialization via random projection
can achieve competitive performance with sentence transformers.

Target: Spearman ρ > 0.70 (within 15% of baseline)
"""

import torch
import numpy as np
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from hdc.projection_encoder import ProjectionHDCEncoder
from typing import List, Tuple
import time
import json
import os


def load_sts_benchmark() -> Tuple[List[str], List[str], List[float]]:
    """Load STS Benchmark dataset"""
    print("Loading STS Benchmark dataset...")
    dataset = load_dataset("mteb/stsbenchmark-sts", split="test")

    sentences1 = []
    sentences2 = []
    scores = []

    for item in dataset:
        sentences1.append(item['sentence1'])
        sentences2.append(item['sentence2'])
        scores.append(item['score'] / 5.0)  # Normalize to [0, 1]

    print(f"✓ Loaded {len(scores)} sentence pairs\n")
    return sentences1, sentences2, scores


def evaluate_projection_hdc(
    sentences1: List[str],
    sentences2: List[str],
    human_scores: List[float],
    hd_dim: int = 10000,
    binary: bool = True
) -> dict:
    """Evaluate Projection HDC encoder"""
    print("=" * 60)
    print("EVALUATING PROJECTION HDC ENCODER (Phase 1.1)")
    print("=" * 60)
    print(f"HD Dimensions: {hd_dim}")
    print(f"Binary: {binary}")
    print(f"Semantic Seed: SentenceTransformer embeddings")
    print()

    encoder = ProjectionHDCEncoder(
        hd_dim=hd_dim,
        binary=binary,
        device='cpu'
    )

    print("\nEncoding sentences...")
    start_time = time.time()

    # Encode all sentences
    all_sentences = list(set(sentences1 + sentences2))
    print(f"  Unique sentences: {len(all_sentences)}")

    sentence_vectors = {}
    for i, sent in enumerate(all_sentences):
        if i % 500 == 0 and i > 0:
            print(f"  Progress: {i}/{len(all_sentences)}")
        vec = encoder.encode([sent])[0]
        sentence_vectors[sent] = vec

    # Compute similarities
    similarities = []
    for sent1, sent2 in zip(sentences1, sentences2):
        vec1 = sentence_vectors[sent1]
        vec2 = sentence_vectors[sent2]
        sim = encoder.cosine_similarity(vec1, vec2)
        similarities.append(sim)

    encoding_time = time.time() - start_time

    # Compute Spearman correlation
    correlation, p_value = spearmanr(human_scores, similarities)

    results = {
        "method": f"Projection HDC (Phase 1.1, binary={binary})",
        "hd_dimensions": hd_dim,
        "spearman_correlation": correlation,
        "p_value": p_value,
        "encoding_time_seconds": encoding_time,
        "pairs_per_second": len(sentences1) / encoding_time,
        "semantic_seed": "SentenceTransformer (all-MiniLM-L6-v2)"
    }

    print(f"\n✓ Encoding complete in {encoding_time:.2f}s")
    print(f"  Speed: {results['pairs_per_second']:.1f} pairs/sec")
    print()

    return results


def evaluate_baseline(
    sentences1: List[str],
    sentences2: List[str],
    human_scores: List[float]
) -> dict:
    """Evaluate SentenceTransformers baseline"""
    print("=" * 60)
    print("EVALUATING BASELINE (SentenceTransformers)")
    print("=" * 60)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding sentences...")
    start_time = time.time()

    embeddings1 = model.encode(sentences1, show_progress_bar=False, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, show_progress_bar=False, convert_to_numpy=True)

    # Compute cosine similarities
    similarities = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        similarities.append(sim)

    encoding_time = time.time() - start_time

    # Compute Spearman correlation
    correlation, p_value = spearmanr(human_scores, similarities)

    results = {
        "method": "SentenceTransformers (Baseline)",
        "dimensions": embeddings1.shape[1],
        "spearman_correlation": correlation,
        "p_value": p_value,
        "encoding_time_seconds": encoding_time,
        "pairs_per_second": len(sentences1) / encoding_time
    }

    print(f"\n✓ Encoding complete in {encoding_time:.2f}s")
    print(f"  Speed: {results['pairs_per_second']:.1f} pairs/sec")
    print()

    return results


def compare_results(projection_results: dict, baseline_results: dict):
    """Compare results and determine success/failure"""
    print("=" * 60)
    print("PHASE 1.1 RESULTS")
    print("=" * 60)
    print()

    baseline_score = baseline_results['spearman_correlation']
    projection_score = projection_results['spearman_correlation']
    gap = baseline_score - projection_score
    gap_percent = (gap / baseline_score) * 100

    print(f"{'Method':<50} {'Spearman ρ':>12}")
    print("-" * 65)
    print(f"{baseline_results['method']:<50} {baseline_score:>12.4f}")
    print(f"{projection_results['method']:<50} {projection_score:>12.4f}")
    print()

    print("ANALYSIS:")
    print(f"  Baseline:        {baseline_score:.4f}")
    print(f"  Projection HDC:  {projection_score:.4f}")
    print(f"  Gap:             {gap:.4f} ({gap_percent:+.1f}%)")
    print()

    # Determine success
    if abs(gap_percent) <= 5:
        status = "✅ SUCCESS"
        message = "Phase 1.1 achieves within 5% of baseline!"
    elif abs(gap_percent) <= 15:
        status = "⚠️  PARTIAL SUCCESS"
        message = "Phase 1.1 within 15%, improvement over Phase 1"
    else:
        status = "❌ FAILURE"
        message = "Phase 1.1 still needs improvement"

    print(f"VERDICT: {status}")
    print(f"  {message}")
    print()

    # Speed comparison
    speed_ratio = projection_results['pairs_per_second'] / baseline_results['pairs_per_second']
    print(f"SPEED:")
    print(f"  Projection HDC is {speed_ratio:.2f}× {'faster' if speed_ratio > 1 else 'slower'}")
    print()

    return {
        "status": status,
        "gap_percent": gap_percent,
        "speed_ratio": speed_ratio
    }


def save_results(projection_results: dict, baseline_results: dict, comparison: dict):
    """Save results to JSON"""
    output = {
        "phase": "1.1",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "approach": "Projection HDC with Semantic Seed",
        "baseline": baseline_results,
        "projection_hdc": projection_results,
        "comparison": comparison
    }

    os.makedirs("hdc/results", exist_ok=True)
    output_file = "hdc/results/phase_1.1_projection.json"

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Results saved to {output_file}")


def main():
    """Run Phase 1.1 benchmark"""
    print("\n" + "=" * 60)
    print("PHASE 1.1: PROJECTION HDC WITH SEMANTIC SEED")
    print("=" * 60)
    print()

    # Load dataset
    sentences1, sentences2, human_scores = load_sts_benchmark()

    # Evaluate baseline
    baseline_results = evaluate_baseline(sentences1, sentences2, human_scores)

    # Evaluate Projection HDC
    projection_results = evaluate_projection_hdc(
        sentences1, sentences2, human_scores,
        hd_dim=10000,
        binary=True
    )

    # Compare
    comparison = compare_results(projection_results, baseline_results)

    # Save
    save_results(projection_results, baseline_results, comparison)

    print("\n" + "=" * 60)
    print("PHASE 1.1 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
