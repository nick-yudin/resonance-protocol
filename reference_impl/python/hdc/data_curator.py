"""
HDC-Based Data Curator: Deduplication and Diversity Sampling

Goal: Improve fine-tuning efficiency by curating training data using HDC representations.

Strategy:
1. Encode all examples into Ternary HDC vectors
2. Remove near-duplicates (cosine similarity > threshold)
3. Cluster remaining examples for diversity
4. Sample from clusters to create curated subset
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from hdc.ternary_encoder import TernaryHDCEncoder
from typing import List, Tuple, Dict
from tqdm import tqdm
import json


class HDCDataCurator:
    """
    Curate training data using HDC representations.

    Parameters:
        hd_dim: Hypervector dimensions (default: 10,000)
        sparsity: Ternary quantization sparsity (default: 0.7)
        dedup_threshold: Cosine similarity threshold for duplicates (default: 0.95)
        device: torch device
    """

    def __init__(
        self,
        hd_dim: int = 10000,
        sparsity: float = 0.7,
        dedup_threshold: float = 0.95,
        device: str = 'cpu'
    ):
        self.hd_dim = hd_dim
        self.sparsity = sparsity
        self.dedup_threshold = dedup_threshold
        self.device = device

        print(f"Initializing HDC Data Curator:")
        print(f"  HD Dimensions: {hd_dim}")
        print(f"  Sparsity: {sparsity}")
        print(f"  Dedup Threshold: {dedup_threshold}")
        print()

        self.encoder = TernaryHDCEncoder(
            hd_dim=hd_dim,
            sparsity=sparsity,
            device=device
        )

    def encode_dataset(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode all texts into HDC vectors.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            (n_samples, hd_dim) ternary array
        """
        print(f"Encoding {len(texts)} texts into HDC vectors...")

        all_vectors = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i+batch_size]
            vectors = self.encoder.encode(batch)
            all_vectors.append(vectors.cpu().numpy())

        embeddings = np.vstack(all_vectors)
        print(f"✓ Encoded {len(embeddings)} samples")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Sparsity: {(embeddings == 0).mean():.1%}")
        print()

        return embeddings

    def remove_duplicates(
        self,
        embeddings: np.ndarray,
        indices: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Remove near-duplicate examples based on HDC similarity.

        Args:
            embeddings: (n_samples, hd_dim) HDC vectors
            indices: Original indices (if None, use range(n_samples))

        Returns:
            unique_embeddings: Deduplicated embeddings
            unique_indices: Indices of unique samples
            duplicate_groups: List of duplicate group sizes
        """
        if indices is None:
            indices = np.arange(len(embeddings))

        print(f"Removing duplicates (threshold={self.dedup_threshold})...")

        # Convert to torch for faster similarity computation
        vectors = torch.FloatTensor(embeddings)

        unique_mask = np.ones(len(embeddings), dtype=bool)
        duplicate_groups = []

        for i in tqdm(range(len(embeddings)), desc="Deduplication"):
            if not unique_mask[i]:
                continue

            # Skip if no remaining vectors
            if i + 1 >= len(embeddings):
                break

            # Compute similarity with all remaining vectors
            vec_i = vectors[i:i+1]
            remaining_vectors = vectors[i+1:]

            if len(remaining_vectors) == 0:
                break

            similarities = torch.nn.functional.cosine_similarity(
                vec_i.expand(len(remaining_vectors), -1),
                remaining_vectors,
                dim=1
            )

            # Mark duplicates
            duplicates = similarities > self.dedup_threshold
            if duplicates.any():
                dup_indices = np.where(duplicates.numpy())[0] + i + 1
                unique_mask[dup_indices] = False
                duplicate_groups.append(len(dup_indices) + 1)

        unique_embeddings = embeddings[unique_mask]
        unique_indices = indices[unique_mask]

        print(f"✓ Deduplication complete:")
        print(f"  Original: {len(embeddings)}")
        print(f"  Unique: {len(unique_embeddings)}")
        print(f"  Removed: {len(embeddings) - len(unique_embeddings)}")
        print(f"  Duplicate groups: {len(duplicate_groups)}")
        if duplicate_groups:
            print(f"  Avg group size: {np.mean(duplicate_groups):.1f}")
        print()

        return unique_embeddings, unique_indices, duplicate_groups

    def cluster_for_diversity(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster embeddings to ensure diversity.

        Args:
            embeddings: (n_samples, hd_dim) HDC vectors
            n_clusters: Number of clusters
            random_state: Random seed

        Returns:
            cluster_labels: (n_samples,) cluster assignments
            cluster_centers: (n_clusters, hd_dim) cluster centroids
        """
        print(f"Clustering for diversity (k={n_clusters})...")

        # K-means clustering in HDC space
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )

        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_

        # Cluster statistics
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)

        print(f"✓ Clustering complete:")
        print(f"  Clusters: {len(unique_labels)}")
        print(f"  Samples per cluster: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        print()

        return cluster_labels, cluster_centers

    def sample_from_clusters(
        self,
        embeddings: np.ndarray,
        indices: np.ndarray,
        cluster_labels: np.ndarray,
        n_samples: int,
        strategy: str = 'nearest_centroid'
    ) -> np.ndarray:
        """
        Sample from clusters to create curated subset.

        Args:
            embeddings: (n_samples, hd_dim) HDC vectors
            indices: Original indices
            cluster_labels: Cluster assignments
            n_samples: Target number of samples
            strategy: 'nearest_centroid' or 'random'

        Returns:
            selected_indices: Indices of selected samples
        """
        print(f"Sampling {n_samples} examples (strategy={strategy})...")

        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters)

        # Determine samples per cluster
        samples_per_cluster = np.full(n_clusters, n_samples // n_clusters)
        remainder = n_samples % n_clusters
        samples_per_cluster[:remainder] += 1

        selected_indices = []

        for cluster_id, n_select in zip(unique_clusters, samples_per_cluster):
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = indices[cluster_mask]

            if strategy == 'nearest_centroid':
                # Compute centroid
                centroid = cluster_embeddings.mean(axis=0, keepdims=True)

                # Find nearest samples to centroid
                similarities = np.dot(cluster_embeddings, centroid.T).flatten()
                norms = np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(centroid)
                similarities = similarities / (norms + 1e-8)

                # Select top-k nearest
                top_k = min(n_select, len(cluster_indices))
                nearest_idx = np.argsort(similarities)[-top_k:]
                selected_indices.extend(cluster_indices[nearest_idx])

            elif strategy == 'random':
                # Random sampling
                n_select = min(n_select, len(cluster_indices))
                chosen = np.random.choice(cluster_indices, size=n_select, replace=False)
                selected_indices.extend(chosen)

        selected_indices = np.array(selected_indices[:n_samples])

        print(f"✓ Sampling complete:")
        print(f"  Selected: {len(selected_indices)}")
        print(f"  Clusters represented: {len(unique_clusters)}")
        print()

        return selected_indices

    def curate(
        self,
        texts: List[str],
        target_size: int,
        sampling_strategy: str = 'nearest_centroid',
        batch_size: int = 32
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full curation pipeline: encode → dedup → cluster → sample.

        Args:
            texts: List of text strings
            target_size: Target curated subset size
            sampling_strategy: 'nearest_centroid' or 'random'
            batch_size: Batch size for encoding

        Returns:
            curated_indices: Indices of selected samples
            stats: Curation statistics
        """
        print("=" * 60)
        print("HDC DATA CURATION PIPELINE")
        print("=" * 60)
        print()

        # Step 1: Encode
        embeddings = self.encode_dataset(texts, batch_size=batch_size)
        original_indices = np.arange(len(texts))

        # Step 2: Deduplicate
        unique_embeddings, unique_indices, duplicate_groups = self.remove_duplicates(
            embeddings, original_indices
        )

        # Step 3: Cluster
        n_clusters = min(target_size, len(unique_embeddings))
        cluster_labels, cluster_centers = self.cluster_for_diversity(
            unique_embeddings, n_clusters
        )

        # Step 4: Sample
        curated_indices = self.sample_from_clusters(
            unique_embeddings,
            unique_indices,
            cluster_labels,
            target_size,
            strategy=sampling_strategy
        )

        # Statistics
        stats = {
            "original_size": len(texts),
            "unique_after_dedup": len(unique_embeddings),
            "duplicates_removed": len(texts) - len(unique_embeddings),
            "duplicate_groups": len(duplicate_groups),
            "avg_duplicate_group_size": float(np.mean(duplicate_groups)) if duplicate_groups else 0,
            "n_clusters": n_clusters,
            "curated_size": len(curated_indices),
            "sampling_strategy": sampling_strategy,
            "dedup_threshold": self.dedup_threshold,
            "sparsity": self.sparsity
        }

        print("=" * 60)
        print("CURATION COMPLETE")
        print("=" * 60)
        print(f"Original size: {stats['original_size']}")
        print(f"Duplicates removed: {stats['duplicates_removed']}")
        print(f"Unique samples: {stats['unique_after_dedup']}")
        print(f"Clusters: {stats['n_clusters']}")
        print(f"Final curated size: {stats['curated_size']}")
        print()

        return curated_indices, stats


def demo():
    """Demo HDC data curator"""
    print("=" * 60)
    print("HDC DATA CURATOR DEMO")
    print("=" * 60)
    print()

    # Simulate a dataset with duplicates and variety
    texts = [
        "How do I learn Python programming?",
        "What's the best way to learn Python?",  # Near-duplicate
        "How can I start learning Python?",  # Near-duplicate
        "Explain quantum computing to a beginner",
        "What is quantum computing?",
        "Translate this text to French",
        "Convert this to French language",  # Near-duplicate
        "Write a poem about nature",
        "Compose a poem on the theme of nature",  # Near-duplicate
        "Calculate the factorial of 10",
        "What is 10 factorial?",
        "Summarize the history of Rome",
        "Give me a brief history of ancient Rome",
        "How do I bake a chocolate cake?",
        "Recipe for chocolate cake",
    ] * 10  # Repeat to create 150 examples

    curator = HDCDataCurator(
        hd_dim=10000,
        sparsity=0.7,
        dedup_threshold=0.90
    )

    # Curate to 20 examples
    curated_indices, stats = curator.curate(
        texts,
        target_size=20,
        sampling_strategy='nearest_centroid',
        batch_size=32
    )

    print("CURATED SUBSET:")
    for i, idx in enumerate(curated_indices[:10]):
        print(f"{i+1}. {texts[idx]}")
    print(f"... ({len(curated_indices)} total)")
    print()

    print("STATISTICS:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demo()
