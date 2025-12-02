"""
SentenceTransformer-Based Data Curator

Baseline curation using conventional dense embeddings (384d) for comparison with HDC.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
from tqdm import tqdm


class STDataCurator:
    """
    Curate training data using SentenceTransformer embeddings.

    Parameters:
        model_name: SentenceTransformer model name
        device: torch device
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu'
    ):
        self.model_name = model_name
        self.device = device

        print(f"Initializing ST Data Curator:")
        print(f"  Model: {model_name}")
        print()

        self.model = SentenceTransformer(model_name, device=device)
        self.embed_dim = self.model.get_sentence_embedding_dimension()

    def encode_dataset(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode all texts into SentenceTransformer embeddings.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            (n_samples, embed_dim) float array
        """
        print(f"Encoding {len(texts)} texts with SentenceTransformer...")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        print(f"✓ Encoded {len(embeddings)} samples")
        print(f"  Shape: {embeddings.shape}")
        print()

        return embeddings

    def cluster_and_sample(
        self,
        embeddings: np.ndarray,
        indices: np.ndarray,
        n_samples: int,
        strategy: str = 'nearest_centroid',
        random_state: int = 42
    ) -> Tuple[np.ndarray, Dict]:
        """
        Cluster embeddings and sample representatives.

        Args:
            embeddings: (n_samples, embed_dim) dense embeddings
            indices: Original indices
            n_samples: Target number of samples
            strategy: 'nearest_centroid' or 'random'
            random_state: Random seed

        Returns:
            selected_indices: Indices of selected samples
            stats: Curation statistics
        """
        print(f"Clustering (k={n_samples}) and sampling (strategy={strategy})...")

        # K-means clustering
        kmeans = KMeans(
            n_clusters=n_samples,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )

        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_

        print(f"✓ Clustering complete")
        print(f"  Inertia: {kmeans.inertia_:.2f}")
        print()

        # Sample from each cluster
        print(f"Sampling representatives from {n_samples} clusters...")

        selected_indices = []
        unique_clusters = np.unique(cluster_labels)

        for cluster_id in tqdm(unique_clusters, desc="Sampling"):
            cluster_mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            cluster_indices = indices[cluster_mask]

            if strategy == 'nearest_centroid':
                # Find sample nearest to centroid
                centroid = cluster_centers[cluster_id:cluster_id+1]

                similarities = np.dot(cluster_embeddings, centroid.T).flatten()
                norms = np.linalg.norm(cluster_embeddings, axis=1) * np.linalg.norm(centroid)
                similarities = similarities / (norms + 1e-8)

                # Select nearest
                nearest_idx = np.argmax(similarities)
                selected_indices.append(cluster_indices[nearest_idx])

            elif strategy == 'random':
                # Random sample from cluster
                chosen = np.random.choice(cluster_indices, size=1)[0]
                selected_indices.append(chosen)

        selected_indices = np.array(selected_indices)

        print(f"✓ Sampling complete:")
        print(f"  Selected: {len(selected_indices)}")
        print()

        stats = {
            "method": "ST-Curated",
            "model": self.model_name,
            "embed_dim": self.embed_dim,
            "n_clusters": n_samples,
            "curated_size": len(selected_indices),
            "sampling_strategy": strategy,
            "inertia": float(kmeans.inertia_)
        }

        return selected_indices, stats

    def curate(
        self,
        texts: List[str],
        target_size: int,
        sampling_strategy: str = 'nearest_centroid',
        batch_size: int = 32
    ) -> Tuple[np.ndarray, Dict]:
        """
        Full curation pipeline: encode → cluster → sample.

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
        print("ST DATA CURATION PIPELINE")
        print("=" * 60)
        print()

        # Step 1: Encode
        embeddings = self.encode_dataset(texts, batch_size=batch_size)
        original_indices = np.arange(len(texts))

        # Step 2: Cluster and sample
        curated_indices, stats = self.cluster_and_sample(
            embeddings,
            original_indices,
            target_size,
            strategy=sampling_strategy
        )

        stats["original_size"] = len(texts)

        print("=" * 60)
        print("ST CURATION COMPLETE")
        print("=" * 60)
        print(f"Original size: {len(texts)}")
        print(f"Final curated size: {len(curated_indices)}")
        print()

        return curated_indices, stats


def demo():
    """Demo ST data curator"""
    print("=" * 60)
    print("ST DATA CURATOR DEMO")
    print("=" * 60)
    print()

    texts = [
        "How do I learn Python programming?",
        "Explain quantum computing to a beginner",
        "Translate this text to French",
        "Write a poem about nature",
        "Calculate the factorial of 10",
        "Summarize the history of Rome",
        "How do I bake a chocolate cake?",
    ] * 20  # 140 examples

    curator = STDataCurator(device='cpu')

    curated_indices, stats = curator.curate(
        texts,
        target_size=20,
        sampling_strategy='nearest_centroid'
    )

    print("CURATED SUBSET:")
    for i, idx in enumerate(curated_indices[:10]):
        print(f"{i+1}. {texts[idx]}")
    print(f"... ({len(curated_indices)} total)")
    print()

    print("STATISTICS:")
    import json
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    demo()
