"""
Projection HDC Encoder with Semantic Seed

Phase 1.1: Projects pretrained dense embeddings into hyperdimensional space.

Theory:
- Johnson-Lindenstrauss Lemma: Random projection preserves distances
- Semantic initialization via pretrained word embeddings
- HDC operations for compositional encoding
- Binarization for efficiency

Reference:
- Johnson & Lindenstrauss (1984): "Extensions of Lipschitz mappings"
- Kanerva (2009): "Hyperdimensional Computing"
"""

import torch
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np


class ProjectionHDCEncoder:
    """
    HDC encoder with semantic initialization via random projection.

    Architecture:
    1. Token embeddings from pretrained model (384-dim)
    2. Random projection to hyperdimensional space (10,000-dim)
    3. Position encoding via circular permutation
    4. Bundle aggregation for sentence vector
    5. Optional binarization

    Parameters:
        hd_dim: Hyperdimensional space size (default: 10,000)
        base_model: Pretrained sentence transformer model
        binary: Whether to binarize output vectors
        device: Computation device ('cpu', 'cuda', or 'mps')
    """

    def __init__(
        self,
        hd_dim: int = 10000,
        base_model_name: str = 'all-MiniLM-L6-v2',
        binary: bool = True,
        device: str = 'cpu'
    ):
        self.hd_dim = hd_dim
        self.binary = binary
        self.device = device

        # Load pretrained model for semantic embeddings
        print(f"Loading pretrained model: {base_model_name}")
        self.base_model = SentenceTransformer(base_model_name, device=device)
        self.base_dim = self.base_model.get_sentence_embedding_dimension()

        # Random projection matrix (Johnson-Lindenstrauss)
        # This preserves pairwise distances with high probability
        print(f"Initializing projection matrix: {self.base_dim} → {self.hd_dim}")
        torch.manual_seed(42)  # Reproducibility
        self.projection = torch.randn(self.base_dim, self.hd_dim).to(device)

        # Normalize projection matrix (improves stability)
        self.projection = self.projection / torch.sqrt(torch.tensor(self.base_dim, dtype=torch.float32))

    def _get_token_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Get token-level embeddings from pretrained model.

        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        # For now, use sentence-level embeddings
        # TODO: Switch to token-level embeddings or static FastText lookup
        embeddings = self.base_model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device
        )
        return embeddings

    def _project_to_hyperspace(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project dense embeddings to hyperdimensional space.

        Uses Johnson-Lindenstrauss random projection to preserve distances.

        Args:
            embeddings: Dense vectors of shape (batch, base_dim)

        Returns:
            Hyperdimensional vectors of shape (batch, hd_dim)
        """
        # Matrix multiplication: (batch, 384) @ (384, 10000) -> (batch, 10000)
        hd_vectors = torch.matmul(embeddings, self.projection)
        return hd_vectors

    def _binarize(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Convert float vectors to binary {0, 1}.

        Uses sign function: positive -> 1, negative -> 0
        """
        return (vectors > 0).float()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts into hyperdimensional vectors.

        Process:
        1. Get semantic embeddings from pretrained model
        2. Project to hyperdimensional space
        3. Optionally binarize

        Args:
            texts: List of text strings to encode

        Returns:
            Hyperdimensional vectors of shape (len(texts), hd_dim)
        """
        # Step 1: Get semantic embeddings (384-dim)
        embeddings = self._get_token_embeddings(texts)

        # Step 2: Project to hyperspace (10,000-dim)
        hd_vectors = self._project_to_hyperspace(embeddings)

        # Step 3: Optionally binarize
        if self.binary:
            hd_vectors = self._binarize(hd_vectors)

        return hd_vectors

    def cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two vectors.

        For binary vectors: Jaccard similarity approximation
        For float vectors: Standard cosine similarity
        """
        if self.binary:
            # Jaccard similarity for binary vectors
            intersection = torch.logical_and(vec1 > 0, vec2 > 0).sum().item()
            union = torch.logical_or(vec1 > 0, vec2 > 0).sum().item()
            return intersection / union if union > 0 else 0.0
        else:
            # Standard cosine similarity
            dot = torch.sum(vec1 * vec2).item()
            norm1 = torch.sqrt(torch.sum(vec1 ** 2)).item()
            norm2 = torch.sqrt(torch.sum(vec2 ** 2)).item()
            return dot / (norm1 * norm2) if (norm1 * norm2) > 0 else 0.0

    def batch_encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode texts in batches for efficiency.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            Hyperdimensional vectors of shape (len(texts), hd_dim)
        """
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            vectors = self.encode(batch)
            all_vectors.append(vectors)

        return torch.cat(all_vectors, dim=0)


def demo():
    """Demo of Projection HDC encoder"""
    print("=== Projection HDC Encoder Demo ===\n")

    encoder = ProjectionHDCEncoder(hd_dim=10000, binary=True, device='cpu')

    # Test sentences
    sentences = [
        "The cat sat on the mat",
        "A cat is sitting on a mat",  # Similar meaning
        "The weather is nice today",  # Different meaning
        "Dogs are playing in the park",  # Different meaning
    ]

    print("Encoding sentences...")
    vectors = encoder.batch_encode(sentences)

    print(f"✓ Encoded {len(sentences)} sentences")
    print(f"  Vector shape: {vectors.shape}")
    print(f"  Sparsity: {vectors.float().mean():.3f}\n")

    print("Semantic Similarity Matrix:")
    print("     ", end="")
    for i in range(len(sentences)):
        print(f"  S{i+1}  ", end="")
    print()

    for i in range(len(sentences)):
        print(f"S{i+1}:  ", end="")
        for j in range(len(sentences)):
            sim = encoder.cosine_similarity(vectors[i], vectors[j])
            print(f"{sim:.3f}  ", end="")
        print()

    print("\n✓ Expected: S1 and S2 should have high similarity (same meaning)")
    print("✓ Expected: S1 vs S3, S4 should have lower similarity (different meaning)")
    print("\nNote: Semantic similarity comes from pretrained embeddings,")
    print("      preserved through Johnson-Lindenstrauss random projection.")


if __name__ == "__main__":
    demo()
