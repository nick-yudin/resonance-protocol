"""
Train a classifier on Ternary HDC representations

Goal: Prove that neural networks can learn on sparse ternary HDC vectors
without significant accuracy loss.

Dataset: SST-2 (Stanford Sentiment Treebank, binary sentiment classification)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from hdc.ternary_encoder import TernaryHDCEncoder
from typing import Tuple, List
import time
import numpy as np


class BaselineClassifier(nn.Module):
    """
    Baseline: SentenceTransformer embeddings (384d) → MLP → binary classification
    """
    def __init__(self, input_dim: int = 384, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # Binary classification
        )

    def forward(self, x):
        return self.mlp(x)


class HDCClassifier(nn.Module):
    """
    HDC: Ternary HDC vectors (10k ternary) → MLP → binary classification
    """
    def __init__(self, input_dim: int = 10000, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # Binary classification
        )

    def forward(self, x):
        return self.mlp(x)


def load_sst2(split: str, max_samples: int = None) -> Tuple[List[str], List[int]]:
    """
    Load SST-2 dataset from HuggingFace.

    Args:
        split: 'train' or 'validation'
        max_samples: Limit dataset size for faster experiments

    Returns:
        texts: List of sentences
        labels: List of labels (0=negative, 1=positive)
    """
    print(f"Loading SST-2 dataset ({split})...")
    dataset = load_dataset("glue", "sst2", split=split, trust_remote_code=True)

    texts = []
    labels = []

    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        texts.append(item['sentence'])
        labels.append(item['label'])

    print(f"✓ Loaded {len(texts)} samples from {split} split\n")
    return texts, labels


def encode_baseline(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Encode texts using SentenceTransformer (baseline).

    Returns:
        embeddings: (n_samples, 384) float32 array
    """
    print("Encoding with SentenceTransformer (baseline)...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"✓ Encoded {len(embeddings)} samples, shape: {embeddings.shape}\n")
    return embeddings


def encode_hdc(texts: List[str], hd_dim: int = 10000, sparsity: float = 0.7) -> np.ndarray:
    """
    Encode texts using Ternary HDC encoder.

    Returns:
        embeddings: (n_samples, 10000) ternary array {-1, 0, +1}
    """
    print("Encoding with Ternary HDC...")
    encoder = TernaryHDCEncoder(hd_dim=hd_dim, sparsity=sparsity, device='cpu')
    vectors = encoder.encode(texts)
    embeddings = vectors.cpu().numpy()
    print(f"✓ Encoded {len(embeddings)} samples, shape: {embeddings.shape}")
    print(f"  Sparsity: {(embeddings == 0).mean():.1%} zeros\n")
    return embeddings


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
    device: str = 'cpu'
) -> dict:
    """
    Train a classifier and return training metrics.

    Returns:
        Dictionary with train/val accuracy and training time
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Training classifier...")
    start_time = time.time()

    best_val_acc = 0.0
    train_history = []
    val_history = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_acc = train_correct / train_total
        train_history.append(train_acc)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_acc = val_correct / val_total
        val_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"  Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    training_time = time.time() - start_time

    print(f"\n✓ Training complete in {training_time:.2f}s")
    print(f"  Best validation accuracy: {best_val_acc:.4f}\n")

    return {
        "best_val_accuracy": best_val_acc,
        "final_train_accuracy": train_history[-1],
        "final_val_accuracy": val_history[-1],
        "training_time_seconds": training_time,
        "train_history": train_history,
        "val_history": val_history
    }


def main():
    """Run baseline vs HDC classifier comparison"""
    print("\n" + "=" * 60)
    print("PHASE 2 FULL: TRAINING ON HDC INPUTS")
    print("=" * 60)
    print()

    # Hyperparameters
    MAX_TRAIN_SAMPLES = 5000  # Limit for faster experiments
    MAX_VAL_SAMPLES = 500
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Load dataset
    train_texts, train_labels = load_sst2('train', max_samples=MAX_TRAIN_SAMPLES)
    val_texts, val_labels = load_sst2('validation', max_samples=MAX_VAL_SAMPLES)

    print("=" * 60)
    print("PIPELINE 1: BASELINE (SentenceTransformer → MLP)")
    print("=" * 60)
    print()

    # Encode with baseline
    train_baseline_emb = encode_baseline(train_texts)
    val_baseline_emb = encode_baseline(val_texts)

    # Create baseline dataloaders
    train_baseline_dataset = TensorDataset(
        torch.FloatTensor(train_baseline_emb),
        torch.LongTensor(train_labels)
    )
    val_baseline_dataset = TensorDataset(
        torch.FloatTensor(val_baseline_emb),
        torch.LongTensor(val_labels)
    )
    train_baseline_loader = DataLoader(train_baseline_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_baseline_loader = DataLoader(val_baseline_dataset, batch_size=BATCH_SIZE)

    # Train baseline
    baseline_model = BaselineClassifier(input_dim=384)
    baseline_results = train_classifier(
        baseline_model,
        train_baseline_loader,
        val_baseline_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    print("=" * 60)
    print("PIPELINE 2: HDC (TernaryHDC → MLP)")
    print("=" * 60)
    print()

    # Encode with HDC
    train_hdc_emb = encode_hdc(train_texts, hd_dim=10000, sparsity=0.7)
    val_hdc_emb = encode_hdc(val_texts, hd_dim=10000, sparsity=0.7)

    # Create HDC dataloaders
    train_hdc_dataset = TensorDataset(
        torch.FloatTensor(train_hdc_emb),
        torch.LongTensor(train_labels)
    )
    val_hdc_dataset = TensorDataset(
        torch.FloatTensor(val_hdc_emb),
        torch.LongTensor(val_labels)
    )
    train_hdc_loader = DataLoader(train_hdc_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_hdc_loader = DataLoader(val_hdc_dataset, batch_size=BATCH_SIZE)

    # Train HDC
    hdc_model = HDCClassifier(input_dim=10000)
    hdc_results = train_classifier(
        hdc_model,
        train_hdc_loader,
        val_hdc_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE
    )

    # Compare results
    print("=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print()

    baseline_acc = baseline_results['best_val_accuracy']
    hdc_acc = hdc_results['best_val_accuracy']
    gap = baseline_acc - hdc_acc
    gap_percent = (gap / baseline_acc) * 100

    print(f"{'Method':<30} {'Val Accuracy':>15} {'Train Time (s)':>18}")
    print("-" * 65)
    print(f"{'Baseline (384d float)':<30} {baseline_acc:>15.4f} {baseline_results['training_time_seconds']:>18.2f}")
    print(f"{'HDC (10k ternary)':<30} {hdc_acc:>15.4f} {hdc_results['training_time_seconds']:>18.2f}")
    print()

    print("ANALYSIS:")
    print(f"  Baseline accuracy:  {baseline_acc:.4f}")
    print(f"  HDC accuracy:       {hdc_acc:.4f}")
    print(f"  Gap:                {gap:.4f} ({gap_percent:+.1f}%)")
    print()

    # Determine success
    if abs(gap_percent) <= 10:
        status = "✅ SUCCESS"
        message = "HDC achieves within 10% of baseline!"
    elif abs(gap_percent) <= 20:
        status = "⚠️  PARTIAL SUCCESS"
        message = "HDC within 20%, but some accuracy loss"
    else:
        status = "❌ FAILURE"
        message = "HDC accuracy gap exceeds 20%"

    print(f"VERDICT: {status}")
    print(f"  {message}")
    print()

    print("=" * 60)
    print("PHASE 2 FULL COMPLETE")
    print("=" * 60)

    return {
        "baseline": baseline_results,
        "hdc": hdc_results,
        "comparison": {
            "gap": gap,
            "gap_percent": gap_percent,
            "status": status
        }
    }


if __name__ == "__main__":
    main()
