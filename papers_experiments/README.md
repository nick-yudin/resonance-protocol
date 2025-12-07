# Paper 1: Cross-Architecture Knowledge Transfer via HDC

## Experiments

This folder contains Jupyter notebooks for reproducing the experiments in Paper 1.

### Experiment 1: Complete Transfer Pipeline
**File**: `Paper1_Experiment1_Complete_Pipeline.ipynb`

Tests the full HDC transfer pipeline:
- Stage 0: Model ceilings (fine-tuned baselines)
- Stage 1: HDC quantization cost
- Stage 2: Alignment methods comparison (None, Procrustes, CCA, Contrastive)
- Stage 3: Model pairs (bidirectional transfer)
- Stage 4: Task generalization (SST-2, AG News)

**Runtime**: ~2-3 hours on Colab GPU

### Experiment 2: Teacher Size Study
**File**: `Paper1_Experiment2_Teacher_Size.ipynb`

Tests whether larger teacher models improve transfer quality:
- DistilBERT (66M) - baseline
- GPT-2 (124M)
- Llama 3.1 8B
- Qwen 2.5 14B

**Runtime**: ~3-4 hours on Colab GPU (with 4-bit quantization)

**Requirements**: 
- Hugging Face account
- Accept Llama license at https://huggingface.co/meta-llama/Llama-3.1-8B
- Add HF_TOKEN to Colab secrets

## Key Results

### Experiment 1
- Contrastive alignment achieves **96% efficiency** (best among tested methods)
- Bidirectional transfer works with **94-99% efficiency**
- Same-family transfers (BERT↔BERT) achieve near-perfect efficiency

### Experiment 2
- **Teacher size does not predict transfer quality** in this setup
- DistilBERT (66M) achieved best results: 78.9% accuracy, 98.8% efficiency
- Architectural compatibility appears more important than model capacity

## Configuration

Both notebooks use similar configuration:
```python
{
    'hdc_dim': 4096,
    'tau': 0.3,  # Ternary threshold
    'train_size': 2000-3000,
    'test_size': 500,
    'anchor_size': 500,
    'seeds': [42, 123, 456],
}
```

## Citation

```bibtex
@article{sep2025transfer,
  title={Cross-Architecture Knowledge Transfer via Hyperdimensional Computing},
  author={Yudin, Nikolay},
  year={2025},
  note={Semantic Event Protocol}
}
```

## Links
- Paper: [papers/paper1/main.tex](../papers/paper1/main.tex)
- Project: https://seprotocol.ai
- GitHub: https://github.com/nick-yudin/SEP
