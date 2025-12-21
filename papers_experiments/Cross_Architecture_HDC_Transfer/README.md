# Cross-Architecture Knowledge Transfer via HDC

Experimental validation of Hyperdimensional Computing (HDC) for transferring learned representations between different neural network architectures.

**Published:** [Zenodo (2025)](https://zenodo.org/records/18009693) | **Paper PDF:** [Cross_Architecture_HDC_Transfer.pdf](../../papers/Cross_Architecture_HDC_Transfer/Cross_Architecture_HDC_Transfer.pdf)

## Overview

These notebooks reproduce the experiments from our research on using HDC as an intermediate representation for cross-architecture knowledge transfer. The core idea: instead of transferring raw embeddings (which are architecture-specific), we project them into a shared HDC space using ternary vectors {-1, 0, +1}.

**Key question**: Can a classifier trained on Teacher's HDC vectors work on Student's HDC vectors?

## Results Summary

### Experiment 1: Complete Transfer Pipeline

| Stage | Finding |
|-------|---------|
| HDC Quantization | 6-10% accuracy loss from float→ternary |
| Alignment Methods | Contrastive: **96%** efficiency vs Procrustes: 79%, CCA: 83% |
| Model Pairs | Bidirectional transfer achieves **94-99%** efficiency |
| Task Generalization | SST-2: 95%, AG News: 100% efficiency |

### Experiment 2: Teacher Size Study

| Teacher | Parameters | Transfer Accuracy | Efficiency |
|---------|------------|-------------------|------------|
| DistilBERT | 66M | **78.9%** | **98.8%** |
| GPT-2 | 124M | 74.3% | 93.9% |
| Llama 3.1 | 8B | 77.7% | 98.3% |
| Qwen 2.5 | 14B | 75.6% | 96.0% |

**Unexpected finding**: Larger teachers did not improve transfer quality. The smallest model (DistilBERT) achieved the best results. We hypothesize this is due to:
1. Architectural compatibility (encoder→encoder works better)
2. Alignment bottleneck (projecting 5120d Qwen embeddings to 512d shared space loses more information than 768d DistilBERT)
3. Task simplicity (SST-2 may not benefit from larger model capacity)

## Visual Results

### Stage 1: HDC Dimension Impact

![HDC Dimension](results/stage1_hdc_dimension.png)

Higher HDC dimensions (4096, 8192) provide better representation capacity with minimal accuracy loss from quantization.

### Stage 2: Alignment Methods Comparison

![Alignment Methods](results/stage2_alignment_methods.png)

Contrastive alignment significantly outperforms classical methods (Procrustes, CCA), achieving 96% transfer efficiency.

### Stage 3: Bidirectional Transfer

![Model Pairs](results/stage3_model_pairs.png)

Transfer works bidirectionally between different architectures (DistilBERT ↔ GPT-2), demonstrating the universality of HDC representations.

### Stage 4: Task Generalization

![Task Generalization](results/stage4_datasets.png)

The HDC transfer approach generalizes across different tasks (SST-2, AG News) with high efficiency.

### Teacher Size Study

![Teacher Size vs Accuracy](results/strong_teacher_size_vs_accuracy.png)

Counterintuitively, larger teacher models do not guarantee better transfer. DistilBERT (66M) outperforms much larger models like Qwen (14B).

### Architecture Overview

![Architecture Diagram](results/architecture_diagram.png)

The complete pipeline: embedding extraction → HDC projection → ternary quantization → alignment → transfer → classification.

## Notebooks

### Experiment 1: Complete Pipeline
**File**: [notebooks/Paper1_Experiment1_Complete_Pipeline.ipynb](notebooks/Paper1_Experiment1_Complete_Pipeline.ipynb)

**Hardware**: Google Colab T4 or L4 GPU (16GB VRAM)

**Runtime**: ~2-3 hours

**What it tests**:
- Stage 0: Fine-tuned model ceilings (baselines)
- Stage 1: HDC dimension sweep (1024, 2048, 4096, 8192)
- Stage 2: Alignment methods (None, Procrustes, CCA, Contrastive)
- Stage 3: Bidirectional transfer between model pairs
- Stage 4: Generalization to different tasks (SST-2, AG News)

### Experiment 2: Teacher Size Study
**File**: [notebooks/Paper1_Experiment2_Teacher_Size.ipynb](notebooks/Paper1_Experiment2_Teacher_Size.ipynb)

**Hardware**: Google Colab A100 GPU (40GB VRAM) — required for Llama 8B and Qwen 14B even with 4-bit quantization

**Runtime**: ~3-4 hours

**What it tests**:
- Does a stronger teacher produce better HDC transfer?
- Teachers: DistilBERT (66M), GPT-2 (124M), Llama 3.1 (8B), Qwen 2.5 (14B)
- Student: DistilBERT (fixed)
- Task: SST-2 sentiment classification

**Requirements**:
- Hugging Face account with access token
- Accept Llama 3.1 license at https://huggingface.co/meta-llama/Llama-3.1-8B
- Add `HF_TOKEN` to Colab secrets (Settings → Secrets)

## Reproduce Our Results

We encourage independent verification of these findings. To run:

1. Open notebook in Google Colab
2. Select appropriate GPU runtime (T4/L4 for Exp1, A100 for Exp2)
3. For Experiment 2: add HF_TOKEN to Colab secrets
4. Run all cells

**Expected outputs**:
- `paper1_experiment1_results.json` — full numerical results
- `paper1_experiment1_results.png` — visualization
- Console output with per-stage summaries

If you get significantly different results, please open an issue — we want to understand why.

## Configuration

Both notebooks use similar hyperparameters:

```python
{
    'hdc_dim': 4096,           # HDC vector dimension
    'tau': 0.3,                # Ternary quantization threshold (× std)
    'train_size': 2000-3000,   # Training samples
    'test_size': 500,          # Test samples  
    'anchor_size': 500,        # Samples for alignment training
    'seeds': [42, 123, 456],   # For statistical significance
    'contrastive_epochs': 20,  # Alignment training epochs
}
```

## Key Implementation Details

**HDC Encoding**:
```python
projected = embeddings @ random_projection  # (N, 768) → (N, 4096)
threshold = 0.3 * std(projected)
ternary[projected > threshold] = +1
ternary[projected < -threshold] = -1
# Result: 32× compression (float32 → int2)
```

**Contrastive Alignment**:
```python
# Project teacher and student to shared space
t_proj = normalize(MLP(teacher_emb))  # (N, 768) → (N, 512)
s_proj = normalize(MLP(student_emb))  # (N, 768) → (N, 512)

# Train: same text should have similar projections
loss = -cosine(t_proj, s_proj) + margin_loss(t_proj, s_proj_shuffled)
```

## Caveats

These results are preliminary and come with important limitations:

1. **Single task dominance**: Most experiments use SST-2 (binary sentiment). Results may differ on more complex tasks.

2. **Alignment bottleneck**: Our contrastive aligner projects to 512d shared space. Larger models with 4096d+ embeddings may lose more information.

3. **No sentence-pair tasks**: We excluded NLI/paraphrase tasks where our HDC encoding performed poorly (near-random accuracy). Different encoding strategies may be needed.

4. **Limited model diversity**: We tested BERT-family and GPT-family. Other architectures (T5, Mamba, etc.) not evaluated.

## Links

- Project: https://seprotocol.ai
- Repository: https://github.com/nick-yudin/SEP
- Contact: 1@seprotocol.ai
