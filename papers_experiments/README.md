# Papers & Experiments

This directory contains reproducible experiments and code that validate claims made in published papers. Each paper has its own subfolder with Jupyter notebooks, results, and detailed documentation.

## Purpose

The goal of this directory is to ensure **full reproducibility** of our research. Every claim in our papers can be verified by running the corresponding notebooks.

## Structure

Each paper has the following structure:

```
papers_experiments/
└── [Paper_Name]/
    ├── README.md          # Detailed experiment documentation
    ├── notebooks/         # Jupyter notebooks with experiments
    │   ├── Experiment1_*.ipynb
    │   └── Experiment2_*.ipynb
    └── results/          # Generated plots and data
        ├── *.png         # Result visualizations
        └── *.json        # Raw numerical results
```

## Published Papers

### 1. Cross-Architecture Knowledge Transfer via HDC

**Publication:** [Zenodo (2025)](https://zenodo.org/records/18009693)

**Paper PDF:** [papers/Cross_Architecture_HDC_Transfer/](../papers/Cross_Architecture_HDC_Transfer/)

**Experiments:** [Cross_Architecture_HDC_Transfer/](./Cross_Architecture_HDC_Transfer/)

**Key Results:**
- HDC quantization: 6-10% accuracy loss from float→ternary
- Contrastive alignment: 96% efficiency vs Procrustes (79%), CCA (83%)
- Bidirectional transfer: 94-99% efficiency across model pairs
- Teacher size study: DistilBERT (66M) achieved best transfer at 98.8% efficiency

**Notebooks:**
- `Experiment1_Complete_Pipeline.ipynb` - Full transfer pipeline validation
- `Experiment2_Teacher_Size.ipynb` - Teacher model size impact study

---

### 2. Encoder-Free Text Classification Using Hyperdimensional Computing

**Publication:** [Zenodo (2025)](https://doi.org/10.5281/zenodo.18025695)

**Paper PDF:** [papers/Encoder_Free_HDC/](../papers/Encoder_Free_HDC/)

**Experiments:** [Encoder_Free_HDC/](./Encoder_Free_HDC/)

**Key Results:**
- Language ID (20 classes): 94.3% accuracy (5.1% gap to BERT)
- Topic classification (4 classes): 76.8% accuracy (17.3% gap to BERT)
- Sentiment analysis (2 classes): 70.7% accuracy (21.7% gap to BERT)
- No neural networks, runs on <4KB memory microcontrollers

**Notebooks:**
- `Paper2_StatisticalValidation.ipynb` - Complete evaluation across 3 tasks with statistical validation

---

## Running Experiments

### Prerequisites

All notebooks are designed to run in **Google Colab** with the following requirements:

**Paper 1 (Cross-Architecture HDC Transfer):**
- **Experiment 1**: T4 or L4 GPU (16GB VRAM), ~2-3 hours runtime
- **Experiment 2**: A100 GPU (40GB VRAM), ~3-4 hours runtime

**Paper 2 (Encoder-Free HDC):**
- **CPU only** (no GPU needed!), ~30-45 minutes runtime

### How to Run

1. Navigate to the paper's experiment directory
2. Read the paper-specific README for detailed instructions
3. Open the notebook in Google Colab
4. Select the appropriate GPU runtime
5. For experiments requiring model access (e.g., Llama):
   - Create a Hugging Face account and accept model licenses
   - Add `HF_TOKEN` to Colab secrets (Settings → Secrets)
6. Run all cells

### Expected Outputs

Each notebook generates:
- `.json` files with numerical results
- `.png` files with visualizations
- Console output with experiment summaries

## Contributing

If you reproduce our experiments and get different results:

1. Check hardware requirements (GPU type, VRAM)
2. Verify random seeds match the paper
3. Open an issue in the main repository with:
   - Your results vs. paper results
   - Hardware configuration
   - Software versions (Python, PyTorch, transformers)

We want to understand any discrepancies to improve reproducibility.

## Citation

If you use these experiments in your research, please cite the corresponding paper. See the paper-specific README for BibTeX entries.

## Links

- **Main Repository**: [github.com/nick-yudin/SEP](https://github.com/nick-yudin/SEP)
- **Website**: [seprotocol.ai](https://seprotocol.ai)
- **Contact**: [1@seprotocol.ai](mailto:1@seprotocol.ai)
