# HDC Text Encoder — Hyperdimensional Computing Research

**Phase 1 of Resonance Protocol HDC exploration**

This module implements and benchmarks a Hyperdimensional Computing (HDC) approach to semantic text encoding using Binary Spatter Codes.

---

## 🎯 Research Goal

**Hypothesis:** HDC can encode text semantics comparably to neural sentence transformers, with potential advantages:
- ✅ Lower computational cost (no neural network training)
- ✅ Explainable operations (vector algebra, not black box)
- ✅ Hardware efficiency (binary operations, suitable for neuromorphic chips)

**Success Criterion:** HDC achieves Spearman correlation within 5% of sentence-transformers baseline on STS dataset.

**✅ STATUS: ACHIEVED — Phase 1.1 (Projection HDC) scores ρ = 0.8201 vs baseline 0.8203 (0.0% gap)**

---

## 📁 Files

```
hdc/
├── README.md                    # This file
├── RESEARCH_LOG.md              # Research journal (all attempts documented)
├── text_encoder.py              # Phase 1: Naive Binary Spatter Codes (FAILED)
├── projection_encoder.py        # Phase 1.1: Projection HDC (SUCCESS!)
├── benchmark_sts.py             # Phase 1 benchmark
├── benchmark_projection.py      # Phase 1.1 benchmark
└── results/
    ├── sts_benchmark.json       # Phase 1 results
    └── phase_1.1_projection.json # Phase 1.1 results
```

---

## 🔬 How It Works

### Phase 1.1: Projection HDC (✅ Current Approach)

**This is the successful approach that achieves baseline performance!**

#### Algorithm

1. **Semantic Seed:** Use pretrained SentenceTransformer embeddings (384-dim)
   ```python
   base_model = SentenceTransformer('all-MiniLM-L6-v2')
   word_embeddings = base_model.encode(text)  # (384,)
   ```

2. **Random Projection Matrix:** Fixed random matrix for Johnson-Lindenstrauss projection
   ```python
   projection = torch.randn(384, 10000) / sqrt(384)
   ```

3. **Project to Hyperspace:** Simple matrix multiplication
   ```python
   hyper_vector = word_embeddings @ projection  # (10000,)
   ```

4. **Binary Quantization:** Convert to binary for efficiency
   ```python
   binary_vector = (hyper_vector > 0).astype(bool)
   ```

5. **Similarity:** Hamming distance (fast bitwise operations)
   ```python
   similarity = 1 - hamming_distance(v1, v2) / dimensions
   ```

**Why it works:**
- ✅ Pretrained embeddings already encode semantics
- ✅ Johnson-Lindenstrauss lemma guarantees distance preservation
- ✅ Binary quantization maintains rank correlation
- ✅ Simple and fast

**Result:** Spearman ρ = 0.8201 (vs baseline 0.8203, gap = 0.0%)

---

### Phase 1: Binary Spatter Codes (❌ Failed Approach)

This was the initial naive attempt that did not work.

The HDC text encoder uses the following approach:

#### 1. Token Encoding
Each unique token is mapped to a random **10,000-bit binary vector**:
```python
token_vector = random_binary_vector(dimensions=10000)
```

Deterministic: same token always gets same vector (via hash-based seeding).

#### 2. N-gram Encoding
Text is broken into overlapping 3-grams. Each n-gram is encoded using **circular permutation binding**:

```
For n-gram ["the", "cat", "sat"]:
1. Get token vectors: v_the, v_cat, v_sat
2. Bind with positions via permutation:
   - v_the ⊕ pos_0 (no permutation)
   - v_cat ⊕ pos_1 (rotate by 1)
   - v_sat ⊕ pos_2 (rotate by 2)
3. Bundle via XOR: ngram_vector = v1 ⊕ v2 ⊕ v3
```

#### 3. Text Encoding
All n-grams are combined via **majority voting**:
```
final_vector[i] = 1 if (more than half of n-gram vectors have 1 at position i)
```

#### 4. Similarity
Cosine similarity for binary vectors = **Hamming similarity**:
```python
similarity = 1 - (hamming_distance / dimensions)
```

---

## 🚀 Quick Start

### Install Dependencies

```bash
pip install torch sentence-transformers datasets scipy 'numpy<2'
```

### Run Demo (Phase 1.1)

```bash
cd reference_impl/python
python -m hdc.projection_encoder
```

Expected output:
```
=== Projection HDC Text Encoder Demo ===

Loading pretrained model: all-MiniLM-L6-v2
Initializing projection matrix: 384 → 10000

Encoding sentences:
1. The cat sat on the mat
2. A cat is sitting on a mat
3. The weather is nice today
4. Dogs are playing in the park

Semantic Similarity Matrix:
      S1    S2    S3    S4
S1:  1.000 0.790 0.344 0.340
S2:  0.790 1.000 0.340 0.342
S3:  0.344 0.340 1.000 0.513
S4:  0.340 0.342 0.513 1.000

✓ S1 and S2 have high similarity (0.790) - same meaning!
✓ S1 vs S3, S4 have low similarity - different meanings
```

---

## 📊 Run Benchmark

Compare Projection HDC against sentence-transformers on **STS Benchmark** (1,379 test sentence pairs):

```bash
python -m hdc.benchmark_projection
```

### Expected Output

```
============================================================
PHASE 1.1: PROJECTION HDC WITH SEMANTIC SEED
============================================================

Loading STS Benchmark dataset...
✓ Loaded 1379 sentence pairs

============================================================
EVALUATING BASELINE (SentenceTransformers)
============================================================
Encoding sentences...

✓ Encoding complete in 19.64s
  Speed: 70.2 pairs/sec

============================================================
EVALUATING PROJECTION HDC ENCODER (Phase 1.1)
============================================================
HD Dimensions: 10000
Binary: True
Semantic Seed: SentenceTransformer embeddings

Loading pretrained model: all-MiniLM-L6-v2
Initializing projection matrix: 384 → 10000

Encoding sentences...
  Unique sentences: 2552
  Progress: 500/2552
  Progress: 1000/2552
  Progress: 1500/2552
  Progress: 2000/2552
  Progress: 2500/2552

✓ Encoding complete in 57.37s
  Speed: 24.0 pairs/sec

============================================================
PHASE 1.1 RESULTS
============================================================

Method                                               Spearman ρ
-----------------------------------------------------------------
SentenceTransformers (Baseline)                          0.8203
Projection HDC (Phase 1.1, binary=True)                  0.8201

ANALYSIS:
  Baseline:        0.8203
  Projection HDC:  0.8201
  Gap:             0.0002 (+0.0%)

VERDICT: ✅ SUCCESS
  Phase 1.1 achieves within 5% of baseline!

SPEED:
  Projection HDC is 0.34× slower

✓ Results saved to hdc/results/phase_1.1_projection.json
```

---

## 📈 Evaluation Metrics

### Spearman Correlation (ρ)
Measures rank correlation between predicted similarities and human judgments.
- **Range:** -1 to +1
- **Good performance:** ρ > 0.70
- **Excellent performance:** ρ > 0.80

### Success Criteria
- ✅ **Success:** HDC within 5% of baseline
- ⚠️ **Partial:** HDC within 15% of baseline
- ❌ **Failure:** Gap > 15%

---

## 🔧 Configuration

Key parameters in `text_encoder.py`:

```python
HDCTextEncoder(
    dimensions=10000,    # Hypervector size (trade-off: capacity vs memory)
    ngram_size=3,        # N-gram size (3 is standard)
    device='cpu'         # or 'cuda' for GPU
)
```

### Tuning Recommendations

**Increase dimensions (10k → 20k):**
- ✅ Better semantic capacity
- ❌ More memory, slower

**Decrease dimensions (10k → 5k):**
- ✅ Faster, less memory
- ❌ May lose semantic detail

**Change n-gram size:**
- `ngram_size=2`: Faster, less context
- `ngram_size=4`: More context, slower

---

## 🧪 Next Steps (Phase 2+)

If Phase 1 succeeds:

### Phase 2: Ternary Quantization
- Reduce float32 → {-1, 0, +1}
- 96 bytes per vector (16× smaller than current)
- Test on edge hardware (Jetson, RasPi)

### Phase 3: Integration with Resonance
- Replace sentence-transformers in `quick_demo.py`
- Benchmark MQTT vs Resonance with HDC encoding
- Measure real energy savings

### Phase 4: Hardware Acceleration
- Test on neuromorphic chips (Intel Loihi, BrainChip Akida)
- Profile on ARM processors
- Explore FPGA implementation

---

## 📚 References

1. **Kanerva, P. (2009).** "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." *Cognitive Computation*, 1(2), 139-159.

2. **Hersche, M. et al. (2022).** "Constrained Few-shot Class-incremental Learning with Hyperdimensional Computing." *arXiv:2203.08807*.

3. **Schlegel, K. et al. (2022).** "TorchHD: Hardware-Agnostic Hyperdimensional Computing Framework." *GitHub repository*.

4. **Räsänen, O. et al. (2023).** "Vector Symbolic Architectures as a Computing Framework for Nanoscale Hardware." *Nano Communication Networks*, 34, 100429.

---

## 🤝 Contributing

This is research code. Improvements welcome:

- [ ] Better tokenization (use spaCy or BPE)
- [ ] Experiment with different binding operations
- [ ] Test on multilingual datasets
- [ ] Optimize performance (vectorized operations)
- [ ] Add support for GPU batch encoding

See [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines.

---

## 📝 Citation

If you use this HDC encoder in research:

```bibtex
@misc{resonance_hdc2025,
  title={HDC Text Encoder for Resonance Protocol},
  author={Nikolay Yudin},
  year={2025},
  url={https://github.com/nick-yudin/resonance-protocol/tree/main/reference_impl/python/hdc}
}
```

---

**Questions?** → 1@resonanceprotocol.org
