# SEMANTIC EVENT PROTOCOL (SEP)

**Event-driven distributed intelligence. Triggered by meaning, not time.**

> **Website:** [seprotocol.ai](https://seprotocol.ai)
> **Author:** Nikolay Yudin ([@Nikolay_Yudin_](https://twitter.com/Nikolay_Yudin_))
> **Contact:** [1@seprotocol.ai](mailto:1@seprotocol.ai)

[![Last Commit](https://img.shields.io/github/last-commit/nick-yudin/SEP)](https://github.com/nick-yudin/SEP)
[![Status](https://img.shields.io/badge/Status-Level%201%20Complete-brightgreen)](https://seprotocol.ai)
[![License](https://img.shields.io/badge/License-Open-blue)](LICENSE)

---

## 🚀 Quick Start

**Three ways to explore SEP:**

### 1️⃣ Read the Specification
Understand the technical foundation: [**Level 1 Specification**](https://seprotocol.ai/docs/specs/v1.0_current/spec-v1-final)

### 2️⃣ Try Interactive Demos
See HDC research results: [**Interactive Demo**](https://seprotocol.ai/demo)

### 3️⃣ Explore Research
Review M2-M4 experimental series: 32× compression, 91.3% cross-lingual transfer, 110% semantic arithmetic, 98.4% vs Knowledge Distillation: [**Research Overview**](https://seprotocol.ai/docs/research)

---

## The Problem

**AI is becoming critical infrastructure. And it's controlled by 3 companies in 1 country.**

- **NVIDIA** controls the hardware
- **USA** controls NVIDIA (export restrictions on chips)
- **OpenAI, Anthropic, Google** control the top models
- **Everyone else** is just a customer — with a kill switch

Training a GPT-4 class model costs ~$100M. 70% goes to GPU compute — thousands of H100s for months. The gap is growing exponentially.

**We don't think "catching up" is the answer. We think the paradigm itself is wrong.**

---

## What is the Semantic Event Protocol (SEP)?

The Semantic Event Protocol is an open standard for **meaning-triggered computing**.

In traditional IoT and AI systems, devices stream data continuously (clock-driven) or poll sensors at fixed intervals. This creates massive noise, latency, and energy waste.

**SEP flips the axiom:**

- **Silence is the default state.** A node transmits nothing until "meaning" changes.
- **Meaning is mathematical.** We use high-dimensional vectors (embeddings) to track state.
- **Events are semantic.** We transmit the change in meaning ($\Delta\mu$), not raw data.

> *"Compute only when it matters."*

---

## 🎯 Core Concepts

### 1. Semantic Filtering

Traditional system:
```
Every 100ms: Send sensor data → 36,000 packets/hour
```

SEP system:
```
Only when meaning changes → 47 packets/hour (99.9% reduction)
```

**How?** Cosine distance in embedding space:
```python
if cosine(v_current, v_last) > threshold:
    transmit()  # Significant change
else:
    silence()   # Noise/synonym
```

---

### 2. Procrustes Alignment

**Problem:** Different nodes use different LLMs (GPT-4, Claude, Llama) with incompatible vector spaces.

**Solution:** Orthogonal Procrustes rotation matrix.

```python
# Nodes share random seed
anchors_A = random_vectors(seed=42)
anchors_B = random_vectors(seed=42)

# Compute rotation
R = orthogonal_procrustes(anchors_A, anchors_B)

# Now they can communicate
v_aligned = v_from_other_node @ R
```

**Result:** Cross-LLM communication without shared training.

---

### 3. Mesh Propagation

Events spread via gossip protocol. No server. No polling.

```
NODE_00 detects event → transmits to neighbors
  → NODE_01 forwards to its neighbors
    → NODE_02 forwards...
      → Entire mesh informed in <100ms
```

**Features:**
- TTL-based hop limiting
- Duplicate suppression via memory
- Energy-efficient (transmit only meaningful events)

---

## 📊 Why This Matters

| Traditional (Clock-Based) | SEP (Meaning-Based) |
|---------------------------|---------------------------|
| Poll every 100ms | Transmit only on change |
| 100% duty cycle | 0.1% duty cycle |
| 2.3MB/hour bandwidth | 18KB/hour bandwidth |
| 6 hour battery life | 3 day battery life |
| 500ms cloud latency | <10ms local mesh |

---

## 📁 Repository Structure

```
SEP/
├── docs/                      # The Single Source of Truth
│   ├── 00_intro/
│   │   └── manifesto.md       # The philosophical foundation (Level 0)
│   ├── 01_specs/
│   │   └── v1.0_current/
│   │       └── spec_v1_final.md  # The technical standard (Level 1)
│   └── 02_research/           # Research documentation (M2-M4 series)
│
├── papers/                    # Published papers (PDFs only)
│   └── Cross_Architecture_HDC_Transfer/
│       └── Cross_Architecture_HDC_Transfer.pdf  # Published paper
│
├── papers_experiments/        # Reproducible experiments for papers
│   └── Cross_Architecture_HDC_Transfer/
│       ├── README.md          # Experiment documentation
│       ├── notebooks/         # Jupyter notebooks
│       └── results/           # Result plots and data
│
├── reference_impl/            # Working code
│   └── python/
│       ├── quick_demo.py      # ⭐ Start here
│       ├── alignment.py       # Procrustes solver
│       ├── gossip.py          # 10-node mesh
│       ├── sender.py          # TCP wire protocol
│       └── receiver.py        # Protobuf deserialization
│
└── website/                   # seprotocol.ai source
```

---

## 🔬 Reference Implementation

**Python** (Level 1 compliant): [`/reference_impl/python`](./reference_impl/python)

Run a working mesh in 3 commands:
```bash
cd reference_impl/python
pip install -r requirements.txt
python quick_demo.py
```

**Status:** ⚠️ Alpha reference implementation
**Tested:** December 2025 — 10-node simulated mesh with 3 embedding backends

[📖 Full Implementation Docs](./reference_impl/python/README.md)

---

## 📖 Documentation

### For Philosophers
**[Manifesto](./docs/00_intro/manifesto.md)** — Why we are abandoning clock-based computing.

### For Engineers
**[Level 1 Specification](./docs/01_specs/v1.0_current/spec_v1_final.md)** — Wire protocol, embeddings, alignment mechanism.

### For Builders
**[Python Reference](./reference_impl/python/README.md)** — Working code with examples.

### For Researchers
**[Papers & Experiments](./papers_experiments/)** — Reproducible experiments validating our research claims.

**Published Work:**
- **"Cross-Architecture Knowledge Transfer via HDC"** (2025) — [Zenodo](https://zenodo.org/records/18009693) | [Experiments](./papers_experiments/Cross_Architecture_HDC_Transfer/)

---

## 🌐 Links

- **Website:** [https://seprotocol.ai](https://seprotocol.ai)
- **Twitter/X:** [@Nikolay_Yudin_](https://twitter.com/Nikolay_Yudin_)
- **Contact:** [1@seprotocol.ai](mailto:1@seprotocol.ai)

---

## ⚠️ Honest Status

### What Works Today

| Component | Status | Evidence |
|-----------|--------|----------|
| Semantic filtering | ✅ Proven | 90%+ reduction in benchmarks |
| Procrustes alignment | ✅ Proven | Cross-model communication |
| Cross-lingual transfer | ✅ Proven | 91.3% across 10 languages (M4c) |
| Semantic arithmetic | ✅ Proven | 110% retention — ternary improves compositionality (M4d) |
| HDC vs Knowledge Distillation | ✅ Proven | 98.4% of KD accuracy with unique properties (M4e) |
| Event-driven architecture | ⚙️ Demonstrated | Simulated energy savings |
| Gossip mesh propagation | ⚙️ Demonstrated | Small-scale tests |

### What We're Researching

| Component | Status | Notes |
|-----------|--------|-------|
| Distributed training on edge | 🔬 Research | DiLoCo, Hivemind show promise. Not production-ready. |
| Ternary computing (10-100×) | ⏳ Waiting | BitNet works. Waiting for ternary hardware. |
| Semantic efficiency for training | 💭 Speculation | Works for inference, not proven for training yet. |
| Governance mechanisms | 📐 Design | "No one controls" needs real mechanism design. |

**Our bet:** New hardware (memristors, neuromorphic chips, in-memory computing) will change the economics of AI. We're building the protocol ready for that hardware.

---

## 📊 Key Experimental Results (M4 Series)

Our latest experiments prove that HDC captures genuine meaning:

| Experiment | Result | What It Proves |
|------------|--------|----------------|
| **Cross-Lingual Transfer** | 91.3% | Train on English → works on Chinese, Arabic, Hindi |
| **Semantic Arithmetic** | 110% | king - man + woman = queen works in ternary |
| **HDC vs KD** | 98.4% | Competitive with standard Knowledge Distillation |
| **Compression** | 32× | Ternary vs float32 with semantic preservation |

**Key insight:** Ternary quantization doesn't just preserve meaning — it *improves* semantic structure.

[📖 Full Research Documentation](https://seprotocol.ai/docs/research)

---

## 🙏 Contributing

This is a research project, not a startup. We're looking for people who see the problem and want to build the alternative.

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for detailed guidelines.

**Quick links:**
- Run the [Quick Start](./reference_impl/python/README.md)
- Read the [Technical Specification](./docs/01_specs/v1.0_current/spec_v1_final.md)
- Join [GitHub Discussions](https://github.com/nick-yudin/SEP/discussions)

**Governance:** All public artifacts maintained in English.

---

## 📜 License

**Code & Implementation:** Apache-2.0 License (see [LICENSE](./LICENSE))

**Documentation & Specification:** Creative Commons Attribution 4.0 International (CC-BY-4.0)

You are free to use, modify, and distribute this work with attribution.

---

## 🎓 Citation

If you use the Semantic Event Protocol (SEP) in research, please cite:

```bibtex
@misc{sep2025,
  title={Semantic Event Protocol (SEP): A Standard for Distributed Edge Intelligence},
  author={Nikolay Yudin},
  year={2025},
  url={https://seprotocol.ai}
}

@misc{yudin2025meaning,
  title={...Until We Found Meaning: Semantic Transfer via Hyperdimensional Computing},
  author={Nikolay Yudin},
  year={2025},
  url={https://github.com/nick-yudin/SEP}
}
```

---

**Author:** Nikolay Yudin
**Initiated:** 2025
**Status:** Level 1 Complete

*"Silence is golden. Meaning is everything."*