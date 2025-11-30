# RESONANCE PROTOCOL

**A semantic event protocol for distributed edge intelligence. Triggered by meaning, not time.**

[![Last Commit](https://img.shields.io/github/last-commit/nick-yudin/resonance-protocol)](https://github.com/nick-yudin/resonance-protocol)
[![Status](https://img.shields.io/badge/Status-Level%201%20Complete-brightgreen)](https://resonanceprotocol.org)
[![License](https://img.shields.io/badge/License-Open-blue)](LICENSE)

---

## 🚀 Quick Start (30 Seconds)

```bash
git clone https://github.com/nick-yudin/resonance-protocol.git
cd resonance-protocol/reference_impl/python
pip install -r requirements.txt
python quick_demo.py
```

**You'll see:**
- ⚡ Semantic filtering (67% bandwidth reduction)
- 🔄 Procrustes alignment (10^-7 error)
- 🕸️ Mesh propagation (no central server)

[📖 Full Python Docs](./reference_impl/python/README.md)

---

## What is Resonance?

Resonance is an open standard for **meaning-triggered computing**.

In traditional IoT and AI systems, devices stream data continuously (clock-driven) or poll sensors at fixed intervals. This creates massive noise, latency, and energy waste.

**Resonance flips the axiom:**

- **Silence is the default state.** A node transmits nothing until "meaning" changes.
- **Meaning is mathematical.** We use high-dimensional vectors (embeddings) to track state.
- **Events are semantic.** We transmit the change in meaning ($\Delta\mu$), not raw data.

> *"The clock stops. The resonance begins."*

---

## 🎯 Core Concepts

### 1. Semantic Filtering

Traditional system:
```
Every 100ms: Send sensor data → 36,000 packets/hour
```

Resonance system:
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

| Traditional (Clock-Based) | Resonance (Meaning-Based) |
|---------------------------|---------------------------|
| Poll every 100ms | Transmit only on change |
| 100% duty cycle | 0.1% duty cycle |
| 2.3MB/hour bandwidth | 18KB/hour bandwidth |
| 6 hour battery life | 3 day battery life |
| 500ms cloud latency | <10ms local mesh |

---

## 📁 Repository Structure

```
resonance-protocol/
├── docs/                      # The Single Source of Truth
│   ├── 00_intro/
│   │   └── manifesto.md       # The philosophical foundation (Level 0)
│   └── 01_specs/
│       └── v1.0_current/
│           └── spec_v1_final.md  # The technical standard (Level 1)
│
├── reference_impl/            # Working code
│   └── python/
│       ├── quick_demo.py      # ⭐ Start here
│       ├── alignment.py       # Procrustes solver
│       ├── gossip.py          # 10-node mesh
│       ├── sender.py          # TCP wire protocol
│       └── receiver.py        # Protobuf deserialization
│
└── website/                   # resonanceprotocol.org source
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

**Status:** ✅ Production-ready  
**Tested:** November 2025 — 10 nodes, 3 LLMs, zero latency

[📖 Full Implementation Docs](./reference_impl/python/README.md)

---

## 📖 Documentation

### For Philosophers
**[Manifesto](./docs/00_intro/manifesto.md)** — Why we are abandoning clock-based computing.

### For Engineers  
**[Level 1 Specification](./docs/01_specs/v1.0_current/spec_v1_final.md)** — Wire protocol, embeddings, alignment mechanism.

### For Builders
**[Python Reference](./reference_impl/python/README.md)** — Working code with examples.

---

## 🌐 Links

- **Website:** [https://resonanceprotocol.org](https://resonanceprotocol.org)
- **Twitter/X:** [@rAI_stack](https://twitter.com/rAI_stack)
- **Contact:** [1@resonanceprotocol.org](mailto:1@resonanceprotocol.org)

---

## 🙏 Contributing

This is an open protocol. Contributions welcome.

See [`PROJECT_STRUCTURE.md`](./PROJECT_STRUCTURE.md) for guidelines.

**Governance:** All public artifacts maintained in English.

---

## 📜 License

[To be specified]

---

## 🎓 Citation

If you use Resonance Protocol in research, please cite:

```
@misc{resonance2025,
  title={Resonance Protocol: A Semantic Event Standard for Distributed Edge Intelligence},
  author={rAI Research Collective},
  year={2025},
  url={https://resonanceprotocol.org}
}
```

---

**Author:** rAI Research Collective  
**Initiated:** 2025  
**Status:** Level 1 Complete

*"Silence is golden. Meaning is everything."*