# Resonance Protocol - Python Reference Implementation

**Level 1 Specification Compliant**

This is the official reference implementation of the Resonance Protocol in Python. It demonstrates all core concepts: semantic filtering, Procrustes alignment, and mesh propagation.

---

## 🔮 Future: Ternary Computing & Compression

Resonance Protocol is designed for **ternary logic systems** and will evolve toward:

### Phase 1: Current (float32)
- 384-dimensional vectors
- 1536 bytes per packet
- Proof of concept on commodity hardware

### Phase 2: Compression (Q1 2025)
- **Ternary quantization**: {-1, 0, +1} weights → 96 bytes (16x smaller)
- **HDC encoding**: 10,000-d binary vectors → 128 bytes
- **BitNet 1.58b integration**: Native ternary models

### Phase 3: Custom Hardware (2025-2026)
- Memristor-based compute-in-memory
- 90nm process + neuromorphic design  
- DVS cameras & silicon cochlea sensors
- <100mW per node, $5-10 cost

**Why ternary?**
- Compatible with BitNet 1.58b (Microsoft Research, 2024)
- Enables stochastic computing (noise becomes a feature)
- Reduces memory bandwidth by 16-32x
- Natural fit for memristor arrays

**See [ROADMAP.md](../../ROADMAP.md) for full technical vision.**

---

## 🚀 Quick Start (30 seconds)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run interactive demo
python quick_demo.py
```

That's it! You'll see:
- ✅ Semantic noise suppression in action
- ✅ Cross-LLM alignment via Procrustes
- ✅ Decentralized mesh propagation

---

## 🎬 See It In Action

[![asciicast](https://asciinema.org/a/Bh7Gt17Pd1YvAeBPENYWqFfSj.svg)](https://asciinema.org/a/Bh7Gt17Pd1YvAeBPENYWqFfSj)

*Interactive demo showing semantic filtering, Procrustes alignment, and mesh propagation in real-time*

---

## 🔥 Killer Proof: MQTT vs Resonance

**Want to see real numbers?**

```bash
python benchmarks/mqtt_vs_resonance.py
```

**Results from 1-hour sensor simulation:**

| Metric | MQTT (Legacy) | Resonance | Improvement |
|--------|--------------|-----------|-------------|
| 📦 Packets sent | 12,000 | 120 | **99.0% reduction** |
| 📊 Bandwidth | 1,500 KB | 180 KB | **88% savings** |
| ⚡ Energy | 75 mAh | 7.2 mAh | **90% savings** |
| 🔋 Battery life | 1.1 days | 11.6 days | **10.5x longer** |

[📖 Full Benchmark Details](./benchmarks/README.md)

---

## 📁 Repository Structure

```
/reference_impl/python/
├── quick_demo.py          # ⭐ Start here - interactive tour
│
├── basic/                 # 📚 Educational examples
│   ├── alignment.py       # Procrustes alignment
│   ├── gossip.py          # 10-node mesh simulation
│   ├── sender.py          # TCP sender
│   ├── receiver.py        # TCP receiver
│   └── README.md          # Learning guide
│
├── benchmarks/            # 🔥 Performance proofs
│   ├── mqtt_vs_resonance.py    # Main benchmark
│   ├── results/           # Generated data
│   └── README.md          # Methodology
│
├── assets/                # 🎬 Media
│   └── demo.cast          # Terminal recording
│
└── requirements.txt       # Dependencies
```

---

## 🎯 Choose Your Path

### Path 1: I want to understand the concepts

```bash
# Interactive tour
python quick_demo.py

# Then explore basics
cd basic
python alignment.py
python gossip.py
```

[📖 Basic Examples Guide](./basic/README.md)

---

### Path 2: I want to see proof it works

```bash
# Run the benchmark
python benchmarks/mqtt_vs_resonance.py

# See the numbers
cat benchmarks/results/comparison.json
```

[📊 Benchmarks Guide](./benchmarks/README.md)

---

### Path 3: I want to build with it

```bash
# Start with sender/receiver
cd basic
python receiver.py  # Terminal 1
python sender.py    # Terminal 2
```

Then read: [Level 1 Specification](../../docs/01_specs/v1.0_current/spec_v1_final.md)

---

## 🔬 How It Works

### 1. Semantic Filtering

```python
# Traditional: Send every reading
for reading in sensor_data:
    mqtt_publish(reading)  # 12,000 transmissions

# Resonance: Send only meaningful changes
for reading in sensor_data:
    if cosine(embedding(reading), last_vector) > 0.35:
        transmit(reading)  # ~120 transmissions
```

**Result:** 99% fewer packets, 90% less energy.

---

### 2. Procrustes Alignment

```python
# Problem: Node A uses GPT-4, Node B uses Llama
# Their vector spaces are rotated

# Solution: Calibration via shared random anchors
R = orthogonal_procrustes(anchors_A, anchors_B)

# Now B can understand A's vectors
aligned = vector_from_A @ R
```

**Result:** Heterogeneous nodes can communicate.

---

### 3. Mesh Propagation

```
NODE_00 detects fire
  → transmits to NODE_01, NODE_02
    → NODE_01 forwards to NODE_03, NODE_04
      → Event reaches all nodes in <100ms
```

**Result:** No server, no single point of failure.

---

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Vector dimension** | 384 | MiniLM-L6-v2 |
| **Semantic threshold** | 0.35 | Tunable |
| **Bandwidth reduction** | 88-99% | vs polling |
| **Energy reduction** | 90-95% | vs always-on |
| **Alignment error** | <10^-5 | Procrustes |
| **Latency** | <10ms | Local mesh |

---

## 🛠️ Requirements

- Python 3.8+
- 2GB RAM (for model)
- No GPU required

**Dependencies:**
```bash
pip install sentence-transformers scipy numpy protobuf
```

---

## 🔗 Next Steps

1. **Run the demos** → Start with `quick_demo.py`
2. **See the proof** → Run `benchmarks/mqtt_vs_resonance.py`
3. **Read the spec** → [Level 1 Documentation](../../docs/01_specs/v1.0_current/spec_v1_final.md)
4. **Explore the manifesto** → [Why this matters](../../docs/00_intro/manifesto.md)
5. **Visit the website** → [resonanceprotocol.org](https://resonanceprotocol.org)

---

## 🐛 Troubleshooting

**Q: Model download fails?**  
A: First run downloads `all-MiniLM-L6-v2` (~80MB). Needs internet.

**Q: Benchmark takes too long?**  
A: Reduce `DURATION_MINUTES` in `mqtt_vs_resonance.py` from 60 to 5.

**Q: Import errors after refactoring?**  
A: Make sure you're running from the `python/` root directory.

---

## 📝 License

This reference implementation is part of the Resonance Protocol project.  
See main repository for license details.

---

## 🙏 Acknowledgments

- **Sentence Transformers:** Nils Reimers & Iryna Gurevych
- **Procrustes Method:** Schönemann (1966)
- **Inspiration:** Biological neural systems, event-driven architectures

---

**Questions?** → 1@resonanceprotocol.org