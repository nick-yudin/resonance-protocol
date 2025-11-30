# Resonance Protocol - Python Reference Implementation

**Level 1 Specification Compliant**

This is the official reference implementation of the Resonance Protocol in Python. It demonstrates all core concepts: semantic filtering, Procrustes alignment, and mesh propagation.

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

## 📁 What's Inside

| File | Purpose | Key Concept |
|------|---------|-------------|
| **`quick_demo.py`** | All-in-one interactive demo | Start here |
| **`alignment.py`** | Procrustes alignment algorithm | Different LLMs can communicate |
| **`gossip.py`** | 10-node mesh simulation | Decentralized propagation |
| **`sender.py`** | TCP sender with semantic filtering | Wire protocol implementation |
| **`receiver.py`** | TCP receiver | Protobuf deserialization |
| **`resonance.proto`** | Protocol buffer specification | Binary wire format |

---

## 🎯 Individual Demos

### Demo 1: Semantic Filtering

**Concept:** Nodes transmit only when meaning changes, not on every clock tick.

```bash
# Terminal 1: Start receiver
python receiver.py

# Terminal 2: Start sender
python sender.py
```

**What happens:**
- Sender processes 6 inputs
- Only 3 are transmitted (50% bandwidth reduction)
- Synonyms like "system startup" vs "system initialization" are suppressed

**Key insight:** `cosine_distance(v1, v2) < threshold → SILENCE`

---

### Demo 2: Procrustes Alignment

**Concept:** Different models (GPT-4, Claude, Llama) have different vector spaces. Procrustes finds the rotation matrix to align them.

```bash
python alignment.py
```

**What happens:**
- Generates 1000 synthetic anchor vectors
- Simulates "alien" node with rotated space
- Computes rotation matrix $R$
- Tests on real sentence: error < 0.00001

**Key insight:** This is how heterogeneous nodes can communicate without retraining.

---

### Demo 3: Mesh Propagation

**Concept:** Events spread through gossip protocol. No central coordinator.

```bash
python gossip.py
```

**What happens:**
- 10 nodes form a mesh topology
- Event injected at NODE_00
- Propagates via TTL-based gossip
- Duplicate suppression via memory cache

**Key insight:** Emergent coordination without a server.

---

## 🔬 How It Works

### 1. Semantic Filtering

```python
# Current input
v_current = model.encode("Fire detected")

# Last transmitted vector
v_last = model.encode("System ready")

# Calculate semantic distance
distance = cosine(v_current, v_last)

if distance > THRESHOLD:  # 0.35
    transmit(v_current)  # Meaningful change
else:
    silence()  # Noise/synonym
```

**Result:** 50-90% reduction in network traffic.

---

### 2. Procrustes Alignment

```python
# Node A and Node B share random seed
seed = 42
np.random.seed(seed)

# Both generate 1000 anchor vectors
anchors_A = np.random.randn(1000, 384)
anchors_B = np.random.randn(1000, 384)

# Node B computes rotation matrix
R = orthogonal_procrustes(anchors_A, anchors_B)

# Now B can decode A's vectors
v_aligned = v_from_A @ R
```

**Result:** Different LLMs can communicate without shared training.

---

### 3. Wire Protocol

```protobuf
message SemanticEvent {
  string source_id = 1;
  int64 created_at = 2;
  repeated float embedding = 3;  // 384 dimensions
  string debug_label = 4;
}
```

**TCP Framing:**
```
[4 bytes: message length][N bytes: protobuf payload]
```

---

## 📊 Performance Characteristics

| Metric | Value | Notes |
|--------|-------|-------|
| **Vector dimension** | 384 | MiniLM-L6-v2 |
| **Semantic threshold** | 0.35 | Tunable per use case |
| **Bandwidth reduction** | 50-90% | vs continuous polling |
| **Alignment error** | <10^-5 | Procrustes accuracy |
| **Latency** | <10ms | Local mesh, no cloud |

---

## 🛠️ Requirements

- Python 3.8+
- 2GB RAM (for sentence-transformers model)
- No GPU required (CPU inference is fast enough)

**Dependencies:**
```
sentence-transformers  # Semantic embeddings
scipy                  # Procrustes solver
numpy                  # Vector operations
protobuf               # Wire protocol
```

---

## 🔗 Next Steps

1. **Read the Specification**  
   [`/docs/01_specs/v1.0_current/spec_v1_final.md`](../../docs/01_specs/v1.0_current/spec_v1_final.md)

2. **Explore the Manifesto**  
   [`/docs/00_intro/manifesto.md`](../../docs/00_intro/manifesto.md)

3. **Visit the Website**  
   [https://resonanceprotocol.org](https://resonanceprotocol.org)

4. **Join the Discussion**  
   Twitter: [@rAI_stack](https://twitter.com/rAI_stack)

---

## 🐛 Troubleshooting

**Q: Model download fails?**  
A: First run downloads `all-MiniLM-L6-v2` (~80MB). Needs internet.

**Q: `receiver.py` says "Address already in use"?**  
A: Kill previous instance: `pkill -f receiver.py` or change PORT in code.

**Q: High CPU usage?**  
A: Normal on first run (model loading). Subsequent runs are fast.

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