# Basic Examples - Learning the Protocol

This folder contains **educational demonstrations** of core Resonance Protocol concepts.

---

## 📚 What's Here

These are the foundational building blocks. Start here to understand how the protocol works.

| File | Concept | Run Time |
|------|---------|----------|
| `alignment.py` | Procrustes alignment for heterogeneous nodes | ~30 sec |
| `gossip.py` | Mesh propagation simulation (10 nodes) | ~10 sec |
| `sender.py` | TCP sender with semantic filtering | Background |
| `receiver.py` | TCP receiver with protobuf | Background |
| `resonance.proto` | Wire protocol specification | N/A |

---

## 🎓 Learning Path

### 1. Start with alignment.py

**Concept:** Different AI models have different vector spaces. How do they communicate?

```bash
cd basic
python alignment.py
```

**You'll learn:**
- How Procrustes rotation works
- Why synthetic anchors solve the problem
- What "alien minds" means

---

### 2. Run gossip.py

**Concept:** Events spread through a mesh without a central server.

```bash
python gossip.py
```

**You'll learn:**
- TTL-based propagation
- Duplicate suppression
- Semantic filtering in action

---

### 3. Try sender + receiver

**Concept:** Wire protocol with TCP framing and Protobuf.

```bash
# Terminal 1
python receiver.py

# Terminal 2
python sender.py
```

**You'll learn:**
- How semantic events are serialized
- TCP stream framing (4-byte length prefix)
- When noise is suppressed vs transmitted

---

## 🔧 Technical Details

### Procrustes Alignment (`alignment.py`)

```python
# Problem: Two nodes have rotated vector spaces
# Solution: Find orthogonal rotation matrix R

R = orthogonal_procrustes(alien_vectors, standard_vectors)

# Now alien node can translate:
aligned = alien_vector @ R
```

**Math:** Minimizes Frobenius norm `||A*R - B||`

---

### Gossip Protocol (`gossip.py`)

```python
class MeshNode:
    def process_event(self, event):
        # 1. Deduplication
        if event.id in self.memory:
            return
        
        # 2. Semantic filtering
        if cosine(event.vector, self.last_vector) < THRESHOLD:
            return  # Silence
        
        # 3. Propagate to neighbors
        for peer in self.peers:
            peer.process_event(event)
```

**Key:** No central coordinator. Pure P2P.

---

### Wire Protocol (`sender.py` + `receiver.py`)

**Message format:**
```
[4 bytes: length][N bytes: protobuf]
```

**Protobuf schema:**
```protobuf
message SemanticEvent {
  string source_id = 1;
  int64 created_at = 2;
  repeated float embedding = 3;
  string debug_label = 4;
}
```

---

## 🎯 Next Steps

After mastering these basics:

1. **Run the full demo:** `python quick_demo.py` (in parent folder)
2. **See the benchmarks:** `python benchmarks/mqtt_vs_resonance.py`
3. **Read the spec:** `/docs/01_specs/v1.0_current/spec_v1_final.md`

---

## 🐛 Common Issues

**Q: "Model download failed"?**  
A: First run downloads `all-MiniLM-L6-v2` (~80MB). Needs internet.

**Q: "Address already in use" in receiver.py?**  
A: Kill old instance: `pkill -f receiver.py`

**Q: gossip.py output too fast?**  
A: Add `time.sleep(0.5)` after line 75 to slow it down.

---

## 📖 Further Reading

- **Procrustes Analysis:** Schönemann (1966)
- **Gossip Protocols:** Demers et al. (1987)
- **Semantic Embeddings:** Reimers & Gurevych (2019)

---

**Questions?** → 1@resonanceprotocol.org