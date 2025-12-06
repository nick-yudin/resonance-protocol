---
id: manifesto
title: Level 0 Manifesto
slug: /manifesto
sidebar_position: 1
---

# Level 0: The SEP Manifesto

## Abstract

The Semantic Event Protocol (SEP) explores an alternative approach to distributed AI: instead of transmitting model weights or raw data, nodes exchange compressed semantic representations. Instead of computing on clock cycles, devices compute when meaning changes.

This document describes the philosophy, the approach, and the current state of experimental validation. We aim for honesty over hype: we share what works, what doesn't, and what remains unknown.

---

## 1. The Problem

Modern AI is centralized. Training a frontier model costs $100M+, requires thousands of GPUs, and concentrates capability in a few organizations. Inference depends on cloud APIs. Edge devices remain passive sensors, not intelligent participants.

This creates:
- **Dependency**: Entire industries rely on API access that can be revoked
- **Inefficiency**: Raw data travels to datacenters for processing
- **Fragility**: No connectivity means no intelligence

We ask: is there another way?

---

## 2. The Hypothesis

What if intelligence could be distributed not by copying models, but by sharing meaning?

The core idea:
- Large models understand meaning (semantic representations)
- This meaning can be compressed into minimal form
- Compressed meaning can be shared between nodes
- Nodes can act on meaning locally, without the full model

**This is a hypothesis under investigation, not a proven system.**

---

## 3. The Approach: Hyperdimensional Computing

We use Hyperdimensional Computing (HDC) as the foundation for semantic compression.

### What is HDC?

HDC represents information as high-dimensional vectors (thousands of dimensions) where:
- Similar meanings have similar vectors
- Vectors can be combined through simple arithmetic
- Extreme quantization (ternary: {-1, 0, +1}) preserves semantic structure

### Why HDC?

1. **Compression**: Ternary vectors are 32x smaller than float32
2. **Composability**: Vectors can be added, subtracted, compared
3. **Hardware efficiency**: Ternary operations need only XOR and popcount
4. **Noise tolerance**: High dimensionality provides redundancy

---

## 4. Core Principles

### Principle 1: Silence is Default

Nodes do not transmit unless meaning changes. No heartbeats, no polling, no redundant data. Silence is the normal state.

### Principle 2: Meaning Over Data

Nodes exchange semantic representations, not raw inputs. A camera doesn't send pixels; it sends "person walking left" as a vector.

### Principle 3: Local Autonomy

Each node maintains its own understanding. No central coordinator required. Nodes can operate independently and sync when connected.

### Principle 4: Threshold-Based Communication

A node transmits only when semantic distance from last transmission exceeds a threshold:

```
if cosine_distance(current_vector, last_vector) > threshold:
    transmit()
else:
    remain_silent()
```

---

## 5. What We Have Tested

All experiments conducted by single author in controlled settings. Results require independent replication and broader validation before any production use.

### Semantic Transfer (M4 Series)

**M4c: Cross-Lingual Transfer**
- Setup: Train classifier on English XNLI (10K examples), test on 10 languages
- Result: 91.3% transfer ratio
- Languages: German, French, Spanish, Russian, Chinese, Arabic, Bulgarian, Hindi, Vietnamese
- Interpretation: Suggests HDC captures language-agnostic meaning

**M4d: Semantic Compositionality**
- Setup: Test word analogies (king - man + woman = queen) in ternary HDC
- Result: 110% retention (ternary outperforms original float embeddings)
- Interpretation: Quantization may act as regularization

**M4e: Comparison with Knowledge Distillation**
- Setup: Compare HDC transfer vs standard KD on SST-2
- Result: HDC achieves 98.4% of KD accuracy (87.3% vs 88.6%)
- Interpretation: HDC competitive while providing unique properties

### Distributed Training (M3 Series)

**M3b: HDC Compression**
- Setup: 2-node distributed training with HDC-compressed sync
- Result: 32x compression (17.5 MB to 271 KB)
- Limitation: 2 nodes only, single task

**M3c: Cross-Architecture Transfer**
- Setup: Transfer from DistilBERT to GPT-2 via HDC
- Result: 93% transfer efficiency on SST-2
- Limitation: Single task, simple classification

### Earlier Experiments (M2 Series)

**M2.6: Compositional Generalization**
- Setup: Test on unseen attribute-object combinations
- Result: 100% accuracy on synthetic task
- Limitation: Toy dataset, not natural language

---

## 6. What We Have Not Tested

Transparency about limitations:

- **Scale**: Maximum 2-10 nodes in experiments, not hundreds or thousands
- **Hardware**: All experiments on standard GPUs, no edge device validation
- **Real-world tasks**: Mostly classification benchmarks, not production workloads
- **Latency**: No systematic measurement of real-time performance
- **Failure modes**: Limited adversarial or stress testing
- **Security**: No formal security analysis
- **Long-term stability**: No experiments beyond hours of runtime

---

## 7. The Vision (Unvalidated)

If the experimental findings generalize, SEP could enable:

**Edge Intelligence**
Devices that understand locally, share meaning efficiently, operate without constant connectivity.

**Mesh Networks**
Swarms of nodes that collectively know more than any individual, learning from each other through semantic exchange.

**Decentralized AI**
Intelligence that no single entity controls, emerging from the collaboration of many autonomous nodes.

**This vision is speculative. The experiments show early promising signals, not a working system.**

---

## 8. How to Participate

SEP is open research. We invite:

- **Replication**: Run our experiments, verify or challenge results
- **Extension**: Test on new tasks, larger scales, different domains
- **Criticism**: Find flaws, identify limitations, improve the approach
- **Application**: Explore real-world use cases

All code and data: [github.com/nick-yudin/SEP](https://github.com/nick-yudin/SEP)

---

## 9. Conclusion

SEP proposes that meaning—not weights, not data—could be the unit of distributed intelligence.

Early experiments suggest:
- Semantic representations survive 32x compression
- Meaning transfers across languages (91%) and architectures (93%)
- Ternary quantization preserves and sometimes improves semantic structure
- HDC is competitive with standard approaches while enabling unique properties

These are preliminary findings from controlled experiments. Significant work remains before any real-world deployment.

We share this work not as a solution, but as a direction worth exploring.

---

*Questions, criticism, collaboration: 1@seprotocol.ai*
