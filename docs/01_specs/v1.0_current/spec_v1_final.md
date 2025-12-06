---
id: spec-v1-final
title: Semantic Event Protocol (SEP) — Level 1 Specification
description: The technical standard for the Semantic Event Protocol (SEP) Level 1 with HDC.
slug: /specs/v1.0_current/spec-v1-final
---

# Semantic Event Protocol (SEP) — Level 1 Specification

**Version:** 2.0.0
**Status:** Experimental (not production-ready)
**Date:** December 2025

---

## 1. Introduction

This document defines the technical specification for SEP Level 1. It describes how nodes encode, compress, and exchange semantic information.

**Important**: This specification is based on small-scale experiments. It has not been validated at production scale or on real edge hardware. Implementations should be considered experimental.

---

## 2. Semantic Encoding

### 2.1 Vector Space

Semantic content is encoded as Hyperdimensional Computing (HDC) vectors:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Dimensions | 4096 | Configurable, tested range: 4096-16384 |
| Values | Ternary {-1, 0, +1} | 2 bits per dimension |
| Sparsity | ~60-70% zeros | Achieved via threshold quantization |

### 2.2 Encoding Pipeline

```
Input Text
    ↓
Sentence Encoder (e.g., all-mpnet-base-v2)
    ↓
Float Embedding (768d)
    ↓
Random Projection (768d → 4096d)
    ↓
Ternary Quantization
    ↓
HDC Vector (4096d ternary)
```

### 2.3 Random Projection

Project float embeddings to HDC space:

```python
# Initialize once per network (shared seed)
projection_matrix = random_normal(768, 4096, seed=NETWORK_SEED)
projection_matrix /= norm(projection_matrix, axis=0)

# Project
hdc_float = embedding @ projection_matrix
```

The Johnson-Lindenstrauss lemma provides theoretical justification: random projections preserve pairwise distances with high probability.

### 2.4 Ternary Quantization

Convert float HDC to ternary:

```python
def quantize(hdc_float):
    threshold = 0.3 * std(hdc_float)
    result = zeros_like(hdc_float)
    result[hdc_float > threshold] = +1
    result[hdc_float < -threshold] = -1
    return result  # {-1, 0, +1}
```

Threshold 0.3 * std found empirically. Different values may work better for specific applications.

---

## 3. Semantic Distance

### 3.1 Distance Metric

Cosine distance between HDC vectors:

```python
def semantic_distance(v1, v2):
    similarity = dot(v1, v2) / (norm(v1) * norm(v2))
    return 1 - similarity
```

### 3.2 Transmission Threshold

Node transmits when distance exceeds threshold:

```python
THRESHOLD = 0.35  # Empirically determined

if semantic_distance(current, last_transmitted) > THRESHOLD:
    transmit(current)
    last_transmitted = current
else:
    remain_silent()
```

Threshold 0.35 balances information preservation vs bandwidth. Applications may tune this value.

---

## 4. Wire Format

### 4.1 Semantic Event

```protobuf
syntax = "proto3";
package sep;

message SemanticEvent {
    string node_id = 1;           // Source node identifier
    int64 timestamp = 2;          // Unix timestamp (ms)
    bytes hdc_vector = 3;         // Compressed ternary vector
    uint32 dimensions = 4;        // HDC dimensionality
    uint32 ttl = 5;               // Time-to-live (hops)
    string label = 6;             // Optional human-readable label
    map<string, string> meta = 7; // Optional metadata
}
```

### 4.2 Vector Compression

Ternary vectors compressed using sparse encoding:

```python
def compress(hdc_ternary):
    # Store only non-zero positions and values
    nonzero_idx = where(hdc_ternary != 0)
    nonzero_val = hdc_ternary[nonzero_idx]

    # Pack: 2 bytes per index, 1 bit per value sign
    return pack(nonzero_idx, nonzero_val)

def decompress(packed, dimensions):
    nonzero_idx, nonzero_val = unpack(packed)
    result = zeros(dimensions)
    result[nonzero_idx] = nonzero_val
    return result
```

Typical compression: 4096d ternary → ~1.5 KB (vs 16 KB uncompressed, vs 3 KB float32 original)

### 4.3 Transport

TCP with length-prefix framing:

```
[4 bytes: payload length (big-endian uint32)]
[N bytes: protobuf payload]
```

---

## 5. Network Behavior

### 5.1 Topology

Mesh network with gossip protocol. No central coordinator.

### 5.2 Event Propagation

```python
def on_receive(event):
    # Deduplicate
    if event.node_id + event.timestamp in seen_cache:
        return
    seen_cache.add(event.node_id + event.timestamp)

    # Check TTL
    if event.ttl <= 0:
        return

    # Process locally
    process(event)

    # Propagate to neighbors
    event.ttl -= 1
    for neighbor in neighbors:
        send(neighbor, event)
```

### 5.3 Semantic Deduplication (Optional)

Nodes may drop semantically redundant events:

```python
def should_propagate(event):
    for recent in recent_events:
        if semantic_distance(event.vector, recent.vector) < THRESHOLD:
            return False  # Too similar to recent event
    return True
```

---

## 6. Experimental Results

Results from controlled experiments. All require independent replication.

### 6.1 Semantic Transfer

| Experiment | Setup | Result | Limitations |
|------------|-------|--------|-------------|
| M4c Cross-Lingual | Train EN, test 10 langs | 91.3% transfer | Single task (XNLI) |
| M4d Compositionality | Word analogies | 110% of baseline | 12 analogies, 71 words |
| M4e vs KD | SST-2 sentiment | 98.4% of KD | Single task |

### 6.2 Compression

| Experiment | Setup | Result | Limitations |
|------------|-------|--------|-------------|
| M3b HDC Compression | 2-node training | 32x compression | 2 nodes only |
| Ternary vs Float | Storage | 32x smaller | No speed benchmark |

### 6.3 Cross-Architecture

| Experiment | Setup | Result | Limitations |
|------------|-------|--------|-------------|
| M3c DistilBERT→GPT-2 | SST-2 transfer | 93% efficiency | Single task |

---

## 7. Reference Implementation

Python reference: [github.com/nick-yudin/SEP](https://github.com/nick-yudin/SEP)

Key modules:
- `hdc_encoder.py`: Encoding pipeline
- `semantic_event.py`: Event handling
- `gossip.py`: Network protocol

**Status**: Reference implementation for experimentation. Not optimized for production.

---

## 8. Compliance

A SEP Level 1 node SHOULD:

1. Use ternary HDC vectors for semantic encoding
2. Apply cosine distance for similarity
3. Implement threshold-based transmission
4. Support protobuf wire format
5. Implement TTL-based propagation
6. Maintain seen-event cache for deduplication

A SEP Level 1 node MAY:

- Use different HDC dimensions (recommended: 4096-16384)
- Adjust threshold based on application needs
- Implement semantic deduplication
- Add application-specific metadata

---

## 9. Known Limitations

This specification has not been validated for:

- Networks larger than 10 nodes
- Real edge hardware (Raspberry Pi, Jetson, microcontrollers)
- Real-time latency requirements
- Adversarial environments
- Long-running stability (>24 hours)
- Tasks beyond text classification

These limitations should be addressed before production deployment.

---

## 10. Future Work

Areas requiring further research:

- **Scale**: Test with 100+ nodes
- **Hardware**: Validate on actual edge devices
- **Tasks**: Extend beyond classification (retrieval, generation)
- **Security**: Formal threat modeling
- **Performance**: Systematic latency/throughput benchmarks

---

## 11. Conclusion

SEP Level 1 provides a foundation for semantic-first distributed communication. The specification is based on promising experimental results but remains unvalidated at scale.

We release this as a starting point for research, not as a production standard.

---

*Feedback: 1@seprotocol.ai*
*Code: github.com/nick-yudin/SEP*
