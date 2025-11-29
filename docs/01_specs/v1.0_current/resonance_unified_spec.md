# Resonance Unified Specification (Level 1)

**Version:** 1.0.0 (Ironclad)

**Status:** Implemented (Python Reference)

**Date:** November 2025



---



## 1. Introduction

This document defines the technical standard for **Resonance Protocol Level 1**. Any node compliant with this specification can join the mesh, filter noise, and exchange semantic events, regardless of the underlying hardware or programming language.



## 2. The Semantic Layer (Layer 7)



### 2.1. Vector Space

All nodes must project data into a shared vector space before processing.

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`

- **Dimensions ($D$):** 384 floating point values.

- **Normalization:** All vectors must be unit-normalized ($||v|| = 1$).



### 2.2. The Silence Mechanism

To ensure energy efficiency ("Silence is Default"), nodes must calculate the Cosine Distance ($d$) between the current input vector ($v_t$) and the last transmitted vector ($v_{t-1}$).



$$d = 1 - \frac{v_t \cdot v_{t-1}}{||v_t|| ||v_{t-1}||}$$



- **Threshold ($\theta$):** **0.35**

- **Logic:**

  - If $d < 0.35$: DROP packet (Noise/Synonym). Update internal state only.

  - If $d \ge 0.35$: TRANSMIT Event (Significant Shift).



---



## 3. The Alignment Layer (Layer 6)



Nodes with different internal representations ("Alien Minds") must align spaces using the **Procrustes Handshake**.



### 3.1. Calibration Protocol

1.  **Seed Exchange:** Nodes agree on a random seed.

2.  **Synthetic Anchors:** Both nodes generate **1000** random vectors (Gaussian Noise) from the seed.

3.  **Solver:** The Receiver computes the Orthogonal Procrustes Matrix ($R$) that minimizes the Frobenius norm between its anchors ($A$) and the sender's anchors ($B$).

    $$\min_R || A R - B ||_F \quad \text{s.t.} \quad R^T R = I$$

4.  **Translation:** All subsequent incoming semantic events $v_{in}$ are rotated:

    $$v_{aligned} = v_{in} \cdot R$$



---



## 4. The Transport Layer (Layer 4)



### 4.1. Wire Format

Data is serialized using **Protocol Buffers v3**.



**`resonance.proto` definition:**

```protobuf

syntax = "proto3";

package resonance;



message SemanticEvent {

  string source_id = 1;       // UUID of the emitter (8 chars)

  int64 created_at = 2;       // Unix Timestamp

  repeated float embedding = 3; // The 384-d vector

  string debug_label = 4;     // Optional text (e.g., "Fire detected")

  int32 ttl = 5;              // Time To Live (default: 3 hops)

}

```

### 4.2. Stream Framing (TCP)

Since TCP is a stream, messages must be framed.

- **Prefix:** 4 bytes (Unsigned Integer, Big-Endian) representing the payload length.

- **Payload:** The binary Protobuf data.



---



## 5. Network Behavior (Gossip)



- **Topology:** Mesh (Ad-hoc).

- **Echo Suppression:** Nodes must maintain a `memory` cache of recently seen `event_id`s. If an ID is in memory, the packet is dropped immediately.

- **Semantic Routing:** Even if a packet is new, a node MAY drop it if the vector content is semantically redundant to the node's current knowledge (distance < $\theta$), preventing information floods.



---



## 6. Reference Implementation

The official Python implementation of this spec is available in:

`/reference_impl/python`



- `sender.py`: Implements Threshold Logic.

- `alignment.py`: Implements Procrustes Solver.

- `gossip.py`: Implements Mesh Propagation.