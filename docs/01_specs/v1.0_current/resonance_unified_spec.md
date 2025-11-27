---
id: unified-spec
title: Level 1 Unified Spec
slug: /unified-spec
sidebar_position: 1
---

# RESONANCE
## Semantic Event Protocol
### Level 1 Unified Specification (Gold Master)

**Version:** 1.0
**Status:** Frozen for Implementation
**Author:** rAI Research Collective
**Date:** November 2025

---

## Abstract

RESONANCE is a protocol for **meaning-triggered intelligence**. Unlike traditional systems driven by clock cycles or raw data streams, Resonance nodes operate in default silence, emitting events only when a semantic threshold is breached.

This Level 1 Specification unifies the visionary "Quiet Mesh" philosophy with a concrete engineering standard. It defines a canonical wire protocol, a 256-dimensional embedding standard, and a "Reference Concept" alignment mechanism that allows heterogeneous AI models (e.g., MobileNet vs. ResNet) to understand each other without sharing weights.

---

## 1. Core Axioms (The "North Star")

The following invariants are non-negotiable foundations of the protocol:

1.  **Silence is Default:** A node transmitting data without a change in meaning is malfunctioning.
2.  **Meaning, Not Data:** We exchange semantic deltas (Δμ), not raw sensor frames.
3.  **Local Autonomy:** Intelligence is pushed to the extreme edge; there is no central cloud brain.
4.  **Trust is Provenance:** Every semantic event is cryptographically signed at the hardware source.

---

## 2. Semantic Representation

To ensure interoperability between devices ranging from microcontrollers to edge servers, we define a canonical semantic space.

### 2.1 The Semantic Vector
State is represented as a fixed-dimension embedding vector `M` in `R^n`.

* **Standard Dimension:** **256-d** (Float32 or Quantized Int8). This is the mandatory baseline for Level 1 compliance.
* **High-Fidelity:** 512-d (Optional for server nodes).
* **Low-Power:** 128-d (Optional for < 1mW sensors).

### 2.2 Semantic Distance
Nodes MUST use **Cosine Distance** to calculate semantic shift:

```math
d(M_t, M_{t-1}) = 1 - (M_t · M_{t-1}) / (||M_t|| ||M_{t-1}||)
```

### 2.3 The Event Trigger
An event is emitted if and only if:

```math
d(M_{current}, M_{last_sent}) > theta
```

**Recommended Thresholds (θ):**
* **0.08:** High Sensitivity (Precision Monitoring)
* **0.15:** Standard Presence/State Change (**Default**)
* **0.25:** Anomaly Detection Only
* **0.40:** Critical Failures Only

---

## 3. Semantic Alignment (The "Rosetta Stone")

The critical challenge is allowing Node A (using Model X) to talk to Node B (using Model Y). We solve this via **Reference Concept Alignment**.

### 3.1 The Concept Set
The protocol defines **100 Canonical Concepts** (see Appendix A) representing universal constants in the physical world (e.g., `REF_002: Fire`, `REF_050: Human Voice`).

### 3.2 The Handshake Protocol (Procrustes Analysis)
When two nodes meet, they do not exchange model weights. They exchange their understanding of the constants.

1.  **Request:** Node A sends its embedding vectors for the 100 Reference Concepts.
2.  **Calculation:** Node B compares A's vectors to its own using Orthogonal Procrustes analysis to find a rotation matrix `R` that minimizes the error:
    `min || X_B - X_A R ||`
3.  **Agreement:** If the alignment error is < 0.20, the connection is established. Node B will apply matrix `R` to all incoming events from Node A.

---

## 4. Canonical Wire Protocol

We use **Protocol Buffers (proto3)** for efficiency and strict typing.

### 4.1 Transport
* **Discovery:** UDP Multicast on port `5353`.
* **Transport:** TCP (Reliable) or QUIC (Low Latency) on port `7741`.

### 4.2 The Semantic Event Schema
This is the atomic unit of the Resonance network.

```protobuf
syntax = "proto3";
package resonance.v1;

message SemanticEvent {
  // --- Core Identity ---
  bytes event_id = 1;           // 16-byte UUID
  uint64 timestamp_us = 2;      // Microseconds since Epoch
  string node_id = 3;           // Source Identifier

  // --- The Meaning ---
  string domain = 4;            // e.g., "vision", "audio", "vibration"
  uint32 embedding_dim = 5;     // e.g., 256
  repeated float delta_vector = 6; // The change in meaning (Δμ)
  
  // --- Confidence & Context ---
  float confidence = 7;         // 0.0 to 1.0
  float distance_from_prev = 8; // The magnitude of the shift
  map<string, string> tags = 9; // Lightweight metadata

  // --- Network Control ---
  uint32 ttl = 10;              // Time-to-Live (hops)
  repeated string trace = 11;   // Loop prevention

  // --- Security ---
  Provenance provenance = 12;
}

message Provenance {
  string model_hash = 1;        // SHA-256 of the model weights
  string hardware_id = 2;       // Hardware Attestation ID
  bytes signature = 3;          // ECDSA Signature of the event
}
```

---

## 5. Hardware Implementation Guide

### 5.1 Current Generation (Level 1 Reference)
For immediate implementation (v1.0), we target available commercial hardware:
* **Compute:** Raspberry Pi 4 / 5 or NVIDIA Jetson Nano.
* **Sensors:** Standard USB Cameras or I2S Microphones.
* **Power Target:** < 2W average (achieved via deep sleep between polling).

### 5.2 Future Target (The "Resonance Native" Chip)
The ultimate goal of the rAI Collective is the **Resonance ASIC**:
* **Architecture:** Neuromorphic / Event-based.
* **Process:** 1-bit logic, asynchronous (clock-less).
* **Power Target:** < 1mW active.
* **Integration:** The sensor *is* the processor.

---

## 6. Implementation Roadmap

1.  **Phase Alpha:** Python SDK implementing the Protobuf schema and Cosine logic.
2.  **Phase Beta:** "The Factory Demo" — 5 Raspberry Pis detecting anomalies in motor vibrations using the Reference Concept alignment.
3.  **Phase Gamma:** Porting the core logic to C++/Rust for microcontroller support (ESP32).

---

## Appendix A: The Canonical Reference Concepts (Subset)

These are the "anchors" used for alignment.

**Vision (000-049):**
* `REF_000`: Person / Human
* `REF_001`: Fire / Flame
* `REF_002`: Vehicle / Car
* `REF_003`: Animal / Dog
* `REF_004`: Plant / Tree
* `REF_005`: Water / Liquid
* ...

**Audio (050-074):**
* `REF_050`: Human Speech
* `REF_051`: Siren / Alarm
* `REF_052`: Glass Breaking
* `REF_053`: Impact / Thud
* ...

**Physics/IMU (075-099):**
* `REF_075`: Vibration (High Freq)
* `REF_076`: Freefall
* `REF_077`: Rotation
* ...

---
*End of Specification Level 1.0*
