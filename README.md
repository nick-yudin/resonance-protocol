# RESONANCE Protocol

**A semantic event protocol for distributed edge intelligence.**
*Triggered by meaning, not time.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status: Draft](https://img.shields.io/badge/Status-Draft-orange)](https://resonanceprotocol.org)

---

## What is Resonance?

Resonance is an open standard for **meaning-triggered computing**.

In traditional IoT and AI systems, devices stream data continuously (clock-driven) or poll sensors at fixed intervals. This creates massive noise, latency, and energy waste.

**Resonance flips the axiom:**
1.  **Silence is the default state.** A node transmits nothing until "meaning" changes.
2.  **Meaning is mathematical.** We use high-dimensional vectors (embeddings) to track state.
3.  **Events are semantic.** We transmit the *change in meaning* ($\Delta\mu$), not raw data.

> "The clock stops. The resonance begins."

---

## Repository Structure

This is a Monorepo containing the core specifications, documentation, and reference implementations.

* [`/docs`](./docs) — **The Single Source of Truth**. Specifications and Manifestos.
    * [`Manifesto`](./docs/00_intro/manifesto.md) — The philosophical foundation (Level 0).
    * [`Specification v1.0`](./docs/01_specs/v1.0_current/resonance_unified_spec.md) — The technical standard (Level 1).
* [`/reference_impl`](./reference_impl) — SDKs and Example Code (Coming Soon).
* [`/website`](./website) — Source code for [resonanceprotocol.org](https://resonanceprotocol.org).

---

## Quick Start

### For Engineers
Read the **[Level 1 Unified Specification](./docs/01_specs/v1.0_current/resonance_unified_spec.md)** to understand the wire protocol, 256-d embeddings, and the Reference Concept Alignment mechanism.

### For Visionaries
Read the **[Manifesto](./docs/00_intro/manifesto.md)** to understand why we are abandoning clock-based computing.

---

## Community & Governance

* **Website:** [https://resonanceprotocol.org](https://resonanceprotocol.org)
* **Twitter/X:** [@rAI_stack](https://twitter.com/rAI_stack)
* **Contact:** 1@resonanceprotocol.org
* **Author:** rAI Research Collective

**Governance Policy:**
All public artifacts in this repository are maintained in **English**. See [PROJECT_STRUCTURE.md](./PROJECT_STRUCTURE.md) for contribution guidelines.
