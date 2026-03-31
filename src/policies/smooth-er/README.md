# SMOOTH: SMOothing I/O Traffic with Hardware support

This repository provides the simulation artifacts and architectural models for the paper:

> **SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference**  

---

## Overview

Running Large Language Models (LLMs) on mobile and edge devices is fundamentally constrained by:

- Limited on-chip SRAM capacity (2–8 MB)
- Low off-chip memory bandwidth

Transformer-based models exhibit **bursty memory behavior** during autoregressive decoding:

- **Compute-bound phases** (e.g., non-linear operations)
- **Memory-bound phases** (e.g., linear layers)

Existing approaches rely on **static compiler-driven memory allocation**, which leads to:

- Memory fragmentation
- Inefficient utilization of transient bandwidth slack

---

## SMOOTH

**SMOOTH** is a hardware-assisted on-chip memory management framework that dynamically optimizes scratchpad usage at runtime.

### Key Contributions

#### 1. Fine-Grained Block-Based Allocation
- Decouples logical tensors from physical SRAM layout
- Eliminates external fragmentation
- Enables flexible and compact data placement

#### 2. Hardware-Driven Early Reclamation
- Tracks tensor usage via hardware signals:
  - `use_cnt`
  - `end_cmd`
- Immediately releases unused memory blocks
- Enables **proactive preloading** of future tensors

---

## Repository Structure

This repository contains multiple branches to compare SMOOTH against different memory management strategies:

| Branch | Description |
|--------|-------------|
| `main` | **SMOOTH-ER (Full version)** with block allocation + early reclamation |
| `smooth` | **SMOOTH-Base**, block allocation only |
| `compiler-ideal` | Ideal compiler-based allocation (best-fit + full preloading) |
| `gemmini` | Pipelined allocation with tile overlap |
| `capuchin` | Cache-like hardware-managed strategy (64B granularity) |

---

## Prerequisites

This project is built on top of **LLMCompass** (an extension of ScaleSim).

### Requirements

- Python 3.8+
- Python libraries:
  - `numpy`
  - `pandas`
- (Optional) **Yosys**
  - Required only for hardware synthesis (area/power analysis)

---

## Getting Started

### 1. Clone Repository


```bash
git clone https://github.com/Seu1ki/SMOOTH.git
cd SMOOTH
````

### 2. Switch Between Baselines

Each branch corresponds to a different memory management strategy:

```bash
git checkout compiler-ideal
```

### 3. Run Simulation

```bash
python run_simulator.py \
  --config configs/mobile_npu.cfg \
  --network workloads/llama2_7b.csv
```

> Note: Adjust the script name depending on your LLMCompass setup.

---

## Hardware Configuration

The default simulation models a **mobile-class NPU**:

* **Core Frequency**: 940 MHz
* **Compute Units**:

  * 1 Core
  * 32×32 Matrix Engine
  * 32-lane Vector Engine
* **SRAM Capacity**:

  * 2 MB / 8 MB / 32 MB
* **DRAM Bandwidth (LPDDR5)**:

  * 16 / 32 / 64 / 128 GB/s

---

## Notes

* This repository focuses on **on-chip memory management for LLM inference**
* Designed for **resource-constrained edge devices**
* Fully compatible with **LLMCompass simulation flow**

---

## Contact

For questions or collaboration inquiries, please open an issue.


