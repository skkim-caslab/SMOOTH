# SMOOTH: Hardware-Assisted Fine-Grained On-Chip Memory Management for Efficient On-Device LLM Inference

This repository contains the official implementation of **SMOOTH**, a hardware-assisted on-chip memory management framework designed to maximize memory bandwidth utilization for on-device LLM inference. SMOOTH introduces a fine-grained block-based memory system and a hardware-driven early reclamation mechanism, significantly reducing Time-to-First-Token (TTFT) and Time-to-Last-Token (TTLT) on memory-constrained mobile SoCs.

This paper is submitted to **ISCA 2026**.

## 1. Environment Setup (Docker Recommended)

SMOOTH relies on the cycle-accurate simulator [LLMCompass](https://github.com/PrincetonUniversity/LLMCompass) for architectural simulation, [Yosys](https://github.com/YosysHQ/yosys) for hardware logic synthesis, and [OpenSTA](https://github.com/The-OpenROAD-Project/OpenSTA) for static timing analysis. 

For your convenience, the **[ASAP7](https://github.com/The-OpenROAD-Project/asap7)** predictive 7nm standard cell library is **already included in this repository** to ensure out-of-the-box hardware synthesis and overhead evaluation.

### 1.1. Docker Environment (For Artifact Evaluation)
To guarantee strict reproducibility of the legacy hardware synthesis environment and avoid any system-level library conflicts (e.g., `glibc`, `libreadline`), **we strongly recommend using our provided Docker setup.**

A `Dockerfile` is already included in the root directory of this repository. Run the following commands in your host machine's terminal to build the image and start the container. 

**Note: All subsequent steps (Sections 2-6) should be executed inside this Docker container.**
<<<<<<< HEAD

```bash
# 1. Build the Docker image
docker build -t isca2026_smooth_ae .

# 2. Run the container and mount the SMOOTH repository
docker run -it --rm --name smooth_ae_env -v $(pwd):/workspace/SMOOTH isca2026_smooth_ae
```

## 2. Data Generation

Generate the baseline and SMOOTH policy data required for the evaluations. This process simulates the execution to gather performance metrics.

```bash
cd $SMOOTH_HOME/src/policies
bash run_all_policies.sh
```

## 3. Reproducing Figures (Latency)

Run the following scripts to plot the TTFT and TTLT latency evaluation figures. The generated plots will be saved in their respective directories.

**Figure 14: Normalized Time-to-First-Token (TTFT)**
```bash
cd $SMOOTH_HOME/src/ae/figure14
python3 plot_ttft.py ../../../data/seq_1/8MB
# Output generated: src/ae/figure14/TTFT_8MB.eps
```

**Figure 16: End-to-End Latency (TTLT)**
```bash
cd $SMOOTH_HOME/src/ae/figure16
python3 plot_latency.py ../../../data/seq_32K/8MB
# Output generated: src/ae/figure16/latency_8MB.png
```

## 4. Hardware Module Synthesis (Verilog)

Use Yosys to synthesize the 5 key hardware modules of SMOOTH and extract microarchitectural overheads. Running the script will generate the synthesis output files (`address_check.out`, `alloc.out`, `bt_lookup.out`, `find_zero.out`, `free.out`) for each respective module.

```bash
cd $SMOOTH_HOME/src/verilog/
bash run_all.sh
```

## 5. Reproducing Figure (Energy)

Plot the energy consumption evaluation across different inference strategies.

**Figure 20: Energy Consumption for N-th Token Generation**
```bash
cd $SMOOTH_HOME/src/ae/figure20
python3 plot_energy.py
# Output generated: src/ae/figure20/energy_8MB.eps
```

## 6. Reproducing Tables (Hardware Overhead)

Extract the Area, Power, and Latency overhead results synthesized during Step 4 to reproduce the tables from the paper.

**Table 1: Area Overhead Estimates**
```bash
cd $SMOOTH_HOME/src/ae/table1
python3 get_area.py
```

**Table 2: Latency and Power Consumption**
```bash
cd $SMOOTH_HOME/src/ae/table2
python3 get_power.py
```
