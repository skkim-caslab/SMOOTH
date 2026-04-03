import os
import re

# =============================================================================
# Configuration & Constants
# =============================================================================
# Directory containing Verilog .out files
verilog_dir = '../../../data/energy/verilog'

# Hardcoded NPU Area (um^2) based on the baseline
NPU_AREA = 13730000.0

# 7nm process SRAM 1-bit area constant 
# (Reverse-calculated from 1,811,939 um^2 / 8MB: approx 0.027 um^2)
BITCELL_AREA_7NM = 0.027

# 1. Calculate SRAM Base Area (8MB)
SRAM_SIZE_BYTES = 8 * 1024 * 1024
SRAM_BASE_BITS = SRAM_SIZE_BYTES * 8
SRAM_BASE_AREA = SRAM_BASE_BITS * BITCELL_AREA_7NM

# 2. Calculate Memory Overhead Area
BLOCK_SIZE_BYTES = 1024
NUM_BLOCKS = SRAM_SIZE_BYTES / BLOCK_SIZE_BYTES

# Overhead breakdown:
# - Virtual address space is 2x the physical memory space.
# - Virtual table entries: p_blk (13b) + cont (13b) + use_cnt (3b) = 29 bits
# - Physical table entries: valid flag = 1 bit
OVERHEAD_BITS = ((13 + 13 + 3) * NUM_BLOCKS * 2) + (1 * NUM_BLOCKS)

# Calculate final memory overhead area in um^2
MEMORY_SRAM_OVERHEAD = OVERHEAD_BITS * BITCELL_AREA_7NM

# List of hardware modules to parse
hw_modules = ['find_zero', 'alloc', 'address_check', 'bt_lookup', 'free']

# =============================================================================
# Data Parsing Function
# =============================================================================
def parse_area(filepath):
    """Parses the Area value from a Verilog .out file."""
    if not os.path.exists(filepath):
        return 0.0
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expression matching "Chip area for module '\module_name': <value>"
    area_match = re.search(r"Chip area for module\s+'[^']+':\s+([\d\.]+)", content)
    if area_match:
        return float(area_match.group(1))
    return 0.0

# =============================================================================
# Compute & Calculate Ratios
# =============================================================================
# Sum up the area for each hardware module (Compute Area Overhead)
compute_area_overhead = 0.0
for mod in hw_modules:
    out_path = os.path.join(verilog_dir, f"{mod}.out")
    compute_area_overhead += parse_area(out_path)

# Calculate ratios relative to the NPU Area (Ratio = Module / NPU_AREA * 100)
sram_base_ratio = (SRAM_BASE_AREA / NPU_AREA) * 100
compute_ratio = (compute_area_overhead / NPU_AREA) * 100
memory_ratio = (MEMORY_SRAM_OVERHEAD / NPU_AREA) * 100

# =============================================================================
# Print Table
# =============================================================================
print("\n" + "="*80)
print(f"{'Area overhead of proposed modules.':^80}")
print("="*80)
print(f"{'Metric':<15} | {'NPU':>12} | {'SRAM':>12} | {'Compute':>15} | {'Memory (SRAM)':>15}")
print("-" * 80)
print(f"{'Area [um^2]':<15} | {NPU_AREA:>12,.0f} | {SRAM_BASE_AREA:>12,.0f} | {compute_area_overhead:>15.0f} | {MEMORY_SRAM_OVERHEAD:>15,.0f}")
print(f"{'Ratio [%]':<15} | {'--':>12} | {sram_base_ratio:>12.1f} | {compute_ratio:>15.4f} | {memory_ratio:>15.3f}")
print("="*80 + "\n")
