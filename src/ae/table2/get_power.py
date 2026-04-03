import os
import re

# =============================================================================
# Configuration
# =============================================================================
verilog_dir = '../../../data/energy/verilog'

# Mapping the actual file name to parse and the short name to display in the table
hw_modules = {
    'find_zero': 'find_zero',
    'alloc': 'alloc',
    'address_check': 'addr_check', # Shorten name to fit table format
    'bt_lookup': 'bt_lookup',
    'free': 'free'
}

# =============================================================================
# Data Parsing
# =============================================================================
def parse_time_power(filepath):
    """Parsing time(ps) and total_power from Verilog .out file"""
    stats = {'time_ps': 0.0, 'power': 0.0}
    if not os.path.exists(filepath):
        return stats
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Time parsing (extract data arrival time value. Assuming EDA tool default unit is ps)
    time_matches = re.findall(r'(\d+\.\d+)\s+data arrival time', content)
    if time_matches:
        stats['time_ps'] = float(time_matches[0])

    # 2. Power parsing (extract Total Power from Total rows)
    power_match = re.search(r'Total\s+[\d\.eE\+\-]+\s+[\d\.eE\+\-]+\s+[\d\.eE\+\-]+\s+([\d\.eE\+\-]+)\s+[\d\.]+%?', content)
    if power_match:
        # If the power unit of the EDA tool is different from the display unit (pW), scaling is required here through multiplication.
        stats['power'] = float(power_match.group(1))

    return stats

# data collection
results = {}
for file_mod, print_mod in hw_modules.items():
    out_path = os.path.join(verilog_dir, f"{file_mod}.out")
    results[print_mod] = parse_time_power(out_path)

# =============================================================================
# Print Table (Horizontal Layout)
# =============================================================================
headers = ["Metric"] + list(hw_modules.values())

print("\n" + "="*85)
print(f"{'Latency and power consumption of each hardware module.':^85}")
print("="*85)

# Header output
header_str = f"{headers[0]:<12} |"
for h in headers[1:]:
    header_str += f" {h:>12} |"
print(header_str)
print("-" * 85)

# Output Time line
time_str = f"{'Time [ps]':<12} |"
for h in headers[1:]:
    val = results[h]['time_ps']
    time_str += f" {val:>12.1f} |"
print(time_str)

# Power line output (with exponential notation, e.g. 1.4e-01)
power_str = f"{'Power [pW]':<12} |"
for h in headers[1:]:
    val = results[h]['power']
    # Convert to x.xe-xx format to mimic LaTeX's x.x * 10^-x form
    formatted_pow = f"{val:>12.1e}" 
    power_str += f" {formatted_pow} |"
print(power_str)

print("="*85 + "\n")
