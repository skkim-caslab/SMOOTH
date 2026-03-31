import pandas as pd
import sys

# Argument: log file name
if len(sys.argv) != 2:
    print("Usage: python script.py <log_file_name>")
    sys.exit(1)

log_file_name = sys.argv[1]

# Output CSV file name
csv_file_name = log_file_name.replace('.log', '.csv')

# Initialize lists for data
cycle = []
bw = []
occ = []
sa = []

# Read log file and parse lines
with open(log_file_name, 'r') as log_file:
    for line in log_file:
        if "total cycle" in line:
            try:
                value = float(line.split(":")[-1].strip())
                cycle.append(value)
            except ValueError:
                continue
        elif "memory bw util[%]" in line:
            try:
                value = float(line.split(":")[-1].strip())
                bw.append(value)
            except ValueError:
                continue
        elif "sram occupancy[%]" in line:
            try:
                value = float(line.split(":")[-1].strip())
                occ.append(value)
            except ValueError:
                continue
        elif "sa util[%]" in line:
            try:
                value = float(line.split(":")[-1].strip())
                sa.append(value)
            except ValueError:
                continue

# Ensure all lists have the same length
max_len = max(len(cycle), len(bw), len(occ), len(sa))
cycle.extend([None] * (max_len - len(cycle)))
bw.extend([None] * (max_len - len(bw)))
occ.extend([None] * (max_len - len(occ)))
sa.extend([None] * (max_len - len(sa)))

# Create DataFrame
data = {
    "Cycle": cycle,
    "Memory BW Util [%]": bw,
    "SRAM Occupancy [%]": occ,
    "SA Util [%]": sa
}
df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv(csv_file_name, index=False)
print(f"Data saved to {csv_file_name}")

