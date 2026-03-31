import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

JSON_PATH = "./Tiles/test_tile/double_512_N_large.json"
MAX_WORKERS = 20
SEQ_LEN_RANGE = range(0, 1)
OUTPUT_PATH = "final/double_6B_w8a8.out"

def run_simulation(seq_len, init_flag=False):
    global JSON_PATH
    try:
        cmd = ["python", "simulate.py", str(seq_len), JSON_PATH]
        if init_flag:
            JSON_PATH = "./Tiles/test_tile/double_512_MN_large.json"
            cmd = ["python", "simulate.py", str(seq_len), JSON_PATH]
            cmd.append("--init")
        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        latency = 0.0
        sa_cycle = 0
        ve_cycle = 0
        total_non_linear_cycle = 1  # Prevent division by zero
        total_linear_cycle = 1      # Prevent division by zero

        for line in result.stdout.splitlines():
            if "Latency" in line:
                try:
                    latency = float(line.split(",")[1].strip())
                except (IndexError, ValueError):
                    pass
            elif "SA cycle" in line:
                try:
                    sa_cycle += int(line.strip().split(":")[-1])
                except:
                    pass
            elif "VE cycle" in line:
                try:
                    ve_cycle += int(line.strip().split(":")[-1])
                except:
                    pass
            elif "NON-linear cycles" in line:
                try:
                    total_non_linear_cycle = int(float(line.strip().split(",")[-1]))
                except:
                    pass
            elif "Linear cycles" in line:
                try:
                    total_linear_cycle = int(float(line.strip().split(",")[-1]))
                except:
                    pass

        return seq_len, latency, sa_cycle, ve_cycle, total_linear_cycle, total_non_linear_cycle

    except subprocess.CalledProcessError:
        print(f"Error at seq_len {seq_len}{' with --init' if init_flag else ''}")
    return seq_len, 0.0, 0, 0, 1, 1

def load_existing_results(path):
    result_dict = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if "," in line:
                    parts = line.strip().split(",")
                    if len(parts) >= 6:
                        try:
                            seq_len = int(parts[0])
                            latency = float(parts[1])
                            sa_cycle = int(parts[2])
                            ve_cycle = int(parts[3])
                            linear_cycle = int(parts[4])
                            non_linear_cycle = int(parts[5])
                            result_dict[seq_len] = (latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle)
                        except ValueError:
                            continue
    return result_dict

def main():
    # Load existing results
    result_dict = load_existing_results(OUTPUT_PATH)

    # Run for seq_len=0 with --init if not already present
    if 0 not in result_dict:
        seq_len, latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle = run_simulation(0, init_flag=True)
        result_dict[0] = (latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle)

    # Only run for missing seq_lens (excluding 0)
    remaining_seq_lens = [s for s in SEQ_LEN_RANGE if s not in result_dict]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_simulation, seq_len): seq_len for seq_len in remaining_seq_lens}
        for future in as_completed(futures):
            seq_len, latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle = future.result()
            result_dict[seq_len] = (latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    # Save all results, with seq_len=0 at the top
    with open(OUTPUT_PATH, "w") as f:
        # Write seq_len=0 first if it exists
        if 0 in result_dict:
            latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle = result_dict[0]
            f.write(f"0,{latency:.3f},{sa_cycle},{ve_cycle},{linear_cycle},{non_linear_cycle}\n")
        # Write the rest in sorted order
        for seq_len in sorted([k for k in result_dict.keys() if k != 0]):
            latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle = result_dict[seq_len]
            f.write(f"{seq_len},{latency:.3f},{sa_cycle},{ve_cycle},{linear_cycle},{non_linear_cycle}\n")

    print(f"Saved full results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
