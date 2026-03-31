import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import sys

################how to run###################
#python run_decode.py llama2_quant
############################################# 

FILE_NAME=sys.argv[1]
JSON_PATH = "./Tiles/test_tile/double_512_N_large.json"
MAX_WORKERS = 46
#SEQ_LEN_RANGE = [1] + list(range(512,32768,512))
#SEQ_LEN_RANGE = [1] 
#SEQ_LEN_RANGE = [1]
#SEQ_LEN_RANGE = [1] + list(range(512, 32768,512))
SEQ_LEN_RANGE = range(8190,8195)
#SEQ_LEN_RANGE = [2048, 4096, 8192, 16*1024, 32*1024]

OUTPUT_PATH = "final/"+FILE_NAME+".out"
CHECKPOINT_INTERVAL = 12

# dim/head 매핑
MODEL_CONFIG = {
    "gemma_2_2b_quant": (1024, 8),
    "gemma_2_2b": (2048, 8),
    "gpt_neo_quant": (1024, 16),
    "gpt_neo": (2048, 16),
    "llama2_quant": (2048, 32),
    "llama2": (4096, 32),
    "tiny_llama_quant": (1024, 32),
    "gpt_xl_quant": (1536, 24),
    "gpt_xl": (3072, 24),
    "gpt_2.7b_quant": (1280, 32),
    "gpt_2.7b": (2560, 32),
    "gpt3_quant": (2560, 40),
    "gpt3": (5120, 40),
#    "bloom": (4096, 32),
}

def run_simulation(seq_len, init_flag=False):
    global JSON_PATH
    
    try:
        dim, head = MODEL_CONFIG[FILE_NAME]
        cmd = ["python", "simulate.py", str(seq_len), JSON_PATH]
        if seq_len == 0:
            JSON_PATH = "./Tiles/test_tile/double_512_MN_large.json"
            cmd = ["python", "simulate.py", str(seq_len), JSON_PATH, str(dim), str(head), "--init"]
        else:
            cmd = ["python", "simulate.py", str(seq_len), JSON_PATH, str(dim), str(head)]
        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#        print("END RUN... PROCESS RESULT")

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

        return True, (seq_len, latency, sa_cycle, ve_cycle, total_linear_cycle, total_non_linear_cycle)

    except subprocess.CalledProcessError:
        print(f"Error at seq_len {seq_len}{' with --init' if init_flag else ''}")
        return False, None

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

def save_results(result_dict, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        # Write seq_len=0 first if it exists
        if 0 in result_dict:
            latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle = result_dict[0]
            f.write(f"0,{latency:.6f},{sa_cycle},{ve_cycle},{linear_cycle},{non_linear_cycle}\n")
        # Write the rest in sorted order
        for seq_len in sorted([k for k in result_dict.keys() if k != 0]):
            latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle = result_dict[seq_len]
            f.write(f"{seq_len},{latency:.6f},{sa_cycle},{ve_cycle},{linear_cycle},{non_linear_cycle}\n")

def main():
    # Load existing results
    result_dict = load_existing_results(OUTPUT_PATH)

    # Only run for missing seq_lens (excluding 0)
    remaining_seq_lens = [s for s in SEQ_LEN_RANGE if s not in result_dict]
    total_remaining = len(remaining_seq_lens)
    processed_count = 0
    checkpoint_buffer = {}  # Buffer to store results until checkpoint

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_simulation, seq_len): seq_len for seq_len in remaining_seq_lens}
        for future in as_completed(futures):
            success, data = future.result()
            if not success:
                continue              # 저장하지 않고 건너뜀

            seq_len, latency, sa, ve, lin, nlin = data
            checkpoint_buffer[seq_len] = (latency, sa, ve, lin, nlin)
            processed_count += 1

            if (processed_count % CHECKPOINT_INTERVAL == 0 or
                    processed_count == total_remaining):
                result_dict.update(checkpoint_buffer)
                save_results(result_dict, OUTPUT_PATH)
                print(f"Checkpoint: saved {processed_count} / "
                      f"{total_remaining} → {OUTPUT_PATH}")
                checkpoint_buffer.clear()

    # Final save if any results remain in buffer
    if checkpoint_buffer:
        result_dict.update(checkpoint_buffer)
        save_results(result_dict, OUTPUT_PATH)
        print(f"Final save: Saved {processed_count} results to {OUTPUT_PATH} ({processed_count}/{total_remaining})")

if __name__ == "__main__":
    main()
