import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

FILE_NAME = sys.argv[1]
TARGET_LEN = int(sys.argv[2])
ROOFTOP_INTERVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 1

# 4번째 인자로 정책(Policy) 이름을 받음
POLICY_NAME = sys.argv[4] if len(sys.argv) > 4 else "smooth"

if TARGET_LEN in [1, -64]:
    DIR_NAME = "seq_1"
elif TARGET_LEN in [32768, -1024]:
    DIR_NAME = "seq_32K"
else:
    DIR_NAME = "seq_default"

# 받은 POLICY_NAME을 경로에 적용
OUTPUT_PATH = f"../../data/{DIR_NAME}/8MB/{POLICY_NAME}/{FILE_NAME}.out"
if TARGET_LEN < 0:
    SEQ_LEN_RANGE = [0]
    PROMPT_LEN = -TARGET_LEN
else:
    # interval(예: 512) 단위로 range 생성 -> 1, 513, 1025 ...
    SEQ_LEN_RANGE = range(1, TARGET_LEN + 1, ROOFTOP_INTERVAL)

DEFAULT_JSON_PATH = "./Tiles/test_tile/double_512_N_large.json"
MAX_WORKERS = 20

# 동적 디렉토리 경로 적용
CHECKPOINT_INTERVAL = 128

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
}

def run_simulation(seq_len, init_flag=False):
    global DEFAULT_JSON_PATH
    
    try:
        dim, head = MODEL_CONFIG[FILE_NAME]
        if seq_len == 0:
            json_path = "./Tiles/test_tile/double_512_MN_large.json"
            prompt_len = PROMPT_LEN
            cmd = ["python", "simulate.py", str(prompt_len), json_path, str(dim), str(head), "--init"]
        else:
            json_path = DEFAULT_JSON_PATH
            cmd = ["python", "simulate.py", str(seq_len), json_path, str(dim), str(head)]

        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=POLICY_NAME)

        latency = 0.0
        sa_cycle = 0
        ve_cycle = 0
        total_non_linear_cycle = 1  
        total_linear_cycle = 1      

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
        if 0 in result_dict:
            latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle = result_dict[0]
            f.write(f"0,{latency:.6f},{sa_cycle},{ve_cycle},{linear_cycle},{non_linear_cycle}\n")
        for seq_len in sorted([k for k in result_dict.keys() if k != 0]):
            latency, sa_cycle, ve_cycle, linear_cycle, non_linear_cycle = result_dict[seq_len]
            f.write(f"{seq_len},{latency:.6f},{sa_cycle},{ve_cycle},{linear_cycle},{non_linear_cycle}\n")

def main():
    result_dict = load_existing_results(OUTPUT_PATH)

    remaining_seq_lens = [s for s in SEQ_LEN_RANGE if s not in result_dict]
    total_remaining = len(remaining_seq_lens)
    processed_count = 0
    checkpoint_buffer = {}  

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_simulation, seq_len): seq_len for seq_len in remaining_seq_lens}
        for future in as_completed(futures):
            success, data = future.result()
            if not success:
                continue              

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

    if checkpoint_buffer:
        result_dict.update(checkpoint_buffer)
        save_results(result_dict, OUTPUT_PATH)
        print(f"Final save: Saved {processed_count} results to {OUTPUT_PATH} ({processed_count}/{total_remaining})")

if __name__ == "__main__":
    main()

