import json
import subprocess
import os
import pandas as pd
import glob
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# --- 설정 (Configuration) ---
BASE_JSON_PATH = "Tiles/test_tile/double_512_N_large.json"
OUTPUT_CSV = "../../../data/motiv/sensitivity_results.csv"

# 병렬 실행 개수
# (디버깅을 위해 안전하게 4 정도로 설정, 시스템이 받쳐주면 늘리세요)
MAX_WORKERS = 24  

# 분석할 Sequence Length
SEQ_LEN = 2048 

# 민감도 분석을 수행할 N, K 범위
N_RANGE = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096] 
K_RANGE = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# 분석할 모델 리스트
TARGET_MODELS = ["gemma_2_2b_quant", "llama2_quant"]

# 모델별 Config
MODEL_CONFIG = {
    "gemma_2_2b_quant": (1024, 8),
    "llama2_quant": (2048, 32),  # [수정 1] Key 이름을 TARGET_MODELS와 일치시킴
}

# 변경할 레이어 이름들
TARGET_LAYERS = [
    "q_projection", "k_projection", "v_projection", 
    "q_mul_k", "a_mul_v", 
    "w0_projection", "w1_projection", "w2_projection"
]

csv_lock = Lock()

def cleanup_stale_temp_files():
    stale_files = glob.glob("Tiles/test_tile/temp_config_*.json")
    if stale_files:
        print(f"Cleaning up {len(stale_files)} stale temporary files...")
        for f in stale_files:
            try:
                os.remove(f)
            except OSError:
                pass

def get_finished_tasks(csv_path):
    finished = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            required_cols = ['Model', 'Tile_N', 'Tile_K']
            if all(col in df.columns for col in required_cols):
                for _, row in df.iterrows():
                    finished.add((row['Model'], int(row['Tile_N']), int(row['Tile_K'])))
        except pd.errors.EmptyDataError:
            pass 
    return finished

def append_result_to_csv(result_dict):
    file_exists = os.path.isfile(OUTPUT_CSV)
    with csv_lock:
        with open(OUTPUT_CSV, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "Tile_N", "Tile_K", "Latency"])
            if not file_exists or os.path.getsize(OUTPUT_CSV) == 0:
                writer.writeheader()
            writer.writerow(result_dict)

def create_unique_json(base_path, n_val, k_val, model_name):
    temp_filename = f"Tiles/test_tile/temp_config_{model_name}_{n_val}_{k_val}.json"
    try:
        with open(base_path, 'r') as f:
            data = json.load(f)
        
        for layer in TARGET_LAYERS:
            if layer in data:
                data[layer]["l1_tile_N"] = n_val
                data[layer]["l1_tile_K"] = k_val
        
        with open(temp_filename, 'w') as f:
            json.dump(data, f, indent=4)
        return temp_filename
    except Exception as e:
        print(f"Error creating JSON {temp_filename}: {e}")
        return None

def run_single_simulation(args):
    model, n, k = args
    json_path = None
    
    # [수정 2] 모든 로직을 try 블록 안으로 이동하여 에러 발생 시 잡아냄
    try:
        # Config 확인
        if model not in MODEL_CONFIG:
            print(f"Error: Model config not found for {model}")
            return None

        dim, head = MODEL_CONFIG[model]
        
        # JSON 생성
        json_path = create_unique_json(BASE_JSON_PATH, n, k, model)
        if not json_path:
            return None

        cmd = [
#            "taskset", "-c", "6-47",
            "python", "simulate.py",
            str(SEQ_LEN),
            json_path,
            str(dim),
            str(head)
        ]
        
        # [수정 3] 환경 변수 설정 (중첩 병렬화 방지)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        
        # [수정 4] timeout 설정 (60초 동안 응답 없으면 강제 종료)
        # capture_output=True 대신 stdout, stderr 분리하여 메모리 버퍼링 이슈 방지 가능하나,
        # 편의상 유지하되 timeout을 둡니다.
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True, 
            env=env,
        )
        
        latency = None
        for line in result.stdout.splitlines():
            if "Latency" in line:
                parts = line.split(",")
                if len(parts) >= 2:
                    latency = float(parts[1].strip())
                    break
        
        if latency is not None:
            res_dict = {
                "Model": model,
                "Tile_N": n,
                "Tile_K": k,
                "Latency": latency
            }
            append_result_to_csv(res_dict)
            return (model, n, k, latency)
        else:
            # Latency 파싱 실패 시 로그 (디버깅용)
            # print(f"DEBUG OUTPUT ({model}): {result.stdout[:100]}...") 
            return None

    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {model} (N={n}, K={k}) took too long.")
        return None
    except subprocess.CalledProcessError as e:
        # simulate.py가 에러를 뱉고 죽은 경우
        # print(f"ERROR in subprocess: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected Error ({model}, {n}, {k}): {e}")
        return None
    finally:
        if json_path and os.path.exists(json_path):
            try:
                os.remove(json_path)
            except OSError:
                pass

def main():
    cleanup_stale_temp_files()
    finished_tasks = get_finished_tasks(OUTPUT_CSV)
    print(f"Found {len(finished_tasks)} already completed tasks in {OUTPUT_CSV}.")

    all_tasks = []
    for model in TARGET_MODELS:
        for n in N_RANGE:
            for k in K_RANGE:
                task = (model, n, k)
                if task in finished_tasks:
                    continue
                all_tasks.append(task)
    
    total_remaining = len(all_tasks)
    if total_remaining == 0:
        print("All tasks are already completed!")
        return

    print(f"Starting Resume Sweep... Remaining tasks: {total_remaining}")
    print(f"Using {MAX_WORKERS} workers.") # flush=True 추가하여 즉시 출력
    print("-" * 50)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_single_simulation, task): task for task in all_tasks}
        
        for future in as_completed(futures):
            # Exception handling in Main Loop
            try:
                res = future.result()
                completed_count += 1
                
                if res:
                    model, n, k, lat = res
                    print(f"[{completed_count}/{total_remaining}] SAVED: {model} (N={n}, K={k}) -> {lat:.4f}", flush=True)
                else:
                    model, n, k = futures[future]
                    print(f"[{completed_count}/{total_remaining}] FAILED: {model} (N={n}, K={k})", flush=True)
            except Exception as e:
                 # 여기까지 에러가 올라오면 정말 치명적인 것
                 print(f"CRITICAL WORKER ERROR: {e}")

    print(f"\nSweep completed. All results updated in {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
