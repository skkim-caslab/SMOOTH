import itertools
from math import floor, log2
import json
import os

# 가능한 값 정의
#m_values = [1, 8, 16]
seq_len = 32*1024
model_dim = 4096 #model dimension
head_num = 32
m_values = [head_num]

#seq_len = 64*1024
#N_list = [16, 64, 2048]
N_list = [128, 256]

SRAM_size = 128*1024
double_buffer = 1
max_sram_allowed = SRAM_size // double_buffer

def get_N_list(max_num):
    final_list = []
    for idx,val in enumerate(N_list):
        if val <= max_num:
            final_list.append(val)
    if len(final_list) == 0:
        final_list = [1]
    return final_list
n_values = {
    "q_projection": get_N_list(model_dim),
    "k_projection": get_N_list(model_dim),
    "v_projection": get_N_list(model_dim),
    "q_mul_k": get_N_list(seq_len), #8192
    "softmax": [1024,2048], #8192
    "a_mul_v": get_N_list(model_dim//head_num), #128 = 2048/16
    "w0_projection": get_N_list(model_dim), #2048
    "w1_projection": get_N_list(4*model_dim), #8192
    "w2_projection": get_N_list(model_dim)
}
k_values = {
    "q_projection": get_N_list(model_dim),
    "k_projection": get_N_list(model_dim),
    "v_projection": get_N_list(model_dim),
    "q_mul_k": get_N_list(model_dim),
    "softmax": [None],
    "a_mul_v": get_N_list(head_num * seq_len),
    "w0_projection": get_N_list(model_dim),
    "w1_projection": get_N_list(model_dim),
    "w2_projection": get_N_list(4*model_dim)
}


# 연산 흐름 정의
ops_sequence = [
    "q_projection", "k_projection", "v_projection", "q_mul_k", "softmax", 
    "a_mul_v", "w0_projection", "w1_projection", "w2_projection"
]

# Softmax 이후 연산 흐름 지정
custom_next_ops = {
    "q_mul_k": "a_mul_v",
    "softmax": None  # softmax에는 next_ops_name이 없음
}
# 총 생성될 파일 수 계산
total_configs = 1
for op in ops_sequence:
    total_configs *= len(n_values[op]) * (1 if op == 'softmax' else len(k_values[op]))
#total_configs *= len(n_values[op])

print(f"총 {total_configs}개의 JSON 파일이 생성될 예정입니다.")

# 사용자 확인 후 진행
user_input = input("계속 진행할까요? (y/n): ").strip().lower()
if user_input != 'y':
    print("파일 생성이 취소되었습니다.")
    exit()
#os.system('rm -rf ./double_tile_size/config*')
# 조합을 생성하여 각 파일로 저장
config_index = 0
for m in m_values:
    n_combinations = itertools.product(*[n_values[op] for op in ops_sequence]) 
    k_combinations = itertools.product(*[k_values[op] for op in ops_sequence])
    
    # k_values가 None이 아닌 경우만 포함
    k_ops = [op for op in ops_sequence if k_values[op] is not None]
#    k_combinations = itertools.product(*[k_values[op] for op in k_ops])
    #print("N",len(list(n_combinations)))
    #print("K",len(list(k_combinations)))

    for n_comb, k_comb in itertools.product(n_combinations, k_combinations):
        json_data = {}
        sram_valid = True
        for idx, op in enumerate(ops_sequence):
            print(idx, op)
            if op != 'softmax': tmp_m = 1
            else: tmp_m = m

            l1_tile_M = tmp_m
            l1_tile_N = n_comb[idx]

            json_data[op] = {
                "l1_tile_M": l1_tile_M,
                "l1_tile_N": l1_tile_N,
            }

            if k_values[op][0] is not None:
                l1_tile_k = k_comb[idx]
                l1_tile_k = min(l1_tile_k, k_values[op][-1])
                l1_tile_k = floor(log2(l1_tile_k))
                l1_tile_k = 2**l1_tile_k
                json_data[op]["l1_tile_K"] = l1_tile_k

                memory_footprint = (l1_tile_M * l1_tile_N) + (l1_tile_N * l1_tile_k) + (l1_tile_M * l1_tile_k)
                if memory_footprint > max_sram_allowed:
                    sram_valid = False
                    break
            
            if op in custom_next_ops:
                if custom_next_ops[op] is not None:
                    json_data[op]["next_ops_name"] = custom_next_ops[op]
            else:
                json_data[op]["next_ops_name"] = "end" if op == ops_sequence[-1] else ops_sequence[idx + 1]
            
        
        # 개별 JSON 파일 생성
        if sram_valid:
            filename = f"./sram_xla_128_256_sram128_seq32K_m6.7B/config_{config_index}.json"
            os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists
            with open(filename, "w") as f:
                json.dump(json_data, f, indent=4)
            print(f"JSON 파일이 생성되었습니다: {filename}")
            config_index += 1
