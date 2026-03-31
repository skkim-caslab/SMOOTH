import pandas as pd
import os
import re
import sys

# 파일 경로 설정
file_path = sys.argv[1]
#print("??",file_path)

#csv_filename = "/home/sadmin/skkim/logs/out/" + file_path[12:-4] + "_parse.csv"
csv_filename = file_path[:-4] + "_parse.csv"
print(csv_filename)

if len(sys.argv) > 2 and sys.argv[2]=="re" and os.path.exists(csv_filename):
    df = pd.read_csv(csv_filename)
else:
    final_list = []
    # 초기화된 딕셔너리 템플릿
    def create_new_dict():
        return {
            "operation": "", 
            "va util": -1, 
            "total cycle": -1, 
            "compute cycle": -1, 
            "io cycle": -1, 
            "current cycle": -1, 
            "memory bw": -1, 
            "sram occupancy": -1, 
            "sa util": -1, 
        }

    new_dict = create_new_dict()
    operation = None

    # 파일 읽기
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip()
        if len(line) < 1:
            continue
        if 'skkim' in line:
            continue

        if line.startswith("simulating"):
            operation = line.split("simulating")[-1].strip()
            final_list.append(new_dict)
            new_dict = create_new_dict()
            new_dict['operation'] = operation

        else:
            if len(line.split(":")) < 2:
                continue

            key = line.split(":")[0].strip()
            value = line.split(":")[-1].strip()

            # 딕셔너리 키와 매칭되는 경우 값 저장
            for dict_key in new_dict.keys():
                if dict_key in key.lower():
                    if new_dict[dict_key] == -1: 
                        new_dict[dict_key] = value
                    else:
                        final_list.append(new_dict)
                        new_dict = create_new_dict()
                        new_dict['operation'] = operation
                        new_dict[dict_key] = value

    df = pd.DataFrame(final_list)
    df = df[df['operation'] != ""] 
    df.to_csv(csv_filename, index=False)




for col in df.columns:
    if col != 'operation':
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df = df.replace(-1, 0)
df = df.replace('-1', 0)
#df.to_csv(file_path, index=False)

op_order = ['layernorm(MHA)', 'q projection', 'k projection', 'v projection', 'q_mul_k', 'softmax', 'a_mul_v',
            'w0_projection', 'layernorm(FFN)', 'w1_projection', 'gelu', 'w2_projection']
if "fa" in file_path:
    op_order = ['layernorm(MHA)', 'q projection(flash attention)', 'k projection(flash attention)', 'v projection(flash attention)', 'flash attention',
                'w0_projection', 'layernorm(FFN)', 'w1_projection', 'gelu', 'w2_projection']

tot_cycle = []
fa_flag = -1 #qkt - 0, softmax - 1, sv - 2 

for op_name in op_order:
    tmp_df = df[df['operation'].isin([op_name])]
    total_cycle_max = int(max(tmp_df['total cycle']))
    tot_cycle.append(total_cycle_max)

if (len(op_order) == 10): #flash attention
    print(op_order[:4] + [""]*3 + op_order[4:])
    print(tot_cycle[:4] + [0]*3 + tot_cycle[4:])
    print("Sum:", sum(tot_cycle))
else:
    print(op_order[:7] + [""] + op_order[7:])
    print(tot_cycle[:7] + [0] + tot_cycle[7:])
    print("Sum:", sum(tot_cycle))


#print(df)
#remote_user = "seulki"   # 원격 사용자 이름
#remote_ip = "10.244.147.174"       # 원격 서버 IP 주소
#remote_path = "~/logs/csv/" # 원격 서버에서 저장할 경로
#password = "intern19!"  # 원격 서버 비밀번호

#scp_command = f"sshpass -p '{password}' scp {csv_filename} {remote_user}@{remote_ip}:{remote_path}"

#exit_code = os.system(scp_command)
