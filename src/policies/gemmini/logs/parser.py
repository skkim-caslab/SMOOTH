import pandas as pd
import os
import re

# 파일 경로 설정
file_path = "nopreload.out"

# 결과를 저장할 리스트
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
csv_filename = file_path[:-4] +"_parse.csv"

df.to_csv( csv_filename, index=False)

print(df)
remote_user = "seulki"   # 원격 사용자 이름
remote_ip = "10.244.147.174"       # 원격 서버 IP 주소
remote_path = "~/logs/" # 원격 서버에서 저장할 경로
password = "intern19!"  # 원격 서버 비밀번호

scp_command = f"sshpass -p '{password}' scp {csv_filename} {remote_user}@{remote_ip}:{remote_path}"

exit_code = os.system(scp_command)
