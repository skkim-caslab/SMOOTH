import json
import pandas as pd
import glob

# JSON 파일들이 저장된 디렉토리 경로 (예: "./json_data/*.json")
json_files = glob.glob("./tile_size/*.json")

# 모든 데이터를 저장할 리스트
data_list = []

# JSON 파일 읽어서 데이터 리스트에 추가
for file in json_files:
    with open(file, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        for op, params in json_data.items():
            entry = {"operation": op}
            for key, value in params.items():
                if key != "next_ops_name":  # 'next_ops_name'은 제외
                    entry[key] = value
            data_list.append(entry)

# DataFrame 생성
df = pd.DataFrame(data_list)

# 같은 operation의 같은 열에 대해 중복 값을 콤마로 결합
df_grouped = df.groupby("operation").agg(lambda x: ', '.join(map(str, sorted(set(x)))))

# CSV 파일로 저장 (원하는 파일명으로 변경 가능)
df_grouped.to_csv("output.csv", encoding="utf-8")

# 표 출력
print(df_grouped)
