#!/bin/bash

# 결과가 저장될 CSV 파일명
OUTPUT_CSV="energy_out/latency_results.csv"

# CSV 헤더 작성
echo "Model,Output_Length,Block_Size,Latency" > $OUTPUT_CSV

# 조건 변수 배열 선언
models=("gpt_neo_quant" "gpt_neo" "llama2_quant" "llama2" "gpt3_quant" "gpt3")
output_lengths=("1K" "8K")
# 512부터 64씩 증가, 마지막 1024는 1K로 표기
blocks=("512" "576" "640" "704" "768" "832" "896" "960" "1K" "1088" "1152" "1216" "1280" "1344" "1408" "1472" "1536" "1600")

for out_len in "${output_lengths[@]}"; do
    for block in "${blocks[@]}"; do
        for model in "${models[@]}"; do
            
            # 경로 조합 (예: energy_out/block1K_output8K/llama2_quant.out)
            filepath="energy_out/block${block}_output${out_len}/${model}.out"
            
            # 파일이 존재하는지 확인
            if [ -f "$filepath" ]; then
                # grep으로 Latency 줄을 찾고, awk로 쉼표 뒤의 값을 추출한 뒤 공백 제거
                latency=$(grep "Latency" "$filepath" | awk -F',' '{print $2}' | tr -d ' ' | tr -d '\r')
                
                if [ -n "$latency" ]; then
                    echo "${model},${out_len},${block},${latency}" >> $OUTPUT_CSV
                else
                    echo "${model},${out_len},${block},Error_No_Value" >> $OUTPUT_CSV
                fi
            else
                # 파일이 없을 경우
                echo "${model},${out_len},${block},File_Not_Found" >> $OUTPUT_CSV
            fi
            
        done
    done
done

echo "추출 완료! $OUTPUT_CSV 파일을 확인해 주세요."
