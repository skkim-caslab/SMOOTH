#!/bin/bash

# 실행할 정책 및 모델 리스트
POLICY_LIST=("smooth" "smooth-er" "capuchin" "gemmini" "compiler-ideal")
MODEL_LIST=(
    gemma_2_2b_quant gemma_2_2b gpt_neo_quant gpt_neo
    llama2_quant llama2 tiny_llama_quant gpt_xl_quant
    gpt_xl gpt_2.7b_quant gpt_2.7b gpt3_quant gpt3
)

ROOFTOP_INTERVAL=512
OUTPUT_LENS=(1025 8193 32769)
BLOCK_SIZES=(256 512 1024 2048 4096)
NUM_CORES=15

(
for POLICY in "${POLICY_LIST[@]}"; do
# 1. figure 14, 16
    for FILE_NAME in "${MODEL_LIST[@]}"; do
#        echo "python run_decode_master.py $FILE_NAME -64 1 $POLICY"
#        echo "python run_decode_master.py $FILE_NAME -1024 1 $POLICY"
        echo "python run_decode_master.py $FILE_NAME 32768 $ROOFTOP_INTERVAL $POLICY"
    done
# 2. figure 20
    for FILE_NAME in "${MODEL_LIST[@]}"; do
        for OUT_LEN in "${OUTPUT_LENS[@]}"; do
            for BLK_SIZE in "${BLOCK_SIZES[@]}"; do
                # 인자: 모델명, 0(에너지모드), 블록사이즈, 정책이름, 출력길이
                echo "python run_decode_master.py $FILE_NAME 0 $BLK_SIZE smooth-er $OUT_LEN"
            done
        done
    done
done
) | xargs -I {} -P "$NUM_CORES" bash -c "{}"

echo "All policies and phases executed successfully!"
