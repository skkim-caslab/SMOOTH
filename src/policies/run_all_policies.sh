#!/bin/bash

# 실행할 정책 및 모델 리스트
POLICY_LIST=("smooth" "smooth-er" "capuchin" "gemmini" "compiler-ideal")
MODEL_LIST=(
    gemma_2_2b_quant gemma_2_2b gpt_neo_quant gpt_neo
    llama2_quant llama2 tiny_llama_quant gpt_xl_quant
    gpt_xl gpt_2.7b_quant gpt_2.7b gpt3_quant gpt3
)

ROOFTOP_INTERVAL=512


# 모든 조합의 커맨드를 생성하여 xargs로 파이프(|) 전달
for POLICY in "${POLICY_LIST[@]}"; do
    for FILE_NAME in "${MODEL_LIST[@]}"; do
#        echo "python run_decode_master.py $FILE_NAME -64 1 $POLICY"
#        echo "python run_decode_master.py $FILE_NAME -1024 1 $POLICY"
        echo "python run_decode_master.py $FILE_NAME 32768 $ROOFTOP_INTERVAL $POLICY"
    done
done | xargs -I {} -P 15 bash -c "{}"

echo "All policies and phases executed successfully!"
