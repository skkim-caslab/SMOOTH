#!/bin/bash

# 실행할 정책 리스트 선언 (폴더명과 일치해야 함)
POLICY_LIST=(
    "smooth"
    "smooth-er"
    "capuchin"
    "compiler-ideal"
    "gemmini"
)

# 모델 리스트
MODEL_LIST=(
    gemma_2_2b_quant gemma_2_2b gpt_neo_quant gpt_neo
    llama2_quant llama2 tiny_llama_quant gpt_xl_quant
    gpt_xl gpt_2.7b_quant gpt_2.7b gpt3_quant gpt3
)

ROOFTOP_INTERVAL=512

for POLICY in "${POLICY_LIST[@]}"; do
    echo "========================================"
    echo " Starting execution for policy: $POLICY"
    echo "========================================"

    ### 1. 기존 동작: figure14. TTFT (Prompt length 64)
    echo "Running Prompt Phase (Length: 64) for $POLICY..."
    for FILE_NAME in "${MODEL_LIST[@]}"; do
        # 3번째 인자: Interval(프롬프트 단계라 1로 고정), 4번째 인자: 정책 이름
        python run_decode_master.py "$FILE_NAME" -64 1 "$POLICY" &
    done
#    wait
#    echo "Finished Original Prompt Phase for $POLICY"
    echo "--------------------------"

    ### 2. 추가 동작: Prompt length 1024
    echo "Running New Prompt Phase (Length: 1024) for $POLICY..."
    for FILE_NAME in "${MODEL_LIST[@]}"; do
        python run_decode_master.py "$FILE_NAME" -1024 1 "$POLICY" &
    done
    wait
    echo "Finished Prompt 1024 Phase for $POLICY"
    echo "--------------------------"

    ### 3. 추가 동작: Decode up to 32K (Rooftop interval 512)
    echo "Running Decode Phase (Up to 32768, Interval: $ROOFTOP_INTERVAL) for $POLICY..."
    for FILE_NAME in "${MODEL_LIST[@]}"; do
        python run_decode_master.py "$FILE_NAME" 32768 "$ROOFTOP_INTERVAL" "$POLICY" &
    done
    wait
    echo "Finished Decode 32K Phase for $POLICY"
    echo "--------------------------"

done

echo "All policies and phases executed successfully!"
