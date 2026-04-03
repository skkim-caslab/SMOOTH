#!/bin/bash

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
# 1. get data for figure14 & figure16
    for FILE_NAME in "${MODEL_LIST[@]}"; do
#        echo "python run_decode_master.py $FILE_NAME -64 1 $POLICY"
        echo "python run_decode_master.py $FILE_NAME -1024 1 $POLICY"
#        echo "python run_decode_master.py $FILE_NAME 32768 $ROOFTOP_INTERVAL $POLICY"
    done
# 2. get data for figure20
done
for OUT_LEN in "${OUTPUT_LENS[@]}"; do
    for BLK_SIZE in "${BLOCK_SIZES[@]}"; do
        echo "python run_decode_master.py llama2_quant 0 $BLK_SIZE smooth-er $OUT_LEN"
    done
done

) | xargs -I {} -P "$NUM_CORES" bash -c "{}"

echo "All policies and phases executed successfully!"
