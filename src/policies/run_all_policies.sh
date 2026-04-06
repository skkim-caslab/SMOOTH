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

# --- Ensure required directories exist for each policy ---
echo "Validating directory structure..."
for POLICY in "${POLICY_LIST[@]}"; do
    # Check if the policy base directory exists
    if [ -d "$POLICY" ]; then
        # Create SRAM directory if it doesn't exist
        if [ ! -d "$POLICY/SRAM" ]; then
            echo "Creating missing directory: $POLICY/SRAM"
            mkdir -p "$POLICY/SRAM"
        fi

        # Create systolic_array_model/temp directory if it doesn't exist
        # mkdir -p handles nested directory creation automatically
        if [ ! -d "$POLICY/systolic_array_model/temp" ]; then
            echo "Creating missing directory: $POLICY/systolic_array_model/temp"
            mkdir -p "$POLICY/systolic_array_model/temp"
        fi
    else
        echo "Warning: Policy directory '$POLICY' not found. Skipping validation."
    fi
done
echo "Directory validation complete."
# ------------------------------------------------------------------

(
for POLICY in "${POLICY_LIST[@]}"; do
# 1. get data for figure14 & figure16
    for FILE_NAME in "${MODEL_LIST[@]}"; do
        echo "python run_decode_master.py $FILE_NAME -64 1 $POLICY"
        echo "python run_decode_master.py $FILE_NAME -512 1 $POLICY"
        echo "python run_decode_master.py $FILE_NAME 32768 $ROOFTOP_INTERVAL $POLICY"
    done
done

# 2. get data for figure20
for OUT_LEN in "${OUTPUT_LENS[@]}"; do
    for BLK_SIZE in "${BLOCK_SIZES[@]}"; do
        echo "python run_decode_master.py llama2_quant 0 $BLK_SIZE smooth-er $OUT_LEN"
    done
done

) | xargs -I {} -P "$NUM_CORES" bash -c "{}"

echo "All policies and phases executed successfully!"
