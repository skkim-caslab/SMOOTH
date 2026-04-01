#!/bin/bash

MODEL_LIST=(
    gemma_2_2b_quant
    gemma_2_2b
    gpt_neo_quant
    gpt_neo
    llama2_quant
    llama2
    tiny_llama_quant
    gpt_xl_quant
    gpt_xl
    gpt_2.7b_quant
    gpt_2.7b
    gpt3_quant
    gpt3
)

### figure14. TTFT
for FILE_NAME in "${MODEL_LIST[@]}"; do
    echo "Running $FILE_NAME..."
    python run_decode.py "$FILE_NAME" -64 &
    echo "Finished $FILE_NAME"
    echo "--------------------------"
done

