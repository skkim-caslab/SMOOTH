#!/bin/bash

for size in 16384 32768; do
    for model in gpt_neo llama2; do
        if [ "$model" = "gpt_neo" ]; then
            seq_len=1024
            datas=16
        else
            seq_len=2048
            datas=32
        fi

        echo "Running simulate.py with size=$size, model=$model"
        python simulate.py $size ./Tiles/test_tile/double_512_N_large.json $seq_len $datas > sram_status/double_${size}_${model}.out &
    done
done

