#!/bin/bash

mkdir -p frag_test  # 결과 디렉토리 없으면 생성

declare -a FILES=(
    "all"
    "base"
    "ffn"
    "flash_atten"
    "qkv_atten"
    "qkv_proj"
)

#declare -a FITS=("first_fit" "best_fit" "worst_fit")
declare -a FITS=("sequential")
#declare -a SUFFIX=("ff" "bf" "wf")
declare -a SUFFIX=("seq")

for file in "${FILES[@]}"; do
    for i in "${!FITS[@]}"; do
        algo="${FITS[$i]}"
        suffix="${SUFFIX[$i]}"
        #input="lifetime/${file}_tile_lifetime_x1.json"
        input="lifetime/${file}_tile_lifetime_8K_x1.json"
        #input="lifetime/${file}_tile_lifetime_x12.json"
        #output="frag_test/${file}_${suffix}.out"
        #output="frag_test/${file}_${suffix}_x12.out"
        output="frag_test/${file}_8K_${suffix}.out"
        echo "Running: python grok_test.py $input $algo > $output"
        python grok_test.py "$input" "$algo" > "$output"
    done
done

