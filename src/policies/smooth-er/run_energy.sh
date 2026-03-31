#!/bin/bash

# 실험할 사이즈 목록 (1K=1024, 8K=8192, 32K=32768)
#SIZES=("1024" "8192" "32768")
#LABELS=("1K" "8K" "32K")
SIZES=("1024" "8192")
LABELS=("1K" "8K")

for i in "${!SIZES[@]}"; do
    SIZE=${SIZES[$i]}
    LABEL=${LABELS[$i]}
    OUT_DIR="energy_out/block1600_output${LABEL}"
    
    # 디렉토리 생성
    mkdir -p "$OUT_DIR"
    
    echo "Running simulation for ${LABEL} (Size: ${SIZE})..."

    # 각 모델별 시뮬레이션 실행 (&를 붙여 백그라운드 병렬 실행)
#    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 1024 8 > "$OUT_DIR/gemma_2_2b_quant.out" &
#    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 2048 8 > "$OUT_DIR/gemma_2_2b.out" &
    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 1024 16 > "$OUT_DIR/gpt_neo_quant.out" &
    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 2048 16 > "$OUT_DIR/gpt_neo.out" &
    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 2048 32 > "$OUT_DIR/llama2_quant.out" &
    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 4096 32 > "$OUT_DIR/llama2.out" &
#    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 1024 32 > "$OUT_DIR/tiny_llama_quant.out" &
#    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 1536 24 > "$OUT_DIR/gpt_xl_quant.out" &
#    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 3072 24 > "$OUT_DIR/gpt_xl.out" &
#    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 1280 32 > "$OUT_DIR/gpt_2.7b_quant.out" &
#    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 2560 32 > "$OUT_DIR/gpt_2.7b.out" &
    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 2560 40 > "$OUT_DIR/gpt3_quant.out" &
    python simulate.py $SIZE ./Tiles/test_tile/double_512_N_large.json 5120 40 > "$OUT_DIR/gpt3.out" &
    
    # 한 사이즈의 모든 모델이 끝날 때까지 기다리려면 아래 주석 해제
    # wait
done

echo "All simulations dispatched."
