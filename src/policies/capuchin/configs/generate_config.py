import json
import os
import itertools

# 설정 가능한 값들
quants = ["w4a8", "w8a8"]
flash_attentions = [True, False]
sram_sizes = ["512KB", "8MB", "64MB"]
#output_token_lengths = [4096, 32768, 131072]
output_token_lengths = [2048]
block_sizes = ["2048", "1", "1024"]

# 출력 디렉토리
output_dir = "generated_configs"
os.makedirs(output_dir, exist_ok=True)

# 이름 변환 규칙
def get_filename(quant, fa, sram, out_len, blk_size):
    fa_str = "fa" if fa else "sa"
    out_len_str = {2048: "2K", 1024:"1K"}[out_len]
    #out_len_str = {4096: "4K", 32768: "32K", 131072: "128K"}[out_len]
    if blk_size == "2048": blk_str = "b2K"
    if blk_size == "1024": blk_str = "b1K"
    if blk_size == "1": blk_str = "b1"
    return f"{quant}_{fa_str}_{sram}_{out_len_str}_{blk_str}.json"

# 조합 생성 및 파일 저장
for quant, fa, sram, out_len, blk_size in itertools.product(
    quants, flash_attentions, sram_sizes, output_token_lengths, block_sizes
):
    config = {
        "quant": quant,
        "flash_attention": fa,
        "sram_size": sram,
        "output_token_length": out_len,
        "block_size": blk_size
    }
    filename = get_filename(quant, fa, sram, out_len, blk_size)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

print(f"✅ {len(quants) * len(flash_attentions) * len(sram_sizes) * len(output_token_lengths) * len(block_sizes)} config files generated in '{output_dir}'")

