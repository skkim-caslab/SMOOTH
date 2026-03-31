python memory_frag.py base_tile.json > frag/512KB_10_base.out & 
python memory_frag.py modified_tile_all_d3fcea4cc603408eb938b9a80d896fa6.json > frag/512KB_10_all.out &
python memory_frag.py modified_tile_ffn_354193762eac4b2dab7504cbe557b3bc.json > frag/512KB_10_ffn.out &
python memory_frag.py modified_tile_flash_attention_15f0b9d183ae45ab9ac55168ef1092c9.json > frag/512KB_10_flash.out &
python memory_frag.py modified_tile_qkv_attention_50dcc35c8f6649a9b26ba44dc25fae17.json  > frag/512KB_10_qkv.out &
