python simulate.py 1 ./Tiles/test_tile/double_512_N_large.json 2048 32 > final_wo_fusion/cache_1_llama2.out &
python simulate.py 1 ./Tiles/test_tile/double_512_N_large.json 1024 16 > final_wo_fusion/cache_1_gpt_neo.out &

python simulate.py 2048 ./Tiles/test_tile/double_512_N_large.json 2048 32 > final_wo_fusion/cache_2048_llama2.out &
python simulate.py 2048 ./Tiles/test_tile/double_512_N_large.json 1024 16 > final_wo_fusion/cache_2048_gpt_neo.out &

python simulate.py 4096 ./Tiles/test_tile/double_512_N_large.json 2048 32 > final_wo_fusion/cache_4096_llama2.out &
python simulate.py 4096 ./Tiles/test_tile/double_512_N_large.json 1024 16 > final_wo_fusion/cache_4096_gpt_neo.out &


python simulate.py 8192 ./Tiles/test_tile/double_512_N_large.json 2048 32 > final_wo_fusion/cache_8192_llama2.out &
python simulate.py 8192 ./Tiles/test_tile/double_512_N_large.json 1024 16 > final_wo_fusion/cache_8192_gpt_neo.out &


python simulate.py 32768 ./Tiles/test_tile/double_512_N_large.json 2048 32 > final_wo_fusion/cache_32768_llama2.out &
python simulate.py 32768 ./Tiles/test_tile/double_512_N_large.json 1024 16 > final_wo_fusion/cache_32768_gpt_neo.out &
