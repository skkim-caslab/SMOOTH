#!/bin/bash


#seq 0 8747 | xargs -P 20 -I {} bash -c '
#    idx={}
#    echo "Running simulation for idx=$idx..."
#    python simulate.py ./Tiles/double_tile_size/config_${idx}.json > /home/sadmin/skkim/logs/out/bs1_bw32/preload/pre_sa_o8192_dou_c${idx}.out
#'

#seq 0 1 | xargs -P 40 -I {} bash -c '
seq 0 8747 | xargs -P 40 -I {} bash -c '
    idx={}
    echo "Running simulation for idx=$idx..."
    python parser.py ../logs/out/bs1_bw32_512KB_seq32K_6.7B/preload/pre_sa_o32K_dou_c${idx}.out > /home/sadmin/skkim/logs/out/bs1_bw32_512KB_seq32K_6.7B/preload/pre_sa_o32K_dou_c${idx}_sum.out
    rm -f /home/sadmin/skkim/logs/out/bs1_bw32_512KB_seq32K_6.7B/preload/pre_sa_o32K_dou_c${idx}.out
    rm -f /home/sadmin/skkim/logs/out/bs1_bw32_512KB_seq32K_6.7B/preload/pre_sa_o32K_dou_c${idx}_parse.csv
'
echo "Simulation completed."

