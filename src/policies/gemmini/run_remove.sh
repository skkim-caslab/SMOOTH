#!/bin/bash


seq 512 8747 | xargs -P 48 -I {} bash -c '
    idx={}
    echo "Running simulation for idx=$idx..."
    rm /home/sadmin/skkim/logs/out/bs1_bw128_128KB_seq32K_6.7B_xla_kn/nopreload/no_sa_o32K_dou_c${idx}.out
'

echo "Simulation completed."


