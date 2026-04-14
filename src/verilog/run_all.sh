#!/bin/bash

dirs=(alloc find_zero address_check bt_lookup free)

for d in "${dirs[@]}"; do
    echo "===== Running in $d ====="
    (
        cd "$d" || exit
        ../yosys -s synth_${d}.ys
        ../sta -exit sta.tcl
    )
done
