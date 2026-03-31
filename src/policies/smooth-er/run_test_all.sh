#!/bin/bash
python simulate.py ./Tiles/tile_size/test_batch_large.json > out/test_batch_large.out&
python simulate.py ./Tiles/tile_size/test_batch_small.json > out/test_batch_small.out&
python simulate.py ./Tiles/tile_size/test_batch_mix.json > out/test_batch_mix.out&
