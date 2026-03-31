# ASMR(Advanced Sram Management foR) LLM Simulator Github Repository

### How to Run

```bash
$ python3 simulate.py
```

Then, this program starts.


## CHECK fragmentation
python fusion_json.py:
  - input: base_tile.json    
  - output: modified_*.json     <fusion tile>

python update_tile_lifetime.py base_tile.json base_tile_lifetime_x1.json
    - input: base_tile.json/modified_tile.json
    - output: tile_lifetime.json
python grok_test.py
    - input: tile_lifetime.json

