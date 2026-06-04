#!/bin/bash
set -e
echo "=== GCT: Granger-Causal Transformer ==="
python -m src.model --config configs/config.yaml --seed 161 --folder results
echo "=== Done ==="
