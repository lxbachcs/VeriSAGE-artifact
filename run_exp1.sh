#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "=== EXP1 full stratified-100 baseline comparison START $(date) ==="
for M in qwen2.5:7b qwen2.5:14b llama3.2:3b; do
  echo "----- model $M $(date) -----"
  python3 exp1_baselines.py "$M" 1,2,3 4 10
done
echo "=== EXP1 DONE $(date) ==="
