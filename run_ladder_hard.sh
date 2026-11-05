#!/usr/bin/env bash
cd "$(dirname "$0")"
for m in llama3.2:3b qwen2.5:7b qwen2.5:14b qwen2.5:72b llama3.3:70b; do
  echo "===== $m $(date +%H:%M:%S) =====" >> results/ladder_hard.log
  python3 experiments_llm.py "$m" 30 6 hard >> results/ladder_hard.log 2>&1
done
echo "HARD LADDER COMPLETE $(date +%H:%M:%S)" >> results/ladder_hard.log
