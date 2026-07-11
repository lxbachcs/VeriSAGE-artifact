#!/usr/bin/env bash
cd "$(dirname "$0")"
for m in qwen2.5:7b qwen2.5:14b qwen2.5:72b; do
  echo "===== $m $(date +%H:%M:%S) =====" >> ../results/hard2.log
  python3 experiments_llm.py "$m" 30 6 hard >> ../results/hard2.log 2>&1
done
echo "HARD2 COMPLETE $(date +%H:%M:%S)" >> ../results/hard2.log
