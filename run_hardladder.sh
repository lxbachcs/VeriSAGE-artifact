#!/usr/bin/env bash
cd "$(dirname "$0")"
for m in qwen2.5:7b qwen2.5:14b llama3.3:70b qwen2.5:72b; do
  echo "===== $m $(date +%H:%M:%S) =====" >> ../results/hardladder.log
  python3 experiments_llm.py "$m" 20 6 hard >> ../results/hardladder.log 2>&1
  echo "done $m $(date +%H:%M:%S)" >> ../results/hardladder.log
done
echo "HARDLADDER COMPLETE $(date +%H:%M:%S)" >> ../results/hardladder.log
