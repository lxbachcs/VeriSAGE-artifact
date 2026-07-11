#!/usr/bin/env bash
# Fast smoke test for the VeriSAGE artifact. NO API key, NO Ollama, NO network:
# it exercises the real C1-C5 verifier on a small labelled Synth-TSR subset and
# then checks that the committed measured results match the manuscript's numbers.
# Runs in a couple of seconds. Exit 0 = OK, non-zero = a check failed.
set -e
cd "$(dirname "$0")"

echo "== [1/2] verifier smoke: 60 labelled Synth-TSR tasks (no LLM) =="
python3 - <<'PY'
import numpy as np, bench
from core import verify
rng = np.random.default_rng(0)
tasks = bench.build_suite(rng, 60)
correct = sum(1 for plan, goal, label, vtype in tasks if verify(plan, goal).ok == label)
acc = 100.0 * correct / len(tasks)
print(f"   verified {len(tasks)} tasks, {correct} correct, accuracy {acc:.1f}%")
assert acc == 100.0, f"verifier smoke FAILED: accuracy {acc:.1f}% != 100%"
print("   [PASS] verifier agrees with ground truth on the smoke subset")
PY

echo
echo "== [2/2] check committed results/*.json against manuscript claims =="
python3 check_results.py

echo
echo "Smoke test complete."
