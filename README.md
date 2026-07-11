# VeriSAGE prototype (executable)

A self-contained implementation of the CSTS verifier and the
propose–verify–repair loop, used to produce the paper's **measured** results.
Pure Python + numpy (verifier/bench/experiments); the LLM proposer talks to a
local [Ollama](https://ollama.com) server.

## Files
- `core.py` — CSTS model + C1–C5 verifier (`verify`) + Hoeffding SLA bound + MC.
- `bench.py` — Synth-TSR generator with ground-truth labels; synthetic repair loop.
- `catalog.py` — fixed service catalogue + goal for the LLM experiment.
- `llm.py` — Ollama proposer (`propose`), robust JSON-array parsing.
- `experiments_core.py` — E1 verifier accuracy · E2 SLA soundness · E3 scalability · E4 repair convergence → `../results/core_results.json`.
- `experiments_llm.py` — Direct-LLM / blind-resample / VeriSAGE with a local model.
- `make_real_figures.py` — writes the manuscript's Fig 3 & Fig 4 from measured results.
- `check_results.py` — asserts `results/*.json` match the manuscript's headline numbers (PASS/FAIL; no LLM/API).
- `run_smoke.sh` — fast smoke test: real verifier on a labelled subset + `check_results.py`; no API/Ollama.
- `requirements.txt` — Python dependencies (numpy + matplotlib; z3-solver optional).

## Install

```bash
pip install -r requirements.txt              # numpy + matplotlib (z3-solver optional)
```

## Run

```bash
bash run_smoke.sh                            # FAST, no API/Ollama: verifier smoke + result checks
python3 check_results.py                     # assert results/*.json match the manuscript (PASS/FAIL)
python3 experiments_core.py                  # recompute E1-E4 measured core results; no LLM needed
python3 experiments_llm.py qwen2.5:7b 50 4    # needs `ollama serve` + the model
python3 make_real_figures.py                  # regenerate real figures from results/*.json
```

## Measured results (this repo; all verified by `check_results.py`)

- Verifier accuracy: **100%** (900 ground-truth Synth-TSR tasks).
- SLA bound soundness: **100%** (32k plans, worst-case latency).
- Verification scaling: **near-linear, R²=0.999** (stress-tested to k=200 services).
- Guidance ≫ sampling, easy payment task (single-shot→blind→VeriSAGE CSR):
  qwen2.5:7b **93→100→100%**, llama3.2:3b **0→0→87%**.

## Extending to frontier models (planned full-scale comparison)
Swap the Ollama endpoint in `llm.py::_call` for a provider API (Anthropic/OpenAI)
and run `experiments_llm.py` over the full suites. See `../manuscript/PLAN.md`.
