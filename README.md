# VeriSAGE prototype (executable artifact)

Prototype of the CSTS verifier and the propose–verify–repair loop from
*VeriSAGE: Verified Agentic Service Composition with SLA, Policy, and Privacy
Guarantees*. It contains the prototype, experiment scripts, and stored results used in the paper. Pure Python +
NumPy for the verifier, benchmarks, and experiments; the LLM planner talks to a
local [Ollama](https://ollama.com) server or a hosted API.

Everything runs from the repository root. All results are written to and read
from `results/` inside this repository.

## Quick check (no LLM, no network, a few seconds)

```bash
pip install -r requirements.txt   # numpy + matplotlib (z3-solver optional)
bash run_smoke.sh                 # verifier on 60 labelled tasks + check_results.py
```

`check_results.py` compares selected committed `results/*.json` values with
expectations embedded in the script and prints PASS/FAIL per check. It does not
parse the manuscript or rerun the experiments. CI runs the same smoke
test on every push (`.github/workflows/smoke.yml`).

## Known limitations

The current prototype has a confirmed C4 dependency-propagation defect and a
narrower C1 check than the formal design. The synthetic smoke test and stored
result checks do not establish general verifier soundness. See
[KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) for the reproducible case and
the distinctions between implementation, offline simulations, and design claims.

## What the prototype implements (and what it does not)

- C1–C5 for **sequential** plans, typed witnesses, the repair loop, the
  Hoeffding SLA bound, and all baselines.
- C2 checks propositional pre/postconditions; full SMT discharge with Z3 is
  benchmarked separately for timing (`exp3_scalability.py`, `exp8_endtoend.py`).
- Policy guards are Boolean functions, not Rego rules evaluated by OPA.
- DAG plans and the path-union SLA bound are covered by the theory only.
- In the LLM loop the planner receives the previous plan plus all witnesses
  (`core.verify_all`) as text; compiled constraints are enforced by the verifier
  gate. The LLM-free experiments (`exp7`, `exp14`) enforce them directly.

## Files

| File | Purpose |
| --- | --- |
| `core.py` | CSTS model, C1–C5 verifier (`verify`, `verify_all`), Hoeffding bound |
| `bench.py` | Synth-TSR generator (labels by construction) and synthetic repair loop |
| `catalog.py`, `catalog_hard.py` | Payment (10 services) and regulated (20 services) catalogues and goals |
| `llm.py` | Planner and repair prompts, JSON-array parsing, Self-Refine / LATS helpers |
| `experiments_core.py` | Verifier accuracy, SLA calibration, scaling, repair convergence → `results/core_results.json` |
| `experiments_llm.py` | Single-shot / blind / VeriSAGE runs with one planner → `results/llm_summary_*.json` |
| `exp1_baselines.py`, `exp1b_selfrefine.py` | Stratified 100-task suite → `results/exp1_baselines_*.json`, `results/raw/` |
| `exp2`–`exp17` | Further experiments (one JSON each in `results/`) |
| `qws_data.py`, `data/` | Real QWS v2.0 dataset and loader (see `data/PROVENANCE.md`) |
| `check_results.py`, `run_smoke.sh` | Claim checks and smoke test |
| `appendix.pdf` | Extended report: proofs, LTL_f semantics, extra SLA bounds |

## Paper claim → script → result file

| Paper (RPE v1.2, §4) | Script | Result file |
| --- | --- | --- |
| Verifier agrees on 900/900 Synth-TSR tasks; per-class recall; reduced-verifier ablation | `experiments_core.py` | `results/core_results.json` (`e1_*`) |
| SLA calibration on 32,000 simulated plans | `experiments_core.py` | `results/core_results.json` (`e2_*`) |
| Bennett −53.7% slack on real QWS | `exp4_bernstein.py` | `results/exp4_bernstein.json` |
| 88.5% feasible vs 59.1% admitted | `exp2_sla_tightness.py` | `results/exp2_sla_tightness.json` |
| Correlated latencies, Cantelli fallback | `exp11_correlation.py` | `results/exp11_correlation.json` |
| 0.27 µs/service to k=200; Z3 0.30–1.11 ms/service | `exp3_scalability.py` | `results/exp3_scalability.json` |
| 13-service plan 4.3–9.1 ms, 100-service 33–70 ms | `exp8_endtoend.py` | `results/exp8_endtoend.json` |
| Table 5(a) stratified suite | `exp1_baselines.py` (+`exp1b_selfrefine.py`) | `results/exp1_baselines_*.json` |
| Table 5(b) hard-task ladder | `experiments_llm.py <model> 20 6 hard` | `results/llm_summary_*_hard.json` |
| Agentic baselines (self-consistency, CRITIC, LATS) | `exp16_agentic.py` | `results/exp16_agentic_*.json` |
| Typed vs check-list vs no feedback | `exp7_repair_ablation.py` | `results/exp7_repair_ablation.json` |
| Priority-ordered multi-defect repair | `exp14_multidefect.py` | `results/exp14_multidefect.json` |
| Contract-noise robustness | `exp17_noisy_contract.py` | `results/exp17_noisy_contract.json` |
| QoS-optimal plan violates C3 | `exp10_qos_baseline.py` | `results/exp10_qos_baseline.json` |
| Bounded planner returns sound INFEASIBLE | `exp9_ucomplete.py` | `results/exp9_ucomplete.json` |
| 0.5 µs suffix re-certification | (recorded) | `results/exp15_recert_timing.json` |

## Re-running experiments

```bash
python3 experiments_core.py                         # no LLM
python3 exp7_repair_ablation.py                      # no LLM (same for exp2-exp14, exp17)
python3 experiments_llm.py qwen2.5:7b 20 6 hard      # needs `ollama serve` + the model
python3 exp1_baselines.py qwen2.5:7b 1,2,3 4         # stratified suite, 3 seeds, T=4
python3 make_real_figures.py                          # writes figures/ from results/
```

Hosted models: set the provider endpoint and key in `llm.py::_call`.
