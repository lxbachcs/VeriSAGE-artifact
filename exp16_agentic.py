"""
EXP16 -- extended head-to-head against modern SOTA agentic baselines, on the same
stratified 100-task suite, same proposer, same per-task LLM budget as EXP1. All
baselines are verifier-free during the loop (the CSTS verifier only SCORES the
final plan); only VeriSAGE uses the sound verifier + typed-counterexample repair.

Baselines (all real, cited):
  * accept-first        : first LLM plan, no verifier.
  * blind               : best-of-T, accept first that passes C1-C5 (verifier-gated).
  * best-of-N (SC)      : draw T plans, pick the MODAL plan by self-consistency
                          [Wang et al. 2023], no verifier gating.
  * Self-Refine         : LLM critiques+revises its OWN plan, no tool
                          [Madaan et al. 2023 / Shinn et al. 2023].
  * CRITIC              : tool-interactive self-correction [Gou et al. 2024] with a
                          SCHEMA/type checker as the tool (C1 only) -- a standard
                          tool, BLIND to ordering/flow/policy/SLA (C2-C5).
  * LATS                : Language Agent Tree Search [Zhou et al. 2024] -- bounded
                          value-guided search with an LLM value function, no verifier.
  * VeriSAGE (ours)     : verify (C1-C5) + typed-counterexample-guided repair.

For fairness, blind and best-of-N share ONE pool of T proposals; Self-Refine,
CRITIC, VeriSAGE each get T-1 revision rounds; LATS gets a ~2T-call budget.
CERTIFIED-SUCCESS = the returned plan passes all five CSTS checks (scored once).

Usage: python3 exp16_agentic.py <model> <seed1,seed2,...> <T> [n_per_stratum]
Outputs results/exp16_agentic_<model>.json (+ raw per-task jsonl).
"""
from __future__ import annotations
import sys, os, json, time
from collections import Counter
import numpy as np
import llm
from core import verify
from catalog import build_catalogue, catalogue_prompt, GOAL_NL
from catalog_hard import build_catalogue_hard, GOAL_NL_HARD
from llm import propose, self_critique, self_refine, value_estimate, propose_next
from catalog import plan_from_names
from exp1_baselines import build_stratified_suite, EASY_STRATA, HARD_STRATA

METHODS = ["accept_first", "blind", "best_of_n", "self_refine", "critic", "lats", "verisage"]


def _ok(names, cat, goal):
    p, _ = plan_from_names(names, cat)
    return verify(p, goal).ok


def _lats(model, cat, desc, gnl, goal, budget, k=2, init=None):
    """LATS (Zhou et al. 2024): a search tree over candidate plans guided by an LLM
    value function, with self-reflection expansion. We root at a real proposal,
    repeatedly select the highest-value node, expand it into k reflected
    alternatives, score each with the LLM value function, and return the
    highest-value plan found. No formal verifier is used in the loop -- the
    selection/return signal is the LLM's own value estimate. `budget` caps LLM
    calls (proposal + value + expansion)."""
    calls = 0
    root = list(init) if init else propose(model, desc, gnl, [], 0.7); calls += 1
    rv = value_estimate(model, desc, gnl, root); calls += 1
    nodes = [(root, rv)]                      # (plan, LLM value)
    best = (root, rv)
    while calls < budget:
        parent, pv = max(nodes, key=lambda x: x[1])   # exploit highest-value node
        for _ in range(k):
            if calls >= budget:
                break
            fb = (f"This plan scored {pv*10:.0f}/10. Produce a better alternative that "
                  "fixes any wrong ordering, missing prerequisites, data-privacy leak, "
                  "policy issue, or budget overrun.")
            child = self_refine(model, desc, gnl, parent, fb, 0.7); calls += 1
            cv = value_estimate(model, desc, gnl, child); calls += 1
            nodes.append((child, cv))
            if cv > best[1]:
                best = (child, cv)
    return best[0]


def run_task(model, cat, desc, gnl, goal, T, lats_budget=24):
    rec = {}
    # ---- shared proposal pool (T plans): initial + T-1 resamples ----
    pool = [propose(model, desc, gnl, [], 0.7)]
    for _ in range(T - 1):
        pool.append(propose(model, desc, gnl, [], 0.9))

    # accept-first: first plan
    rec["accept_first"] = _ok(pool[0], cat, goal)

    # blind: first pool plan that passes C1-C5
    rec["blind"] = any(_ok(nm, cat, goal) for nm in pool)

    # best-of-N (self-consistency): modal plan, no verifier gating
    modal = Counter(tuple(nm) for nm in pool).most_common(1)[0][0]
    rec["best_of_n"] = _ok(list(modal), cat, goal)

    # Self-Refine: LLM self-critique loop, no tool; stop when it says OK
    sr = list(pool[0])
    for _ in range(T - 1):
        fb = self_critique(model, desc, gnl, sr, 0.5)
        if fb.strip().upper().startswith("OK"):
            break
        sr = self_refine(model, desc, gnl, sr, fb, 0.5)
    rec["self_refine"] = _ok(sr, cat, goal)

    # CRITIC: schema/type checker (C1 only) as the critique tool; blind to C2-C5
    cr = list(pool[0])
    for _ in range(T - 1):
        v = verify(plan_from_names(cr, cat)[0], goal)
        if v.check != "C1":                      # schema tool sees only C1 -> stops
            break
        tool = f"Schema/type checker: {v.witness}. Add the missing producer or input."
        cr = self_refine(model, desc, gnl, cr, tool, 0.5)
    rec["critic"] = _ok(cr, cat, goal)

    # LATS: value-function-guided tree search (rooted at the shared proposal), no verifier
    lats_plan = _lats(model, cat, desc, gnl, goal, budget=lats_budget, init=pool[0])
    rec["lats"] = _ok(lats_plan, cat, goal)

    # VeriSAGE: verify (C1-C5) + typed-counterexample repair
    from core import verify_all
    vs = list(pool[0]); plan_i = plan_from_names(vs, cat)[0]; ok = verify(plan_i, goal).ok
    it = 0
    while not ok and it < T - 1:
        w = verify_all(plan_i, goal)
        if not w:
            ok = True; break
        nm = propose(model, desc, gnl, w, 0.5, prev_plan=vs)
        vs = nm if nm else vs
        plan_i = plan_from_names(vs, cat)[0]; it += 1; ok = verify(plan_i, goal).ok
    rec["verisage"] = bool(ok)
    return {k: bool(v) for k, v in rec.items()}


def run_suite(model, seed, T, n_per):
    os.environ["VERISAGE_LLM_SEED"] = str(seed * 100000)
    llm._CALL_N = 0
    ecat = build_catalogue(); edesc = catalogue_prompt(ecat)
    hcat = build_catalogue_hard(); hdesc = catalogue_prompt(hcat)
    tasks = build_stratified_suite(n_per)
    recs = []; t0 = time.time()
    for i, (suite, si, j, goal) in enumerate(tasks):
        cat, desc, gnl = (ecat, edesc, GOAL_NL) if suite == "easy" else (hcat, hdesc, GOAL_NL_HARD)
        gnl = gnl + f" The end-to-end latency budget is {goal.budget:.0f} ms."
        # LATS budget scaled to plan depth: easy ~6-service, hard ~13-service plans
        lbudget = 22 if suite == "easy" else 46
        r = run_task(model, cat, desc, gnl, goal, T, lats_budget=lbudget)
        r["suite"] = suite
        recs.append(r)
        if (i + 1) % 10 == 0:
            cur = {m[:5]: round(100 * np.mean([x[m] for x in recs]), 1) for m in METHODS}
            print(f"[{model} seed={seed}] {i+1}/{len(tasks)} {time.time()-t0:.0f}s {cur}", flush=True)
    return recs


def summarize(recs):
    def rate(key, sub=None):
        rr = [r for r in recs if (sub is None or r["suite"] == sub)]
        return round(100 * float(np.mean([r[key] for r in rr])), 2)
    out = {"n": len(recs)}
    for m in METHODS:
        out[m + "_CSR"] = rate(m)
        out[m + "_CSR_easy"] = rate(m, "easy")
        out[m + "_CSR_hard"] = rate(m, "hard")
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2", "3"])]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_per = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    os.makedirs("results/raw", exist_ok=True)
    per_seed, all_recs = [], []
    for sd in seeds:
        recs = run_suite(model, sd, T, n_per); all_recs += recs
        s = summarize(recs); s["seed"] = sd; per_seed.append(s)
        print(f"  seed {sd}: {s}", flush=True)

    def agg(f):
        vals = [ps[f] for ps in per_seed]
        return {"mean": round(float(np.mean(vals)), 2), "std": round(float(np.std(vals)), 2)}
    fields = [k for k in per_seed[0] if k.endswith(("_CSR", "_easy", "_hard"))]
    out = {"model": model, "seeds": seeds, "T": T,
           "n_tasks": len(build_stratified_suite(n_per)),
           "methods": METHODS, "per_seed": per_seed,
           "aggregate": {f: agg(f) for f in fields}}
    tag = model.replace(":", "_").replace("/", "_")
    with open(f"results/exp16_agentic_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(f"results/raw/exp16_{tag}.jsonl", "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print("\n=== EXP16 AGENTIC BASELINES (mean over seeds) ===")
    for m in METHODS:
        a = out["aggregate"][m + "_CSR"]
        print(f"  {m:14s} CSR={a['mean']:5.1f}±{a['std']:.1f}  "
              f"(easy {out['aggregate'][m+'_CSR_easy']['mean']:.1f}, "
              f"hard {out['aggregate'][m+'_CSR_hard']['mean']:.1f})")
    print(f"wrote results/exp16_agentic_{tag}.json")


if __name__ == "__main__":
    main()
