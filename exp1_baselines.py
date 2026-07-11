"""
EXP1 -- stratified ~100-task head-to-head baseline comparison (Stanford review
Q8 / primary ask). Real LLM proposer via Ollama.

We build a STRATIFIED 100-task suite over the two implemented catalogues (the
easy 6-service payment task and the hard 13-service regulated pipeline), stratified
by SLA-budget tightness (5 strata each) so tasks span easy->hard and loose->tight.
For every task we run three methods that share the SAME proposer and per-task LLM
budget T:
  * ACCEPT-FIRST  (no verifier): take the first LLM plan as the answer; "success"
    = it happens to be admissible (this is what a no-verifier agent ships).
  * BLIND-RESAMPLE (no typed feedback): draw up to T independent proposals, accept
    the first that passes C1--C5 (verifier-gated selection, but no counterexample).
  * VERISAGE (verifier + typed-counterexample-guided repair).
CERTIFIED-SUCCESS = the returned plan passes all five checks (accept-first is only
"certified" when its first plan happens to be admissible).

We repeat the whole suite over several proposer seeds (Ollama `seed`, set via
VERISAGE_LLM_SEED) and report mean +/- std per method. Honest: this is a stratified
subset over the two implemented catalogues, not the full 340-task benchmark whose
open suites (APIs.guru/HSC/QWS/Enterprise-WF) are described but not built here.

Usage: python3 exp1_baselines.py <model> <seed1,seed2,...> <T> [n_per_stratum]
Outputs results/exp1_baselines_<model>.json (+ raw per-task jsonl).
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np
from core import Goal
from catalog import build_catalogue, catalogue_prompt, GOAL_NL
from catalog_hard import build_catalogue_hard, GOAL_NL_HARD
from experiments_llm import run_task

EASY_STRATA = [1.5, 1.8, 2.1, 2.4, 2.7]     # budget-slack multipliers (tight->loose)
HARD_STRATA = [1.20, 1.35, 1.50, 1.65, 1.80]


def easy_goal(slack_mult: float) -> Goal:
    ub = 30.0; n_min = 6
    mu = n_min * ub / 2
    slack = slack_mult * (n_min * ub ** 2) ** 0.5
    return Goal(init_fields=set(), goal_fields={"receipt.id", "metric"},
                goal_props={"receipted"},
                phi=[("precede", "authenticate", "charge"),
                     ("response", "charge", "receipt")],
                budget=float(mu + slack), delta=0.05, init_level={})


def hard_goal(slack_mult: float) -> Goal:
    UB = 24.0; n_req = 13
    mu = n_req * UB / 2
    slack = slack_mult * (n_req * UB ** 2) ** 0.5
    return Goal(init_fields=set(),
                goal_fields={"receipt.id", "metric", "audit.id"},
                goal_props={"receipted"},
                phi=[("precede", "authenticate", "charge"),
                     ("precede", "consent", "charge"),
                     ("precede", "kyc", "charge"),
                     ("precede", "fraud", "charge"),
                     ("response", "charge", "receipt")],
                budget=float(mu + slack), delta=0.05, init_level={})


def build_stratified_suite(n_per_stratum=10):
    """Return list of (suite, stratum_idx, task_idx, goal). Fixed & reproducible."""
    tasks = []
    for si, mult in enumerate(EASY_STRATA):
        for j in range(n_per_stratum):
            tasks.append(("easy", si, j, easy_goal(mult)))
    for si, mult in enumerate(HARD_STRATA):
        for j in range(n_per_stratum):
            tasks.append(("hard", si, j, hard_goal(mult)))
    return tasks


def run_suite(model, seed, T, n_per_stratum):
    import llm
    os.environ["VERISAGE_LLM_SEED"] = str(seed * 100000)   # well-separated base
    llm._CALL_N = 0                                          # reset per replica
    easy_cat = build_catalogue(); easy_desc = catalogue_prompt(easy_cat)
    hard_cat = build_catalogue_hard(); hard_desc = catalogue_prompt(hard_cat)
    tasks = build_stratified_suite(n_per_stratum)
    recs = []
    t0 = time.time()
    for i, (suite, si, j, goal) in enumerate(tasks):
        if suite == "easy":
            cat, desc, gnl = easy_cat, easy_desc, GOAL_NL
        else:
            cat, desc, gnl = hard_cat, hard_desc, GOAL_NL_HARD
        # include the numeric SLA budget in the goal so budget strata are
        # genuinely different tasks (and repair has a concrete target)
        gnl = gnl + f" The end-to-end latency budget is {goal.budget:.0f} ms."
        r = run_task(model, cat, desc, gnl, goal, T, seed=1000 + i)
        r.update({"suite": suite, "stratum": si, "seed_replica": seed})
        recs.append(r)
        if (i + 1) % 10 == 0:
            cur = {k[:6]: round(100 * np.mean([x[k] for x in recs]), 1) for k in
                   ("direct_certified", "blind_certified", "verisage_certified")}
            print(f"[{model} seed={seed}] {i+1}/{len(tasks)}  {time.time()-t0:.0f}s  {cur}",
                  flush=True)
    return recs


def summarize(recs):
    def rate(key, sub=None):
        rr = [r for r in recs if (sub is None or r["suite"] == sub)]
        return round(100 * float(np.mean([r[key] for r in rr])), 2)
    out = {"n": len(recs)}
    for key, nm in (("direct_certified", "accept_first"),
                    ("blind_certified", "blind_resample"),
                    ("selfrefine_certified", "self_refine"),
                    ("verisage_certified", "verisage")):
        out[nm + "_CSR"] = rate(key)
        out[nm + "_CSR_easy"] = rate(key, "easy")
        out[nm + "_CSR_hard"] = rate(key, "hard")
    out["verisage_mean_iters"] = round(float(np.mean([r["verisage_iters"] for r in recs])), 3)
    out["accept_first_hallucinated_pct"] = round(
        100 * float(np.mean([r["direct_hallucinated"] for r in recs])), 2)
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    seeds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2", "3"])]
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    n_per = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    os.makedirs("../results/raw", exist_ok=True)
    per_seed, all_recs = [], []
    for sd in seeds:
        recs = run_suite(model, sd, T, n_per)
        all_recs += recs
        s = summarize(recs); s["seed"] = sd
        per_seed.append(s)
        print(f"  seed {sd} summary: {s}", flush=True)

    def agg(field):
        vals = [ps[field] for ps in per_seed]
        return {"mean": round(float(np.mean(vals)), 2), "std": round(float(np.std(vals)), 2)}
    fields = [k for k in per_seed[0] if k.endswith("_CSR") or k.endswith("_easy")
              or k.endswith("_hard") or k in ("verisage_mean_iters",
                                              "accept_first_hallucinated_pct")]
    out = {"model": model, "seeds": seeds, "T": T,
           "n_tasks": len(build_stratified_suite(n_per)),
           "strata": {"easy": EASY_STRATA, "hard": HARD_STRATA},
           "per_seed": per_seed,
           "aggregate": {f: agg(f) for f in fields}}
    tag = model.replace(":", "_").replace("/", "_")
    with open(f"../results/exp1_baselines_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    with open(f"../results/raw/exp1_{tag}.jsonl", "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print("\n=== EXP1 AGGREGATE (mean +/- std over seeds) ===")
    for m, nm in (("accept_first", "accept-first (no verifier)"),
                  ("blind_resample", "blind resample"),
                  ("self_refine", "Self-Refine (LLM self-critique)"),
                  ("verisage", "VeriSAGE (typed repair)")):
        a = out["aggregate"][m + "_CSR"]
        ae = out["aggregate"][m + "_CSR_easy"]; ah = out["aggregate"][m + "_CSR_hard"]
        print(f"  {nm:32s} CSR={a['mean']:.1f}+/-{a['std']:.1f}  "
              f"(easy {ae['mean']:.1f}, hard {ah['mean']:.1f})")
    print(f"  VeriSAGE mean repair iters = {out['aggregate']['verisage_mean_iters']['mean']}")
    print(f"wrote ../results/exp1_baselines_{tag}.json")


if __name__ == "__main__":
    main()
