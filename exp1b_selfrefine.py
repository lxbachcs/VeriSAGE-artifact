"""
EXP1b -- backfill the Self-Refine (Madaan'23) baseline for a proposer whose
original exp1 run predates Self-Refine being added, WITHOUT re-running or
perturbing the other (already-published) methods for that model.

It reuses EXACTLY exp1's stratified 100-task suite and the same Self-Refine flow
as experiments_llm.run_task (LLM self-critique/revise, no verifier in the loop;
the verifier scores only the final plan). It writes self_refine_CSR / _easy /
_hard into the model's existing results/exp1_baselines_<model>.json aggregate.

Usage: python3 exp1b_selfrefine.py <model> [seed] [T] [n_per_stratum]
"""
from __future__ import annotations
import sys, os, json, time
import numpy as np

from core import verify
from catalog import build_catalogue, catalogue_prompt, GOAL_NL, plan_from_names
from catalog_hard import build_catalogue_hard, GOAL_NL_HARD
from llm import propose, self_critique, self_refine
from exp1_baselines import build_stratified_suite


def selfrefine_certified(model, cat, desc, gnl, goal, T):
    names = propose(model, desc, gnl, [], 0.7)          # same first proposal as exp1
    sr = list(names)
    for _ in range(T - 1):
        fb = self_critique(model, desc, gnl, sr, 0.5)
        if fb.strip().upper().startswith("OK"):
            break
        sr = self_refine(model, desc, gnl, sr, fb, 0.5)
    plan, _ = plan_from_names(sr, cat)
    return bool(verify(plan, goal).ok)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma2:9b"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_per = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    os.environ["VERISAGE_LLM_SEED"] = str(seed * 100000)   # match exp1 convention
    easy_cat = build_catalogue(); easy_desc = catalogue_prompt(easy_cat)
    hard_cat = build_catalogue_hard(); hard_desc = catalogue_prompt(hard_cat)
    tasks = build_stratified_suite(n_per)

    recs = []
    t0 = time.time()
    for i, (suite, si, j, goal) in enumerate(tasks):
        if suite == "easy":
            cat, desc, gnl = easy_cat, easy_desc, GOAL_NL
        else:
            cat, desc, gnl = hard_cat, hard_desc, GOAL_NL_HARD
        gnl = gnl + f" The end-to-end latency budget is {goal.budget:.0f} ms."
        ok = selfrefine_certified(model, cat, desc, gnl, goal, T)
        recs.append({"suite": suite, "ok": ok})
        if (i + 1) % 10 == 0:
            r = round(100 * np.mean([x["ok"] for x in recs]), 1)
            print(f"[{model} self-refine] {i+1}/{len(tasks)} {time.time()-t0:.0f}s  CSR={r}", flush=True)

    def rate(sub=None):
        rr = [r for r in recs if sub is None or r["suite"] == sub]
        return round(100 * float(np.mean([r["ok"] for r in rr])), 2)
    res = {"self_refine_CSR": rate(), "self_refine_CSR_easy": rate("easy"),
           "self_refine_CSR_hard": rate("hard")}
    print("== self-refine backfill ==", res)

    # write into the existing exp1 json aggregate (mean/std format), std=0 (1 seed)
    tag = model.replace(":", "_").replace("/", "_")
    path = f"../results/exp1_baselines_{tag}.json"
    with open(path) as f:
        d = json.load(f)
    for k, v in res.items():
        d["aggregate"][k] = {"mean": v, "std": 0.0}
    d["self_refine_backfill"] = {"seed": seed, "T": T, "n_per": n_per, **res}
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"wrote self_refine_* into {path}")


if __name__ == "__main__":
    main()
