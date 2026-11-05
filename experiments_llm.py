"""
Real LLM experiment: run Direct-LLM (single-shot), blind-resample (best-of-T),
and VeriSAGE (verify + typed-counterexample-guided repair) with a LOCAL model
(ollama) as the proposer, on the catalogue task. Emits per-task JSONL and a
summary. This produces MEASURED numbers for the paper's headline ablation
(single-shot vs blind-resample vs guided repair) and for Direct-LLM vs VeriSAGE.
"""
from __future__ import annotations
import sys, json, time
import numpy as np
from core import verify, verify_all
from catalog import build_catalogue, catalogue_prompt, GOAL_NL, make_goal, plan_from_names
from llm import propose, self_critique, self_refine

REPAIR_HINT = {
    "C1": "every service's required inputs must be produced by an earlier service (include readCard/tokenize as needed)",
    "C2": "authenticate must run (sets 'authed') before charging; charge before receipt",
    "C3": "authenticate BEFORE chargeCard; emailReceipt AFTER chargeCard",
    "C4": "run tokenize before analyticsSink so the raw card number never reaches the public sink",
    "C5": "remove unnecessary services to stay within the SLA latency budget, and ensure guards pass",
}


def _hint(v):
    # feed the specific typed counterexample witness plus a short directive
    base = REPAIR_HINT.get(v.check, "")
    return f"[{v.check}] {v.witness}." + (" " + base if base else "")


def run_task(model, cat, cat_desc, goal_nl, goal, T, seed):
    rng = np.random.default_rng(seed)
    rec = {"seed": int(seed)}

    # ---- Direct-LLM (single shot, accepted as-is) ----
    names = propose(model, cat_desc, goal_nl, [], 0.7)
    plan, halluc = plan_from_names(names, cat)
    v = verify(plan, goal)
    rec["direct_certified"] = bool(v.ok)
    rec["direct_hallucinated"] = halluc > 0
    rec["direct_schema_ok"] = v.check != "C1"          # passed schema stage

    # ---- w/o repair == single-shot verify+reject (same first proposal) ----
    rec["single_shot_certified"] = bool(v.ok)

    # ---- blind resample: up to T independent proposals, accept first admissible ----
    blind_ok = v.ok
    for _ in range(T - 1):
        if blind_ok:
            break
        nm = propose(model, cat_desc, goal_nl, [], 0.9)
        p, _h = plan_from_names(nm, cat)
        blind_ok = verify(p, goal).ok
    rec["blind_certified"] = bool(blind_ok)

    # ---- Self-Refine (Madaan'23) / Reflexion: the LLM critiques and revises its
    #      OWN plan, with NO formal verifier in the loop. The self-critique decides
    #      when to stop (it says 'OK'); the verifier is used ONLY to score the
    #      final plan, never shown to the loop. This isolates "LLM self-reflection"
    #      from VeriSAGE's typed counterexamples. ----
    sr_names = list(names)
    sr_iters = 0
    for _ in range(T - 1):
        fb = self_critique(model, cat_desc, goal_nl, sr_names, 0.5)
        if fb.strip().upper().startswith("OK"):
            break                                   # model believes it is correct
        sr_names = self_refine(model, cat_desc, goal_nl, sr_names, fb, 0.5)
        sr_iters += 1
    sr_plan, _h = plan_from_names(sr_names, cat)
    rec["selfrefine_certified"] = bool(verify(sr_plan, goal).ok)
    rec["selfrefine_iters"] = int(sr_iters)

    # ---- VeriSAGE: verify + typed-counterexample-guided repair (patches the
    #      previous plan given the specific witness -- proper CEGIS step) ----
    vs_ok, iters = v.ok, 0
    cur_names = list(names)
    plan_i = plan
    while not vs_ok and iters < T - 1:
        witnesses = verify_all(plan_i, goal)          # full typed diagnostic
        if not witnesses:
            vs_ok = True
            break
        nm = propose(model, cat_desc, goal_nl, witnesses, 0.5, prev_plan=cur_names)
        cur_names = nm if nm else cur_names
        plan_i, _h = plan_from_names(cur_names, cat)
        iters += 1
        vs_ok = verify(plan_i, goal).ok
    rec["verisage_certified"] = bool(vs_ok)
    rec["verisage_iters"] = int(iters)
    rec["verisage_pc"] = float(verify(plan_i, goal).proof_coverage if vs_ok else 0.0)
    return rec


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    suite = sys.argv[4] if len(sys.argv) > 4 else "easy"
    if suite == "hard":
        from catalog_hard import build_catalogue_hard, GOAL_NL_HARD, make_goal_hard
        cat = build_catalogue_hard(); goal_nl = GOAL_NL_HARD; mk_goal = make_goal_hard
    else:
        cat = build_catalogue(); goal_nl = GOAL_NL; mk_goal = make_goal
    cat_desc = catalogue_prompt(cat)
    recs = []
    t0 = time.time()
    for i in range(n):
        goal = mk_goal(np.random.default_rng(1000 + i))
        r = run_task(model, cat, cat_desc, goal_nl, goal, T, seed=i)
        recs.append(r)
        if (i + 1) % 10 == 0:
            done = i + 1
            cur = {k: round(100 * np.mean([x[k] for x in recs]), 1)
                   for k in ("direct_certified", "single_shot_certified",
                             "blind_certified", "selfrefine_certified",
                             "verisage_certified")}
            print(f"[{model}] {done}/{n}  {time.time()-t0:.0f}s  {cur}", flush=True)

    out = {
        "model": model, "n": n, "T": T, "suite": suite,
        "direct_CSR": 100 * np.mean([r["direct_certified"] for r in recs]),
        "single_shot_CSR": 100 * np.mean([r["single_shot_certified"] for r in recs]),
        "blind_CSR": 100 * np.mean([r["blind_certified"] for r in recs]),
        "selfrefine_CSR": 100 * np.mean([r["selfrefine_certified"] for r in recs]),
        "verisage_CSR": 100 * np.mean([r["verisage_certified"] for r in recs]),
        "selfrefine_mean_iters": float(np.mean([r["selfrefine_iters"] for r in recs])),
        "direct_HAR": 100 * np.mean([r["direct_hallucinated"] for r in recs]),
        "direct_SV": 100 * np.mean([r["direct_schema_ok"] for r in recs]),
        "verisage_mean_iters": float(np.mean([r["verisage_iters"] for r in recs])),
        "verisage_PC": 100 * np.mean([r["verisage_pc"] for r in recs if r["verisage_certified"]] or [0]),
        "wall_s": round(time.time() - t0, 1),
    }
    import os
    os.makedirs("results/raw", exist_ok=True)
    tag = model.replace(':', '_').replace('/', '_') + ("" if suite == "easy" else f"_{suite}")
    with open(f"results/raw/llm_{tag}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    with open(f"results/llm_summary_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
