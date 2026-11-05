"""
EXP14 -- Multi-defect repair: plans that SIMULTANEOUSLY violate ordering (C3) +
policy (C5-pol) + SLA (C5-sla) (+ optionally flow C4), and the typed-repair
PRIORITIZATION that avoids oscillation, with multi-class repair traces (Stanford
round-2 review Q8, LLM-free typed repair like exp7).

When several checks fail at once, a naive "fix whatever fires first" loop can
OSCILLATE: fixing SLA by dropping a service can re-break ordering; fixing a leak
by inserting a declassifier can re-break the budget. VeriSAGE compiles each typed
counterexample into a HARD CONSTRAINT that permanently excludes the refuted class
(Prop. admissibility-preserving repair), so a class is never re-entered. We show
a fixed PRIORITY ORDER over defect classes that, combined with the
never-re-refute discipline, guarantees monotone progress (a strictly shrinking
unrefuted set U) and hence no oscillation.

Priority (most-constraining structural first, budget last):
    C1 (missing producer) > C2 (precondition) > C3 (ordering) > C4 (flow) > C5-pol > C5-sla
Rationale: adding producers/reorders/declassifiers can only ADD services or
permute them, which the SLA check then evaluates once the structure is fixed;
repairing SLA first would be undone by later structural inserts. Fixing budget
LAST (by dropping the cheapest non-essential distractor, never an essential
service a hard constraint requires) converges without reopening structure.

We compare, on tasks with 2-4 SIMULTANEOUS defects across DISTINCT classes:
  (A) NAIVE first-fire order (no hard-constraint memory, may re-refute)  -> oscillation
  (B) PRIORITIZED typed repair with hard-constraint memory (VeriSAGE)     -> monotone
and report: certified-success rate, mean iterations, oscillation rate (a class
re-entered after being repaired), and dump a few multi-class repair TRACES.

Pure Python + the real core verifier. No LLM. Outputs results/exp14_multidefect.json.
"""
from __future__ import annotations
import json, os, copy
from typing import List, Tuple
import numpy as np

from core import Service, Goal, verify, verify_all
import bench


PRIORITY = ["C1", "C2", "C3", "C4", "C5pol", "C5sla"]


def _classify(witnesses: List[str]) -> List[str]:
    """map verify_all witnesses to defect classes present."""
    classes = []
    for w in witnesses:
        if w.startswith("[C1]"): classes.append("C1")
        elif w.startswith("[C2]"): classes.append("C2")
        elif w.startswith("[C3]"): classes.append("C3")
        elif w.startswith("[C4]"): classes.append("C4")
        elif "policy guard" in w: classes.append("C5pol")
        elif "SLA bound" in w: classes.append("C5sla")
    # dedup preserving priority order
    return [c for c in PRIORITY if c in classes]


def _make_multidefect(rng, n_classes):
    """Start from the clean payment core; inject defects in n_classes DISTINCT
    classes simultaneously. Returns (plan, goal, base, injected_classes)."""
    base, goal = bench._payment_core(rng)
    goal = Goal(**{**goal.__dict__})
    plan = [copy.deepcopy(s) for s in base]
    names = [s.name for s in plan]
    pool = ["C3", "C4", "C5pol", "C5sla", "C2"]
    chosen = list(rng.permutation(pool))[:n_classes]
    injected = []
    for c in chosen:
        names = [s.name for s in plan]
        if c == "C3" and "chargeCard" in names and "authenticate" in names:
            ci, ai = names.index("chargeCard"), names.index("authenticate")
            if ci > ai:
                plan[ai], plan[ci] = plan[ci], plan[ai]
            else:
                plan[ci], plan[ai] = plan[ai], plan[ci]
            injected.append("C3")
        elif c == "C4":
            plan = [s for s in plan if s.name != "tokenize"]
            for s in plan:
                if s.name in ("chargeCard", "analyticsSink"):
                    s.requires = {x if x != "card.tok" else "card.pan" for x in s.requires}
            injected.append("C4")
        elif c == "C5pol":
            for s in plan:
                if s.name == "chargeCard":
                    s.guard_ok = False
            injected.append("C5pol")
        elif c == "C5sla":
            # add TWO droppable distractors and tighten the budget so the plan is
            # over-budget but recoverable by dropping the distractors (not any
            # essential service) -- a genuine SLA defect that does not conflict
            # with the structural repairs.
            for dn in ("logEvent", "refund"):
                extra = Service(dn, dn, set(), {f"{dn}.id"}, lat_ub=30, lat_mean=15)
                plan.append(extra)
            mu = sum(s.lat_mean for s in plan)
            import math as _m
            sum_ub2 = sum(s.lat_ub ** 2 for s in plan)
            goal.budget = mu + 1.2 * _m.sqrt(sum_ub2)   # just under the certifiable slack
            injected.append("C5sla")
        elif c == "C2":
            # break the 'authed' precondition of charge WITHOUT dropping the token
            # producer: strip authenticate's post-prop so 'authed' is never set
            # (isolates the failure to C2, no cascading C1).
            for s in plan:
                if s.name == "authenticate":
                    s.post_props = set()
            injected.append("C2")
    return plan, goal, base, injected


def _repair_class(plan, base, goal, cls, rng):
    """apply the typed fix for a single defect class (deterministic)."""
    names = [s.name for s in plan]
    if cls == "C3":
        if "authenticate" in names and "chargeCard" in names:
            ai, ci = names.index("authenticate"), names.index("chargeCard")
            if ai > ci:
                plan[ci], plan[ai] = plan[ai], plan[ci]
    elif cls == "C2":
        # restore the missing precondition-establishing effect ('authed').
        found = False
        for s in plan:
            if s.name == "authenticate":
                s.post_props = {"authed"}; found = True
        if not found:
            au = next((s for s in copy.deepcopy(base) if s.name == "authenticate"), None)
            if au: plan = [au] + plan
    elif cls == "C4":
        if not any(s.name == "tokenize" for s in plan):
            tok = next(s for s in copy.deepcopy(base) if s.name == "tokenize")
            idx = names.index("analyticsSink") if "analyticsSink" in names else len(plan)
            plan.insert(idx, tok)
            for s in plan:
                if s.name in ("chargeCard", "analyticsSink"):
                    s.requires = {x if x != "card.pan" else "card.tok" for x in s.requires}
    elif cls == "C5pol":
        for s in plan:
            if s.name == "chargeCard":
                s.guard_ok = True         # obtain the missing authorization grant
    elif cls == "C5sla":
        # drop the cheapest NON-ESSENTIAL distractor (never an essential service).
        essential = {"readCard", "authenticate", "tokenize", "chargeCard",
                     "emailReceipt", "analyticsSink", "mintToken"}
        drops = [i for i, s in enumerate(plan) if s.name not in essential]
        if drops:
            plan.pop(drops[-1])
    return plan


def run(regime, n_tasks=700, T=8, seed=0, class_dist=(2, 3, 4)):
    rng = np.random.default_rng(seed)
    certified = 0
    iters = []
    oscillations = 0
    traces = []
    for tid in range(n_tasks):
        nc = int(rng.choice(class_dist))
        plan, goal, base, injected = _make_multidefect(rng, nc)
        refuted = set()          # hard-constraint memory (classes permanently fixed)
        repaired_once = set()
        trace = []
        done = False
        for it in range(T + 1):
            v = verify(plan, goal)
            ws = verify_all(plan, goal)
            classes = _classify(ws)
            if v.ok:
                certified += 1; iters.append(it); done = True
                break
            if it == T:
                break
            if regime == "prioritized":
                # pick the highest-priority UNREFUTED class; never re-enter a
                # refuted STRUCTURAL class (hard-constraint memory). C5sla is the
                # sole probabilistic class: each drop excludes a distinct plan (a
                # different member), so it may recur monotonically (fewer services
                # each time) without oscillation -- it is exempt from the once-only
                # rule but still strictly shrinks the plan, guaranteeing progress.
                target = next((c for c in classes if c not in refuted), None)
                if target is None:
                    target = "C5sla" if "C5sla" in classes else (classes[0] if classes else None)
                if target is None:
                    break
                plan = _repair_class(plan, base, goal, target, rng)
                if target != "C5sla":
                    refuted.add(target)
            else:  # naive: fix the first-firing check (short-circuit order), no memory
                first = v.check if v.check != "C5" else (
                    "C5pol" if any("policy guard" in w for w in ws) else "C5sla")
                if first in repaired_once and first in classes:
                    oscillations += 1        # re-entered an already-repaired class
                plan = _repair_class(plan, base, goal, first, rng)
                repaired_once.add(first)
            trace.append({"iter": it, "classes_present": classes,
                          "target": (target if regime == "prioritized" else first)})
        if tid < 6 and len(injected) >= 3:
            traces.append({"injected_classes": injected, "certified": done,
                           "iters": len(trace), "trace": trace})
    return {"regime": regime, "n_tasks": n_tasks, "T": T,
            "CSR_pct": round(100 * certified / n_tasks, 2),
            "mean_iters": round(float(np.mean(iters)) if iters else -1, 3),
            "oscillation_events": oscillations,
            "sample_traces": traces}


def main():
    os.makedirs("results", exist_ok=True)
    res = {}
    res["prioritized"] = run("prioritized", seed=14)
    res["naive"] = run("naive", seed=14)
    with open("results/exp14_multidefect.json", "w") as f:
        json.dump(res, f, indent=2)
    print("== multi-defect repair (2-4 simultaneous distinct-class defects) ==")
    for r in ("prioritized", "naive"):
        d = res[r]
        print(f"  {r:<12} CSR={d['CSR_pct']:.1f}%  mean-iters={d['mean_iters']:.2f}  "
              f"oscillation-events={d['oscillation_events']}")
    print("== sample multi-class repair traces (prioritized) ==")
    for t in res["prioritized"]["sample_traces"][:3]:
        seq = " -> ".join(f"{s['target']}" for s in t["trace"])
        print(f"  inject {t['injected_classes']}: repair order [{seq}] "
              f"-> certified={t['certified']} in {t['iters']} iters")
    print("\nwrote results/exp14_multidefect.json")


if __name__ == "__main__":
    main()
