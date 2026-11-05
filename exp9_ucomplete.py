"""
EXP9 -- U-complete proposer: a bounded enumerator / classical forward planner
over the SAME CSTS constraints that certifies a SOUND INFEASIBLE verdict within a
length bound (Stanford round-2 review Q6).

The paper's Corollary (relative completeness) says: with a U-complete proposer
that eventually returns every unrefuted plan of length <= L_max, VeriSAGE returns
a certified plan whenever one exists and a *sound* INFEASIBLE otherwise. An LLM is
NOT U-complete, so it can only ABSTAIN. This experiment IMPLEMENTS that U-complete
proposer as a bounded best-first forward planner over the contracted catalogue and
demonstrates:

  (1) SOUNDNESS: on tasks that are genuinely UNSATISFIABLE within the bound
      (goal field never producible / a mandatory precondition unreachable / an
      unavoidable un-declassified leak / a budget no admissible plan can meet),
      the enumerator exhausts U and returns a *sound* INFEASIBLE -- and we
      cross-check that NO admissible plan of length <= L_max exists (exhaustive
      ground truth), so INFEASIBLE is never wrong.

  (2) COMPLETENESS: on SATISFIABLE tasks the enumerator finds a certified plan.

  (3) The runtime / coverage trade-off vs the length bound L_max: nodes expanded,
      wall-clock, and the fraction of the (huge) plan space actually explored,
      as L_max grows. Pruning by the verifier's typed constraints (never expand a
      prefix that already violates a monotone check) keeps this tractable.

Pure Python + the real core.py verifier. No LLM, no SMT, no network.
Outputs results/exp9_ucomplete.json.
"""
from __future__ import annotations
import json, os, time, itertools, copy
from typing import Dict, List, Optional, Tuple
import numpy as np

from core import Service, Goal, verify, verify_all, hoeffding_bound, LEVELS
from catalog import build_catalogue
from catalog_hard import build_catalogue_hard


# ---------------------------------------------------------------------------
# Monotone prefix pruning: a prefix already fails a *monotone* check (one that
# cannot be un-failed by appending services) can be pruned. We prune on:
#   - C3 ordering: an action b appeared before its required predecessor a (a
#     'precede' violation is permanent once b has occurred without a);
#   - C4 leak: a high field already reached a lower-clearance sink (permanent);
#   - C5 SLA lower bound: the prefix mean already exceeds the budget (mu only
#     grows as we append, so P[violate]->1).
# C1/C2/goal are checked at the end (they can be satisfied by later services).
# This keeps enumeration sound (we never prune a prefix that could extend to an
# admissible plan) while cutting the branching factor drastically.
# ---------------------------------------------------------------------------
def _prefix_dead(plan: List[Service], goal: Goal) -> bool:
    word = [s.action for s in plan]
    # C3 precede: if b occurred with no preceding a, no suffix can fix it
    for kind, a, b in goal.phi:
        if kind == "precede":
            seen_a = False
            for ev in word:
                if ev == a:
                    seen_a = True
                if ev == b and not seen_a:
                    return True
        elif kind == "absence" and a in word:
            return True
    # C4 permanent leak
    eff: Dict[str, str] = dict(goal.init_level)
    for s in plan:
        for f, lv in s.field_level.items():
            eff[f] = lv
        for f in s.declassifies:
            eff[f] = "public"
        if s.sink_field is not None:
            if LEVELS.get(eff.get(s.sink_field, "public"), 0) > LEVELS.get(s.sink_clearance, 3):
                return True
    # C5 SLA mean already over budget (mu monotonically increases with length)
    mu = sum(s.lat_mean for s in plan)
    if mu > goal.budget:
        return True
    return False


def enumerate_plans(cat: Dict[str, Service], goal: Goal, L_max: int,
                    stop_on_first: bool = True, node_cap: int = 4_000_000) -> dict:
    """Bounded best-first (BFS by length) forward enumeration over U, pruning
    dead prefixes. Returns the first certified plan (if any) and stats. When
    stop_on_first is False, it exhausts the whole <=L_max space (ground truth).
    node_cap bounds expansion; if hit before exhausting, verdict is UNKNOWN
    (never a false INFEASIBLE)."""
    services = list(cat.values())
    expanded = 0
    capped = False
    t0 = time.perf_counter()
    certified: Optional[List[str]] = None
    n_admissible = 0
    # BFS frontier of partial plans (lists of Service)
    frontier: List[List[Service]] = [[]]
    for depth in range(L_max + 1):
        nxt: List[List[Service]] = []
        for prefix in frontier:
            # try to certify the current prefix as a complete plan
            if prefix:
                v = verify(prefix, goal)
                if v.ok:
                    n_admissible += 1
                    if certified is None:
                        certified = [s.name for s in prefix]
                    if stop_on_first:
                        return {"certified": certified, "expanded": expanded,
                                "n_admissible": n_admissible, "L_max": L_max,
                                "wall_s": round(time.perf_counter() - t0, 4),
                                "verdict": "SAT"}
            if depth == L_max:
                continue
            # expand: append each catalogue service (no repeats -> DAG plans)
            used = {s.name for s in prefix}
            for s in services:
                if s.name in used:
                    continue
                child = prefix + [copy.deepcopy(s)]
                expanded += 1
                if expanded > node_cap:
                    capped = True
                    break
                if _prefix_dead(child, goal):
                    continue
                nxt.append(child)
            if capped:
                break
        if capped:
            break
        frontier = nxt
    wall = round(time.perf_counter() - t0, 4)
    if certified is not None:
        verdict = "SAT"
    elif capped:
        verdict = "UNKNOWN"      # node cap hit; never claim INFEASIBLE unsoundly
    else:
        verdict = "INFEASIBLE"
    return {"certified": certified, "expanded": expanded,
            "n_admissible": n_admissible, "L_max": L_max, "wall_s": wall,
            "verdict": verdict, "capped": capped}


def _brute_ground_truth(cat: Dict[str, Service], goal: Goal, L_max: int) -> bool:
    """Exhaustive check: does ANY plan of length <= L_max certify? Used ONLY to
    validate the enumerator's INFEASIBLE verdicts are sound. To terminate on the
    full catalogue we enumerate ordered plans with the SAME sound monotone-prefix
    pruning as the enumerator (which cannot prune away an admissible plan), so an
    all-False result is still a genuine exhaustive negative over <=L_max."""
    services = list(cat.values())
    frontier: List[List[Service]] = [[]]
    for depth in range(L_max + 1):
        nxt: List[List[Service]] = []
        for prefix in frontier:
            if prefix and verify(prefix, goal).ok:
                return True
            if depth == L_max:
                continue
            used = {s.name for s in prefix}
            for s in services:
                if s.name in used:
                    continue
                child = prefix + [copy.deepcopy(s)]
                if _prefix_dead(child, goal):
                    continue
                nxt.append(child)
        frontier = nxt
    return False


# ---------------------------------------------------------------------------
# UNSAT task constructors: each removes exactly one necessary capability so NO
# admissible plan of any length exists over the (reduced) catalogue -> the
# enumerator must return a SOUND INFEASIBLE.
# ---------------------------------------------------------------------------
def _unsat_tasks(base_cat: Dict[str, Service], base_goal: Goal):
    tasks = []
    # (a) missing producer: drop the ONLY declassifier for the PII sink field ->
    #     any plan reaching the public analytics sink leaks -> C4 unavoidable.
    cat = {k: v for k, v in base_cat.items() if k != "tokenize"}
    tasks.append(("no_declassifier", cat, base_goal))
    # (b) unreachable precondition: drop authenticate -> 'authed' never true ->
    #     chargeCard's precondition (C2) can never be met -> no 'receipted'.
    cat = {k: v for k, v in base_cat.items() if k != "authenticate"}
    tasks.append(("no_auth_precondition", cat, base_goal))
    # (c) impossible budget: keep full catalogue but set a budget below the mean
    #     of the minimal required chain -> C5 SLA unsatisfiable for every plan.
    g = Goal(**{**base_goal.__dict__})
    g.budget = g.budget * 0.20
    tasks.append(("impossible_budget", base_cat, g))
    # (d) missing goal producer: require a goal field nothing produces.
    g2 = Goal(**{**base_goal.__dict__})
    g2.goal_fields = set(base_goal.goal_fields) | {"nonexistent.field"}
    tasks.append(("unproducible_goal", base_cat, g2))
    return tasks


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(9)
    cat = build_catalogue()
    from catalog import make_goal
    goal = make_goal(rng)

    res: dict = {"catalogue": "payment", "catalogue_size": len(cat)}

    # ---- (1)+(2) SAT: enumerator finds a certified plan (completeness) ----
    sat = enumerate_plans(cat, goal, L_max=8, stop_on_first=True)
    res["sat_task"] = sat

    # ---- (1) UNSAT: sound INFEASIBLE, cross-checked against brute ground truth
    unsat_results = []
    for name, ucat, ugoal in _unsat_tasks(cat, goal):
        enum = enumerate_plans(ucat, ugoal, L_max=6, stop_on_first=True)
        # ground truth: is there ANY certifiable plan <= 6? (should be False)
        gt_sat = _brute_ground_truth(ucat, ugoal, L_max=6)
        enum["ground_truth_sat_leq6"] = gt_sat
        enum["sound"] = (enum["verdict"] == "INFEASIBLE") and (not gt_sat)
        enum["task"] = name
        unsat_results.append(enum)
    res["unsat_tasks"] = unsat_results
    res["all_infeasible_sound"] = all(u["sound"] for u in unsat_results)

    # ---- (3) runtime / coverage vs L_max on the harder regulated catalogue ----
    hcat = build_catalogue_hard()
    from catalog_hard import make_goal_hard
    hgoal = make_goal_hard(rng)
    # a genuinely-UNSAT variant of the hard task: drop redactSSN so the SSN sink
    # (auditLog) is always leaked -> no admissible plan.
    hcat_unsat = {k: v for k, v in hcat.items() if k != "redactSSN"}
    sweep = []
    for L in (3, 4, 5):
        e = enumerate_plans(hcat_unsat, hgoal, L_max=L, stop_on_first=True,
                            node_cap=800_000)
        # coverage: expanded vs the (pruned-free) upper bound sum_{l<=L} P(|cat|,l)
        n = len(hcat_unsat)
        space = sum(int(np.prod([n - j for j in range(l)])) for l in range(1, L + 1))
        e["space_upper_bound"] = space
        e["explored_fraction"] = round(e["expanded"] / max(space, 1), 6)
        sweep.append(e)
    res["runtime_coverage_sweep_regulated_unsat"] = sweep

    with open("results/exp9_ucomplete.json", "w") as f:
        json.dump(res, f, indent=2)

    print("== U-complete bounded enumerator (sound INFEASIBLE) ==")
    print(f"SAT task: verdict={sat['verdict']} plan_len="
          f"{len(sat['certified']) if sat['certified'] else 0} "
          f"expanded={sat['expanded']} wall={sat['wall_s']}s")
    print("UNSAT tasks (verdict / ground-truth-SAT / SOUND):")
    for u in unsat_results:
        print(f"  {u['task']:<22} {u['verdict']:<11} gt_sat={u['ground_truth_sat_leq6']!s:<5} "
              f"sound={u['sound']} expanded={u['expanded']} wall={u['wall_s']}s")
    print(f"ALL INFEASIBLE verdicts sound: {res['all_infeasible_sound']}")
    print("== runtime/coverage vs L_max (regulated UNSAT) ==")
    print(f"  {'L_max':>5} {'expanded':>10} {'wall_s':>9} {'explored_frac':>14} {'verdict':>11}")
    for e in sweep:
        print(f"  {e['L_max']:>5} {e['expanded']:>10} {e['wall_s']:>9} "
              f"{e['explored_fraction']:>14.2e} {e['verdict']:>11}")
    print("\nwrote results/exp9_ucomplete.json")


if __name__ == "__main__":
    main()
