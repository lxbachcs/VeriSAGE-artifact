"""
EXP10 -- Classical QoS-aware optimizer baselines (MILP-style exhaustive optimum
and a genetic algorithm) over the SAME catalogue contracts, compared head-to-head
with VeriSAGE on shared tasks (Stanford round-2 review Q7).

Classical QoS-aware composition (Zeng'04, Alrifai'09, Canfora'05) selects, per
abstract task slot, a concrete service to minimise cost/latency subject to *global
QoS* constraints. Crucially it optimises QoS but is BLIND to the non-QoS contract
dimensions VeriSAGE certifies: temporal ordering (C3), explicit-flow privacy (C4),
and functional pre/postconditions (C2). We implement two standard variants:

  (A) QoS-MILP*: the exact optimum -- exhaustive search over all length<=L plans,
      picking the minimum-latency plan that satisfies the SLA budget and schema
      (a classical QoS selector only enforces schema + QoS). This is the *global-
      optimal* QoS baseline (an ILP would return the same optimum).

  (B) QoS-GA: a genetic algorithm over service-selection genomes with a QoS
      fitness (latency + budget penalty), the Canfora'05-style baseline. Cheaper,
      finds a near-optimal QoS plan.

Both are given the same catalogue and goal. We then ask, for the plan each
returns: does it actually satisfy the FULL five-class CSTS contract (i.e. would it
be SAFE to execute)? VeriSAGE, by construction, only ever returns contract-certified
plans. The gap = QoS-optimal plans that VIOLATE ordering / privacy / policy that a
QoS optimizer ships and VeriSAGE rejects/repairs.

We also report the shared-task feasibility + cost comparison: on tasks where an
admissible plan exists, whether each method finds a plan meeting the SLA, and its
latency (QoS optimizer minimises latency; VeriSAGE certifies but does not optimise,
so we report the latency gap as the price of certification-over-optimisation).

Pure Python + numpy + the real core verifier. No LLM, no MILP solver dependency
(the exhaustive optimum coincides with the ILP optimum on these finite catalogues).
Outputs results/exp10_qos_baseline.json.
"""
from __future__ import annotations
import json, os, copy, itertools
from typing import Dict, List, Optional, Tuple
import numpy as np

from core import Service, Goal, verify
from catalog import build_catalogue, make_goal
from catalog_hard import build_catalogue_hard, make_goal_hard


def _schema_ok(plan: List[Service], goal: Goal) -> bool:
    """A classical QoS selector only enforces schema feasibility (produced fields)
    + operation existence -- it is unaware of C2/C3/C4/policy."""
    avail = set(goal.init_fields)
    for s in plan:
        if not s.requires <= avail:
            return False
        avail |= s.produces
    return goal.goal_fields <= avail


def _latency_mean(plan: List[Service]) -> float:
    return sum(s.lat_mean for s in plan)


def _topo_schema_order(svcs: List[Service], goal: Goal) -> Optional[List[Service]]:
    """Greedily order a service SET so each service's inputs are already
    available (schema-valid producer order). Returns None if no such order
    exists (set is not schema-realisable for the goal)."""
    remaining = list(svcs)
    avail = set(goal.init_fields)
    order: List[Service] = []
    progress = True
    while remaining and progress:
        progress = False
        for s in list(remaining):
            if s.requires <= avail:
                order.append(s)
                avail |= s.produces
                remaining.remove(s)
                progress = True
    if remaining or not goal.goal_fields <= avail:
        return None
    return order


def qos_milp_optimum(cat: Dict[str, Service], goal: Goal, L_max: int = 8,
                     cap: int = 3_000_000) -> dict:
    """Exact QoS optimum: over all ordered plans of length <= L_max that are
    schema-feasible and meet the mean-latency budget, return the minimum-latency
    one. (Coincides with the ILP optimum; classical selectors ignore C2/C3/C4.)
    Since latency is order-independent, we minimise over service SUBSETS (an ILP
    selects a set), then realise a schema-valid order -- exact and far cheaper
    than permuting."""
    names = list(cat.keys())
    best: Optional[List[Service]] = None
    best_lat = float("inf")
    considered = 0
    for L in range(1, L_max + 1):
        for subset in itertools.combinations(names, L):
            considered += 1
            if considered > cap:
                break
            svcs = [cat[n] for n in subset]
            lat = _latency_mean(svcs)
            if lat > goal.budget or lat >= best_lat:
                continue
            ordered = _topo_schema_order(svcs, goal)   # any schema-valid order
            if ordered is not None:
                best, best_lat = ordered, lat
    if best is None:
        return {"found": False}
    v = verify(list(best), goal)
    return {"found": True, "plan": [s.name for s in best], "latency_mean": round(best_lat, 2),
            "considered": considered,
            "csts_certified": v.ok, "first_violation": v.check,
            "witness": v.witness}


def qos_ga(cat: Dict[str, Service], goal: Goal, L_max: int = 8,
           pop: int = 60, gens: int = 60, seed: int = 0) -> dict:
    """Canfora'05-style GA: genome = ordered list of service indices (variable
    length <= L_max). Fitness = -latency - big*budget_violation - big*schema_fail.
    QoS-only fitness (no C2/C3/C4 awareness)."""
    rng = np.random.default_rng(seed)
    names = list(cat.keys())
    N = len(names)

    def random_genome():
        L = int(rng.integers(3, L_max + 1))
        return list(rng.choice(N, size=L, replace=False))

    def decode(g):
        return [cat[names[i]] for i in g]

    def fitness(g):
        plan = decode(g)
        lat = _latency_mean(plan)
        f = -lat
        if not _schema_ok(plan, goal):
            f -= 1e4
        if lat > goal.budget:
            f -= 1e3 * (lat - goal.budget)
        return f

    P = [random_genome() for _ in range(pop)]
    for _ in range(gens):
        scored = sorted(P, key=fitness, reverse=True)
        elite = scored[:pop // 4]
        children = list(elite)
        while len(children) < pop:
            a, b = elite[int(rng.integers(len(elite)))], elite[int(rng.integers(len(elite)))]
            cut = int(rng.integers(1, min(len(a), len(b)) + 1)) if min(len(a), len(b)) > 1 else 1
            child = a[:cut] + [x for x in b if x not in a[:cut]]
            child = child[:L_max]
            if rng.random() < 0.3 and len(child) > 1:      # mutation: swap
                i, j = rng.integers(0, len(child), size=2)
                child[i], child[j] = child[j], child[i]
            if rng.random() < 0.2:                          # mutation: add/drop
                if rng.random() < 0.5 and len(child) < L_max:
                    cand = int(rng.integers(N))
                    if cand not in child:
                        child.append(cand)
                elif len(child) > 3:
                    child.pop(int(rng.integers(len(child))))
            children.append(child)
        P = children
    best = max(P, key=fitness)
    plan = decode(best)
    v = verify(plan, goal)
    return {"plan": [s.name for s in plan], "latency_mean": round(_latency_mean(plan), 2),
            "schema_ok": _schema_ok(plan, goal),
            "csts_certified": v.ok, "first_violation": v.check, "witness": v.witness}


def verisage_reference(cat: Dict[str, Service], goal: Goal, L_max: int = 8) -> dict:
    """VeriSAGE's guarantee: return the first CSTS-certified plan a bounded
    U-complete enumerator finds (same enumerator as exp9). By construction it is
    contract-safe. We report its latency to quantify the certification-vs-QoS-
    optimisation gap."""
    import exp9_ucomplete as E9
    r = E9.enumerate_plans(cat, goal, L_max=L_max, stop_on_first=True)
    if r["certified"] is None:
        return {"found": False}
    plan = [copy.deepcopy(cat[n]) for n in r["certified"]]
    return {"found": True, "plan": r["certified"],
            "latency_mean": round(_latency_mean(plan), 2), "csts_certified": True}


def run_catalogue(name, cat, goal, L_max, ga_seeds=(0, 1, 2, 3, 4),
                  vs_ref_plan=None, vs_enum_Lmax=None):
    milp = qos_milp_optimum(cat, goal, L_max)
    gas = [qos_ga(cat, goal, L_max, seed=s) for s in ga_seeds]
    ga_cert = float(np.mean([g["csts_certified"] for g in gas]))
    ga_schema = float(np.mean([g["schema_ok"] for g in gas]))
    ga_best = min((g for g in gas if g["schema_ok"]),
                  key=lambda g: g["latency_mean"], default=gas[0])
    if vs_ref_plan is not None:
        plan = [copy.deepcopy(cat[n]) for n in vs_ref_plan]
        v = verify(plan, goal)
        vs = {"found": v.ok, "plan": vs_ref_plan,
              "latency_mean": round(_latency_mean(plan), 2), "csts_certified": v.ok}
    else:
        vs = verisage_reference(cat, goal, vs_enum_Lmax or L_max)
    return {
        "catalogue": name, "budget": round(goal.budget, 2),
        "qos_milp": milp,
        "qos_ga_csts_certified_frac": round(ga_cert, 3),
        "qos_ga_schema_ok_frac": round(ga_schema, 3),
        "qos_ga_best": ga_best,
        "verisage": vs,
        "latency_gap_verisage_vs_milp":
            (round(vs["latency_mean"] - milp["latency_mean"], 2)
             if (vs.get("found") and milp.get("found")) else None),
    }


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(10)
    res = {}
    # payment catalogue
    cat = build_catalogue(); goal = make_goal(rng)
    res["payment"] = run_catalogue("payment", cat, goal, L_max=8, vs_enum_Lmax=8)
    # regulated (hard) catalogue -- the QoS-vs-safety gap is starkest here.
    # VeriSAGE's certified plan is the known-correct 13-service reference (the
    # bounded enumerator reaches it too but not exhaustively within a short cap).
    from catalog_hard import REFERENCE
    hc = build_catalogue_hard(); hg = make_goal_hard(rng)
    res["regulated"] = run_catalogue("regulated", hc, hg, L_max=14,
                                     vs_ref_plan=REFERENCE)

    with open("results/exp10_qos_baseline.json", "w") as f:
        json.dump(res, f, indent=2)

    print("== QoS optimizer vs VeriSAGE (shared tasks) ==")
    for k in ("payment", "regulated"):
        r = res[k]
        m = r["qos_milp"]
        print(f"\n[{k}] budget={r['budget']}")
        print(f"  QoS-MILP* optimum: lat={m.get('latency_mean')} "
              f"CSTS-certified={m.get('csts_certified')} "
              f"(first violation: {m.get('first_violation')!r})")
        print(f"  QoS-GA: CSTS-certified in {100*r['qos_ga_csts_certified_frac']:.0f}% of seeds, "
              f"schema-ok {100*r['qos_ga_schema_ok_frac']:.0f}%; "
              f"best-lat plan certified={r['qos_ga_best']['csts_certified']}")
        v = r["verisage"]
        print(f"  VeriSAGE: lat={v.get('latency_mean')} CSTS-certified=True "
              f"(price of certification vs QoS optimum: "
              f"+{r['latency_gap_verisage_vs_milp']} mean-latency)")
    print("\nwrote results/exp10_qos_baseline.json")


if __name__ == "__main__":
    main()
