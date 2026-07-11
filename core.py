"""
VeriSAGE core: an executable prototype of the Contracted Service Transition
System (CSTS) and the five-check verifier (C1-C5) described in the manuscript.

This is a *real* implementation (pure Python + numpy) used to produce measured
numbers for the paper's verifier-accuracy, SLA-calibration, scalability, and
repair-convergence results. It runs SEQUENTIAL compositions, which is the exact
case for the SLA theorem (critical path = the whole chain); the DAG union bound
of the manuscript is a strict generalization and is not exercised here.

Design mirrors the paper:
  - a contracted service carries In/Out field sets, an action label, pre/post
    propositions, a field-dependency (taint) map with declassifiers, a bounded
    QoS latency (support [0, ub]), field privacy levels, and a policy guard;
  - a state carries the available-field record R(q) and the true-proposition set;
  - admissibility = C1 (schema) & C2 (functional) & C3 (LTL_f ordering) &
    C4 (explicit-flow) & C5 (policy + Hoeffding SLA bound).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional
import math

# privacy lattice: public < internal < pii < secret
LEVELS = {"public": 0, "internal": 1, "pii": 2, "secret": 3}


@dataclass
class Service:
    name: str
    action: str                       # LTL_f event label
    requires: Set[str]                # input fields needed available
    produces: Set[str]                # output fields added to R(q)
    pre_props: Set[str] = field(default_factory=set)
    post_props: Set[str] = field(default_factory=set)
    lat_ub: float = 100.0             # latency support upper bound u_i (ms)
    lat_mean: float = 50.0            # E[latency]; sampled Uniform[0, ub] => ub/2
    taint: Dict[str, Set[str]] = field(default_factory=dict)   # out_field -> in_fields
    declassifies: Set[str] = field(default_factory=set)        # fields lowered to public
    field_level: Dict[str, str] = field(default_factory=dict)  # produced field -> level
    guard_ok: bool = True             # policy guard evaluates allow under context
    sink_field: Optional[str] = None  # if a sink, the field it exports
    sink_clearance: str = "secret"    # clearance of the sink


@dataclass
class Goal:
    init_fields: Set[str]
    goal_fields: Set[str]
    goal_props: Set[str]
    phi: List[Tuple[str, str, str]]   # ("precede",a,b) or ("response",a,b) or ("absence",a,"")
    budget: float                     # SLA latency budget B_ell
    delta: float = 0.05               # target risk
    init_level: Dict[str, str] = field(default_factory=dict)


@dataclass
class Verdict:
    ok: bool
    check: str = ""                   # first failing check id ("C1".."C5") or ""
    witness: str = ""                 # typed counterexample description
    sla_bound: float = 1.0            # certified Hoeffding upper bound
    mu: float = 0.0
    proof_coverage: float = 1.0       # fraction of declared hard constraints certified


def hoeffding_bound(mu: float, ubs: List[float], budget: float) -> float:
    """P[sum latency > budget] <= exp(-2 (B-mu)^2 / sum u_i^2), for B>=mu."""
    if budget < mu:
        return 1.0
    denom = sum(u * u for u in ubs)
    if denom == 0:
        return 0.0
    return math.exp(-2.0 * (budget - mu) ** 2 / denom)


def verify(plan: List[Service], goal: Goal) -> Verdict:
    """Run C1-C5 in increasing cost order; short-circuit on first failure."""
    declared = 0   # count of declared hard constraints (for proof coverage)
    discharged = 0

    # rolling state
    avail: Set[str] = set(goal.init_fields)
    props: Set[str] = set()
    level: Dict[str, str] = dict(goal.init_level)

    # ---- C1 schema: R(q) ⪯ In(s) i.e. every required input is available ----
    for s in plan:
        declared += 1
        missing = s.requires - avail
        if missing:
            return Verdict(False, "C1", f"{s.name} missing input(s) {sorted(missing)}",
                           proof_coverage=discharged / max(declared, 1))
        discharged += 1
        avail |= s.produces
        for f, lv in s.field_level.items():
            level[f] = lv
    declared += 1
    if not goal.goal_fields <= avail:              # goal schema R(q_k) ⪯ Out_G
        return Verdict(False, "C1", f"goal fields {sorted(goal.goal_fields - avail)} not produced",
                       proof_coverage=discharged / max(declared, 1))
    discharged += 1

    # ---- C2 functional: pre before, post at end ----
    props = set()
    for s in plan:
        declared += 1
        if not s.pre_props <= props:
            return Verdict(False, "C2", f"{s.name} precondition {sorted(s.pre_props - props)} unmet",
                           proof_coverage=discharged / max(declared, 1))
        discharged += 1
        props |= s.post_props
    declared += 1
    if not goal.goal_props <= props:
        return Verdict(False, "C2", f"goal postcondition {sorted(goal.goal_props - props)} unmet",
                       proof_coverage=discharged / max(declared, 1))
    discharged += 1

    # ---- C3 ordering: finite-trace LTL_f over the action word ----
    word = [s.action for s in plan]
    for kind, a, b in goal.phi:
        declared += 1
        if kind == "precede":          # every b preceded by an a
            seen_a = False
            bad = False
            for ev in word:
                if ev == a:
                    seen_a = True
                if ev == b and not seen_a:
                    bad = True
                    break
            if bad:
                return Verdict(False, "C3", f"'{b}' occurs before '{a}'",
                               proof_coverage=discharged / max(declared, 1))
        elif kind == "response":       # every a eventually followed by b
            bad = False
            for i, ev in enumerate(word):
                if ev == a and b not in word[i + 1:]:
                    bad = True
                    break
            if bad:
                return Verdict(False, "C3", f"'{a}' not followed by '{b}'",
                               proof_coverage=discharged / max(declared, 1))
        elif kind == "absence":
            if a in word:
                return Verdict(False, "C3", f"forbidden action '{a}' present",
                               proof_coverage=discharged / max(declared, 1))
        discharged += 1

    # ---- C4 explicit-flow: no high field reaches a lower-clearance sink ----
    # effective level, lowered by declassifiers as we walk the plan.
    eff: Dict[str, str] = dict(goal.init_level)
    for s in plan:
        for f, lv in s.field_level.items():
            eff[f] = lv
        for f in s.declassifies:
            eff[f] = "public"
        if s.sink_field is not None:
            declared += 1
            fl = LEVELS.get(eff.get(s.sink_field, "public"), 0)
            cl = LEVELS.get(s.sink_clearance, 3)
            if fl > cl:
                return Verdict(False, "C4",
                               f"field '{s.sink_field}' ({eff.get(s.sink_field)}) "
                               f"reaches sink '{s.name}' cleared for {s.sink_clearance}",
                               proof_coverage=discharged / max(declared, 1))
            discharged += 1

    # ---- C5 policy: every guard allows ----
    for s in plan:
        declared += 1
        if not s.guard_ok:
            return Verdict(False, "C5", f"policy guard of '{s.name}' denies",
                           proof_coverage=discharged / max(declared, 1))
        discharged += 1

    # ---- C5 SLA: certified Hoeffding bound <= delta ----
    declared += 1
    mu = sum(s.lat_mean for s in plan)
    ubs = [s.lat_ub for s in plan]
    bound = hoeffding_bound(mu, ubs, goal.budget)
    if bound > goal.delta:
        return Verdict(False, "C5", f"SLA bound {bound:.3f} > delta {goal.delta}",
                       sla_bound=bound, mu=mu,
                       proof_coverage=discharged / max(declared, 1))
    discharged += 1

    return Verdict(True, "", "", sla_bound=bound, mu=mu, proof_coverage=1.0)


def verify_all(plan: List[Service], goal: Goal) -> List[str]:
    """Collect ALL violation witnesses across C1-C5 (no short-circuit), so a
    repair round gets the complete diagnostic. Empty list => admissible."""
    w: List[str] = []
    avail: Set[str] = set(goal.init_fields)
    level: Dict[str, str] = dict(goal.init_level)
    for s in plan:                                   # C1 schema (per edge)
        miss = s.requires - avail
        if miss:
            w.append(f"[C1] {s.name} needs input(s) {sorted(miss)} not yet produced")
        avail |= s.produces
    if not goal.goal_fields <= avail:
        w.append(f"[C1] goal outputs {sorted(goal.goal_fields - avail)} never produced")
    props: Set[str] = set()                          # C2 functional
    for s in plan:
        if not s.pre_props <= props:
            w.append(f"[C2] {s.name} precondition {sorted(s.pre_props - props)} unmet")
        props |= s.post_props
    if not goal.goal_props <= props:
        w.append(f"[C2] goal postcondition {sorted(goal.goal_props - props)} unmet")
    word = [s.action for s in plan]                  # C3 ordering
    for kind, a, b in goal.phi:
        if kind == "precede":
            seen = False
            for ev in word:
                if ev == a: seen = True
                if ev == b and not seen:
                    w.append(f"[C3] '{b}' occurs before required '{a}'"); break
        elif kind == "response":
            for i, ev in enumerate(word):
                if ev == a and b not in word[i + 1:]:
                    w.append(f"[C3] '{a}' not eventually followed by '{b}'"); break
        elif kind == "absence" and a in word:
            w.append(f"[C3] forbidden action '{a}' present")
    eff: Dict[str, str] = dict(goal.init_level)      # C4 explicit-flow
    for s in plan:
        for f, lv in s.field_level.items(): eff[f] = lv
        for f in s.declassifies: eff[f] = "public"
        if s.sink_field is not None:
            if LEVELS.get(eff.get(s.sink_field, "public"), 0) > LEVELS.get(s.sink_clearance, 3):
                w.append(f"[C4] '{s.sink_field}' reaches sink '{s.name}' un-declassified")
    for s in plan:                                   # C5 policy
        if not s.guard_ok:
            w.append(f"[C5] policy guard of '{s.name}' denies")
    mu = sum(s.lat_mean for s in plan)               # C5 SLA
    bound = hoeffding_bound(mu, [s.lat_ub for s in plan], goal.budget)
    if bound > goal.delta:
        w.append(f"[C5] SLA bound {bound:.3f} > {goal.delta} (too many services; remove unneeded ones)")
    return w


def empirical_sla_violation(plan: List[Service], budget: float, rng, n: int = 2000) -> float:
    """Monte-Carlo P[sum latency > budget], latency_i ~ Uniform(0, u_i)."""
    ubs = [s.lat_ub for s in plan]
    draws = rng.random((n, len(ubs))) * ubs           # broadcast Uniform[0,u_i]
    totals = draws.sum(axis=1)
    return float((totals > budget).mean())
