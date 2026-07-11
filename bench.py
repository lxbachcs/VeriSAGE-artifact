"""
Synth-TSR generator: composition tasks with GROUND-TRUTH admissible/inadmissible
labels, obtained by construction (clean plans vs. plans with one injected
violation of a chosen check). Used to measure real verifier accuracy, and to
drive the synthetic propose-verify-repair loop.

Domain mirrors the manuscript's worked example (payment + PII handling) with
randomized filler services so plan length and difficulty vary.
"""
from __future__ import annotations
from typing import List, Tuple
import copy
from core import Service, Goal, verify

VIOLATIONS = ["C1", "C2", "C3", "C4", "C5pol", "C5sla"]


def _filler(i: int, produces: str, requires_prev: str, ub: float) -> Service:
    return Service(
        name=f"prep{i}", action=f"prep{i}",
        requires={requires_prev} if requires_prev else set(),
        produces={produces},
        lat_ub=ub, lat_mean=ub / 2.0,
        field_level={produces: "internal"},
    )


def _payment_core(rng) -> Tuple[List[Service], Goal]:
    """A clean, admissible payment composition with n_filler prep steps."""
    n_filler = rng.integers(0, 6)
    ub = float(rng.integers(20, 60))

    plan: List[Service] = []
    prev = ""
    # filler chain producing fields d0..d{n-1}, so plan length varies
    for i in range(int(n_filler)):
        s = _filler(i, f"d{i}", prev, ub)
        plan.append(s)
        prev = f"d{i}"

    authenticate = Service(
        "authenticate", "authenticate", requires=set(), produces={"token"},
        post_props={"authed"}, lat_ub=ub, lat_mean=ub / 2.0,
        field_level={"token": "internal"})
    tokenize = Service(
        "tokenize", "tokenize", requires={"card.pan"}, produces={"card.tok"},
        lat_ub=ub, lat_mean=ub / 2.0, declassifies={"card.pan"},
        field_level={"card.tok": "public"})
    charge = Service(
        "chargeCard", "charge", requires={"token", "card.tok"}, produces={"charge.id"},
        pre_props={"authed"}, post_props={"charged"}, lat_ub=ub, lat_mean=ub / 2.0,
        field_level={"charge.id": "internal"})
    receipt = Service(
        "emailReceipt", "receipt", requires={"charge.id"}, produces={"receipt.id"},
        pre_props={"charged"}, post_props={"receipted"}, lat_ub=ub, lat_mean=ub / 2.0,
        field_level={"receipt.id": "public"})
    analytics = Service(
        "analyticsSink", "analytics", requires={"card.tok"}, produces={"metric"},
        lat_ub=ub, lat_mean=ub / 2.0, sink_field="card.pan", sink_clearance="public")

    # provide card.pan at the start (pii) via a card-reader service
    reader = Service("readCard", "readcard", requires=set(), produces={"card.pan"},
                     lat_ub=ub, lat_mean=ub / 2.0, field_level={"card.pan": "pii"})

    plan += [reader, authenticate, tokenize, charge, receipt, analytics]

    mu = sum(s.lat_mean for s in plan)
    budget = mu + rng.uniform(1.5, 4.0) * (sum(s.lat_ub ** 2 for s in plan)) ** 0.5
    goal = Goal(
        init_fields=set(), goal_fields={"receipt.id"}, goal_props={"receipted"},
        phi=[("precede", "authenticate", "charge"),
             ("response", "charge", "receipt")],
        budget=float(budget), delta=0.05,
        init_level={})
    return plan, goal


def gen_task(rng, force_admissible: bool = None) -> Tuple[List[Service], Goal, bool, str]:
    """Return (plan, goal, is_admissible_label, violation_type)."""
    plan, goal = _payment_core(rng)
    make_bad = (not force_admissible) if force_admissible is not None else bool(rng.integers(0, 2))
    if not make_bad:
        return plan, goal, True, "none"

    v = VIOLATIONS[rng.integers(0, len(VIOLATIONS))]
    p = copy.deepcopy(plan)
    if v == "C1":                      # drop the card reader -> missing input
        p = [s for s in p if s.name != "readCard"]
    elif v == "C2":                    # drop authenticate -> 'authed' precondition unmet
        p = [s for s in p if s.name != "authenticate"]
        # keep 'token' producer removed too so charge's C1 still... no: keep token via readCard?
        # authenticate produced token; removing it also drops token. Re-add a token producer
        # so the failure is isolated to C2 (pre_props), not C1.
        tok = Service("mintToken", "minttoken", requires=set(), produces={"token"},
                      lat_ub=30, lat_mean=15, field_level={"token": "internal"})
        p = [tok] + p
    elif v == "C3":                    # move charge before authenticate
        names = [s.name for s in p]
        ci, ai = names.index("chargeCard"), names.index("authenticate")
        if ci > ai:
            p[ai], p[ci] = p[ci], p[ai]
    elif v == "C4":                    # drop tokenize -> pii card.pan reaches public sink
        p = [s for s in p if s.name != "tokenize"]
        # charge requires card.tok; give it card.pan instead so C1 still passes and C4 is the failure
        for s in p:
            if s.name == "chargeCard":
                s.requires = {"token", "card.pan"}
            if s.name == "analyticsSink":
                s.requires = {"card.pan"}
    elif v == "C5pol":                 # a guard denies
        for s in p:
            if s.name == "chargeCard":
                s.guard_ok = False
    elif v == "C5sla":                 # tighten budget below the certifiable bound
        goal = Goal(**{**goal.__dict__})
        goal.budget = goal.budget * 0.35
    return p, goal, False, v


def build_suite(rng, n: int) -> List[Tuple[List[Service], Goal, bool, str]]:
    return [gen_task(rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# Synthetic propose-verify-repair loop (no LLM): a heuristic proposer that may
# emit ordering/flow mistakes, corrected by the verifier's typed counterexamples.
# ---------------------------------------------------------------------------
def synthetic_repair_loop(rng, goal: Goal, T: int = 6):
    """
    Start from a plausible-but-often-wrong first proposal (the payment core with
    random perturbations); on each failure, apply the repair implied by the typed
    counterexample and re-propose. Returns (certified, iters).
    """
    base, _ = _payment_core(rng)          # a correct scaffold to repair toward
    # first proposal: randomly perturb (drop tokenize / reorder / drop reader),
    # emulating an LLM's plausible-but-unverified guess.
    plan = copy.deepcopy(base)
    forbidden = set()                     # accumulated hard constraints (R)

    # perturbations, each fixable by one repair
    perturb = rng.random()
    if perturb < 0.30:                    # reorder charge before authenticate
        names = [s.name for s in plan]
        ci, ai = names.index("chargeCard"), names.index("authenticate")
        plan[ai], plan[ci] = plan[ci], plan[ai]
    if rng.random() < 0.30:               # drop tokenize (pii leak)
        plan = [s for s in plan if s.name != "tokenize"]
        for s in plan:
            if s.name in ("chargeCard", "analyticsSink"):
                s.requires = {x if x != "card.tok" else "card.pan" for x in s.requires}
    if rng.random() < 0.15:               # drop reader (missing input)
        plan = [s for s in plan if s.name != "readCard"]

    for it in range(T):
        v = verify(plan, goal)
        if v.ok:
            return True, it
        # apply the repair implied by the counterexample (CEGIS step)
        plan = _apply_repair(plan, base, goal, v, forbidden, rng)
        if plan is None:
            return False, it + 1          # proposer exhausted -> abstain/infeasible
    v = verify(plan, goal)
    return (v.ok, T)


def _apply_repair(plan, base, goal, v, forbidden, rng):
    """Turn a typed counterexample into a targeted fix toward the correct scaffold."""
    names = [s.name for s in plan]
    if v.check == "C3":                   # fix ordering: authenticate before charge
        if "authenticate" in names and "chargeCard" in names:
            ai, ci = names.index("authenticate"), names.index("chargeCard")
            if ai > ci:
                plan[ci], plan[ai] = plan[ai], plan[ci]
                return plan
    if v.check == "C4":                   # insert tokenize declassifier, restore card.tok
        if not any(s.name == "tokenize" for s in plan):
            tok = next(s for s in copy.deepcopy(base) if s.name == "tokenize")
            # insert before the sink
            idx = names.index("analyticsSink") if "analyticsSink" in names else len(plan)
            plan.insert(idx, tok)
            for s in plan:
                if s.name in ("chargeCard", "analyticsSink"):
                    s.requires = {x if x != "card.pan" else "card.tok" for x in s.requires}
            return plan
    if v.check in ("C1", "C2"):           # add the missing producer from the scaffold
        for s in copy.deepcopy(base):
            if s.name not in names:
                return [s] + plan
    if v.check == "C5":                   # relax nothing we can; give up (abstain)
        return None
    # default: rebuild from the correct scaffold (last-resort repair)
    return copy.deepcopy(base)
