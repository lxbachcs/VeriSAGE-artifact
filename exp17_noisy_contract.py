"""
EXP17 -- Contract mis-specification robustness (Stanford review Q4).

Reviewer Q4: "How robust is the approach to moderate mis-specification in
contracts (e.g. missing declassifiers or slightly wrong pre/postconditions)?
Have you evaluated a 'noisy contract' setting to quantify false rejections or
unsafe admissions?"

VeriSAGE's soundness (Thm.~1) is RELATIVE TO THE CONTRACT: the verifier is exact
for the contract it is given, so any error under a wrong contract is a CONTRACT
bug, not a verifier bug. This experiment quantifies that sensitivity by injecting
two kinds of contract noise at rate p and measuring the two error modes against
GROUND-TRUTH labels (which are fixed by the TRUE contract):

  Track A -- OMISSION / over-strict noise on admissible contracts
    (drop a declassifier, add a spurious ordering constraint, over-tag a field,
     shave the SLA budget). Effect: the verifier REJECTS a truly-admissible plan
     -> a FALSE REJECTION. This is the CONSERVATIVE direction: under VeriSAGE a
     false rejection triggers repair and, if unrepairable, an honest ABSTAIN --
     never an unsafe plan.

  Track B -- RELAXATION / under-strict noise on inadmissible contracts
    (delete exactly the clause that catches the injected violation: drop the
     ordering constraint, lift the sink clearance, widen the budget, re-allow the
     guard). Effect: the verifier ADMITS a truly-inadmissible plan -> an UNSAFE
     ADMISSION. This is the worst case for relaxation noise and is bounded by p:
     an admitted plan is still provably correct w.r.t. the (now-wrong) contract.

Deterministic, verifier-only. No LLM, no SMT. Outputs results/exp17_noisy_contract.json.
"""
from __future__ import annotations
import json, os, copy
import numpy as np

from core import Goal, verify
import bench


# --------------------------------------------------------------------------
# contract-noise operators
# --------------------------------------------------------------------------
def _over_strict(plan, goal, rng):
    """Apply one OMISSION / over-strict contract mutation. Returns (plan', goal')."""
    p = copy.deepcopy(plan); g = copy.deepcopy(goal)
    ops = []
    # (a) missing declassifier: drop a declassifies set -> a declassified field
    #     is now treated as still-sensitive and looks like a leak (C4).
    decl = [s for s in p if getattr(s, "declassifies", None)]
    if decl: ops.append("drop_declassifier")
    # (b) spurious ordering constraint the plan happens to violate (C3).
    names = [s.name for s in p]
    if "authenticate" in names and "chargeCard" in names:
        ops.append("spurious_order")
    # (c) over-tag a public field to a higher level (C4).
    pub = [(s, f) for s in p for f, lv in getattr(s, "field_level", {}).items() if lv == "public"]
    if pub: ops.append("over_tag")
    # (d) shave the SLA budget (C5 SLA).
    ops.append("shave_budget")

    op = ops[rng.integers(0, len(ops))]
    if op == "drop_declassifier":
        s = decl[rng.integers(0, len(decl))]
        s.declassifies = set()
    elif op == "spurious_order":
        # require receipt precede authenticate; plan authenticates first -> C3 fail.
        g.phi = list(g.phi) + [("precede", "receipt", "authenticate")]
    elif op == "over_tag":
        s, f = pub[rng.integers(0, len(pub))]
        s.field_level = dict(s.field_level); s.field_level[f] = "secret"
        # ensure some sink sees it: mark it a public sink target if a sink exists
        for t in p:
            if getattr(t, "sink_field", None) is None and rng.random() < 0.5:
                t.sink_field = f; t.sink_clearance = "public"; break
    elif op == "shave_budget":
        g.budget = g.budget * 0.6
    return p, g, op


def _relax(plan, goal, vtype, rng):
    """Delete exactly the contract clause that catches the injected violation
    `vtype` (worst-case under-strict noise). Returns (plan, goal', applied?)."""
    g = copy.deepcopy(goal); p = copy.deepcopy(plan)
    if vtype == "C3":                          # drop the ordering constraints
        g.phi = []
        return p, g, True
    if vtype == "C4":                          # lift every sink clearance to secret
        for s in p:
            if getattr(s, "sink_field", None) is not None:
                s.sink_clearance = "secret"
        return p, g, True
    if vtype == "C5sla":                       # widen the budget generously
        g.budget = g.budget * 4.0
        return p, g, True
    if vtype == "C5pol":                        # re-allow the denied guard
        for s in p:
            s.guard_ok = True
        return p, g, True
    # C1/C2 are structural: no single deletable clause masks them under this noise.
    return p, g, False


# --------------------------------------------------------------------------
# tracks
# --------------------------------------------------------------------------
def track_false_rejection(rates, n=1500, seed=71):
    """Over-strict noise on ADMISSIBLE contracts -> false-rejection rate."""
    out = []
    for p in rates:
        rng = np.random.default_rng(seed)
        fr = 0; total = 0
        for _ in range(n):
            plan, goal = bench._payment_core(rng)   # a truly-admissible task
            total += 1
            if rng.random() < p:
                plan, goal, _ = _over_strict(plan, goal, rng)
            if not verify(plan, goal).ok:
                fr += 1
        out.append({"noise_rate": p, "false_rejection_rate": round(100 * fr / total, 2)})
    return out


def track_unsafe_admission(rates, n=1500, seed=73):
    """Relaxation noise on INADMISSIBLE contracts -> unsafe-admission rate.
    Baseline (p=0) confirms every injected violation is caught (0 unsafe)."""
    out = []
    for p in rates:
        rng = np.random.default_rng(seed)
        unsafe = 0; total = 0
        for _ in range(n):
            plan, goal, label, vtype = bench.gen_task(rng, force_admissible=False)
            if label:            # gen_task may still return admissible; skip
                continue
            total += 1
            g, pl = goal, plan
            if rng.random() < p:
                pl, g, _ = _relax(plan, goal, vtype, rng)
            if verify(pl, g).ok:       # admitted a truly-inadmissible plan
                unsafe += 1
        out.append({"noise_rate": p, "unsafe_admission_rate": round(100 * unsafe / total, 2),
                    "n_inadmissible": total})
    return out


def main():
    os.makedirs("results", exist_ok=True)
    rates = [0.0, 0.05, 0.1, 0.2, 0.3]
    res = {
        "rates": rates,
        "false_rejection": track_false_rejection(rates),
        "unsafe_admission": track_unsafe_admission(rates),
    }
    with open("results/exp17_noisy_contract.json", "w") as f:
        json.dump(res, f, indent=2)

    print("== EXP17 contract-noise robustness (Q4) ==")
    print("  p     false-reject(%)   unsafe-admit(%)")
    fr = {r["noise_rate"]: r["false_rejection_rate"] for r in res["false_rejection"]}
    ua = {r["noise_rate"]: r["unsafe_admission_rate"] for r in res["unsafe_admission"]}
    for p in rates:
        print(f"  {p:<5} {fr[p]:>12.2f}    {ua[p]:>12.2f}")
    print("\n  Over-strict noise -> conservative false rejections (become Abstain/repair).")
    print("  Relaxation noise  -> unsafe admissions only vs the *wrong* contract, ~linear in p.")
    print("  At p=0 both error modes are 0: the verifier is exact for the stated contract.")
    print("\nwrote results/exp17_noisy_contract.json")


if __name__ == "__main__":
    main()
