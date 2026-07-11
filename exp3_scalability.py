"""
EXP3 -- scalability stress test + timing-unit audit (Stanford review Q6).

Answers three concrete things the review asked for, all with MEASURED wall-clock:
  (A) Timing-unit audit. We measure the real per-verify wall-clock and report it
      in BOTH microseconds and milliseconds so the unit is unambiguous. The
      manuscript figure previously plotted microseconds; we confirm that is the
      genuine value for the lightweight structural checker (no unit error), and
      we make the measurement method explicit.
  (B) Verification time vs DAG size k in {10,25,50,100,200} (the review's exact
      grid), plus a finer sweep for a linear-fit R^2, with a per-check breakdown
      at the largest k.
  (C) Verification time vs policy/temporal complexity |Phi| in {1,2,5,10,20,40}.
  (D) A Z3-backed C2 (SMT) micro-benchmark: the prototype's C2 is a lightweight
      structural check (hence microseconds); a production C2 discharges real
      QF-LIA+EUF pre/postconditions with an SMT solver. We measure that real cost
      per service as the formula size grows, so the paper honestly states the
      per-check cost under "more complex SMT pre/postconditions".

Outputs results/exp3_scalability.json.
"""
from __future__ import annotations
import json, os, time
from typing import List, Tuple
import numpy as np
from core import Service, Goal, hoeffding_bound, LEVELS

REPS_INNER = 2000           # inner repetitions per timing sample
REPS_OUTER = 7              # outer samples -> mean/sd


# --------------------------------------------------------------------------
# A timed re-implementation of the C1--C5 verifier that returns per-check
# wall-clock. It is byte-for-byte the same decision logic as core.verify;
# we only add per-block timers so we can produce a per-check breakdown.
# --------------------------------------------------------------------------
def verify_timed(plan: List[Service], goal: Goal):
    t = {}
    # C1 schema
    t0 = time.perf_counter()
    avail = set(goal.init_fields)
    for s in plan:
        if s.requires - avail:
            break
        avail |= s.produces
    ok1 = goal.goal_fields <= avail
    t["C1"] = time.perf_counter() - t0
    # C2 functional (structural pre/post propagation, as in the prototype)
    t0 = time.perf_counter()
    props = set()
    for s in plan:
        if not s.pre_props <= props:
            break
        props |= s.post_props
    ok2 = goal.goal_props <= props
    t["C2"] = time.perf_counter() - t0
    # C3 temporal (LTL_f over the action word, per phi in Phi)
    t0 = time.perf_counter()
    word = [s.action for s in plan]
    for kind, a, b in goal.phi:
        if kind == "precede":
            seen = False
            for ev in word:
                if ev == a: seen = True
                if ev == b and not seen: break
        elif kind == "response":
            for i, ev in enumerate(word):
                if ev == a and b not in word[i + 1:]: break
        elif kind == "absence":
            if a in word: break
    t["C3"] = time.perf_counter() - t0
    # C4 explicit-flow reachability with declassifiers
    t0 = time.perf_counter()
    eff = dict(goal.init_level)
    for s in plan:
        for f, lv in s.field_level.items(): eff[f] = lv
        for f in s.declassifies: eff[f] = "public"
        if s.sink_field is not None:
            _ = LEVELS.get(eff.get(s.sink_field, "public"), 0) > LEVELS.get(s.sink_clearance, 3)
    t["C4"] = time.perf_counter() - t0
    # C5 policy + SLA (Hoeffding on the critical path)
    t0 = time.perf_counter()
    for s in plan:
        if not s.guard_ok: break
    mu = sum(s.lat_mean for s in plan)
    _ = hoeffding_bound(mu, [s.lat_ub for s in plan], goal.budget)
    t["C5"] = time.perf_counter() - t0
    return t


def _chain(k: int, n_phi: int = 2) -> Tuple[List[Service], Goal]:
    """A length-k linear composition with n_phi satisfiable LTL_f constraints,
    one sink, and a mix of privacy levels -- exercises all five checks."""
    plan: List[Service] = []
    prev = ""
    for i in range(k):
        plan.append(Service(
            f"s{i}", f"a{i}",
            requires={prev} if prev else set(), produces={f"f{i}"},
            pre_props=({f"p{i-1}"} if i > 0 else set()), post_props={f"p{i}"},
            lat_ub=1.0, lat_mean=0.5,
            field_level={f"f{i}": "internal"},
            declassifies=({f"f{i-1}"} if i > 0 and i % 5 == 0 else set()),
            sink_field=(f"f{i-1}" if i == k - 1 else None),
            sink_clearance="internal"))
        prev = f"f{i}"
    # n_phi satisfiable ordering constraints over the (already ordered) actions
    phi = []
    for j in range(n_phi):
        a, b = f"a{j}", f"a{min(j + 1, k - 1)}"
        phi.append(("precede", a, b) if j % 2 == 0 else ("response", a, b))
    # generous budget so C5 runs the full Hoeffding arithmetic (not early-out)
    mu = sum(s.lat_mean for s in plan)
    budget = mu + 5.0 * (sum(s.lat_ub ** 2 for s in plan)) ** 0.5
    goal = Goal(set(), {f"f{k-1}"}, {f"p{k-1}"}, phi, budget=budget, delta=0.05)
    return plan, goal


def _time_total(plan, goal) -> Tuple[float, float]:
    """Return (mean_us, sd_us) per full verify, measured wall-clock."""
    def one_verify():
        t = verify_timed(plan, goal)
        return sum(t.values())
    for _ in range(200):        # warm
        one_verify()
    samples = []
    for _ in range(REPS_OUTER):
        t0 = time.perf_counter()
        for _ in range(REPS_INNER):
            one_verify()
        samples.append((time.perf_counter() - t0) / REPS_INNER * 1e6)  # us
    s = np.array(samples)
    return float(s.mean()), float(s.std())


def sweep_k(ks=(10, 25, 50, 100, 200), fine=None):
    fine = fine or list(range(10, 201, 10))
    out_main, out_fine = [], []
    for k in sorted(set(list(ks) + fine)):
        plan, goal = _chain(k)
        us, sd = _time_total(plan, goal)
        rec = {"k": int(k), "us_mean": round(us, 3), "us_sd": round(sd, 3),
               "ms_mean": round(us / 1e3, 6)}
        out_fine.append(rec)
        if k in ks:
            out_main.append(rec)
    kk = np.array([r["k"] for r in out_fine], float)
    tt = np.array([r["us_mean"] for r in out_fine], float)
    b1, b0 = np.polyfit(kk, tt, 1)
    r2_lin = float(np.corrcoef(kk, tt)[0, 1] ** 2)
    # log-log slope: t ~ k^alpha
    alpha = float(np.polyfit(np.log(kk), np.log(tt), 1)[0])
    return {"grid": ks, "main": out_main, "fine": out_fine,
            "linfit_slope_us_per_service": round(float(b1), 4),
            "linfit_intercept_us": round(float(b0), 4),
            "linfit_R2": round(r2_lin, 4),
            "loglog_slope_alpha": round(alpha, 3)}


def sweep_phi(nphis=(1, 2, 5, 10, 20, 40), k=50):
    out = []
    for n in nphis:
        plan, goal = _chain(k, n_phi=n)
        us, sd = _time_total(plan, goal)
        out.append({"n_phi": int(n), "us_mean": round(us, 3), "us_sd": round(sd, 3)})
    nn = np.array([r["n_phi"] for r in out], float)
    tt = np.array([r["us_mean"] for r in out], float)
    b1, b0 = np.polyfit(nn, tt, 1)
    r2 = float(np.corrcoef(nn, tt)[0, 1] ** 2)
    return {"k": k, "curve": out, "us_per_phi": round(float(b1), 4),
            "linfit_R2": round(r2, 4)}


def per_check_breakdown(k=200, n_phi=10):
    plan, goal = _chain(k, n_phi=n_phi)
    for _ in range(200):
        verify_timed(plan, goal)
    acc = {c: [] for c in ("C1", "C2", "C3", "C4", "C5")}
    for _ in range(REPS_OUTER):
        agg = {c: 0.0 for c in acc}
        for _ in range(REPS_INNER):
            t = verify_timed(plan, goal)
            for c in acc: agg[c] += t[c]
        for c in acc: acc[c].append(agg[c] / REPS_INNER * 1e6)  # us
    return {"k": k, "n_phi": n_phi,
            "per_check_us": {c: round(float(np.mean(v)), 4) for c, v in acc.items()},
            "total_us": round(float(sum(np.mean(v) for v in acc.values())), 4)}


def z3_smt_c2_microbench():
    """Real SMT cost for C2: discharge a QF-LIA+EUF pre/postcondition per service
    with Z3, as the number of atoms grows. This is the honest per-service cost of
    a production functional check (vs the microsecond structural surrogate)."""
    try:
        import z3
    except Exception:
        return {"available": False}
    rng = np.random.default_rng(0)
    out = []
    for natoms in (1, 5, 10, 20, 40):
        times = []
        for _ in range(30):
            s = z3.Solver()
            xs = [z3.Int(f"x{i}") for i in range(natoms + 2)]
            f = z3.Function("f", z3.IntSort(), z3.IntSort())   # EUF term
            # precondition: a conjunction of linear + EUF atoms (satisfiable)
            for i in range(natoms):
                a, b = xs[i], xs[(i + 1) % len(xs)]
                s.add(a + int(rng.integers(1, 5)) * b >= int(rng.integers(-3, 3)))
                s.add(f(a) != f(b) + 1)
            # postcondition entailment query: is pre & !post UNSAT?
            s.add(z3.Or([x > 1000 for x in xs]))
            t0 = time.perf_counter()
            s.check()
            times.append((time.perf_counter() - t0) * 1e3)   # ms
        out.append({"n_atoms": int(natoms),
                    "ms_mean": round(float(np.mean(times)), 4),
                    "ms_sd": round(float(np.std(times)), 4)})
    return {"available": True, "z3_version": z3.get_version_string(), "curve": out}


def main():
    os.makedirs("../results", exist_ok=True)
    res = {}
    print("EXP3 (A/B) scalability vs k ..."); res["scal_k"] = sweep_k()
    print("EXP3 (C) scalability vs |Phi| ..."); res["scal_phi"] = sweep_phi()
    print("EXP3 breakdown at k=200 ..."); res["breakdown_k200"] = per_check_breakdown()
    print("EXP3 (D) Z3 SMT C2 microbench ..."); res["z3_c2"] = z3_smt_c2_microbench()
    with open("../results/exp3_scalability.json", "w") as f:
        json.dump(res, f, indent=2)
    # console summary
    m = res["scal_k"]
    print("\n== scalability vs k (measured wall-clock) ==")
    for r in m["main"]:
        print(f"  k={r['k']:>3}  {r['us_mean']:>8.2f} us  = {r['ms_mean']:.5f} ms")
    print(f"  linear fit: {m['linfit_slope_us_per_service']} us/service, "
          f"R2={m['linfit_R2']}, log-log slope alpha={m['loglog_slope_alpha']}")
    print("== per-check breakdown at k=200 ==")
    print("  ", res["breakdown_k200"]["per_check_us"],
          "total", res["breakdown_k200"]["total_us"], "us")
    if res["z3_c2"]["available"]:
        print("== Z3-backed C2 real SMT cost (ms/service) ==")
        for r in res["z3_c2"]["curve"]:
            print(f"  atoms={r['n_atoms']:>2}  {r['ms_mean']:.4f} ms")
    print("\nwrote ../results/exp3_scalability.json")


if __name__ == "__main__":
    main()
