"""
EXP8 -- Full end-to-end verification timing with realistic SMT clause loads
(Stanford review Q7).

The manuscript reports MICROSECOND structural verification times (RQ3). Q7 asks
for the honest END-TO-END time once C2 discharges real QF-LIA+EUF pre/post-
conditions with an SMT solver, over a realistic clause-count distribution.

This experiment composes two ALREADY-MEASURED ingredients from exp3_scalability
(results/exp3_scalability.json), so it needs no heavy re-timing and no fabrication:
  (i)  the measured STRUCTURAL per-check breakdown (C1,C3,C4,C5 + structural-C2),
       microseconds, from "breakdown_k200" and the per-service structural slope
       ("linfit_slope_us_per_service"); and
  (ii) the measured Z3 per-service SMT cost vs clause/atom count ("z3_c2" curve,
       z3 4.15.4 on the author workstation).

For each representative plan length k we sample a realistic per-service atom count
from three regimes (light/mixed/heavy), price each service's C2 by interpolating
the measured Z3 curve, add the measured structural cost of the non-C2 checks, and
report end-to-end mean / p50 / p95 in MILLISECONDS. The source of every input
number is recorded under "sources".

Outputs results/exp8_endtoend.json.
"""
from __future__ import annotations
import json, os
import numpy as np

E3 = "../results/exp3_scalability.json"


def _structural_nonc2_us_per_service(e3):
    """Per-service structural cost of C1+C3+C4+C5 (exclude C2), from the measured
    k=200 breakdown, in microseconds/service."""
    b = e3["breakdown_k200"]["per_check_us"]
    k = e3["breakdown_k200"]["k"]
    return (b["C1"] + b["C3"] + b["C4"] + b["C5"]) / k


def _smt_curve(e3):
    z = e3.get("z3_c2", {})
    if not z.get("available"):
        return None, "z3 curve unavailable"
    return ({int(r["n_atoms"]): r["ms_mean"] for r in z["curve"]},
            f"measured z3 {z.get('z3_version','?')} (author workstation)")


def _smt_cost_ms(n_atoms, curve):
    xs = sorted(curve)
    if n_atoms <= xs[0]:
        return curve[xs[0]]
    if n_atoms >= xs[-1]:
        x0, x1 = xs[-2], xs[-1]
        slope = (curve[x1] - curve[x0]) / (x1 - x0)
        return curve[x1] + slope * (n_atoms - x1)
    for i in range(len(xs) - 1):
        if xs[i] <= n_atoms <= xs[i + 1]:
            f = (n_atoms - xs[i]) / (xs[i + 1] - xs[i])
            return curve[xs[i]] + f * (curve[xs[i + 1]] - curve[xs[i]])
    return curve[xs[-1]]


def endtoend(ks=(6, 13, 25, 50, 100), seed=51):
    rng = np.random.default_rng(seed)
    e3 = json.load(open(E3))
    struct_us_ps = _structural_nonc2_us_per_service(e3)
    curve, src = _smt_curve(e3)
    dists = {
        "light": lambda: int(max(1, rng.poisson(3))),
        "mixed": lambda: int(max(1, rng.poisson(8))),
        "heavy": lambda: int(max(1, np.clip(rng.normal(25, 8), 1, 60))),
    }
    out = {"sources": {"structural": "measured exp3 breakdown_k200 (us)",
                       "smt_c2": src},
           "structural_nonC2_us_per_service": round(struct_us_ps, 4),
           "by_k": []}
    for k in ks:
        struct_ms = struct_us_ps * k / 1e3            # non-C2 structural, ms
        rec = {"k": k, "structural_nonC2_ms": round(struct_ms, 5), "regimes": {}}
        if curve is None:
            rec["regimes"] = {"note": "z3 curve unavailable"}
            out["by_k"].append(rec); continue
        for name, sampler in dists.items():
            totals = []
            for _ in range(4000):
                clauses = [sampler() for _ in range(k)]
                smt_ms = sum(_smt_cost_ms(c, curve) for c in clauses)
                totals.append(struct_ms + smt_ms)
            totals = np.array(totals)
            rec["regimes"][name] = {
                "mean_ms": round(float(totals.mean()), 4),
                "p50_ms": round(float(np.percentile(totals, 50)), 4),
                "p95_ms": round(float(np.percentile(totals, 95)), 4),
                "mean_ms_per_service": round(float(totals.mean() / k), 4),
            }
        out["by_k"].append(rec)
    return out


def main():
    os.makedirs("../results", exist_ok=True)
    res = endtoend()
    with open("../results/exp8_endtoend.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"structural nonC2 = {res['structural_nonC2_us_per_service']} us/service")
    print(f"SMT source: {res['sources']['smt_c2']}")
    print("== end-to-end verify time (structural + real Z3 C2), ms ==")
    print(f"  {'k':>4} {'struct(nonC2)':>13} | light p50/p95 | mixed p50/p95 | heavy p50/p95 (ms)")
    for r in res["by_k"]:
        g = r["regimes"]
        if "light" in g:
            print(f"  {r['k']:>4} {r['structural_nonC2_ms']:>13.4f} | "
                  f"{g['light']['p50_ms']:.2f}/{g['light']['p95_ms']:.2f} | "
                  f"{g['mixed']['p50_ms']:.2f}/{g['mixed']['p95_ms']:.2f} | "
                  f"{g['heavy']['p50_ms']:.2f}/{g['heavy']['p95_ms']:.2f}")
    print("\nwrote ../results/exp8_endtoend.json")


if __name__ == "__main__":
    main()
