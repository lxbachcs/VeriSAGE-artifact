"""
EXP2 -- SLA-bound tightness / conservatism analysis (Stanford review Q4).

Soundness of the Hoeffding SLA bound is already established (E2: empirical MC
violation never exceeds the certified delta). This experiment measures the OTHER
side: how *loose* the bound is, and what that conservatism costs in practice.

We report three things, all measured:
  (1) Gap: certified Hoeffding bound delta vs the empirical violation rate, under
      a realistic Uniform[0,u_i] latency model AND the two-point worst case
      {0,u_i} (which is the Hoeffding-extremal case). The ratio empirical/delta is
      the tightness.
  (2) Acceptance-rate impact: over a distribution of plans/budgets, what fraction
      of plans that TRULY meet the SLA (empirical <= delta) are nonetheless
      REJECTED by C5 because the conservative bound exceeds delta. This is the
      false-rejection rate induced purely by bound conservatism, plus the budget
      inflation factor (how much extra latency headroom Hoeffding demands).
  (3) Repair-iteration impact: a plan padded with optional services is shrunk one
      service per repair step until C5 passes; we count iterations under the
      conservative Hoeffding gate vs an oracle empirical gate.

Outputs results/exp2_sla_tightness.json.
"""
from __future__ import annotations
import json, os
import numpy as np
from core import hoeffding_bound


def _mc_violation(us: np.ndarray, budget: float, model: str, rng, n: int = 20000) -> float:
    """Empirical P[sum latency > budget]. model in {uniform, twopoint}."""
    m = len(us)
    if model == "uniform":
        draws = rng.random((n, m)) * us            # Uniform[0,u_i]
    else:                                          # two-point {0,u_i} w.p. 1/2
        draws = (rng.random((n, m)) < 0.5) * us
    return float((draws.sum(axis=1) > budget).mean())


def _mean_lat(us: np.ndarray, model: str) -> float:
    return float(0.5 * us.sum())   # both models have mean u_i/2


# --------------------------------------------------------------------------
# (1) bound-vs-empirical gap, per target delta, per latency model
# --------------------------------------------------------------------------
def gap_analysis(n_plans=3000, seed=11):
    rng = np.random.default_rng(seed)
    targets = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
    res = {}
    for model in ("uniform", "twopoint"):
        curve = []
        for t in targets:
            emps = []
            for _ in range(n_plans):
                m = int(rng.integers(3, 10))
                us = rng.uniform(0.5, 1.5, size=m)
                mu = _mean_lat(us, model)
                S2 = (us ** 2).sum()
                # budget so the certified Hoeffding bound == t
                B = mu + np.sqrt(-0.5 * S2 * np.log(t))
                emps.append(_mc_violation(us, B, model, rng, n=6000))
            emps = np.array(emps)
            curve.append({"delta": t,
                          "empirical_mean": round(float(emps.mean()), 5),
                          "empirical_p95": round(float(np.percentile(emps, 95)), 5),
                          "tightness_emp_over_delta": round(float(emps.mean() / t), 4),
                          "gap_delta_minus_emp": round(float(t - emps.mean()), 5),
                          "sound_frac": round(float((emps <= t + 1e-9).mean()), 4)})
        res[model] = curve
    return res


# --------------------------------------------------------------------------
# (2) acceptance-rate impact: conservative (false) rejections at fixed delta
# --------------------------------------------------------------------------
def acceptance_impact(delta=0.05, n_plans=8000, model="uniform", seed=12):
    rng = np.random.default_rng(seed)
    n_true_feasible = 0
    n_cons_reject = 0        # truly feasible but Hoeffding rejects
    n_hoeff_accept = 0
    infl = []                # budget inflation factor on truly-feasible plans
    for _ in range(n_plans):
        m = int(rng.integers(4, 12))
        us = rng.uniform(0.5, 1.5, size=m)
        mu = _mean_lat(us, model)
        S2 = (us ** 2).sum()
        # draw a budget spanning genuinely-infeasible..loose relative to the
        # Hoeffding-required slack (lower end makes the true tail exceed delta)
        s = rng.uniform(0.15, 2.2)
        B = mu + s * np.sqrt(0.5 * np.log(1 / delta) * S2)
        hoeff = hoeffding_bound(mu, list(us), B)
        true_v = _mc_violation(us, B, model, rng, n=20000)
        hoeff_ok = hoeff <= delta
        true_ok = true_v <= delta
        if hoeff_ok:
            n_hoeff_accept += 1
        if true_ok:
            n_true_feasible += 1
            if not hoeff_ok:
                n_cons_reject += 1
            # budget inflation: slack Hoeffding demands vs slack truth demands
            B_hoeff = mu + np.sqrt(0.5 * np.log(1 / delta) * S2)   # min B for hoeff<=delta
            # min B for true<=delta: the (1-delta) quantile of the sum (MC)
            if model == "uniform":
                draws = rng.random((20000, m)) * us
            else:
                draws = (rng.random((20000, m)) < 0.5) * us
            B_true = float(np.quantile(draws.sum(axis=1), 1 - delta))
            if B_true > mu:
                infl.append((B_hoeff - mu) / (B_true - mu))
    return {
        "delta": delta, "model": model, "n_plans": n_plans,
        "hoeffding_acceptance_rate": round(100 * n_hoeff_accept / n_plans, 2),
        "true_feasible_rate": round(100 * n_true_feasible / n_plans, 2),
        "conservative_rejections_pct_of_all": round(100 * n_cons_reject / n_plans, 2),
        "false_rejection_rate_among_feasible":
            round(100 * n_cons_reject / max(n_true_feasible, 1), 2),
        "budget_inflation_factor_mean": round(float(np.mean(infl)), 3),
        "budget_inflation_factor_median": round(float(np.median(infl)), 3),
    }


# --------------------------------------------------------------------------
# (3) repair-iteration impact: shrink a padded plan until C5 passes
# --------------------------------------------------------------------------
def repair_impact(delta=0.05, n_tasks=3000, model="uniform", seed=13):
    """A plan has n_core required services + n_pad optional ones. Each repair
    iteration drops one optional service. Count iterations to pass C5 under the
    conservative Hoeffding gate vs an oracle (empirical) gate; the budget is
    fixed at the value that makes the CORE plan truly just-feasible."""
    rng = np.random.default_rng(seed)
    hoeff_iters, oracle_iters, extra = [], [], 0
    hoeff_infeasible = 0
    for _ in range(n_tasks):
        n_core = int(rng.integers(4, 8))
        n_pad = int(rng.integers(1, 6))
        us_core = rng.uniform(0.5, 1.5, size=n_core)
        us_pad = rng.uniform(0.5, 1.5, size=n_pad)
        # budget: core plan truly meets delta with a little room
        muC = _mean_lat(us_core, model); S2C = (us_core ** 2).sum()
        B = muC + 1.15 * np.sqrt(0.5 * np.log(1 / delta) * S2C)

        def passes(gate, us):
            mu = _mean_lat(us, model); S2 = (us ** 2).sum()
            if gate == "hoeff":
                return hoeffding_bound(mu, list(us), B) <= delta
            return _mc_violation(us, B, model, rng, n=8000) <= delta

        # order pad drops deterministically (largest u first = greedy repair)
        order = np.argsort(-us_pad)
        for gate, store in (("hoeff", hoeff_iters), ("oracle", oracle_iters)):
            us = np.concatenate([us_core, us_pad])
            it = 0
            dropped = 0
            while not passes(gate, us) and dropped < n_pad:
                us = np.concatenate([us_core, np.delete(us_pad, order[:dropped + 1])])
                dropped += 1
                it += 1
            store.append(it if passes(gate, us) else -1)
        if hoeff_iters[-1] > oracle_iters[-1]:
            extra += 1
        if hoeff_iters[-1] == -1:
            hoeff_infeasible += 1
    hi = np.array([x for x in hoeff_iters if x >= 0])
    oi = np.array([x for x in oracle_iters if x >= 0])
    return {
        "delta": delta, "model": model, "n_tasks": n_tasks,
        "mean_iters_hoeffding": round(float(hi.mean()), 3),
        "mean_iters_oracle": round(float(oi.mean()), 3),
        "extra_iters_pct": round(100 * extra / n_tasks, 2),
        "hoeffding_could_not_satisfy_pct": round(100 * hoeff_infeasible / n_tasks, 2),
    }


def main():
    os.makedirs("results", exist_ok=True)
    res = {}
    print("EXP2 (1) gap analysis ...");        res["gap"] = gap_analysis()
    print("EXP2 (2) acceptance impact ...")
    res["acceptance_uniform"] = acceptance_impact(model="uniform")
    res["acceptance_twopoint"] = acceptance_impact(model="twopoint")
    print("EXP2 (3) repair impact ...")
    res["repair_uniform"] = repair_impact(model="uniform")
    with open("results/exp2_sla_tightness.json", "w") as f:
        json.dump(res, f, indent=2)

    print("\n== (1) bound vs empirical (Uniform latencies) ==")
    for r in res["gap"]["uniform"]:
        print(f"  delta={r['delta']:.2f}  emp={r['empirical_mean']:.4f}  "
              f"tightness(emp/delta)={r['tightness_emp_over_delta']:.3f}  sound={r['sound_frac']}")
    print("== (1) bound vs empirical (two-point worst case) ==")
    for r in res["gap"]["twopoint"]:
        print(f"  delta={r['delta']:.2f}  emp={r['empirical_mean']:.4f}  "
              f"tightness(emp/delta)={r['tightness_emp_over_delta']:.3f}")
    a = res["acceptance_uniform"]
    print("\n== (2) acceptance impact @ delta=0.05 (Uniform) ==")
    print(f"  Hoeffding acceptance={a['hoeffding_acceptance_rate']}%  "
          f"true-feasible={a['true_feasible_rate']}%")
    print(f"  false rejections among feasible={a['false_rejection_rate_among_feasible']}%  "
          f"budget inflation x{a['budget_inflation_factor_median']} (median)")
    r = res["repair_uniform"]
    print("== (3) repair iterations @ delta=0.05 (Uniform) ==")
    print(f"  mean iters Hoeffding={r['mean_iters_hoeffding']} vs oracle={r['mean_iters_oracle']}  "
          f"extra-iter tasks={r['extra_iters_pct']}%")
    print("\nwrote results/exp2_sla_tightness.json")


if __name__ == "__main__":
    main()
