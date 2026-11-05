"""
Non-LLM measured experiments (fast, deterministic):
  E1  verifier accuracy on Synth-TSR ground-truth labels
  E2  SLA-bound soundness/calibration: predicted Hoeffding bound vs empirical MC
  E3  scalability: verification time vs composition length
  E4  synthetic propose-verify-repair convergence (iteration distribution)
Outputs results/core_results.json (consumed by make_real_figures.py).
"""
from __future__ import annotations
import json, os, time
import numpy as np
from core import Service, Goal, verify, verify_all, hoeffding_bound
import bench


def e1_verifier_accuracy(n=900, seed=0):
    rng = np.random.default_rng(seed)
    tasks = bench.build_suite(rng, n)
    correct = 0
    per_check_caught = {}
    n_adm = n_inadm = 0
    for plan, goal, label, vtype in tasks:
        v = verify(plan, goal)
        pred_admissible = v.ok
        if label:
            n_adm += 1
        else:
            n_inadm += 1
            per_check_caught.setdefault(vtype, [0, 0])
            per_check_caught[vtype][1] += 1
            if not pred_admissible:
                per_check_caught[vtype][0] += 1
        if pred_admissible == label:
            correct += 1
    acc = 100.0 * correct / n
    return {"n": n, "accuracy": round(acc, 2), "n_admissible": n_adm,
            "n_inadmissible": n_inadm,
            "per_violation_recall": {k: round(100 * c / t, 1)
                                     for k, (c, t) in per_check_caught.items()}}


def e1b_reduced_verifier_ablation(n=900, seed=0):
    """New baseline: how much does a REDUCED verifier (the checks a schema-only or
    typed-tool/consistency-only approach would run) miss? For each cumulative
    prefix of {C1,C2,C3,C4,C5} we admit a plan iff NONE of its true violations
    falls in the enabled set, then measure recall on the 900 labeled instances.
    C1+C2-only (schema+SMT, the 'typed tool' baseline) catches only the schema and
    functional classes; the temporal/flow/policy/SLA classes are exactly what a
    sound multi-class verifier adds."""
    rng = np.random.default_rng(seed)
    tasks = bench.build_suite(rng, n)
    configs = {"C1 only": {"C1"}, "C1+C2 (schema+SMT)": {"C1", "C2"},
               "C1-C3 (+temporal)": {"C1", "C2", "C3"},
               "C1-C4 (+flow)": {"C1", "C2", "C3", "C4"},
               "C1-C5 (full)": {"C1", "C2", "C3", "C4", "C5"}}
    caught = {k: 0 for k in configs}
    n_inadm = 0
    for plan, goal, label, vtype in tasks:
        if label:
            continue
        n_inadm += 1
        wit = verify_all(plan, goal)                       # all true violations
        classes = {w[1:3] for w in wit}                    # {'C1','C3',...}
        for k, enabled in configs.items():
            if classes & enabled:                          # some violation is visible
                caught[k] += 1
    return {"n_inadmissible": n_inadm,
            "recall": {k: round(100 * c / n_inadm, 1) for k, c in caught.items()}}


def _rand_plan(rng, m, u_lo=0.5, u_hi=1.5):
    plan = []
    prev = ""
    for i in range(m):
        ub = float(rng.uniform(u_lo, u_hi))
        plan.append(Service(f"s{i}", f"a{i}", {prev} if prev else set(), {f"f{i}"},
                            lat_ub=ub, lat_mean=ub / 2.0,
                            field_level={f"f{i}": "internal"}))
        prev = f"f{i}"
    goal = Goal(set(), {f"f{m-1}"}, set(), [], budget=0.0, delta=0.05)
    return plan, goal


def e2_sla_calibration(n_plans=4000, mc=4000, seed=1):
    """
    Two-point worst-case latency (Hoeffding-extremal): ell_i in {0, u_i} w.p. 1/2.
    For each random plan choose a budget giving a target predicted bound; measure
    the empirical MC violation. Report (predicted, empirical) pairs -> soundness
    means empirical <= predicted.
    """
    rng = np.random.default_rng(seed)
    targets = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
    bins = {t: [] for t in targets}
    for _ in range(n_plans):
        m = int(rng.integers(3, 9))
        us = rng.uniform(0.5, 1.5, size=m)
        mu = 0.5 * us.sum()                       # two-point mean = u_i/2
        S2 = (us ** 2).sum()
        t = targets[rng.integers(0, len(targets))]
        # budget so predicted bound == t:  exp(-2 (B-mu)^2 / S2) = t
        B = mu + np.sqrt(-0.5 * S2 * np.log(t))
        pred = hoeffding_bound(mu, list(us), B)
        # empirical: two-point draws
        draws = (rng.random((mc, m)) < 0.5) * us   # {0,u_i}
        emp = float((draws.sum(axis=1) > B).mean())
        bins[t].append((pred, emp))
    curve = []
    for t in targets:
        preds = np.array([p for p, e in bins[t]])
        emps = np.array([e for p, e in bins[t]])
        curve.append({"predicted": round(float(preds.mean()), 4),
                      "empirical": round(float(emps.mean()), 4),
                      "empirical_sd": round(float(emps.std()), 4),
                      "sound_frac": round(float((emps <= preds + 1e-9).mean()), 4)})
    overall_sound = float(np.mean([c["sound_frac"] for c in curve]))
    return {"curve": curve, "overall_sound_frac": round(overall_sound, 4)}


def e3_scalability(lengths=range(2, 13), reps=4000, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for m in lengths:
        plan, goal = _rand_plan(rng, m)
        # warm
        verify(plan, goal)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                verify(plan, goal)
            times.append((time.perf_counter() - t0) / reps * 1e3)   # ms/verify
        times = np.array(times)
        out.append({"length": int(m), "ms_mean": round(float(times.mean()), 4),
                    "ms_sd": round(float(times.std()), 4)})
    return out


def e4_repair_convergence(n=3000, T=6, seed=3):
    rng = np.random.default_rng(seed)
    iters, certified = [], 0
    for i in range(n):
        goal = bench._payment_core(np.random.default_rng(5000 + i))[1]
        ok, it = bench.synthetic_repair_loop(np.random.default_rng(7000 + i), goal, T=T)
        if ok:
            certified += 1
            iters.append(it)
    iters = np.array(iters)
    hist = [int((iters == k).sum()) for k in range(0, T + 1)]
    frac = [round(100 * h / len(iters), 1) for h in hist]
    return {"n": n, "certified_frac": round(100 * certified / n, 1),
            "mean_iters": round(float(iters.mean()), 3),
            "le1": round(100 * float((iters <= 1).mean()), 1),
            "le3": round(100 * float((iters <= 3).mean()), 1),
            "hist_frac": frac}


def main():
    os.makedirs("results", exist_ok=True)
    res = {}
    print("E1 verifier accuracy ..."); res["e1_verifier_accuracy"] = e1_verifier_accuracy()
    print("E1b reduced-verifier ablation ..."); res["e1b_reduced_verifier"] = e1b_reduced_verifier_ablation()
    print("E2 SLA calibration ...");    res["e2_sla_calibration"] = e2_sla_calibration()
    print("E3 scalability ...");        res["e3_scalability"] = e3_scalability()
    print("E4 repair convergence ..."); res["e4_repair_convergence"] = e4_repair_convergence()
    with open("results/core_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
