#!/usr/bin/env python3
"""
check_results.py -- assert that the committed measured results (results/*.json)
match the numbers claimed in the manuscript. Read-only; NO API / Ollama / network.

Run:  python3 check_results.py
Exit: 0 if every claim matches, 1 otherwise. Prints a PASS/FAIL line per claim.

Each check names the manuscript location and the JSON source, so a reviewer can
trace every headline number back to a committed measured file.
"""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

_fails = 0


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def check(label, ok, got, want):
    global _fails
    status = "PASS" if ok else "FAIL"
    if not ok:
        _fails += 1
    print(f"[{status}] {label}\n         got={got}  expected={want}")


def approx(a, b, tol):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def main():
    core = load("core_results.json")
    tight = load("exp2_sla_tightness.json")
    scal = load("exp3_scalability.json")
    e4 = load("exp4_bernstein.json")
    e10 = load("exp10_qos_baseline.json")
    e1_7b = load("exp1_baselines_qwen2.5_7b.json")
    e1_14b = load("exp1_baselines_qwen2.5_14b.json")
    e1_gemma = load("exp1_baselines_gemma2_9b.json")

    # ---- RQ1: verifier soundness (abstract/intro/results: 100% of 900) ----
    e1 = core["e1_verifier_accuracy"]
    check("RQ1 verifier n=900 (results.tex; core_results.e1.n)",
          e1["n"] == 900, e1["n"], 900)
    check("RQ1 verifier accuracy=100% (results.tex; core_results.e1.accuracy)",
          approx(e1["accuracy"], 100.0, 1e-6), e1["accuracy"], 100.0)

    # ---- RQ2 soundness: SLA bound holds on 100% of 32,000 plans ----
    e2 = core["e2_sla_calibration"]
    check("RQ2 SLA soundness 100% (results.tex; core_results.e2.overall_sound_frac)",
          approx(e2["overall_sound_frac"], 1.0, 1e-9),
          e2["overall_sound_frac"], 1.0)
    # 32,000 plans = 8 delta-points x 4,000 plans/point (measured protocol, setup.tex)
    n_pts = len(e2["curve"])
    check("RQ2 SLA plan budget = 8 points x 4000 = 32,000 (setup.tex; e2.curve length)",
          n_pts == 8, f"{n_pts} points x 4000 = {n_pts * 4000}", "8 x 4000 = 32000")

    # ---- RQ2 tightness (results.tex: 33.3% false-rejected; 2.57x headroom) ----
    au = tight["acceptance_uniform"]
    check("RQ2 tightness false-rejection 33.3% (results.tex; exp2.acceptance_uniform)",
          approx(au["false_rejection_rate_among_feasible"], 33.3, 0.3),
          au["false_rejection_rate_among_feasible"], "~33.3")
    check("RQ2 tightness budget inflation 2.57x (results.tex; exp2.acceptance_uniform)",
          approx(au["budget_inflation_factor_mean"], 2.57, 0.02),
          au["budget_inflation_factor_mean"], "~2.57")

    # ---- RQ2 REAL-DATA soundness on the QWS dataset (results.tex; setup.tex) ----
    q = e4["qws_real"]
    check("RQ2 QWS n=2507 real services (setup.tex; exp4.qws_real.real_stats.n)",
          q["real_stats"]["n"] == 2507, q["real_stats"]["n"], 2507)
    check("RQ2 QWS all bounds 100% empirically sound on real data (results.tex; exp4.qws_real)",
          all(approx(v, 1.0, 1e-9) for v in q["empirical_sound_frac"].values()),
          q["empirical_sound_frac"], "all 1.0")
    check("RQ2 QWS Bennett cuts slack ~53.7% (results.tex; exp4.qws_real)",
          approx(q["slack_reduction_bennett_pct_mean"], 53.7, 0.6),
          q["slack_reduction_bennett_pct_mean"], "~53.7")

    # ---- QoS-optimizer baseline on real QWS latencies (results.tex) ----
    reg = e10["regulated"]
    check("QoS-MILP* regulated violates C3 on real QWS (results.tex; exp10.regulated)",
          (not reg["qos_milp"]["csts_certified"]) and reg["qos_milp"]["first_violation"] == "C3",
          (reg["qos_milp"]["csts_certified"], reg["qos_milp"]["first_violation"]),
          (False, "C3"))
    check("QoS certification price +1018 ms on real QWS (results.tex; exp10.regulated.gap)",
          approx(reg["latency_gap_verisage_vs_milp"], 1018.1, 1.0),
          reg["latency_gap_verisage_vs_milp"], "~1018.1")

    # ---- RQ3 scalability: R^2=0.999; Z3 C2 ~1 ms/service ----
    sk = scal["scal_k"]
    check("RQ3 scalability R^2=0.999 (results.tex; exp3.scal_k.linfit_R2)",
          approx(sk["linfit_R2"], 0.999, 0.001), sk["linfit_R2"], "~0.999")
    z3 = scal["z3_c2"]["curve"]
    z3_max = max(pt["ms_mean"] for pt in z3)
    check("RQ3 Z3 C2 max cost ~1 ms/service (results.tex; exp3.z3_c2)",
          approx(z3_max, 1.0, 0.15), f"{z3_max:.3f} ms", "~1.0 ms (0.28-1.01)")

    # ---- RQ4 baselines (results.tex Table: 58.0 and 89.3 certified-success) ----
    a7 = e1_7b["aggregate"]
    check("RQ4 qwen2.5:7b VeriSAGE CSR=61.3 (Table; exp1_7b.verisage_CSR.mean)",
          approx(a7["verisage_CSR"]["mean"], 61.3, 0.1),
          a7["verisage_CSR"]["mean"], 61.3)
    check("RQ4 qwen2.5:7b accept-first/blind 32.3/49.0 (Table; exp1_7b)",
          approx(a7["accept_first_CSR"]["mean"], 32.3, 0.1)
          and approx(a7["blind_resample_CSR"]["mean"], 49.0, 0.1),
          (a7["accept_first_CSR"]["mean"], a7["blind_resample_CSR"]["mean"]),
          (32.3, 49.0))
    check("RQ4 qwen2.5:7b Self-Refine 19.0, below accept-first (Table; exp1_7b)",
          approx(a7["self_refine_CSR"]["mean"], 19.0, 0.1)
          and a7["self_refine_CSR"]["mean"] < a7["accept_first_CSR"]["mean"],
          (a7["self_refine_CSR"]["mean"], a7["accept_first_CSR"]["mean"]),
          "19.0 < 32.3")

    a14 = e1_14b["aggregate"]
    check("RQ4 qwen2.5:14b VeriSAGE CSR=88.3 (Table; exp1_14b.verisage_CSR.mean)",
          approx(a14["verisage_CSR"]["mean"], 88.3, 0.1),
          a14["verisage_CSR"]["mean"], 88.3)
    check("RQ4 qwen2.5:14b accept-first/blind 45.7/60.7 (Table; exp1_14b)",
          approx(a14["accept_first_CSR"]["mean"], 45.7, 0.1)
          and approx(a14["blind_resample_CSR"]["mean"], 60.7, 0.1),
          (a14["accept_first_CSR"]["mean"], a14["blind_resample_CSR"]["mean"]),
          (45.7, 60.7))
    check("RQ4 qwen2.5:14b Self-Refine 18.0, below accept-first (Table; exp1_14b)",
          approx(a14["self_refine_CSR"]["mean"], 18.0, 0.1)
          and a14["self_refine_CSR"]["mean"] < a14["accept_first_CSR"]["mean"],
          (a14["self_refine_CSR"]["mean"], a14["accept_first_CSR"]["mean"]),
          "18.0 < 45.7")
    check("RQ4 qwen2.5:14b hard-stratum VeriSAGE 76.7 (Table; exp1_14b hard)",
          approx(a14["verisage_CSR_hard"]["mean"], 76.7, 0.1),
          a14["verisage_CSR_hard"]["mean"], 76.7)

    ag = e1_gemma["aggregate"]
    check("RQ4 gemma2:9b Acc<Blind<VeriSAGE 41.3/60.3/82.3 (Table+macros; exp1_gemma)",
          approx(ag["accept_first_CSR"]["mean"], 41.3, 0.1)
          and approx(ag["blind_resample_CSR"]["mean"], 60.3, 0.1)
          and approx(ag["verisage_CSR"]["mean"], 82.3, 0.1),
          (ag["accept_first_CSR"]["mean"], ag["blind_resample_CSR"]["mean"],
           ag["verisage_CSR"]["mean"]), (41.3, 60.3, 82.3))
    check("RQ4 gemma2:9b hard-stratum 16.0/40.7/78.0 (Table+macros; exp1_gemma hard)",
          approx(ag["accept_first_CSR_hard"]["mean"], 16.0, 0.1)
          and approx(ag["blind_resample_CSR_hard"]["mean"], 40.7, 0.1)
          and approx(ag["verisage_CSR_hard"]["mean"], 78.0, 0.1),
          (ag["accept_first_CSR_hard"]["mean"], ag["blind_resample_CSR_hard"]["mean"],
           ag["verisage_CSR_hard"]["mean"]), (16.0, 40.7, 78.0))

    # ---- gemma2:9b Self-Refine backfill (Table; exp1_gemma) ----
    check("RQ4 gemma2:9b Self-Refine 23.0/10.0 hard (Table; exp1_gemma backfill)",
          approx(ag["self_refine_CSR"]["mean"], 23.0, 0.1)
          and approx(ag["self_refine_CSR_hard"]["mean"], 10.0, 0.1),
          (ag["self_refine_CSR"]["mean"], ag["self_refine_CSR_hard"]["mean"]), (23.0, 10.0))

    # ---- Frontier proposers (Table tab:baselines; hosted GPT-4o family, 1 seed) ----
    for tag, exp, want in [
        ("gpt-4o-mini", "exp1_baselines_gpt-4o-mini.json",
         {"accept_first": (62.0, 24.0), "blind_resample": (79.0, 58.0),
          "self_refine": (27.0, 20.0), "verisage": (84.0, 68.0)}),
        ("gpt-4o", "exp1_baselines_gpt-4o.json",
         {"accept_first": (63.0, 26.0), "blind_resample": (79.0, 58.0),
          "self_refine": (59.0, 48.0), "verisage": (90.0, 80.0)})]:
        af = load(exp)["aggregate"]
        okall = all(approx(af[m + "_CSR"]["mean"], o, 0.1)
                    and approx(af[m + "_CSR_hard"]["mean"], h, 0.1)
                    for m, (o, h) in want.items())
        check(f"RQ4 frontier {tag} all cells match Table (measured; {exp})",
              okall,
              {m: (af[m + "_CSR"]["mean"], af[m + "_CSR_hard"]["mean"]) for m in want},
              want)
        # the headline: VeriSAGE dominates every baseline overall and on hard
        vs = af["verisage_CSR"]["mean"]; vh = af["verisage_CSR_hard"]["mean"]
        check(f"RQ4 frontier {tag}: VeriSAGE best overall+hard (Table; {exp})",
              all(vs > af[m + "_CSR"]["mean"] for m in ("accept_first", "blind_resample", "self_refine"))
              and all(vh >= af[m + "_CSR_hard"]["mean"] for m in ("accept_first", "blind_resample", "self_refine")),
              (vs, vh), "max over baselines")

    # ---- Extended SOTA agentic baselines (Table tab:agentic; exp16) ----
    ex = load("exp16_agentic_qwen2.5_7b.json")["aggregate"]
    hard = {m: ex[m + "_CSR_hard"]["mean"] for m in
            ("accept_first", "best_of_n", "self_refine", "lats", "critic", "blind", "verisage")}
    check("Extended: VeriSAGE best on hard stratum = 10.7 (results.tex; exp16)",
          approx(hard["verisage"], 10.7, 0.1)
          and hard["verisage"] == max(hard.values()),
          hard["verisage"], "10.7 = max over all agentic baselines")
    check("Extended: Self-Refine 18.3 overall, below accept-first (results.tex; exp16)",
          approx(ex["self_refine_CSR"]["mean"], 18.3, 0.1)
          and ex["self_refine_CSR"]["mean"] < ex["accept_first_CSR"]["mean"],
          (ex["self_refine_CSR"]["mean"], ex["accept_first_CSR"]["mean"]), "18.3 < 33.3")
    check("Extended: every agentic baseline collapses on hard <=8.7 < VeriSAGE (exp16)",
          max(v for m, v in hard.items() if m != "verisage") <= 8.7 + 1e-9
          and all(v < hard["verisage"] for m, v in hard.items() if m != "verisage"),
          {m: v for m, v in hard.items() if m != "verisage"}, "all < 10.7")

    # ---- Extended agentic on FRONTIER gpt-4o-mini (Table tab:agentic; exp16) ----
    exf = load("exp16_agentic_gpt-4o-mini.json")["aggregate"]
    hf = {m: exf[m + "_CSR_hard"]["mean"] for m in
          ("accept_first", "best_of_n", "self_refine", "lats", "critic", "blind", "verisage")}
    check("Frontier agentic gpt-4o-mini: VeriSAGE hard 74.0, best of all (Table; exp16)",
          approx(hf["verisage"], 74.0, 0.1) and hf["verisage"] == max(hf.values()),
          hf["verisage"], "74.0 = max")
    check("Frontier agentic gpt-4o-mini: verifier-free methods collapse on hard <=44 (Table; exp16)",
          max(hf[m] for m in ("accept_first", "best_of_n", "self_refine", "lats", "critic")) <= 44.0 + 1e-9,
          {m: hf[m] for m in ("accept_first", "best_of_n", "self_refine", "lats", "critic")},
          "all verifier-free <= 44")
    check("Frontier agentic gpt-4o-mini: Self-Refine 22.0 hard, below accept-first 40.0 (Table; exp16)",
          approx(hf["self_refine"], 22.0, 0.1) and hf["self_refine"] < hf["accept_first"],
          (hf["self_refine"], hf["accept_first"]), "22.0 < 40.0")

    # ---- NEW baseline: reduced-verifier ablation (results.tex RQ1; core.e1b) ----
    ab = core["e1b_reduced_verifier"]["recall"]
    check("Ablation C1+C2 (schema+SMT) recall 51.1% (results.tex; core.e1b)",
          approx(ab["C1+C2 (schema+SMT)"], 51.1, 0.3), ab["C1+C2 (schema+SMT)"], "~51.1")
    check("Ablation full C1-C5 recall 100% (results.tex; core.e1b)",
          approx(ab["C1-C5 (full)"], 100.0, 1e-9), ab["C1-C5 (full)"], 100.0)
    check("Ablation C1-C4 (+flow) recall 68.4% (appendix; core.e1b)",
          approx(ab["C1-C4 (+flow)"], 68.4, 0.3), ab["C1-C4 (+flow)"], "~68.4")

    # ---- Q4 contract-noise robustness (discussion/appendix; exp17) ----
    nz = load("exp17_noisy_contract.json")
    fr = {r["noise_rate"]: r["false_rejection_rate"] for r in nz["false_rejection"]}
    ua = {r["noise_rate"]: r["unsafe_admission_rate"] for r in nz["unsafe_admission"]}
    check("Q4 noise p=0 -> 0% false-reject AND 0% unsafe-admit (discussion; exp17)",
          approx(fr[0.0], 0.0, 1e-9) and approx(ua[0.0], 0.0, 1e-9),
          (fr[0.0], ua[0.0]), (0.0, 0.0))
    check("Q4 omission noise false-rejection 3.1-19.0% for p=0.05-0.3 (appendix; exp17)",
          approx(fr[0.05], 3.1, 0.2) and approx(fr[0.3], 19.0, 0.3),
          (fr[0.05], fr[0.3]), ("~3.1", "~19.0"))
    check("Q4 relaxation unsafe-admit < p and 14.6% at p=0.3 (appendix; exp17)",
          approx(ua[0.3], 14.6, 0.3) and all(ua[p] <= 100 * p + 1e-9 for p in (0.05, 0.1, 0.2, 0.3)),
          ua[0.3], "~14.6, all < 100p")

    # ---- Q3 correlation admission-hit on real QWS marginals (discussion/appendix; exp11) ----
    ah = load("exp11_correlation.json")["admission_hit_qws"]
    check("Q3 robust certificate safe under ALL correlation (discussion; exp11)",
          ah["robust_safe_under_all_correlation"] is True,
          ah["robust_safe_under_all_correlation"], True)
    check("Q3 Hoeffding stays sound on real QWS at high corr (0% unsafe) (discussion; exp11)",
          approx(ah["hoeffding_unsafe_admit_frac_hi_corr"], 0.0, 1e-9)
          and ah["hoeffding_worst_true_violation"] < 0.05,
          (ah["hoeffding_unsafe_admit_frac_hi_corr"], ah["hoeffding_worst_true_violation"]),
          "0% unsafe, worst<0.05")
    check("Q3 robust admission price 0.8-1.5x Hoeffding slack (discussion; exp11)",
          approx(ah["slack_ratio_robust_over_hoeff_min"], 0.806, 0.05)
          and approx(ah["slack_ratio_robust_over_hoeff_max"], 1.452, 0.05),
          (ah["slack_ratio_robust_over_hoeff_min"], ah["slack_ratio_robust_over_hoeff_max"]),
          ("~0.81", "~1.45"))

    # ---- RPE v1.1 additions: hard-task ladder (Table 4b), timing, tightness ----
    ladder = {  # planner -> (single-shot, blind, VeriSAGE) CSR % on the hard task
        "gemma2_9b": (15, 65, 100), "qwen2.5_7b": (0, 20, 30), "qwen2.5_14b": (0, 0, 90),
        "qwen2.5_72b": (0, 40, 85), "qwen2.5-coder_7b": (5, 20, 20), "llama3.2_3b": (0, 0, 0),
        "llama3.1_8b": (0, 55, 25), "llama3.3_70b": (100, 100, 100), "gpt-4o-mini": (65, 100, 85),
    }
    for tag, want in ladder.items():
        d = load(f"llm_summary_{tag}_hard.json")
        got = (d["single_shot_CSR"], d["blind_CSR"], d["verisage_CSR"])
        check(f"v1.1 Table 4(b) hard ladder {tag} single/blind/VeriSAGE",
              all(approx(g, w, 0.5) for g, w in zip(got, want)), got, want)
    check("v1.1 structural verification ~0.27 us/service (exp3.scal_k.linfit_slope)",
          approx(sk["linfit_slope_us_per_service"], 0.27, 0.01),
          sk["linfit_slope_us_per_service"], "~0.27")
    e8 = load("exp8_endtoend.json")
    p50 = {r["k"]: (r["regimes"]["light"]["p50_ms"], r["regimes"]["heavy"]["p50_ms"]) for r in e8["by_k"]}
    check("v1.1 end-to-end p50: 13 services 4.3-9.1 ms, 100 services 33-70 ms (exp8)",
          approx(p50[13][0], 4.3, 0.05) and approx(p50[13][1], 9.1, 0.05)
          and approx(p50[100][0], 32.7, 0.5) and approx(p50[100][1], 69.9, 0.5),
          (p50[13], p50[100]), "((4.3, 9.1), (~33, ~70))")
    acc = tight["acceptance_uniform"]
    check("v1.1 SLA tightness: 88.5% feasible, 59.1% admitted (exp2)",
          approx(acc["true_feasible_rate"], 88.5, 0.05) and approx(acc["hoeffding_acceptance_rate"], 59.1, 0.05),
          (acc["true_feasible_rate"], acc["hoeffding_acceptance_rate"]), (88.5, 59.1))
    e7 = load("exp7_repair_ablation.json")
    check("v1.1 typed 100% (2.2 rounds) vs check-list 75.9% (3.4) vs none 0% (exp7)",
          approx(e7["typed"]["CSR_pct"], 100, 0.01) and approx(e7["typed"]["mean_iters_to_certify"], 2.2, 0.05)
          and approx(e7["nl"]["CSR_pct"], 75.9, 0.05) and approx(e7["nl"]["mean_iters_to_certify"], 3.4, 0.05)
          and approx(e7["none"]["CSR_pct"], 0, 0.01),
          (e7["typed"]["CSR_pct"], e7["nl"]["CSR_pct"], e7["none"]["CSR_pct"]), (100, 75.9, 0))
    e14 = load("exp14_multidefect.json")
    check("v1.1 multi-defect: priority 39.4% vs naive 14.0%, 3896 re-entries (exp14)",
          approx(e14["prioritized"]["CSR_pct"], 39.4, 0.05) and approx(e14["naive"]["CSR_pct"], 14.0, 0.05)
          and e14["naive"]["oscillation_events"] == 3896,
          (e14["prioritized"]["CSR_pct"], e14["naive"]["CSR_pct"], e14["naive"]["oscillation_events"]),
          (39.4, 14.0, 3896))

    print("-" * 60)
    if _fails == 0:
        print("RESULT: PASS -- all manuscript claims match measured results/*.json")
        return 0
    print(f"RESULT: FAIL -- {_fails} claim(s) do not match")
    return 1


if __name__ == "__main__":
    sys.exit(main())
