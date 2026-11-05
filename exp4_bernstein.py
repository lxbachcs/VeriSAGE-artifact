"""
EXP4 -- Tighter distribution-free SLA bounds: Bernstein / Bennett vs Hoeffding
(Stanford review Q9).

Hoeffding uses only the support widths u_i; it ignores variance. When the true
per-service latency variance sigma_i^2 is small relative to u_i^2 (the common
case for concentrated, near-deterministic microservice latencies) a
variance-aware bound (Bernstein / Bennett) certifies the SAME delta at a MUCH
smaller budget slack, i.e. it accepts strictly more truly-feasible plans while
remaining distribution-free (it needs only a variance bound and support, no
distributional form).

For a sum S = sum_i ell_i of independent, bounded ell_i in [0, u_i] with
E[S]=mu, variance sum_i sigma_i^2 = V, and per-term deviation bounded by
b = max_i (u_i - E[ell_i]) <= max_i u_i, the classical bounds on the upper tail
P[S - mu > t] are:

  Hoeffding : exp( -2 t^2 / sum_i u_i^2 )
  Bernstein : exp( - t^2 / ( 2 V + (2/3) b t ) )
  Bennett   : exp( - V/b^2 * h( b t / V ) ),  h(x) = (1+x)ln(1+x) - x

Bennett >= Bernstein (Bennett is tighter; Bernstein is the classic quadratic
relaxation of Bennett). All three are distribution-free given (u_i, sigma_i, b).

We invert each bound to the *minimum budget slack* s = t needed to certify a
target delta, and report:
  (1) slack ratio Hoeffding/Bernstein and Hoeffding/Bennett across latency
      concentration regimes (variance fraction rho = sigma_i^2 / (u_i^2/4),
      rho=1 is the two-point worst case, rho->0 is near-deterministic);
  (2) re-analysis of the 32k-plan Monte-Carlo calibration: for each of the 32k
      plans, what slack each bound needs, and the acceptance-rate lift of
      Bernstein/Bennett over Hoeffding at fixed delta, with EMPIRICAL soundness
      verified by Monte-Carlo (the tighter bounds must still never be violated).

Distribution-free is preserved: the variance V is itself bounded from the fitted
QWS support the paper already uses (sigma_i^2 <= u_i^2/4 for any dist on [0,u_i];
tighter if a variance upper bound is known), so no distributional assumption is
added -- only a *second moment* certificate on top of the support certificate.

Outputs results/exp4_bernstein.json.
"""
from __future__ import annotations
import json, os, math
import numpy as np
import qws_data


# --------------------------------------------------------------------------
# distribution-free upper-tail bounds; return P[S-mu > t]
# --------------------------------------------------------------------------
def hoeffding(t, us):
    S2 = float(np.sum(np.square(us)))
    if t <= 0:
        return 1.0
    return math.exp(-2.0 * t * t / S2)


def bernstein(t, V, b):
    if t <= 0:
        return 1.0
    return math.exp(-(t * t) / (2.0 * V + (2.0 / 3.0) * b * t))


def _h(x):
    return (1.0 + x) * math.log(1.0 + x) - x


def bennett(t, V, b):
    if t <= 0:
        return 1.0
    if b <= 0:
        return 0.0
    return math.exp(-(V / (b * b)) * _h(b * t / V))


# invert a monotone-decreasing-in-t bound to the min slack t* with bound(t*)=delta
def _invert(fn, delta, hi):
    lo, hi = 0.0, float(hi)
    # ensure fn(hi) <= delta
    for _ in range(60):
        if fn(hi) <= delta:
            break
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if fn(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


# --------------------------------------------------------------------------
# (1) slack ratio across concentration regimes
# --------------------------------------------------------------------------
def slack_by_concentration(delta=0.05, m=8, u=1.0, seed=7):
    """Homogeneous plan of m services, support [0,u]. Sweep the variance fraction
    rho in (0,1]: sigma_i^2 = rho * (u^2/4). Mean fixed at u/2 (worst-case b=u/2).
    Report min slack for each bound and the Hoeffding/tighter ratio."""
    us = np.full(m, u)
    S2 = float(np.sum(us ** 2))
    hi = 5.0 * math.sqrt(S2)
    out = []
    for rho in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01):
        sig2 = rho * (u * u / 4.0)
        V = m * sig2
        b = u / 2.0                              # worst-case centered deviation
        t_h = _invert(lambda t: hoeffding(t, us), delta, hi)
        t_be = _invert(lambda t: bernstein(t, V, b), delta, hi)
        t_bn = _invert(lambda t: bennett(t, V, b), delta, hi)
        out.append({
            "rho_variance_fraction": rho,
            "slack_hoeffding": round(t_h, 4),
            "slack_bernstein": round(t_be, 4),
            "slack_bennett": round(t_bn, 4),
            "ratio_hoeff_over_bernstein": round(t_h / t_be, 3),
            "ratio_hoeff_over_bennett": round(t_h / t_bn, 3),
            "slack_reduction_bennett_pct": round(100 * (t_h - t_bn) / t_h, 2),
        })
    return {"delta": delta, "m": m, "u": u, "curve": out}


# --------------------------------------------------------------------------
# (2) re-analysis of the 32k-plan Monte-Carlo calibration
# --------------------------------------------------------------------------
def _slack_hoeffding_vec(S2, delta):
    """t s.t. exp(-2 t^2/S2)=delta  ->  t = sqrt(-0.5 S2 ln delta)."""
    return np.sqrt(-0.5 * S2 * math.log(delta))


def _slack_bernstein_vec(V, b, delta):
    """Bernstein tail exp(-t^2/(2V + (2/3) b t))=delta.  Let L=ln(1/delta).
    t^2 - (2/3 b L) t - 2 V L = 0  ->  t = (a + sqrt(a^2 + 8 V L))/2, a=(2/3)bL."""
    L = math.log(1.0 / delta)
    a = (2.0 / 3.0) * b * L
    return 0.5 * (a + np.sqrt(a * a + 8.0 * V * L))


def _slack_bennett_scalar(V, b, delta, hi):
    """Bennett needs numeric inversion (transcendental); scalar bisection."""
    return _invert(lambda t: bennett(t, V, b), delta, hi)


def recalibrate_32k(delta=0.05, n_plans=32000, mc=3000, seed=21):
    """For 32k random plans and several realistic latency models, compute the
    min slack each bound needs to certify delta (vectorized for Hoeffding &
    Bernstein; Bennett by bisection), verify EMPIRICAL soundness by Monte-Carlo
    on a subset at each bound's tight budget, and report the budget saving of the
    tighter bounds over Hoeffding at a fixed delta."""
    rng = np.random.default_rng(seed)
    # draw 32k plans ONCE (shared support), then evaluate each latency model on
    # the same plans so the comparison is apples-to-apples.
    ms = rng.integers(3, 10, size=n_plans)
    plans = [rng.uniform(0.5, 1.5, size=int(m)) for m in ms]
    S2 = np.array([float(np.sum(us ** 2)) for us in plans])
    b_arr = np.array([float(np.max(0.5 * us)) for us in plans])   # centered dev bound = max u_i/2
    mus = np.array([float(0.5 * us.sum()) for us in plans])

    var1 = {"uniform": lambda u: u * u / 12.0,
            "twopoint": lambda u: u * u / 4.0,
            "beta22": lambda u: u * u * (0.05)}   # Beta(2,2) var=1/20 of u^2
    sampler = {
        "uniform":  lambda us, n: rng.random((n, len(us))) * us,
        "twopoint": lambda us, n: (rng.random((n, len(us))) < 0.5) * us,
        "beta22":   lambda us, n: rng.beta(2, 2, (n, len(us))) * us,
    }
    t_h = _slack_hoeffding_vec(S2, delta)
    hi_arr = 5.0 * np.sqrt(S2)

    res = {}
    for name in ("uniform", "twopoint", "beta22"):
        V = np.array([float(np.sum(var1[name](us))) for us in plans])
        t_be = _slack_bernstein_vec(V, b_arr, delta)
        # Bennett per plan (bisection, but cheap scalar loop)
        t_bn = np.array([_slack_bennett_scalar(V[i], b_arr[i], delta, hi_arr[i])
                         for i in range(n_plans)])
        ratio_be = t_h / t_be
        ratio_bn = t_h / t_bn
        # soundness MC on every 16th plan at each bound's tight budget
        sound = {"hoeffding": 0, "bernstein": 0, "bennett": 0}
        idxs = range(0, n_plans, 16)
        for i in idxs:
            draws = sampler[name](plans[i], mc)
            tot = draws.sum(axis=1)
            for nm, t in (("hoeffding", t_h[i]), ("bernstein", t_be[i]), ("bennett", t_bn[i])):
                if float((tot > mus[i] + t).mean()) <= delta + 1e-9:
                    sound[nm] += 1
        checked = len(list(idxs))
        res[name] = {
            "delta": delta, "n_plans": n_plans, "mc_checked": checked,
            "slack_ratio_hoeff_over_bernstein_mean": round(float(ratio_be.mean()), 3),
            "slack_ratio_hoeff_over_bennett_mean": round(float(ratio_bn.mean()), 3),
            "slack_reduction_bernstein_pct_mean": round(100 * float((1 - 1 / ratio_be).mean()), 2),
            "slack_reduction_bennett_pct_mean": round(100 * float((1 - 1 / ratio_bn).mean()), 2),
            "mean_budget_saving_bennett_pct": round(100 * float((1 - 1 / ratio_bn).mean()), 2),
            "empirical_sound_frac": {k: round(v / checked, 4) for k, v in sound.items()},
        }
    return res


# --------------------------------------------------------------------------
# (3) REAL-DATA soundness: latencies bootstrapped from the QWS dataset
# --------------------------------------------------------------------------
def recalibrate_qws_real(delta=0.05, n_plans=32000, mc=3000, seed=23):
    """Same distribution-free bounds, but each service's per-invocation latency
    is a bootstrap draw from the REAL QWS response-time population (2507 measured
    services, heavy right tail). The support u_i is the observed real maximum and
    the variance is the real second moment; both are read off the data, so no
    distributional form is assumed. We verify the bounds remain EMPIRICALLY SOUND
    on the real distribution and report the variance-aware slack advantage that
    the heavy tail (loose support, small relative variance) makes decisive."""
    rng = np.random.default_rng(seed)
    st = qws_data.qws_stats()
    u = float(st["max"])                 # real observed support upper bound (ms)
    mu_i = float(st["mean"])             # real per-service mean latency (ms)
    var_i = float(st["var"])             # real per-service variance
    b = u - mu_i                         # centered deviation bound
    ms = rng.integers(3, 10, size=n_plans)
    S2 = (ms * (u * u)).astype(float)                 # sum u_i^2
    mus = (ms * mu_i).astype(float)                   # E[sum]
    V = (ms * var_i).astype(float)                    # sum var
    hi_arr = 5.0 * np.sqrt(S2)
    t_h = _slack_hoeffding_vec(S2, delta)
    t_be = _slack_bernstein_vec(V, b, delta)
    t_bn = np.array([_slack_bennett_scalar(V[i], b, delta, hi_arr[i]) for i in range(n_plans)])
    ratio_be = t_h / t_be
    ratio_bn = t_h / t_bn
    # empirical soundness on REAL bootstrapped latencies, every 16th plan
    sound = {"hoeffding": 0, "bernstein": 0, "bennett": 0}
    idxs = list(range(0, n_plans, 16))
    for i in idxs:
        draws = qws_data.bootstrap_latencies(rng, (mc, int(ms[i])))
        tot = draws.sum(axis=1)
        for nm, t in (("hoeffding", t_h[i]), ("bernstein", t_be[i]), ("bennett", t_bn[i])):
            if float((tot > mus[i] + t).mean()) <= delta + 1e-9:
                sound[nm] += 1
    checked = len(idxs)
    return {
        "dataset": "QWS v2.0 (2507 real web services), response time (ms)",
        "real_stats": {k: round(st[k], 3) for k in ("n", "mean", "std", "p95", "p99", "max", "skew")},
        "delta": delta, "n_plans": n_plans, "mc_checked": checked,
        "support_ub_ms": round(u, 1),
        "slack_ratio_hoeff_over_bernstein_mean": round(float(ratio_be.mean()), 3),
        "slack_ratio_hoeff_over_bennett_mean": round(float(ratio_bn.mean()), 3),
        "slack_reduction_bennett_pct_mean": round(100 * float((1 - 1 / ratio_bn).mean()), 2),
        "empirical_sound_frac": {k: round(v / checked, 4) for k, v in sound.items()},
    }


def main():
    os.makedirs("results", exist_ok=True)
    res = {}
    print("EXP4 (1) slack by concentration ...")
    res["slack_by_concentration"] = slack_by_concentration()
    print("EXP4 (2) 32k recalibration (Bernstein/Bennett) ...")
    res["recalibrate_32k"] = recalibrate_32k()
    print("EXP4 (3) REAL-DATA (QWS) soundness ...")
    res["qws_real"] = recalibrate_qws_real()
    with open("results/exp4_bernstein.json", "w") as f:
        json.dump(res, f, indent=2)

    print("\n== (1) slack ratio vs variance fraction rho (delta=0.05, m=8) ==")
    for r in res["slack_by_concentration"]["curve"]:
        print(f"  rho={r['rho_variance_fraction']:.2f}  "
              f"Hoeff/Bennett slack ratio={r['ratio_hoeff_over_bennett']:.2f}x  "
              f"(Bennett cuts slack {r['slack_reduction_bennett_pct']:.1f}%)")
    print("== (2) 32k recalibration ==")
    for name, r in res["recalibrate_32k"].items():
        print(f"  [{name}] Hoeff/Bennett slack {r['slack_ratio_hoeff_over_bennett_mean']:.2f}x, "
              f"budget saving {r['mean_budget_saving_bennett_pct']:.1f}%, "
              f"sound frac {r['empirical_sound_frac']}")
    q = res["qws_real"]
    print("== (3) REAL-DATA soundness (QWS, 2507 real services) ==")
    print(f"  support u={q['support_ub_ms']}ms (real max), skew={q['real_stats']['skew']}")
    print(f"  Hoeff/Bennett slack {q['slack_ratio_hoeff_over_bennett_mean']:.2f}x, "
          f"Bennett cuts slack {q['slack_reduction_bennett_pct_mean']:.1f}%")
    print(f"  empirical soundness on real bootstrapped latencies: {q['empirical_sound_frac']}")
    print("\nwrote results/exp4_bernstein.json")


if __name__ == "__main__":
    main()
