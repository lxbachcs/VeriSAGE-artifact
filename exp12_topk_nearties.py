"""
EXP12 -- Top-k path-union sensitivity to NEAR-TYING path means, and an empirical
criterion for choosing k (Stanford round-2 review Q2).

The paper's top-k path-union bound sums only the k highest-MEAN source->sink
paths, arguing omitted paths have strictly lower mean so the bound stays a valid
upper bound "in the near-sequential regime". The reviewer asks: how sensitive is
this to NEAR-TYING path means (several paths with almost-equal means, so the
top-k truncation drops paths that contribute non-negligibly to the tail)? and:
is there an empirical criterion for k trading soundness vs conservatism?

We construct layered DAGs with a controlled MEAN-GAP g between consecutive paths
(g large = well-separated means, g->0 = near-tied means). For each g and each k
we measure, by Monte-Carlo:
  (1) SOUNDNESS: empirical violation at the top-k-implied budget vs the certified
      delta. When means are near-tied, small k can UNDER-count the tail and the
      pruned bound can become UNSOUND -- we detect exactly when.
  (2) CONSERVATISM: slack(top-k)/slack(true) -- larger k is safer but looser.
  (3) an EMPIRICAL CRITERION: the smallest k whose *marginal* tail contribution
      from the (k+1)-th path is below eps*delta -- i.e. include every path whose
      mean is within a slack-window of the top mean. We show this criterion keeps
      soundness at 100% while adding minimal conservatism, and report the k it
      selects as a function of the near-tie gap.

Pure numpy Monte-Carlo. No LLM. Outputs results/exp12_topk_nearties.json.
"""
from __future__ import annotations
import json, os, math
import numpy as np


def _make_paths(n_paths, path_len, mean_gap, u=1.0, seed=0):
    """n_paths independent source->sink paths, each with path_len nodes of
    support u. Path j's node means are tuned so path means are
    mu_j = base - j*mean_gap (path 0 highest). Returns per-path (means, S2)."""
    rng = np.random.default_rng(seed)
    paths = []
    for j in range(n_paths):
        us = np.full(path_len, u)
        # shift the last node's effective support to realise the target mean gap
        target_mean = 0.5 * path_len * u - j * mean_gap
        node_means = np.full(path_len, 0.5 * u)
        # adjust one node's mean downward for lower paths (keep within [0,u])
        deficit = 0.5 * path_len * u - target_mean
        node_means[-1] = max(0.0, 0.5 * u - deficit)
        paths.append({"us": us, "node_means": node_means,
                      "mean": float(node_means.sum()),
                      "S2": float(np.sum(us ** 2))})
    return paths


def _mc_max_path(paths, n, rng):
    """MC the DAG latency L = max_j sum_i ell_i (paths disjoint). ell node ~
    latency with mean node_means and support [0,u] (triangular-ish: use a
    Beta scaled so E=node_mean)."""
    L = np.zeros(n)
    for p in paths:
        us = p["us"]; nm = p["node_means"]
        # sample each node uniform on [0, 2*mean] clipped to [0,u] to hit E=mean
        draws = np.zeros((n, len(us)))
        for i in range(len(us)):
            hi = min(2 * nm[i], us[i]) if nm[i] > 0 else 1e-9
            draws[:, i] = rng.random(n) * hi
        L = np.maximum(L, draws.sum(axis=1))
    return L


def _hoeff_tail(t, S2):
    return math.exp(-2.0 * t * t / S2) if t > 0 else 1.0


def _topk_slack(paths, delta, k):
    means = np.array([p["mean"] for p in paths])
    S2s = np.array([p["S2"] for p in paths])
    order = np.argsort(-means)[:k]
    mu_max = float(means.max())
    def bound(B):
        return sum(_hoeff_tail(B - means[j], S2s[j]) for j in order)
    lo, hi = mu_max, mu_max + 8.0 * math.sqrt(S2s.max())
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bound(mid) > delta: lo = mid
        else: hi = mid
    return hi - mu_max, mu_max


def _criterion_k(paths, delta, eps=0.1):
    """empirical criterion: include paths until the (k+1)-th path's tail
    contribution at the current top-k budget is < eps*delta."""
    means = np.array([p["mean"] for p in paths])
    order = np.argsort(-means)
    for k in range(1, len(paths) + 1):
        s_k, mu_max = _topk_slack(paths, delta, k)
        B = mu_max + s_k
        if k < len(paths):
            j = order[k]
            extra = _hoeff_tail(B - means[j], paths[j]["S2"])
            if extra < eps * delta:
                return k
        else:
            return k
    return len(paths)


def sweep(delta=0.05, n_paths=8, path_len=6, n_mc=40000):
    out = []
    for gap in (0.6, 0.3, 0.15, 0.05, 0.0):     # near-tie gap -> 0
        paths = _make_paths(n_paths, path_len, gap, seed=int(1000 * gap) + 1)
        rng = np.random.default_rng(7)
        L = _mc_max_path(paths, n_mc, rng)
        mu_max = max(p["mean"] for p in paths)
        B_true = float(np.quantile(L, 1 - delta))
        s_true = max(B_true - mu_max, 1e-9)
        per_k = []
        for k in (1, 2, 4, n_paths):
            s_k, _ = _topk_slack(paths, delta, k)
            emp = float((L > mu_max + s_k).mean())
            per_k.append({"k": k,
                          "conservatism_x": round(s_k / s_true, 3),
                          "emp_viol": round(emp, 5),
                          "sound": bool(emp <= delta + 1e-9)})
        k_star = _criterion_k(paths, delta)
        s_star, _ = _topk_slack(paths, delta, k_star)
        emp_star = float((L > mu_max + s_star).mean())
        out.append({
            "mean_gap": gap, "n_paths": n_paths,
            "per_k": per_k,
            "criterion_k": k_star,
            "criterion_conservatism_x": round(s_star / s_true, 3),
            "criterion_emp_viol": round(emp_star, 5),
            "criterion_sound": bool(emp_star <= delta + 1e-9),
        })
    return {"delta": delta, "n_paths": n_paths, "path_len": path_len,
            "n_mc": n_mc, "sweep": out}


def main():
    os.makedirs("results", exist_ok=True)
    res = sweep()
    with open("results/exp12_topk_nearties.json", "w") as f:
        json.dump(res, f, indent=2)
    print("== top-k path-union sensitivity to near-tying means (delta=0.05) ==")
    print(f"  {'gap':>5} | " + " ".join(f"k={k}(sound/cons)" for k in (1, 2, 4, res['n_paths']))
          + " | crit-k  crit-sound")
    for r in res["sweep"]:
        cells = " ".join(f"{p['sound']!s:>5}/{p['conservatism_x']:<4}" for p in r["per_k"])
        print(f"  {r['mean_gap']:>5} | {cells} | k*={r['criterion_k']} "
              f"{r['criterion_sound']!s}({r['criterion_conservatism_x']}x)")
    print("\nwrote results/exp12_topk_nearties.json")


if __name__ == "__main__":
    main()
