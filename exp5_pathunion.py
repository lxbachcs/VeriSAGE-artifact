"""
EXP5 -- Path-union conservatism for parallel DAGs + a path-pruning heuristic
(Stanford review Q9, second half).

For a parallel DAG the latency is L = max_{p in Pi} sum_{i in p} ell_i over the
set Pi of source->sink paths. C5 certifies the *union bound*
    P[L > B] <= sum_{p in Pi} P[ sum_{i in p} ell_i > B ]
                 <= sum_{p in Pi} exp(-2 (B - mu_p)^2 / S2_p)       (Hoeffding)
which double-counts shared prefixes and over-counts when many paths are similar.

We measure, on random layered DAGs:
  (1) the conservatism ratio  (union-bound-implied slack) / (true MC slack) as a
      function of the number of parallel paths |Pi| and their correlation via
      shared edges;
  (2) the effect of the paper's k-highest-mean path-pruning heuristic: C5 sums
      only the top-k paths by mean. We verify this stays SOUND (empirical
      violation <= certified delta) and quantify how much slack it recovers vs
      the full union bound;
  (3) how conservative the union bound is on the near-sequential DAGs LLMs
      actually emit (|Pi| small) -- the regime the paper claims is benign.

Everything is distribution-free / Monte-Carlo-checked; no LLM.
Outputs results/exp5_pathunion.json.
"""
from __future__ import annotations
import json, os, math
import numpy as np


def _layered_dag(rng, n_layers, width, u_lo=0.5, u_hi=1.5):
    """A layered DAG: one source, one sink, `n_layers` middle layers each of
    `width` parallel nodes, fully connected layer-to-layer. Enumerate all
    source->sink paths (there are width**n_layers). Each node has bounded latency
    ell in [0,u]. Returns (paths as lists of node-indices, us array)."""
    # node ids: 0 = source, then layers, last = sink
    nodes = [("src", 0)]
    layer_nodes = []
    idx = 1
    for _ in range(n_layers):
        lyr = list(range(idx, idx + width))
        layer_nodes.append(lyr)
        idx += width
    sink = idx
    n_nodes = idx + 1
    us = rng.uniform(u_lo, u_hi, size=n_nodes)
    us[0] = 0.0            # source is a no-op
    us[sink] = rng.uniform(u_lo, u_hi)
    # enumerate paths: pick one node per layer
    import itertools
    paths = []
    for combo in itertools.product(*layer_nodes):
        paths.append([0] + list(combo) + [sink])
    return paths, us, sink


def _mc_dag_latency(paths, us, n, rng):
    """Monte-Carlo the DAG latency L=max_path sum ell, ell_i~Uniform[0,u_i]."""
    n_nodes = len(us)
    draws = rng.random((n, n_nodes)) * us      # per-node latency samples
    # path sums
    L = np.zeros(n)
    for p in paths:
        s = draws[:, p].sum(axis=1)
        L = np.maximum(L, s)
    return L


def _hoeff_tail(t, S2):
    return math.exp(-2.0 * t * t / S2) if t > 0 else 1.0


def _union_slack(paths, us, delta, topk=None):
    """Min budget slack s (above the max path mean) so the union Hoeffding bound
    over `paths` is <= delta. If topk given, only the top-k highest-mean paths
    are summed (the paper's pruning heuristic)."""
    means = np.array([0.5 * us[p].sum() for p in paths])
    S2s = np.array([float(np.sum(us[p] ** 2)) for p in paths])
    mu_max = float(means.max())
    order = np.argsort(-means)
    if topk is not None:
        order = order[:topk]
    # union bound as a function of absolute budget B; find min B with bound<=delta
    def bound(B):
        tot = 0.0
        for j in order:
            t = B - means[j]
            tot += _hoeff_tail(t, S2s[j])
        return tot
    lo, hi = mu_max, mu_max + 6.0 * math.sqrt(S2s.max())
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bound(mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi - mu_max, mu_max          # slack above max-path-mean, and that mean


def conservatism_sweep(delta=0.05, n_mc=40000, seed=31):
    rng = np.random.default_rng(seed)
    out = []
    # (n_layers, width) -> |Pi| = width**n_layers
    configs = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 3), (3, 2), (2, 4), (3, 3)]
    for n_layers, width in configs:
        n_paths = width ** n_layers
        rows = []
        for _ in range(30):                # 30 random DAGs per config
            paths, us, sink = _layered_dag(rng, n_layers, width)
            mu_max = max(0.5 * us[p].sum() for p in paths)
            # true slack from MC: (1-delta) quantile of L, minus mu_max
            L = _mc_dag_latency(paths, us, n_mc, rng)
            B_true = float(np.quantile(L, 1 - delta))
            s_true = max(B_true - mu_max, 1e-9)
            s_full, _ = _union_slack(paths, us, delta)             # full union
            s_top1, _ = _union_slack(paths, us, delta, topk=1)     # prune to 1 path
            s_top2, _ = _union_slack(paths, us, delta, topk=min(2, n_paths))
            # soundness of pruned bounds: empirical viol at the pruned budget
            emp_top1 = float((L > mu_max + s_top1).mean())
            emp_top2 = float((L > mu_max + s_top2).mean())
            emp_full = float((L > mu_max + s_full).mean())
            rows.append((s_full / s_true, s_top1 / s_true, s_top2 / s_true,
                         emp_full, emp_top1, emp_top2, s_full, s_top1))
        rows = np.array(rows)
        out.append({
            "n_layers": n_layers, "width": width, "n_paths": n_paths,
            "conservatism_full_union_x": round(float(rows[:, 0].mean()), 3),
            "conservatism_top1_prune_x": round(float(rows[:, 1].mean()), 3),
            "conservatism_top2_prune_x": round(float(rows[:, 2].mean()), 3),
            "slack_recovered_top1_vs_full_pct":
                round(100 * float((1 - rows[:, 7] / rows[:, 6]).mean()), 2),
            "empirical_viol_full_max": round(float(rows[:, 3].max()), 5),
            "empirical_viol_top1_max": round(float(rows[:, 4].max()), 5),
            "empirical_viol_top2_max": round(float(rows[:, 5].max()), 5),
            "top1_sound_frac": round(float((rows[:, 4] <= delta + 1e-9).mean()), 3),
            "top2_sound_frac": round(float((rows[:, 5] <= delta + 1e-9).mean()), 3),
            "full_sound_frac": round(float((rows[:, 3] <= delta + 1e-9).mean()), 3),
        })
    return {"delta": delta, "n_mc": n_mc, "configs": out}


def main():
    os.makedirs("../results", exist_ok=True)
    res = {"conservatism": conservatism_sweep()}
    with open("../results/exp5_pathunion.json", "w") as f:
        json.dump(res, f, indent=2)
    print("== path-union conservatism (slack_union / slack_true), delta=0.05 ==")
    print(f"{'|Pi|':>5} {'full':>7} {'top1':>7} {'top2':>7}  {'full_sound':>10} {'top1_sound':>10}")
    for c in res["conservatism"]["configs"]:
        print(f"{c['n_paths']:>5} {c['conservatism_full_union_x']:>7.2f} "
              f"{c['conservatism_top1_prune_x']:>7.2f} {c['conservatism_top2_prune_x']:>7.2f}  "
              f"{c['full_sound_frac']:>10.2f} {c['top1_sound_frac']:>10.2f}")
    print("\nwrote ../results/exp5_pathunion.json")


if __name__ == "__main__":
    main()
