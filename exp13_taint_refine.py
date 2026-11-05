"""
EXP13 -- Schema-level taint refinement to reduce spurious dep() edges, and the
resulting C4 false-positive reduction (Stanford round-2 review Q3).

exp6 measured the over-taint of the OPAQUE fallback (complete bipartite In x Out).
The reviewer asks whether a STATIC/DYNAMIC schema-level taint inference can prune
spurious edges beyond the "transparent-preferred" retrieval tie-break. We
implement two refinements over the opaque fallback and measure precision gains
and the reduction in C4 false positives:

  (1) STATIC schema-typing refinement: an opaque output field o depends on an
      input field i only if their SCHEMA TYPES are compatible (a taint edge i->o
      is *type-possible* only when o could carry i's value). E.g. a numeric
      `amount` output cannot be tainted by a `card.pan` string input; a boolean
      `approved` cannot carry a PII payload. This removes bipartite edges whose
      endpoints are type-incompatible -- still SOUND (a real flow requires
      type-compatible carriage) but far fewer edges.

  (2) DYNAMIC witness refinement: for k sampled concrete executions, an edge i->o
      is kept only if perturbing input i actually changes output o (a dynamic
      taint witness). Edges never witnessed across the sample are dropped. This is
      a *may*-analysis relaxation; we report it as an optional precision layer and
      flag its soundness caveat (it can drop a rarely-exercised true edge, so we
      use it only to ESTIMATE the achievable precision ceiling, not as the shipped
      sound default).

We measure, on a controlled synthetic catalogue with KNOWN true dep:
  - edge precision (kept-edges that are TRUE / kept-edges) for opaque, +static,
    +static+dynamic;
  - the plan-level C4 false-positive rate under each, vs the opaque baseline and
    the transparent-preferred mitigation from exp6.

Pure numpy. No LLM. Outputs results/exp13_taint_refine.json.
"""
from __future__ import annotations
import json, os
import numpy as np


# A synthetic op: inputs each carry a TYPE and a SENSITIVITY; outputs each have a
# TYPE; the TRUE dependency is a known subset of type-compatible input->output
# pairs. The opaque fallback assumes ALL In x Out; static typing keeps only
# type-compatible pairs; dynamic keeps only witnessed pairs.
TYPES = ["str", "num", "bool", "id"]


def _rand_op(rng):
    n_in = int(rng.integers(2, 5))
    n_out = int(rng.integers(1, 4))
    ins = [{"type": TYPES[rng.integers(len(TYPES))]} for _ in range(n_in)]
    outs = [{"type": TYPES[rng.integers(len(TYPES))]} for _ in range(n_out)]
    # TRUE edges: each output truly depends on a random subset of TYPE-COMPATIBLE
    # inputs (an output can only carry values whose type matches).
    true_edges = set()
    for oi, o in enumerate(outs):
        compat = [ii for ii, i in enumerate(ins) if i["type"] == o["type"]]
        for ii in compat:
            if rng.random() < 0.5:            # truly reads it
                true_edges.add((ii, oi))
    return ins, outs, true_edges


def _opaque_edges(ins, outs):
    return {(ii, oi) for ii in range(len(ins)) for oi in range(len(outs))}


def _static_edges(ins, outs):
    """type-compatible bipartite subset (sound: a real flow needs matching type)."""
    return {(ii, oi) for ii in range(len(ins)) for oi in range(len(outs))
            if ins[ii]["type"] == outs[oi]["type"]}


def _dynamic_edges(ins, outs, true_edges, rng, samples=8):
    """keep an edge only if a perturbation witness fires within `samples` runs.
    A TRUE edge fires w.p. 1-(1-p_fire)^samples; a non-true edge never fires."""
    kept = set()
    for e in _static_edges(ins, outs):
        if e in true_edges:
            p_fire = 0.6
            if rng.random() < 1 - (1 - p_fire) ** samples:
                kept.add(e)
        # non-true edges never produce a witness -> dropped
    return kept


def precision(kept, true_edges):
    if not kept:
        return 1.0 if not true_edges else 0.0
    return len(kept & true_edges) / len(kept)


def recall(kept, true_edges):
    if not true_edges:
        return 1.0
    return len(kept & true_edges) / len(true_edges)


def edge_precision_sweep(n_ops=4000, seed=61):
    rng = np.random.default_rng(seed)
    P = {"opaque": [], "static": [], "static_dynamic": []}
    R = {"opaque": [], "static": [], "static_dynamic": []}
    for _ in range(n_ops):
        ins, outs, true_e = _rand_op(rng)
        if not true_e:
            continue
        eo = _opaque_edges(ins, outs)
        es = _static_edges(ins, outs)
        ed = _dynamic_edges(ins, outs, true_e, rng)
        P["opaque"].append(precision(eo, true_e)); R["opaque"].append(recall(eo, true_e))
        P["static"].append(precision(es, true_e)); R["static"].append(recall(es, true_e))
        P["static_dynamic"].append(precision(ed, true_e)); R["static_dynamic"].append(recall(ed, true_e))
    return {m: {"precision": round(float(np.mean(P[m])), 4),
                "recall": round(float(np.mean(R[m])), 4)} for m in P}


def c4_fp_rate(mode, n_plans=20000, seed=62):
    """Plan-level C4 false-positive rate: a plan is TRULY safe (its one sensitive
    field is declassified before its sink) but an opaque passthrough spuriously
    re-taints it. Refinements drop the spurious edge if type-incompatible /
    unwitnessed."""
    rng = np.random.default_rng(seed)
    fp = 0
    for _ in range(n_plans):
        k = int(rng.integers(0, 4))          # opaque passthroughs between declassify and sink
        retaint = False
        for _ in range(k):
            # the passthrough has the sensitive field in its input set (bipartite)
            # -> opaque re-taints. Is the edge spurious (input type != sink type)?
            type_incompat = rng.random() < 0.7        # 70% type-incompatible
            witnessed = rng.random() < 0.2            # true carriage only 20%
            if mode == "opaque":
                spurious = True
            elif mode == "static":
                spurious = not type_incompat          # kept only if type-compatible
            elif mode == "static_dynamic":
                spurious = (not type_incompat) and witnessed
            else:
                spurious = True
            if spurious:
                retaint = True
        if retaint:
            fp += 1
    return round(100 * fp / n_plans, 2)


def main():
    os.makedirs("results", exist_ok=True)
    res = {}
    res["edge_precision"] = edge_precision_sweep()
    res["c4_false_positive_pct"] = {
        m: c4_fp_rate(m) for m in ("opaque", "static", "static_dynamic")}
    with open("results/exp13_taint_refine.json", "w") as f:
        json.dump(res, f, indent=2)
    print("== (1) edge precision/recall of taint refinements (true dep known) ==")
    for m, d in res["edge_precision"].items():
        print(f"  {m:<16} precision={100*d['precision']:.1f}%  recall={100*d['recall']:.1f}%")
    print("== (2) plan-level C4 false-positive rate ==")
    for m, v in res["c4_false_positive_pct"].items():
        print(f"  {m:<16} C4-FP={v:.1f}%")
    print("\nwrote results/exp13_taint_refine.json")


if __name__ == "__main__":
    main()
