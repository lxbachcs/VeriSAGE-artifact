"""
A fixed service catalogue + goal for the LLM-proposer experiment. The LLM is
given the catalogue and an NL goal and must emit an ordered list of service
names; we map names -> Service objects and run the verifier. Hallucinated names
(not in the catalogue) are dropped and counted (HAR).
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import copy, os
from core import Service, Goal


def _qws_on() -> bool:
    """Ground catalogue QoS in the REAL QWS dataset when VERISAGE_QWS=1."""
    return os.environ.get("VERISAGE_QWS", "0") == "1"


def _qws_scale() -> float:
    """Real mean service latency (ms) from QWS, used as the QoS unit when grounded."""
    import qws_data
    return qws_data.qws_stats()["mean"]


def _apply_qws(C: Dict[str, Service], seed: int = 1234) -> None:
    """Reassign each service a real per-service latency profile drawn from the
    QWS response-time population (deterministic, seeded). lat_mean = a real
    measured response time; lat_ub = 2*mean keeps the Uniform[0,ub] support
    model the C5 certificate assumes. Distractor SLA tension is preserved
    because budgets scale with the same real unit (see make_goal)."""
    import numpy as np, qws_data
    rng = np.random.default_rng(seed)
    means = qws_data.sample_service_means(rng, len(C))
    for s, m in zip(C.values(), means):
        s.lat_mean = float(m)
        s.lat_ub = float(2.0 * m)


def build_catalogue() -> Dict[str, Service]:
    ub = 30.0
    C: Dict[str, Service] = {}
    C["readCard"] = Service("readCard", "readcard", set(), {"card.pan"},
                            lat_ub=ub, lat_mean=ub / 2, field_level={"card.pan": "pii"})
    C["authenticate"] = Service("authenticate", "authenticate", set(), {"token"},
                                post_props={"authed"}, lat_ub=ub, lat_mean=ub / 2,
                                field_level={"token": "internal"})
    C["tokenize"] = Service("tokenize", "tokenize", {"card.pan"}, {"card.tok"},
                            declassifies={"card.pan"}, lat_ub=ub, lat_mean=ub / 2,
                            field_level={"card.tok": "public"})
    C["chargeCard"] = Service("chargeCard", "charge", {"token", "card.tok"}, {"charge.id"},
                              pre_props={"authed"}, post_props={"charged"},
                              lat_ub=ub, lat_mean=ub / 2, field_level={"charge.id": "internal"})
    C["emailReceipt"] = Service("emailReceipt", "receipt", {"charge.id"}, {"receipt.id"},
                                pre_props={"charged"}, post_props={"receipted"},
                                lat_ub=ub, lat_mean=ub / 2, field_level={"receipt.id": "public"})
    C["analyticsSink"] = Service("analyticsSink", "analytics", {"card.tok"}, {"metric"},
                                 lat_ub=ub, lat_mean=ub / 2,
                                 sink_field="card.pan", sink_clearance="public")
    # distractors (plausible but usually unnecessary; padding them can breach SLA)
    C["fraudCheck"] = Service("fraudCheck", "fraud", {"token"}, {"fraud.ok"}, lat_ub=ub, lat_mean=ub / 2)
    C["shipOrder"] = Service("shipOrder", "ship", {"charge.id"}, {"ship.id"}, lat_ub=ub, lat_mean=ub / 2)
    C["logEvent"] = Service("logEvent", "log", set(), {"log.id"}, lat_ub=ub, lat_mean=ub / 2)
    C["refund"] = Service("refund", "refund", {"charge.id"}, {"refund.id"}, lat_ub=ub, lat_mean=ub / 2)
    if _qws_on():
        _apply_qws(C)
    return C


GOAL_NL = (
    "Compose services (only from the catalogue, in a valid execution order) to "
    "charge the customer's card, email them a receipt, and record an anonymized "
    "analytics metric -- while protecting the customer's raw payment data and "
    "staying within the latency SLA budget. Follow standard secure-payment "
    "practice."
)


def catalogue_prompt(cat: Dict[str, Service]) -> str:
    lines = []
    for s in cat.values():
        req = ",".join(sorted(s.requires)) or "-"
        prod = ",".join(sorted(s.produces)) or "-"
        extra = []
        if s.declassifies:
            extra.append(f"declassifies {','.join(s.declassifies)}")
        if s.sink_field:
            extra.append(f"PUBLIC sink of {s.sink_field}")
        if s.pre_props:
            extra.append(f"needs {','.join(s.pre_props)}")
        tag = f" [{'; '.join(extra)}]" if extra else ""
        lines.append(f"  {s.name}: in=[{req}] out=[{prod}]{tag}")
    return "\n".join(lines)


def make_goal(rng) -> Goal:
    # budget: enough for the ~6-service minimal correct plan, tight enough that
    # padding 2-3 distractors risks an SLA-bound failure.
    ub = (2.0 * _qws_scale()) if _qws_on() else 30.0   # real QWS unit when grounded
    n_min = 6                      # readCard,authenticate,tokenize,charge,receipt,analytics
    mu_min = n_min * ub / 2
    slack = rng.uniform(1.8, 2.6) * ((n_min) * ub ** 2) ** 0.5
    return Goal(init_fields=set(), goal_fields={"receipt.id", "metric"},
                goal_props={"receipted"},
                phi=[("precede", "authenticate", "charge"),
                     ("response", "charge", "receipt")],
                budget=float(mu_min + slack), delta=0.05, init_level={})


def plan_from_names(names: List[str], cat: Dict[str, Service]) -> Tuple[List[Service], int]:
    """Map proposed names to Services; return (plan, n_hallucinated)."""
    plan, halluc = [], 0
    for nm in names:
        key = _match(nm, cat)
        if key is None:
            halluc += 1
        else:
            plan.append(copy.deepcopy(cat[key]))
    return plan, halluc


def _match(nm: str, cat: Dict[str, Service]):
    nm2 = nm.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    for k in cat:
        if k.lower() == nm2:
            return k
    for k in cat:                  # loose contains match
        if k.lower() in nm2 or nm2 in k.lower():
            return k
    return None
