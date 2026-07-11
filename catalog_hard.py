"""
HARD suite: a regulated payment + PII pipeline. Larger catalogue with distractors
and traps; multiple sensitive fields (card.pan:pii, ssn:secret, email:pii) each
needing its OWN declassifier before its OWN sink; several interacting ordering /
precondition constraints (consent, KYC, fraud, auth all before charge); and a
tight SLA budget so any padding breaks it. Designed so even strong proposers slip
single-shot, giving the repair loop a visible recovery margin.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import copy
from core import Service, Goal
from catalog import _qws_on, _qws_scale, _apply_qws

UB = 24.0


def _s(name, action, req, prod, **kw):
    return Service(name, action, set(req), set(prod), lat_ub=UB, lat_mean=UB / 2, **kw)


def build_catalogue_hard() -> Dict[str, Service]:
    C: Dict[str, Service] = {}
    # sources of sensitive data
    C["readCard"]     = _s("readCard", "readcard", [], ["card.pan"], field_level={"card.pan": "pii"})
    C["readCustomer"] = _s("readCustomer", "readcust", [], ["ssn", "email"],
                           field_level={"ssn": "secret", "email": "pii"})
    C["readAddress"]  = _s("readAddress", "readaddr", [], ["address"], field_level={"address": "internal"})
    # identity / compliance (all must precede charge)
    C["authenticate"] = _s("authenticate", "authenticate", [], ["token"], post_props={"authed"},
                           field_level={"token": "internal"})
    C["getConsent"]   = _s("getConsent", "consent", ["email"], ["consent.id"], post_props={"consented"},
                           field_level={"consent.id": "internal"})
    C["kycCheck"]     = _s("kycCheck", "kyc", ["ssn", "token"], ["kyc.ok"], pre_props={"authed"},
                           post_props={"kyc_ok"}, field_level={"kyc.ok": "internal"})
    C["fraudCheck"]   = _s("fraudCheck", "fraud", ["token"], ["fraud.ok"], post_props={"fraud_ok"},
                           field_level={"fraud.ok": "internal"})
    # declassifiers (one per sensitive field)
    C["tokenize"]     = _s("tokenize", "tokenize", ["card.pan"], ["card.tok"],
                           declassifies={"card.pan"}, field_level={"card.tok": "public"})
    C["redactSSN"]    = _s("redactSSN", "redact", ["ssn"], ["ssn.red"],
                           declassifies={"ssn"}, field_level={"ssn.red": "internal"})
    C["hashEmail"]    = _s("hashEmail", "hashemail", ["email"], ["email.hash"],
                           declassifies={"email"}, field_level={"email.hash": "public"})
    # core actions
    C["chargeCard"]   = _s("chargeCard", "charge", ["token", "card.tok"], ["charge.id"],
                           pre_props={"authed", "consented", "kyc_ok", "fraud_ok"},
                           post_props={"charged"}, field_level={"charge.id": "internal"})
    C["emailReceipt"] = _s("emailReceipt", "receipt", ["charge.id", "email.hash"], ["receipt.id"],
                           pre_props={"charged"}, post_props={"receipted"},
                           field_level={"receipt.id": "public"})
    C["shipOrder"]    = _s("shipOrder", "ship", ["charge.id", "address"], ["ship.id"],
                           pre_props={"charged"}, post_props={"shipped"}, field_level={"ship.id": "internal"})
    # sinks (each leaks its field unless declassified first)
    C["analyticsSink"] = _s("analyticsSink", "analytics", [], ["metric"],
                            sink_field="card.pan", sink_clearance="public")
    C["auditLog"]      = _s("auditLog", "audit", [], ["audit.id"],
                            sink_field="ssn", sink_clearance="internal")
    # distractors / traps
    C["quickCharge"]  = _s("quickCharge", "charge", ["card.pan"], ["charge.id"],
                           post_props={"charged"}, field_level={"charge.id": "internal"})  # skips auth: trap
    C["logEvent"]     = _s("logEvent", "log", [], ["log.id"])
    C["notifyUser"]   = _s("notifyUser", "notify", ["email"], ["notif.id"])
    C["refund"]       = _s("refund", "refund", ["charge.id"], ["refund.id"])
    C["upsell"]       = _s("upsell", "upsell", ["charge.id"], ["offer.id"])
    if _qws_on():
        _apply_qws(C, seed=5678)
    return C


GOAL_NL_HARD = (
    "Compose services (only from the catalogue, in a valid execution order) to "
    "charge the customer, email them a receipt, publish an anonymized analytics "
    "metric, and write a compliance audit record. Protect all sensitive customer "
    "data (card number, SSN, email) end-to-end, satisfy all regulatory "
    "preconditions before charging, and stay within the latency SLA budget. Use "
    "standard secure, compliant payment practice; include no unnecessary services."
)


def make_goal_hard(rng) -> Goal:
    # minimal required plan is ~13 services; budget admits it but not padding.
    n_req = 13
    ub_unit = (2.0 * _qws_scale()) if _qws_on() else UB   # real QWS unit when grounded
    mu = n_req * ub_unit / 2
    sum_ub2 = n_req * ub_unit ** 2
    slack = rng.uniform(1.35, 1.7) * (sum_ub2 ** 0.5)
    return Goal(
        init_fields=set(),
        goal_fields={"receipt.id", "metric", "audit.id"},
        goal_props={"receipted"},
        phi=[("precede", "authenticate", "charge"),
             ("precede", "consent", "charge"),
             ("precede", "kyc", "charge"),
             ("precede", "fraud", "charge"),
             ("response", "charge", "receipt")],
        budget=float(mu + slack), delta=0.05, init_level={})


# a known-correct reference plan (for self-test)
REFERENCE = ["readCard", "readCustomer", "authenticate", "getConsent", "kycCheck",
             "fraudCheck", "tokenize", "redactSSN", "hashEmail", "chargeCard",
             "emailReceipt", "analyticsSink", "auditLog"]
