"""
qws_data.py -- loader for the REAL QWS (Quality of Web Service) dataset.

The QWS Dataset v2.0 (Al-Masri & Mahmoud, 2007/2008) reports measured Quality of
Service for 2,507 real Web services discovered on the public Web. We use its
measured per-service *Response Time* (column 1, milliseconds) as the empirical
population of real service latencies that grounds VeriSAGE's SLA/QoS experiments
(RQ2 soundness, the QoS-optimizer baseline). No distributional form is assumed:
the distribution-free C5 certificate needs only a support bound and (for the
variance-aware bounds) a second-moment bound, both read off the real data.

Source: https://qwsdata.github.io/  (Zenodo mirror), retrieved 2026-07;
committed verbatim at data/qws2_dataset.txt with data/PROVENANCE.md.

Real distribution (Response Time, ms): n=2507, mean=383.8, median=226.6,
P95=1340.4, P99=3287.7, max=4989.7, skew=+4.39 (heavy right tail -- a genuine
stress test for a distribution-free tail bound).
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import Tuple
import numpy as np

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qws2_dataset.txt")

# QWS v2 column order (comma-separated, first nine numeric):
# 1 ResponseTime  2 Availability  3 Throughput  4 Successability  5 Reliability
# 6 Compliance    7 BestPractices 8 Latency      9 Documentation
_COL_RESPONSE_TIME = 0
_COL_LATENCY = 7


@lru_cache(maxsize=2)
def load_qws(column: int = _COL_RESPONSE_TIME) -> np.ndarray:
    """Return the real per-service latency population (ms) as a float array."""
    vals = []
    with open(_DATA) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            try:
                vals.append(float(parts[column]))
            except (ValueError, IndexError):
                continue
    arr = np.asarray(vals, dtype=float)
    if arr.size < 2000:
        raise RuntimeError(f"QWS load failed: only {arr.size} rows parsed from {_DATA}")
    return arr


def qws_stats(column: int = _COL_RESPONSE_TIME) -> dict:
    a = load_qws(column)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
        "var": float(a.var(ddof=0)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
        "min": float(a.min()),
        "skew": float(((a - a.mean()) ** 3).mean() / a.std() ** 3),
    }


def support_ub(column: int = _COL_RESPONSE_TIME) -> float:
    """A real, dataset-derived support upper bound u_i for a single service:
    the observed maximum real response time. Every real latency <= this, so the
    distribution-free bound stays honest."""
    return float(load_qws(column).max())


def bootstrap_latencies(rng: np.random.Generator, shape) -> np.ndarray:
    """Draw i.i.d. per-invocation service latencies (ms) by bootstrap resampling
    the real QWS response-time population -- the actual measured distribution,
    heavy tail and all. Used as the *empirical* latency model to verify the
    distribution-free SLA bounds hold on real data."""
    pop = load_qws()
    idx = rng.integers(0, pop.size, size=shape)
    return pop[idx]


def sample_service_means(rng: np.random.Generator, n: int) -> np.ndarray:
    """Assign n catalogue services a real mean latency each, by resampling QWS
    response times. Grounds the catalogue QoS contracts in measured data."""
    return bootstrap_latencies(rng, (n,))
