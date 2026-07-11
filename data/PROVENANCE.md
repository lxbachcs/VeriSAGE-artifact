# Real dataset provenance

## QWS Dataset v2.0 (`qws2_dataset.txt`)

- **What:** measured Quality of Service for **2,507 real Web services** found on
  the public Web. Comma-separated; the first nine fields are numeric QoS
  attributes: (1) Response Time [ms], (2) Availability, (3) Throughput,
  (4) Successability, (5) Reliability, (6) Compliance, (7) Best Practices,
  (8) Latency [ms], (9) Documentation, then (10) Service Name, (11) WSDL Address.
- **Source:** Eyhab Al-Masri and Qusay H. Mahmoud, University of Guelph.
  Official page: https://qwsdata.github.io/ (Zenodo `qws2.zip`). Retrieved
  2026-07 from a public verbatim mirror; committed unmodified here.
- **Citation:** E. Al-Masri and Q. H. Mahmoud, "Investigating Web Services on the
  World Wide Web," WWW 2008 / "QoS-based Discovery and Ranking of Web Services,"
  ICCCN 2007.
- **How VeriSAGE uses it:** the measured **Response Time** column is the empirical
  population of real service latencies (ms) that grounds the SLA/QoS experiments.
  - *RQ2 soundness (`exp4_bernstein.py`, `qws_real` model):* per-invocation
    service latencies are bootstrap draws from this real population; the
    distribution-free C5 bounds (Hoeffding / Bernstein / Bennett) are verified to
    still hold empirically on the real heavy-tailed data (skew +4.39).
  - *QoS-optimizer baseline (`exp10_qos_baseline.py`):* catalogue service
    latencies are grounded in real QWS response times.
  Loader: `qws_data.py`. No distributional form is assumed; only a support bound
  (observed max) and a second-moment bound are read off the data.

Real Response-Time distribution: n=2507, mean=383.8 ms, median=226.6 ms,
P95=1340.4 ms, P99=3287.7 ms, max=4989.7 ms.
