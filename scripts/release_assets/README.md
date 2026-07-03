---
license: cc-by-4.0
pretty_name: "EWAT v5 — Microservice Telemetry (Train Ticket + Chaos Mesh)"
tags:
  - microservices
  - observability
  - anomaly-detection
  - concept-drift
  - time-series
  - graph
  - kubernetes
size_categories:
  - n<1K
---

# EWAT v5 — Early-Warning & Anomaly-Typing Telemetry Dataset

Multi-modal telemetry (**metrics + traces + logs**) collected on a Kubernetes
deployment of the **Train Ticket** microservice benchmark (41 Spring Cloud
services), with **Chaos Mesh** ground-truth fault injection across **24
scenarios**. Each episode is an aligned time series of a signal tensor **S(t)**,
a **dynamic service graph G(t)**, a missingness mask, and per-timestep
regime/intensity labels — with a **held-out novelty split** (5 scenarios test-only).

Built to bridge the *benchmark-vs-production* gap in microservice RCA / early-warning research (Fu et al., 2025).

`🔒 leak-audited` — every file passes an infrastructure-leak audit; see [`leak_audit.json`](leak_audit.json).

## Data at a glance

| | |
|---|---|
| Episodes | **409** (611 collected − 202 rejected for >50 % missing) |
| Split | train 224 / val 47 / test 138 (stratified; held-out → test-only) |
| Services (N) | 41 |
| Timesteps (T) | variable per episode (~50–120, 30 s grid) |
| Signal features | 18 (schema v5.1, see `schema.json`) |
| Scenarios | **24** (15 mono + 4 compositional + 5 held-out: 3 novel chaos + F1/F3 bugs) |
| Ground truth | Chaos Mesh scenario + injection timeline (model-independent) |

### Signal schema (per timestep, per service) — `M[0:10] | T[10:14] | L[14:18]`

- **Metrics**: cpu_util, ram_util, latency_p99, error_rate_http, net_sat,
  disk_io, mem_limit_ratio, jvm_heap_ratio, jvm_gc_util, jvm_threads_blocked
- **Traces**: abnormal_span_rate, trace_depth, fan_out, latency_cv
- **Logs**: log_error_rate, restart_count, semantic_anomaly, lexical_entropy

`adjacency.npz` holds G(t) `(T, N, N, 3)`: edge dims = (volume, latency_med, error_rate).

### Labels (`labels.parquet`)

`regime` (normal | injection | drift_anomaly | recovery), `scenario`,
`category`, `drift_flag`, `is_injection`, `intensity_t` ∈ [0,1],
`fault_type` (chaos | bug), `bug_id`, `held_out_flag`.

## Quickstart

```python
from load_ewat import load_episode, iter_split

# one episode
signal, mask, adjacency, labels, services = load_episode("data/<episode_id>")
print(signal.shape, adjacency.shape, len(services))   # (T, 41, 18) (T, 41, 41, 3) 41

# iterate a split
for ep_id, signal, mask, adjacency, labels, services in iter_split(".", "test"):
    ...
```

`signal` is imputed; `mask` is `True` where a value was imputed. Instance
normalization is intentionally **not** applied — do it yourself if needed.

## Files

```
data/<episode_id>/  signal.npz, signal_mask.npz, adjacency.npz,
                    labels.parquet, services.json, metadata.json
dataset.json, index.parquet, split.json, services.json, summary.csv
schema.json, load_ewat.py, DATASHEET.md, LICENSE, CITATION.cff, SHA256SUMS
```

Verify integrity: `sha256sum -c SHA256SUMS`.

## Limitations (read before benchmarking)

Synthetic load; single topology (Train Ticket); ~23–30/41 services traced per
episode (rest imputed, mask provided); **F1** is an honest telemetry-invisible
negative; `mem_limit_ratio` replaces `oom_events` on the collection cluster.
Full discussion in [`DATASHEET.md`](DATASHEET.md).

## License & citation

CC-BY-4.0 (see [`LICENSE`](LICENSE)). If you use this dataset, cite it (see
[`CITATION.cff`](CITATION.cff)) **and** the upstream projects:

- **Train Ticket** — FudanSELab, https://github.com/FudanSELab/train-ticket (Apache-2.0)
- **Chaos Mesh** — https://github.com/chaos-mesh/chaos-mesh (Apache-2.0)
