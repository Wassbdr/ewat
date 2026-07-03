# Datasheet — EWAT v5 telemetry dataset (Train Ticket)

Following the *Datasheets for Datasets* framework (Gebru et al., 2021).

---

## Motivation

**For what purpose was the dataset created?**
To support research on **early warning and anomaly typing** in Kubernetes
microservice architectures — detecting and classifying failure precursors
*before* an outage, and separating benign operational drift (deployments,
autoscaling) from genuine anomalies. It was built to bridge the
*benchmark-vs-production* gap highlighted by Fu et al. (2025): most RCA datasets
are either synthetic toys or closed production traces. EWAT v5 offers a public,
reproducible middle ground — a realistic 41-service application under
ground-truth fault injection.

**Who created it and who funded it?**
Collected by Wassim Badraoui during a research internship, on a shared
Kubernetes research cluster. The underlying application (Train Ticket) and the
fault-injection tool (Chaos Mesh) are third-party open-source projects (Apache-2.0).

---

## Composition

**What do instances represent?**
Each **episode** is one time-series recording of the whole application under a
single fault scenario (or a benign/normal run), spanning a
baseline → ramp-up → injection → recovery timeline.

**How many instances?**
**409 episodes** across **24 scenarios** (611 collected; 202 rejected for >50 % missing
values). Split: **train 224 / val 47 / test 138**. Authoritative counts and the list of
rejected episodes (with reasons) are in `dataset.json`; per-episode index in `index.parquet`.

**What does each instance contain?** (per episode, under `data/<episode_id>/`)

| Artifact | Shape / type | Content |
|---|---|---|
| `signal.npz` (`signal`) | `(T, N, 18)` float32 | Multi-modal signal S(t), imputed |
| `signal_mask.npz` (`missing_mask`) | `(T, N, 18)` bool | `True` where imputed |
| `adjacency.npz` (`adjacency`) | `(T, N, N, 3)` float32 | Dynamic service graph G(t): volume, latency_med, error_rate |
| `labels.parquet` | `T` rows | Per-timestep regime/intensity/scenario/fault labels |
| `services.json` | `N` names | Canonical service order (axis N) |
| `metadata.json` | — | Scenario, timeline boundaries, feature names, quality snapshot |

`N = 41` services; `T` varies per episode (~50–120 steps on a 30 s grid; the exact length
is the first axis of each tensor).

**Signal schema (v5.1, 18 features)** — `M[0:10] | T[10:14] | L[14:18]`:

- **Metrics (M)**: cpu_util, ram_util, latency_p99, error_rate_http, net_sat,
  disk_io, mem_limit_ratio, jvm_heap_ratio, jvm_gc_util, jvm_threads_blocked
- **Traces (T)**: abnormal_span_rate, trace_depth, fan_out, latency_cv
- **Logs (L)**: log_error_rate, restart_count, semantic_anomaly, lexical_entropy

`latency_p99` / `error_rate_http` are derived from Jaeger spans (Train Ticket is
not service-meshed). `semantic_anomaly` is a **scalar** distance of log lines to
the per-service normal centroid (SentenceBERT) — **the raw log text is not
retained** in the dataset.

**Labels** (`labels.parquet`): `regime`
(normal | injection | drift_anomaly | recovery), `category`, `scenario`,
`drift_flag`, `is_injection`, `intensity_t` ∈ [0,1], `fault_type` (chaos | bug),
`bug_id`, `held_out_flag`.

**Scenarios (24)**:
- **15 mono-cause**: contention (cpu_stress, cpu_starvation, memory_stress,
  memory_pressure), gray failures (network_delay/loss/corrupt/duplicate,
  time_skew, net_delay_central), hard failures (network_partition, pod_kill,
  pod_failure, container_kill, dns_error).
- **4 compositional**: cpu_then_mem, net_loss_concurrent, pod_kill_under_delay,
  cascade_dns_net.
- **5 held-out (test-only, novelty)**: 3 novel chaos primitives (held_io_latency,
  held_net_bandwidth, held_kernel_fault) + 2 real bugs — **F3** (OOM,
  telemetry-visible: restart_count + ram_util + heap collapse) and **F1** (silent
  logic bug — see *Limitations*).

**Splits**: stratified train/val/test, with all `held_out_flag=True` episodes
routed to **test only** (novelty evaluation). See `split.json`.

**Are there missing values?** Yes — trace coverage is ~23–30/41 services per
episode; untraced services are imputed with an explicit `missing_mask`. All 41
services have metrics.

**Is the dataset self-contained?** Yes. No external resources are required; a
standalone loader (`load_ewat.py`) is included.

**Does it contain confidential / personal data?** No. The workload is fully
synthetic (automated query traffic against a fictional train-ticketing app), no
real users, no PII. Infrastructure identifiers (IPs, node names, cluster DNS,
telemetry endpoints, filesystem paths) have been **stripped** during packaging;
see `leak_audit.json` (audit report, must be clean) and the sanitization gate in
the release tooling.

---

## Collection process

**How was the data acquired?**
- **Metrics** via Prometheus (cAdvisor) + a JMX Prometheus javaagent for JVM.
- **Traces** via Jaeger (application spans).
- **Logs** via Loki (promtail), reduced to per-service scalar features.

Load was generated by a fork of `train-ticket-auto-query` (weighted mix of
booking/query flows). Faults were injected with **Chaos Mesh** (infrastructure
faults) or by **container-image swap** (real code-bug reproductions). Each
episode follows a fixed timeline (baseline → pre → injection with ramped
intensity → recovery), with the exact boundaries recorded in
`metadata.json:boundaries`.

**Over what timeframe?** Collected in mid-2026 on a single cluster.

---

## Preprocessing / labeling

- **Aggregation** per service uses differentiated rules (max for saturation,
  volume-weighted sums for rates, P99 over the union of raw distributions for
  latency, median for structural features) — never a naive mean, never a
  percentile-of-percentiles.
- Missing values are imputed **after** the mask is recorded; instance
  normalization is intentionally left to the user (not baked in).
- Labels come from the known injection timeline and the Chaos Mesh scenario —
  this is **independent ground truth**, not derived from any model.

Both raw (masked) and imputed signals exist during build; the release ships the
imputed `signal.npz` + `missing_mask` only. Raw dumps are **not** distributed.

---

## Uses

**Intended uses**: anomaly detection, failure typing/clustering, drift
detection, precursor prediction, graph neural networks on dynamic service
graphs, transfer/novelty (held-out) evaluation, ontology learning.

**Uses to avoid / caveats**: This is a *single-topology* dataset under
*synthetic* load — absolute performance numbers do not transfer directly to
arbitrary production systems. Treat held-out scenarios as the honest novelty
test; in-distribution scenario classification can be optimistic.

---

## Limitations (declared honestly)

- **Synthetic load** — automated query traffic, not organic user behavior.
- **Single topology** — Train Ticket only (N=41). No cross-application variety.
- **Partial trace coverage** — ~23–30/41 services traced per episode; the rest
  are imputed at activity-zero (mask provided).
- **F1 is telemetry-invisible** — it is a data-correctness logic bug with no
  resource/error/log signature; it is included as an **honest negative** (a
  boundary of passive-telemetry detection), consistent with the literature that
  logic bugs require business-level oracles.
- **`mem_limit_ratio` instead of `oom_events`** — the collection cluster's
  cAdvisor does not surface OOM events (all-zero series), so memory pressure is
  captured as `working_set / limit` ∈ [0,1] instead.

---

## Distribution & maintenance

- **License**: CC-BY-4.0 (see `LICENSE`). Derived from Train Ticket and Chaos
  Mesh (Apache-2.0) — please cite them (see `CITATION.cff`).
- **Distribution**: Zenodo (archival DOI) + Hugging Face Datasets (ML access).
- **Integrity**: every data file is checksummed in `SHA256SUMS`.
- **Contact / errata**: via the associated Zenodo record.
