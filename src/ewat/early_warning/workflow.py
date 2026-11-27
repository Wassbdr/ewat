"""Train-or-load, single-episode replay and the seeded synthetic benchmark."""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ewat.early_warning.pipeline import EarlyWarningPipeline, PipelineConfig
from ewat.synthetic.generator import BENIGN_DRIFT, FAULT_TYPES, SCENARIOS, generate_episode

TRAIN_SEED0 = 10_000  # training episodes: seeds 10_000 … 10_000 + n_train - 1
N_TRAIN_PER_SCENARIO = 30
DEFAULT_MODEL = Path(os.environ.get("EWAT_MODEL", Path.home() / ".cache" / "ewat" / "synthetic.pt"))


def hardware() -> str:
    cpu = platform.processor() or platform.machine()
    return f"{cpu}, {os.cpu_count()} CPU threads, no GPU, torch {torch.__version__}"


def train(path: Path = DEFAULT_MODEL, log: bool = True) -> EarlyWarningPipeline:
    episodes = [
        generate_episode(s, TRAIN_SEED0 + i) for s in SCENARIOS for i in range(N_TRAIN_PER_SCENARIO)
    ]
    t0 = time.perf_counter()
    pipe = EarlyWarningPipeline.fit(episodes, PipelineConfig(), log=log)
    if log:
        print(
            f"trained on {len(episodes)} synthetic episodes (seeds {TRAIN_SEED0}+) "
            f"in {time.perf_counter() - t0:.1f} s  σ={pipe.sigma:.3f}  ε={pipe.epsilon:.4f}"
        )
    pipe.save(path)
    return pipe


def get_pipeline(
    path: Path = DEFAULT_MODEL, retrain: bool = False, log: bool = True
) -> EarlyWarningPipeline:
    path = Path(path)
    if path.exists() and not retrain:
        return EarlyWarningPipeline.load(path)
    if log:
        print(f"no model at {path} — training one (≈20 s on CPU)")
    return train(path, log=log)


def first_alert_outcome(episode, run) -> dict:
    a = run.first_alert
    out = {
        "scenario": episode.scenario,
        "seed": episode.seed,
        "target": episode.target,
        "drift_start": episode.drift_start,
        "fault_start": episode.fault_start,
        "alert_step": a.t if a else None,
        "alert_type": a.alert_type if a else None,
    }
    if episode.fault_start is None:
        out["false_alarm"] = a is not None
    else:
        early = a is not None and a.t < episode.fault_start
        out["false_alarm"] = early
        out["detected"] = a is not None and not early
        out["delay_steps"] = (a.t - episode.fault_start) if out["detected"] else None
        out["type_correct"] = (a.alert_type == episode.scenario) if out["detected"] else None
    # MMD-only baseline: alert on 2 consecutive steps above ε, no typing.
    above = run.array("mmd2") >= run.steps[0].epsilon
    above &= run.array("regime") != "warmup"
    two = np.flatnonzero(above[1:] & above[:-1])
    out["mmd_only_alert_step"] = int(two[0] + 1) if len(two) else None
    return out


def benchmark(pipe: EarlyWarningPipeline, n_per_scenario: int = 50, seed0: int = 0) -> dict:
    rows, lat = [], []
    for s in SCENARIOS:
        for i in range(n_per_scenario):
            e = generate_episode(s, seed0 + i)
            r = pipe.run(e.signal, e.adjacency)
            lat.extend(r.array("latency_ms").tolist())
            rows.append(first_alert_outcome(e, r))

    def rate(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    drift_rows = [r for r in rows if r["scenario"] == BENIGN_DRIFT]
    fault_rows = [r for r in rows if r["scenario"] != BENIGN_DRIFT]
    per_type = {}
    for f in FAULT_TYPES:
        fr = [r for r in fault_rows if r["scenario"] == f]
        d = [r["delay_steps"] for r in fr if r["detected"]]
        per_type[f] = {
            "episodes": len(fr),
            "detection_rate": rate([r["detected"] for r in fr]),
            "type_accuracy": rate([r["type_correct"] for r in fr]),
            "median_delay_s": float(np.median(d)) * 30.0 if d else None,
        }
    delays = [r["delay_steps"] for r in fault_rows if r["detected"]]

    # MMD-only baseline false alarm: fires before the fault (or at all, drift-only)
    def mmd_fa(r):
        a = r["mmd_only_alert_step"]
        return a is not None and (r["fault_start"] is None or a < r["fault_start"])

    lat = np.array(lat)
    return {
        "setup": {
            "data": "synthetic (ewat.synthetic), 6 services × 17 features, 30 s steps",
            "eval_seeds": f"{seed0}..{seed0 + n_per_scenario - 1} per scenario",
            "train_seeds": f"{TRAIN_SEED0}..{TRAIN_SEED0 + N_TRAIN_PER_SCENARIO - 1} per scenario",
            "episodes": len(rows),
            "episodes_per_scenario": n_per_scenario,
            "hardware": hardware(),
            "config": pipe.config.__dict__,
            "epsilon": pipe.epsilon,
            "sigma": pipe.sigma,
        },
        "false_alarm_rate_drift_only": rate([r["false_alarm"] for r in drift_rows]),
        "false_alarm_rate_all": rate([r["false_alarm"] for r in rows]),
        "mmd_only_false_alarm_rate_drift_only": rate([mmd_fa(r) for r in drift_rows]),
        "detection_rate": rate([r["detected"] for r in fault_rows]),
        "median_delay_s": float(np.median(delays)) * 30.0 if delays else None,
        "p90_delay_s": round(float(np.percentile(delays, 90)) * 30.0, 1) if delays else None,
        "type_accuracy": rate([r["type_correct"] for r in fault_rows]),
        "per_type": per_type,
        "step_latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
        },
        "episodes": rows,
    }
