"""Checks of the synthetic early-warning pipeline (CPU, ≈25 s)."""

import dataclasses

import numpy as np
import pytest

from ewat.early_warning import CLASSES, EarlyWarningPipeline, PipelineConfig
from ewat.early_warning.workflow import (
    N_TRAIN_PER_SCENARIO,
    TRAIN_SEED0,
    benchmark,
    first_alert_outcome,
)
from ewat.synthetic import BENIGN_DRIFT, FAULT_TYPES, SCENARIOS, generate_episode


@pytest.fixture(scope="module")
def pipe():
    """The shipped configuration (same training set as `ewat train`, ≈15 s on CPU)."""
    train = [
        generate_episode(s, TRAIN_SEED0 + i) for s in SCENARIOS for i in range(N_TRAIN_PER_SCENARIO)
    ]
    return EarlyWarningPipeline.fit(train, PipelineConfig())


def test_classes():
    assert CLASSES == ["normal", BENIGN_DRIFT, *FAULT_TYPES]


def test_threshold_calibrated(pipe):
    assert pipe.epsilon > 0 and pipe.sigma > 0


def test_run_is_causal(pipe):
    e = generate_episode("fail_slow_latency", 0)
    full = pipe.run(e.signal, e.adjacency)
    cut = e.fault_start + 3
    part = pipe.run(e.signal[:cut], e.adjacency[:cut])
    np.testing.assert_allclose(full.array("mmd2")[:cut], part.array("mmd2"), rtol=1e-5)
    assert full.steps[0].regime == "warmup"


def test_too_short_window(pipe):
    e = generate_episode("fail_slow_latency", 0)
    with pytest.raises(ValueError):
        pipe.run(e.signal[:5], e.adjacency[:5])


def test_benign_drift_is_not_an_alert(pipe):
    alarms = 0
    for seed in range(5):
        e = generate_episode(BENIGN_DRIFT, seed)
        alarms += pipe.run(e.signal, e.adjacency).first_alert is not None
    assert alarms == 0


def test_latency_fault_is_detected_and_typed(pipe):
    hits = 0
    for seed in range(5):
        e = generate_episode("fail_slow_latency", seed)
        out = first_alert_outcome(e, pipe.run(e.signal, e.adjacency))
        hits += bool(out["detected"] and out["type_correct"])
    assert hits >= 3


def test_save_load_roundtrip(pipe, tmp_path):
    path = tmp_path / "m.pt"
    pipe.save(path)
    other = EarlyWarningPipeline.load(path)
    e = generate_episode("intermittent_error", 2)
    a, b = pipe.run(e.signal, e.adjacency), other.run(e.signal, e.adjacency)
    np.testing.assert_allclose(a.array("mmd2"), b.array("mmd2"), rtol=1e-6)
    assert [s.alert_type for s in a.steps] == [s.alert_type for s in b.steps]


def test_benchmark_schema(pipe):
    res = benchmark(pipe, n_per_scenario=1)
    assert res["setup"]["episodes"] == len(SCENARIOS)
    for key in ("false_alarm_rate_drift_only", "detection_rate", "type_accuracy", "median_delay_s"):
        assert key in res
    assert set(res["per_type"]) == set(FAULT_TYPES)


def test_config_is_a_dataclass():
    cfg = dataclasses.replace(PipelineConfig(), sigma_scale=1.0)
    assert cfg.sigma_scale == 1.0
