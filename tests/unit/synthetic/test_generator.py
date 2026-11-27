import numpy as np
import pytest

from ewat.synthetic import (
    BENIGN_DRIFT,
    FAULT_TYPES,
    SCENARIOS,
    SERVICES,
    call_graph_adjacency,
    generate_episode,
)
from telemetry.feature_names import M_CPU_UTIL, M_ERROR_RATE, M_LATENCY_P99, SIGNAL_DIM


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_shapes_and_phases(scenario):
    e = generate_episode(scenario, 3)
    T = len(e.phase)
    assert e.signal.shape == (T, len(SERVICES), SIGNAL_DIM)
    assert e.adjacency.shape == (T, len(SERVICES), len(SERVICES), 3)
    assert np.isfinite(e.signal).all()
    assert (e.phase[: e.drift_start] == "normal").all()
    if scenario == BENIGN_DRIFT:
        assert e.fault_start is None and e.target is None
        assert set(e.phase) == {"normal", "drift"}
    else:
        assert (e.phase[e.fault_start :] == "fault").all()
        assert e.target in SERVICES and e.target != "load-generator"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_seeded_and_deterministic(scenario):
    a, b = generate_episode(scenario, 7), generate_episode(scenario, 7)
    c = generate_episode(scenario, 8)
    np.testing.assert_array_equal(a.signal, b.signal)
    assert not np.array_equal(a.signal[: len(c.signal)], c.signal[: len(a.signal)])


@pytest.mark.parametrize(
    ("scenario", "feat"),
    [
        ("fail_slow_latency", M_LATENCY_P99),
        ("intermittent_error", M_ERROR_RATE),
        ("cpu_starvation", M_CPU_UTIL),
    ],
)
def test_fault_moves_its_metric_on_target(scenario, feat):
    e = generate_episode(scenario, 1)
    j = SERVICES.index(e.target)
    before = e.signal[e.fault_start - 10 : e.fault_start, j, feat].mean()
    after = e.signal[e.fault_start + 5 :, j, feat].mean()
    assert after > before * 1.2


def test_unknown_scenario_rejected():
    with pytest.raises(ValueError):
        generate_episode("nope", 0)


def test_call_graph_adjacency_shape():
    e = generate_episode(FAULT_TYPES[0], 0)
    adj = call_graph_adjacency(e.signal)
    assert adj.shape == e.adjacency.shape
    assert np.isfinite(adj).all()
