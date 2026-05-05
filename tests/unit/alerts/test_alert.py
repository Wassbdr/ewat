"""Tests for Alert dataclass."""

import json

import pytest

from ewat.alerts.alert import Alert

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_alert_basic_construction():
    a = Alert(cluster_id=3, probability=0.8, horizon_steps=6, horizon_seconds=180.0)
    assert a.cluster_id == 3
    assert a.probability == pytest.approx(0.8)
    assert a.horizon_steps == 6
    assert a.horizon_seconds == pytest.approx(180.0)


def test_alert_default_fiche_is_empty_dict():
    a = Alert(cluster_id=0, probability=0.5, horizon_steps=2, horizon_seconds=60.0)
    assert a.fiche == {}


def test_alert_default_timestamp_zero():
    a = Alert(cluster_id=0, probability=0.5, horizon_steps=2, horizon_seconds=60.0)
    assert a.timestamp == pytest.approx(0.0)


def test_alert_default_episode_id_empty():
    a = Alert(cluster_id=0, probability=0.5, horizon_steps=2, horizon_seconds=60.0)
    assert a.episode_id == ""


def test_alert_probability_boundaries():
    Alert(cluster_id=0, probability=0.0, horizon_steps=1, horizon_seconds=30.0)
    Alert(cluster_id=0, probability=1.0, horizon_steps=1, horizon_seconds=30.0)


def test_alert_probability_invalid_raises():
    with pytest.raises(ValueError, match="probability"):
        Alert(cluster_id=0, probability=1.1, horizon_steps=1, horizon_seconds=30.0)


def test_alert_probability_negative_raises():
    with pytest.raises(ValueError, match="probability"):
        Alert(cluster_id=0, probability=-0.1, horizon_steps=1, horizon_seconds=30.0)


def test_alert_horizon_zero_raises():
    with pytest.raises(ValueError, match="horizon_steps"):
        Alert(cluster_id=0, probability=0.5, horizon_steps=0, horizon_seconds=0.0)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_to_dict_keys():
    a = Alert(cluster_id=2, probability=0.75, horizon_steps=4, horizon_seconds=120.0,
               fiche={"top_feature": "cpu_util"}, timestamp=1000.0, episode_id="ep_001")
    d = a.to_dict()
    assert set(d.keys()) == {
        "cluster_id", "probability", "horizon_steps", "horizon_seconds",
        "fiche", "timestamp", "episode_id",
    }


def test_to_dict_values():
    a = Alert(cluster_id=2, probability=0.75, horizon_steps=4, horizon_seconds=120.0,
               fiche={"top_feature": "cpu_util"}, timestamp=1000.0, episode_id="ep_001")
    d = a.to_dict()
    assert d["cluster_id"] == 2
    assert d["probability"] == pytest.approx(0.75)
    assert d["fiche"]["top_feature"] == "cpu_util"
    assert d["episode_id"] == "ep_001"


def test_from_dict_roundtrip():
    a = Alert(cluster_id=5, probability=0.9, horizon_steps=8, horizon_seconds=240.0,
               fiche={"k": "v"}, timestamp=42.0, episode_id="ep_XYZ")
    d = a.to_dict()
    a2 = Alert.from_dict(d)
    assert a2.cluster_id == a.cluster_id
    assert a2.probability == pytest.approx(a.probability)
    assert a2.horizon_steps == a.horizon_steps
    assert a2.fiche == a.fiche
    assert a2.episode_id == a.episode_id


def test_from_dict_optional_fields_default():
    d = {"cluster_id": 0, "probability": 0.5, "horizon_steps": 2, "horizon_seconds": 60.0}
    a = Alert.from_dict(d)
    assert a.fiche == {}
    assert a.timestamp == pytest.approx(0.0)
    assert a.episode_id == ""


def test_to_dict_is_json_serializable():
    a = Alert(cluster_id=1, probability=0.6, horizon_steps=2, horizon_seconds=60.0,
               fiche={"features": [0.1, 0.2]})
    json.dumps(a.to_dict())  # must not raise
