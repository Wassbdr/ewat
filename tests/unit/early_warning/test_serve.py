import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from ewat.early_warning import EarlyWarningPipeline, PipelineConfig, serve  # noqa: E402
from ewat.synthetic import SCENARIOS, generate_episode  # noqa: E402


@pytest.fixture(scope="module")
def client():
    train = [generate_episode(s, 10_000 + i) for s in SCENARIOS for i in range(4)]
    serve._pipe = EarlyWarningPipeline.fit(train, PipelineConfig(train_iters=60))
    return TestClient(serve.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and len(r.json()["services"]) == 6


def test_predict_window(client):
    e = generate_episode("fail_slow_latency", 0)
    r = client.post("/predict", json={"signal": np.round(e.signal[:60], 4).tolist()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert {"drift_score", "threshold", "fault_type", "latency_ms", "regime"} <= set(body)
    assert body["latency_ms"] > 0 and body["steps"] == 60


def test_predict_rejects_bad_shape(client):
    r = client.post("/predict", json={"signal": [[[0.0] * 3] * 6] * 20})
    assert r.status_code == 422
