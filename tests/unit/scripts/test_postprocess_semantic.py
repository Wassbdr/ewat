from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.postprocess_semantic import postprocess_run
from telemetry.feature_names import L_SEMANTIC_ANOMALY


def test_postprocess_run_injects_semantic_column(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_20260101_000000"
    run_dir.mkdir(parents=True, exist_ok=True)

    signal = np.zeros((2, 1, 17), dtype=np.float32)
    np.savez_compressed(run_dir / "signal.npz", signal=signal)
    (run_dir / "services.json").write_text(json.dumps(["svc-a"]), encoding="utf-8")
    (run_dir / "metadata.json").write_text(
        json.dumps({"semantic_mode": "offline", "semantic_postprocessed": False}),
        encoding="utf-8",
    )
    (run_dir / "raw_logs.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"timestamp": 100.0, "service_name": "svc-a", "body": "INFO normal"}),
                json.dumps({"timestamp": 130.0, "service_name": "svc-a", "body": "ERROR boom"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    labels = pd.DataFrame(
        [
            {"timestamp": 100.0, "regime": "normal", "scenario": "normal"},
            {"timestamp": 130.0, "regime": "injection", "scenario": "crash"},
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda _path: labels)

    class _DummyScorer:
        def score(self, _lines):
            return 0.77

    monkeypatch.setattr(
        "scripts.postprocess_semantic._build_service_scorers",
        lambda **_kwargs: {"svc-a": _DummyScorer()},
    )

    postprocess_run(
        run_dir=run_dir,
        window_s=120.0,
        model_name="all-MiniLM-L6-v2",
        batch_size=64,
    )

    enriched = np.load(run_dir / "signal.npz")["signal"]
    assert np.isclose(enriched[0, 0, L_SEMANTIC_ANOMALY], 0.77)
    assert np.isclose(enriched[1, 0, L_SEMANTIC_ANOMALY], 0.77)

    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["semantic_postprocessed"] is True
