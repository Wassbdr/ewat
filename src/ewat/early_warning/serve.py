"""Small FastAPI service: POST a metrics window, get drift + fault type + latency.

uvicorn ewat.early_warning.serve:app      # or: ewat serve
"""

from __future__ import annotations

import time

import numpy as np

from ewat.synthetic.generator import SERVICES, call_graph_adjacency
from telemetry.feature_names import FEATURE_NAMES

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("the API needs the serve extra: pip install '.[serve]'") from exc

from ewat.early_warning.workflow import get_pipeline


class WindowRequest(BaseModel):
    signal: list[list[list[float]]] = Field(
        ..., description="metrics window, shape [T][6 services][17 features]; T ≥ 15 steps of 30 s"
    )
    adjacency: list[list[list[list[float]]]] | None = Field(
        None, description="optional [T][6][6][3]; derived from the call graph when omitted"
    )


class WindowResponse(BaseModel):
    drift_score: float
    threshold: float
    drift: bool
    regime: str
    alert: bool
    fault_type: str | None
    probability: float | None
    first_alert_step: int | None
    latency_ms: float
    steps: int


app = FastAPI(title="EWAT early warning", version="0.1.0")
_pipe = None


def pipeline():
    global _pipe
    if _pipe is None:
        _pipe = get_pipeline(log=False)
    return _pipe


@app.get("/health")
def health() -> dict:
    p = pipeline()
    return {"status": "ok", "services": SERVICES, "features": FEATURE_NAMES, "epsilon": p.epsilon}


@app.post("/predict", response_model=WindowResponse)
def predict(req: WindowRequest) -> WindowResponse:
    p = pipeline()
    signal = np.asarray(req.signal, dtype=np.float32)
    if signal.ndim != 3 or signal.shape[1:] != (len(SERVICES), len(FEATURE_NAMES)):
        raise HTTPException(
            422,
            f"signal must be [T][{len(SERVICES)}][{len(FEATURE_NAMES)}], got {list(signal.shape)}",
        )
    need = p.config.window_ref + p.config.window_cur
    if signal.shape[0] < need:
        raise HTTPException(422, f"need at least {need} steps, got {signal.shape[0]}")
    adj = (
        np.asarray(req.adjacency, dtype=np.float32)
        if req.adjacency
        else call_graph_adjacency(signal)
    )
    t0 = time.perf_counter()
    run = p.run(signal, adj)
    latency = (time.perf_counter() - t0) * 1000.0
    last, first = run.steps[-1], run.first_alert
    shown = first or last
    return WindowResponse(
        drift_score=last.mmd2,
        threshold=last.epsilon,
        drift=bool(last.mmd2 >= last.epsilon),
        regime=last.regime,
        alert=first is not None,
        fault_type=first.alert_type if first else None,
        probability=shown.probability,
        first_alert_step=first.t if first else None,
        latency_ms=latency,
        steps=int(signal.shape[0]),
    )
