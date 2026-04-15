"""Optional real-cluster E2E test for labeled data collection.

This test is opt-in and disabled by default because it requires:
- Kubernetes access to the target cluster
- Chaos scenarios to be executable in namespace ewat
- Existing observability stack reachable from the cluster

Enable with:
    EWAT_E2E_REAL=1 python -m pytest tests/integration/test_collection_e2e_real.py -q
"""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.skipif(
    os.getenv("EWAT_E2E_REAL") != "1",
    reason="Set EWAT_E2E_REAL=1 to run real-cluster integration test.",
)


def _extract_run_dir(stdout: str) -> str:
    """Extract run_dir from collect_labeled stdout payload."""
    try:
        payload = json.loads(stdout)
        run_dir = payload.get("run_dir", "")
        if run_dir:
            return run_dir
    except json.JSONDecodeError:
        pass

    match = re.search(r'"run_dir"\s*:\s*"([^"]+)"', stdout)
    if not match:
        raise AssertionError("Unable to parse run_dir from collect_labeled output")
    return match.group(1)


def _is_endpoint_resolvable(endpoint: str) -> bool:
    """Return True when endpoint host can be resolved from this environment."""
    host = urlparse(endpoint).hostname
    if not host:
        return False
    try:
        socket.getaddrinfo(host, None)
        return True
    except socket.gaierror:
        return False


@pytest.mark.integration
@pytest.mark.timeout(720)
def test_collect_labeled_real(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    python_bin = sys.executable

    base_cfg = OmegaConf.load(repo_root / "configs/default.yaml")
    prom_endpoint = str(base_cfg.telemetry.prometheus.endpoint)
    if not _is_endpoint_resolvable(prom_endpoint):
        pytest.skip(
            "Prometheus endpoint host is not resolvable from this execution context: "
            f"{prom_endpoint}"
        )

    config_path = tmp_path / "collection_real.yaml"
    config_path.write_text(
        "\n".join(
            [
                "collection:",
                "  namespace: ewat",
                "  output_root: data/raw",
                "  sample_interval_s: 10",
                "  baseline_s: 10s",
                "  pre_injection_s: 10s",
                "  recovery_s: 10s",
                "  cool_down_s: 0s",
                "  repetitions: 5",
                "  scenarios:",
                "    - crash",
            ]
        ),
        encoding="utf-8",
    )

    run_dir_path: Path | None = None
    try:
        collect = subprocess.run(
            [
                python_bin,
                "scripts/collect_labeled.py",
                "--config",
                str(config_path),
                "--base-config",
                "configs/default.yaml",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )

        run_dir = _extract_run_dir(collect.stdout)
        run_dir_path = Path(run_dir)
        assert run_dir_path.exists(), f"Run directory missing: {run_dir_path}"

        for required in [
            "metadata.json",
            "signal.npz",
            "signal_mask.npz",
            "adjacency.npz",
            "labels.parquet",
            "graph_stats.csv",
            "services.json",
        ]:
            assert (run_dir_path / required).exists(), f"Missing artifact: {required}"

        with np.load(run_dir_path / "signal.npz") as payload:
            signal = payload["signal"]
        with np.load(run_dir_path / "adjacency.npz") as payload:
            adjacency = payload["adjacency"]
        metadata = json.loads((run_dir_path / "metadata.json").read_text(encoding="utf-8"))

        assert signal.ndim == 3
        assert adjacency.ndim == 4
        if signal.shape[0] == 0:
            prom_endpoint = (
                metadata.get("config", {})
                .get("telemetry", {})
                .get("prometheus", {})
                .get("endpoint", "unknown")
            )
            pytest.skip(
                "No timesteps collected in real run. "
                "Likely telemetry endpoints are unreachable from this execution context "
                f"(prometheus={prom_endpoint})."
            )

        subprocess.run(
            [
                python_bin,
                "scripts/validate_dataset.py",
                str(run_dir_path),
                "--min-coverage-episodes",
                "1",
                "--min-distribution-episodes",
                "1",
                "--max-nan-ratio",
                "1.0",
                "--min-baseline-edges",
                "1",
            ],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        if run_dir_path is not None and run_dir_path.exists():
            shutil.rmtree(run_dir_path)
