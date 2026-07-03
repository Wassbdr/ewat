"""Unit tests for scripts/audit_leak_v5.py — the publication leak gate."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.audit_leak_v5 import audit


def _write_episode(d, meta: dict, labels: dict | None = None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps(meta))
    (d / "services.json").write_text(json.dumps(["ts-order", "ts-travel"]))
    np.savez_compressed(d / "signal.npz", signal=np.zeros((2, 2, 18), np.float32))
    if labels is not None:
        pd.DataFrame(labels).to_parquet(d / "labels.parquet")


@pytest.fixture
def clean_release(tmp_path):
    _write_episode(
        tmp_path / "data" / "ep1",
        {"scenario": {"name": "cpu_stress", "file": "contention/cpu_stress.yaml"}},
        {"scenario": ["cpu_stress"], "target_service": ["ts-order"]},
    )
    return tmp_path


def test_clean_release_passes(clean_release):
    report = audit(clean_release, allowlist=set())
    assert report["clean"], report["findings"]
    assert report["n_findings"] == 0


@pytest.mark.parametrize(
    "leak, pattern",
    [
        ("172.16.203.12", "private_ipv4"),
        ("10.43.0.10", "private_ipv4"),
        ("rancher.devolab.lan", "internal_dns"),
        ("observit-cluster1-workers-58w74-mwxb2", "node_name"),
        ("/home/wassimbadraoui/repos/ewat", "home_path"),
        ("prometheus-server:9090", "telemetry_endpoint"),
        ("https://rancher.devolab.lan/k8s/clusters/c-m-x", "cluster_api"),
    ],
)
def test_infra_leak_detected(tmp_path, leak, pattern):
    _write_episode(tmp_path / "data" / "ep1", {"scenario": {"name": "x"}, "host": leak})
    report = audit(tmp_path, allowlist=set())
    assert not report["clean"]
    assert any(f["pattern"] == pattern for f in report["findings"]), report["findings"]


def test_forbidden_base_config_key_detected(tmp_path):
    _write_episode(
        tmp_path / "data" / "ep1",
        {"scenario": {"name": "x"}, "base_config": {"cluster": {"name": "obs"}}},
    )
    report = audit(tmp_path, allowlist=set())
    assert not report["clean"]
    assert any(f["pattern"] == "forbidden_key" for f in report["findings"])


def test_raw_signal_in_release_detected(tmp_path):
    d = tmp_path / "data" / "ep1"
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"scenario": {"name": "x"}}))
    np.savez_compressed(d / "signal.npz", signal=np.zeros((2, 2, 18), np.float32),
                        signal_raw=np.zeros((2, 2, 18), np.float32))
    report = audit(tmp_path, allowlist=set())
    assert not report["clean"]
    assert any(f["pattern"] == "raw_signal_in_release" for f in report["findings"])


def test_author_identity_allowed_in_doc_files(tmp_path):
    (tmp_path / "CITATION.cff").write_text("authors:\n  - family-names: Badraoui\n")
    _write_episode(tmp_path / "data" / "ep1", {"scenario": {"name": "x"}})
    report = audit(tmp_path, allowlist=set())
    # Author name in a doc file is legitimate, not a leak.
    assert report["clean"], report["findings"]


def test_author_identity_flagged_in_data_files(tmp_path):
    _write_episode(tmp_path / "data" / "ep1", {"scenario": {"name": "wassim-test"}})
    report = audit(tmp_path, allowlist=set())
    assert not report["clean"]
    assert any(f["pattern"] == "author_identity" for f in report["findings"])


def test_infra_leak_in_doc_file_still_fails(tmp_path):
    # A doc file tolerates the author name but NOT infrastructure leaks.
    (tmp_path / "README.md").write_text("host: 172.16.203.12\n")
    _write_episode(tmp_path / "data" / "ep1", {"scenario": {"name": "x"}})
    report = audit(tmp_path, allowlist=set())
    assert not report["clean"]
    assert any(f["pattern"] == "private_ipv4" for f in report["findings"])
