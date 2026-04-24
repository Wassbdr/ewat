"""EWAT — Phase 1: record raw telemetry around Chaos Mesh injections.

Pipeline overview
=================

For each ``(scenario, rep)`` pair defined in ``configs/collection.yaml``:

1. Wait ``baseline_s`` seconds of normal traffic  (regime = normal).
2. Wait ``pre_injection_s`` seconds (regime = pre).
3. Apply the Chaos Mesh manifest and wait for ``scenario.duration``
   (regime = injection).
4. Delete the manifest and wait ``recovery_s`` seconds (regime = recovery).
5. After the episode completes, issue **one bulk range query per source**
   to Prometheus / Jaeger / Loki covering ``[t_baseline_start, t_recovery_end]``
   and dump the raw JSON responses to disk.

Absolutely no feature engineering (S(t), G(t)) happens here. The raw dumps
are consumed by Phase 2 (``scripts/build_features.py``) which can be
re-run with different aggregation rules, time grids or service sets without
touching the cluster.

Output layout
=============

::

    data/raw/episode_<scenario>_<rep>_<YYYYmmddTHHMMSSZ>/
    ├── episode.json            # phase timestamps, scenario metadata, chaos apply/delete timings
    ├── prometheus_range.json.gz
    ├── jaeger_spans.json.gz
    ├── loki_logs.json.gz
    └── manifest.json           # per-source availability, file sizes, errors

Usage
=====

::

    python -m scripts.record_episode \
        --config configs/collection.yaml \
        --base-config configs/default.yaml \
        --endpoint-mode local-portforward
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import signal as signal_mod
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.chaos_injector import ChaosInjector  # noqa: E402
from telemetry.recorder import (  # noqa: E402
    JaegerDump,
    LokiDump,
    PrometheusDump,
    TelemetryRecorder,
)

logger = logging.getLogger(__name__)


_LOCAL_PORT_FORWARD_ENDPOINTS: dict[str, str] = {
    "prometheus": "http://127.0.0.1:19090",
    "jaeger": "http://127.0.0.1:16686",
    "loki": "http://127.0.0.1:13100",
}


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
#
# Set by SIGINT/SIGTERM handlers. Checked at safe points in the main loop so
# the current episode finishes cleanly (chaos delete + dump) before exit.
_shutdown_requested: bool = False


def _install_signal_handlers() -> None:
    def _handler(signum: int, _frame: Any) -> None:  # noqa: ANN001
        global _shutdown_requested
        _shutdown_requested = True
        sig_name = signal_mod.Signals(signum).name
        logger.warning(
            "Received %s — will finish current episode then exit cleanly",
            sig_name,
        )

    signal_mod.signal(signal_mod.SIGINT, _handler)
    signal_mod.signal(signal_mod.SIGTERM, _handler)


# ---------------------------------------------------------------------------
# Checkpoint (idempotent resume)
# ---------------------------------------------------------------------------


class _Checkpoint:
    """Append-only JSONL record of successfully-completed (scenario, rep) pairs.

    Matched on ``(scenario, rep)`` rather than ``episode_id`` because the
    latter contains a timestamp that changes on every run. A restart after
    crash therefore skips episodes already written+quality-gated to disk.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._done: set[tuple[str, int]] = set()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scenario = entry.get("scenario")
                rep = entry.get("rep")
                if isinstance(scenario, str) and isinstance(rep, int):
                    self._done.add((scenario, rep))
        if self._done:
            logger.info(
                "checkpoint: loaded %d completed (scenario, rep) pairs from %s",
                len(self._done), self._path,
            )

    def is_done(self, scenario: str, rep: int) -> bool:
        return (scenario, rep) in self._done

    def mark_done(self, scenario: str, rep: int, episode_id: str) -> None:
        self._done.add((scenario, rep))
        entry = {
            "scenario": scenario,
            "rep": rep,
            "episode_id": episode_id,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass

    def reset(self) -> None:
        if self._path.exists():
            self._path.unlink()
        self._done.clear()


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def _check_episode_quality(
    manifest: dict[str, Any],
    *,
    enable_prometheus: bool,
    enable_jaeger: bool,
    enable_loki: bool,
) -> tuple[bool, list[str]]:
    """Post-dump sanity check on the episode manifest.

    Returns ``(ok, reasons)``. ``reasons`` is the list of modalities that
    failed their minimal expectation. An episode passes only if every
    enabled modality returned at least one non-empty result.
    """
    reasons: list[str] = []
    sources = manifest.get("sources", {}) or {}

    if enable_prometheus:
        prom = sources.get("prometheus", {}) or {}
        if prom.get("skipped"):
            reasons.append("prometheus-skipped")
        elif not prom.get("queries_ok"):
            reasons.append("prometheus-no-queries")

    if enable_jaeger:
        jae = sources.get("jaeger", {}) or {}
        if jae.get("skipped"):
            reasons.append("jaeger-skipped")
        elif int(jae.get("n_traces_total", 0)) <= 0:
            reasons.append("jaeger-empty")

    if enable_loki:
        loki = sources.get("loki", {}) or {}
        if loki.get("skipped"):
            reasons.append("loki-skipped")
        elif int(loki.get("n_lines", 0)) <= 0:
            reasons.append("loki-empty")

    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# On-demand port-forward management
# ---------------------------------------------------------------------------


class _PortForward:
    """Manage a single kubectl port-forward subprocess.

    Each instance owns at most one ``kubectl port-forward`` process.
    Calling :meth:`start` kills any existing process first, then opens a
    fresh SPDY tunnel and waits for the local port to become reachable.

    Designed for on-demand use: start before a bulk dump, stop after.
    This avoids the SPDY tunnel degradation that kills long-lived
    port-forwards (~1-2 h under heavy Jaeger payloads).
    """

    def __init__(
        self,
        target: str,
        namespace: str,
        local_port: int,
        remote_port: int,
        name: str = "",
    ) -> None:
        self._target = target
        self._ns = namespace
        self._local = local_port
        self._remote = remote_port
        self._name = name or target
        self._proc: subprocess.Popen[bytes] | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, ready_timeout: float = 15.0) -> None:
        """Start a fresh port-forward. Kills any existing one first."""
        self.stop()
        self._kill_orphans()

        cmd = [
            "kubectl", "port-forward",
            "-n", self._ns,
            self._target,
            f"{self._local}:{self._remote}",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        logger.info("port-forward started: %s → 127.0.0.1:%d  (pid=%d)",
                     self._name, self._local, self._proc.pid)

        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"port-forward for {self._name} exited immediately "
                    f"(rc={self._proc.returncode})"
                )
            try:
                with socket.create_connection(("127.0.0.1", self._local), timeout=1.0):
                    logger.info("port-forward ready: %s", self._name)
                    return
            except OSError:
                time.sleep(0.5)
        self.stop()
        raise RuntimeError(
            f"port-forward for {self._name} not reachable after {ready_timeout}s"
        )

    def stop(self) -> None:
        """Terminate the owned port-forward process, if any."""
        if self._proc is None:
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal_mod.SIGTERM)
        except OSError:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal_mod.SIGKILL)
            except OSError:
                pass
        self._proc = None

    def _kill_orphans(self) -> None:
        """Kill any kubectl port-forward processes using our local port."""
        try:
            out = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{self._local}"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return
        for pid_s in out.splitlines():
            try:
                os.kill(int(pid_s), signal_mod.SIGKILL)
            except (OSError, ValueError):
                pass

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def __del__(self) -> None:
        self.stop()


class PortForwardGroup:
    """Manages port-forwards for prometheus, jaeger, and loki as a unit."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._pfs: dict[str, _PortForward] = {}
        for name, spec in config.items():
            self._pfs[name] = _PortForward(
                target=str(spec["target"]),
                namespace=str(spec["namespace"]),
                local_port=int(spec["local_port"]),
                remote_port=int(spec["remote_port"]),
                name=name,
            )

    def start_all(self) -> None:
        """Start all port-forwards with fresh SPDY tunnels."""
        for name, pf in self._pfs.items():
            try:
                pf.start()
            except RuntimeError:
                logger.exception("failed to start port-forward for %s", name)
                raise

    def stop_all(self) -> None:
        for pf in self._pfs.values():
            pf.stop()

    def restart_all(self) -> None:
        """Kill all existing port-forwards and start fresh ones."""
        self.stop_all()
        time.sleep(1.0)
        self.start_all()


# ---------------------------------------------------------------------------
# Phase boundaries
# ---------------------------------------------------------------------------


@dataclass
class PhaseBoundaries:
    """Unix timestamps for each phase of one episode.

    ``apply_returned_at`` and ``delete_returned_at`` record when ``kubectl``
    returned control, which is the best wall-clock approximation of the
    chaos on/off edge without watching the CRD status field. The clocks
    are the host running this script (same VM as the port-forwards).
    """

    baseline_start: float
    baseline_end: float
    pre_start: float
    pre_end: float
    injection_start: float
    apply_returned_at: float
    injection_end: float
    delete_returned_at: float
    recovery_start: float
    recovery_end: float


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_duration(duration: str | int | float) -> float:
    if isinstance(duration, (int, float)):
        return float(duration)
    text = str(duration).strip().lower()
    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    if text.endswith("s"):
        return float(text[:-1])
    if text.endswith("m"):
        return float(text[:-1]) * 60.0
    if text.endswith("h"):
        return float(text[:-1]) * 3600.0
    return float(text)


def _apply_endpoint_mode(base_cfg: Any, endpoint_mode: str, collection_cfg: Any) -> None:
    if endpoint_mode == "local-portforward":
        base_cfg.telemetry.prometheus.endpoint = _LOCAL_PORT_FORWARD_ENDPOINTS["prometheus"]
        base_cfg.telemetry.jaeger.endpoint = _LOCAL_PORT_FORWARD_ENDPOINTS["jaeger"]
        base_cfg.telemetry.loki.endpoint = _LOCAL_PORT_FORWARD_ENDPOINTS["loki"]
        return

    if endpoint_mode == "nodeport":
        np_cfg = OmegaConf.to_container(collection_cfg.get("nodeport", {}), resolve=True) or {}
        node_ip = str(np_cfg.get("node_ip", "")).strip()
        if not node_ip:
            raise SystemExit(
                "collection.nodeport.node_ip must be set when --endpoint-mode=nodeport. "
                "Fill it in configs/collection.yaml with a Ready worker node IP.",
            )
        prom_port = int(np_cfg.get("prometheus_port", 31090))
        jaeger_port = int(np_cfg.get("jaeger_port", 31686))
        loki_port = int(np_cfg.get("loki_port", 31100))
        base_cfg.telemetry.prometheus.endpoint = f"http://{node_ip}:{prom_port}"
        base_cfg.telemetry.jaeger.endpoint = f"http://{node_ip}:{jaeger_port}"
        base_cfg.telemetry.loki.endpoint = f"http://{node_ip}:{loki_port}"
        return

    # "cluster" mode: leave base_cfg untouched (uses in-cluster service DNS).


def _safe_git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None
    commit = proc.stdout.strip()
    return commit or None


def _sleep_with_status(duration_s: float, regime: str, episode_id: str) -> None:
    """``time.sleep`` variant that logs a progress ping every 30 s."""
    if duration_s <= 0:
        return
    start = time.time()
    deadline = start + duration_s
    ping_every = 30.0
    next_ping = start + ping_every
    while True:
        now = time.time()
        remaining = deadline - now
        if remaining <= 0:
            return
        if now >= next_ping:
            logger.info(
                "  [%s] %s elapsed=%.0fs remaining=%.0fs",
                episode_id,
                regime,
                now - start,
                remaining,
            )
            next_ping = now + ping_every
        time.sleep(min(1.0, remaining))


# ---------------------------------------------------------------------------
# Episode orchestration
# ---------------------------------------------------------------------------


def _run_episode(
    *,
    scenario_name: str,
    rep: int,
    injector: ChaosInjector,
    baseline_s: float,
    pre_s: float,
    recovery_s: float,
    dry_run: bool,
) -> tuple[str, PhaseBoundaries, dict[str, Any]]:
    """Execute the four-phase timeline of one episode.

    Returns
    -------
    episode_id:
        Stable identifier used for the output directory name.
    boundaries:
        Per-phase Unix timestamps.
    scenario_info:
        Scenario metadata (category, target services, chaos file, duration).
    """
    spec = injector.get_scenario(scenario_name)
    inject_s = 0.0 if dry_run else _parse_duration(spec.duration)

    utc_tag = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    episode_id = f"episode_{scenario_name}_{rep:03d}_{utc_tag}"

    logger.info("[%s] starting  scenario=%s  rep=%d  injection=%.1fs",
                episode_id, scenario_name, rep, inject_s)

    baseline_start = time.time()
    _sleep_with_status(baseline_s, "baseline", episode_id)
    baseline_end = time.time()

    pre_start = baseline_end
    _sleep_with_status(pre_s, "pre", episode_id)
    pre_end = time.time()

    injection_start = pre_end
    apply_returned_at = injection_start
    try:
        injector.apply(scenario_name)
        apply_returned_at = time.time()
    except Exception as exc:
        logger.error("[%s] chaos apply failed: %s", episode_id, exc)
        raise

    try:
        _sleep_with_status(inject_s, "injection", episode_id)
    finally:
        injection_end = time.time()
        try:
            injector.delete(scenario_name)
        except Exception as exc:
            logger.warning("[%s] chaos delete failed: %s", episode_id, exc)
        delete_returned_at = time.time()

    recovery_start = delete_returned_at
    _sleep_with_status(recovery_s, "recovery", episode_id)
    recovery_end = time.time()

    boundaries = PhaseBoundaries(
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        pre_start=pre_start,
        pre_end=pre_end,
        injection_start=injection_start,
        apply_returned_at=apply_returned_at,
        injection_end=injection_end,
        delete_returned_at=delete_returned_at,
        recovery_start=recovery_start,
        recovery_end=recovery_end,
    )

    scenario_info: dict[str, Any] = {
        "name": spec.name,
        "category": spec.category,
        "kind": spec.kind,
        "file": spec.file,
        "duration_nominal_s": inject_s,
        "targets": list(spec.targets),
        "description": spec.description,
    }
    return episode_id, boundaries, scenario_info


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@dataclass
class PersistStats:
    path: str
    size_bytes: int
    sha256: str
    elapsed_s: float
    errors: dict[str, str] = field(default_factory=dict)


def _write_gz_json(path: Path, payload: Any, start_s: float) -> PersistStats:
    """Write ``payload`` as gzip-compressed JSON. Returns sha256 + size."""
    raw = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    h = hashlib.sha256(raw)
    with gzip.open(path, "wb", compresslevel=6) as f:
        f.write(raw)
    return PersistStats(
        path=path.name,
        size_bytes=path.stat().st_size,
        sha256=h.hexdigest(),
        elapsed_s=time.time() - start_s,
    )


def _record_and_persist(
    *,
    recorder: TelemetryRecorder,
    episode_dir: Path,
    services: list[str],
    t_start: float,
    t_end: float,
    enable_prometheus: bool,
    enable_jaeger: bool,
    enable_loki: bool,
) -> dict[str, Any]:
    """Issue the 3 bulk dumps and persist them to ``episode_dir``."""
    manifest: dict[str, Any] = {"sources": {}}

    # Prometheus ---------------------------------------------------------
    if enable_prometheus:
        logger.info("  fetch prometheus range [t..t+%.0fs]", t_end - t_start)
        t0 = time.time()
        prom: PrometheusDump = recorder.record_prometheus(t_start, t_end)
        stats = _write_gz_json(
            episode_dir / "prometheus_range.json.gz",
            {
                "start_unix_s": prom.start_unix_s,
                "end_unix_s": prom.end_unix_s,
                "step_s": prom.step_s,
                "namespace": prom.namespace,
                "window": prom.window,
                "endpoint": prom.endpoint,
                "results": prom.results,
                "fallback_used": prom.fallback_used,
                "errors": prom.errors,
                "elapsed_s": prom.elapsed_s,
            },
            t0,
        )
        manifest["sources"]["prometheus"] = {
            **asdict(stats),
            "errors": prom.errors,
            "queries_ok": sorted(prom.results.keys()),
            "fallback_used": prom.fallback_used,
            "fetch_elapsed_s": prom.elapsed_s,
        }
    else:
        manifest["sources"]["prometheus"] = {"skipped": True}

    # Jaeger -------------------------------------------------------------
    if enable_jaeger:
        logger.info("  fetch jaeger traces for %d services", len(services))
        t0 = time.time()
        jae: JaegerDump = recorder.record_jaeger(t_start, t_end, services)
        stats = _write_gz_json(
            episode_dir / "jaeger_spans.json.gz",
            {
                "start_unix_s": jae.start_unix_s,
                "end_unix_s": jae.end_unix_s,
                "endpoint": jae.endpoint,
                "services_queried": jae.services_queried,
                "per_service_counts": jae.per_service_counts,
                "traces": jae.traces,
                "errors": jae.errors,
                "elapsed_s": jae.elapsed_s,
            },
            t0,
        )
        manifest["sources"]["jaeger"] = {
            **asdict(stats),
            "errors": jae.errors,
            "n_traces_total": len(jae.traces),
            "per_service_counts": jae.per_service_counts,
            "fetch_elapsed_s": jae.elapsed_s,
        }
    else:
        manifest["sources"]["jaeger"] = {"skipped": True}

    # Loki ---------------------------------------------------------------
    if enable_loki:
        logger.info("  fetch loki logs")
        t0 = time.time()
        loki: LokiDump = recorder.record_loki(t_start, t_end)
        stats = _write_gz_json(
            episode_dir / "loki_logs.json.gz",
            {
                "start_unix_s": loki.start_unix_s,
                "end_unix_s": loki.end_unix_s,
                "endpoint": loki.endpoint,
                "namespace": loki.namespace,
                "streams": loki.streams,
                "n_lines": loki.n_lines,
                "truncated": loki.truncated,
                "errors": loki.errors,
                "elapsed_s": loki.elapsed_s,
            },
            t0,
        )
        manifest["sources"]["loki"] = {
            **asdict(stats),
            "errors": loki.errors,
            "n_lines": loki.n_lines,
            "truncated": loki.truncated,
            "fetch_elapsed_s": loki.elapsed_s,
        }
    else:
        manifest["sources"]["loki"] = {"skipped": True}

    return manifest


def _write_episode_json(
    episode_dir: Path,
    *,
    episode_id: str,
    scenario_info: dict[str, Any],
    boundaries: PhaseBoundaries,
    services: list[str],
    cluster_cfg: dict[str, Any],
    collection_cfg: dict[str, Any],
    endpoint_mode: str,
    recorder_params: dict[str, Any],
) -> None:
    payload = {
        "episode_id": episode_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _safe_git_commit(),
        "host": platform.node(),
        "python_version": sys.version.split()[0],
        "scenario": scenario_info,
        "boundaries": asdict(boundaries),
        "canonical_services": services,
        "cluster": cluster_cfg,
        "collection": collection_cfg,
        "endpoint_mode": endpoint_mode,
        "recorder_params": recorder_params,
    }
    with (episode_dir / "episode.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Any:
    return OmegaConf.load(path)


def main() -> None:  # noqa: C901 - single-process orchestrator
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _cli()

    cfg = _load_yaml(Path(args.config))
    base_cfg = _load_yaml(Path(args.base_config))
    _apply_endpoint_mode(base_cfg, args.endpoint_mode, cfg.collection)

    _install_signal_handlers()

    collection_cfg = cfg.collection
    namespace = str(collection_cfg.get("namespace", "ewat"))
    canonical_services = [str(s) for s in collection_cfg.get("canonical_services", [])]
    if not canonical_services:
        raise SystemExit("collection.canonical_services must be non-empty")

    output_root = Path(collection_cfg.get("output_root", "data/raw"))
    output_root = output_root if output_root.is_absolute() else REPO_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = _Checkpoint(output_root / ".checkpoint.jsonl")
    if args.reset_checkpoint:
        logger.warning("--reset-checkpoint: wiping %s", checkpoint._path)
        checkpoint.reset()

    consecutive_failures = 0
    max_consecutive_failures = int(args.max_consecutive_failures)

    baseline_s = _parse_duration(collection_cfg.get("baseline_s", "5m"))
    pre_s = _parse_duration(collection_cfg.get("pre_injection_s", "1m"))
    recovery_s = _parse_duration(collection_cfg.get("recovery_s", "2m"))
    cool_down_s = _parse_duration(collection_cfg.get("cool_down_s", "0s"))
    repetitions = int(collection_cfg.get("repetitions", 1))
    scenarios = [str(s) for s in collection_cfg.get("scenarios", [])]
    if not scenarios:
        raise SystemExit("collection.scenarios must be non-empty")

    # Scenario filter from CLI
    if args.scenarios:
        requested = {s.strip() for s in args.scenarios.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s in requested]
        if not scenarios:
            raise SystemExit(f"no scenarios match --scenarios={args.scenarios}")

    prom_endpoint = str(base_cfg.telemetry.prometheus.get("endpoint", ""))
    jaeger_endpoint = str(base_cfg.telemetry.jaeger.get("endpoint", ""))
    loki_endpoint = str(base_cfg.telemetry.loki.get("endpoint", ""))
    prom_step_s = int(base_cfg.telemetry.prometheus.get("scrape_interval_s", 15))
    prom_window = str(collection_cfg.get("prom_rate_window", "2m"))
    recorder_params = {
        "prom_step_s": prom_step_s,
        "prom_rate_window": prom_window,
        "prom_timeout_s": float(collection_cfg.get("prom_timeout_s", 30.0)),
        "jaeger_timeout_s": float(collection_cfg.get("jaeger_timeout_s", 30.0)),
        "loki_timeout_s": float(collection_cfg.get("loki_timeout_s", 30.0)),
        "jaeger_limit": int(collection_cfg.get("jaeger_limit", 1500)),
        "loki_limit": int(collection_cfg.get("loki_limit", 5000)),
    }
    recorder = TelemetryRecorder(
        prometheus_endpoint=prom_endpoint,
        jaeger_endpoint=jaeger_endpoint,
        loki_endpoint=loki_endpoint,
        namespace=namespace,
        **recorder_params,
    )

    injector = ChaosInjector(namespace=namespace, dry_run=args.dry_run)

    cluster_cfg = OmegaConf.to_container(base_cfg.get("cluster", {}), resolve=True) or {}
    collection_cfg_out = OmegaConf.to_container(collection_cfg, resolve=True) or {}

    # Port-forward management (opt-in)
    pf_group: PortForwardGroup | None = None
    if args.manage_port_forwards and args.endpoint_mode == "local-portforward":
        pf_cfg = OmegaConf.to_container(
            collection_cfg.get("port_forwards", {}), resolve=True,
        ) or {}
        if pf_cfg:
            pf_group = PortForwardGroup(pf_cfg)
            logger.info("port-forward management enabled for: %s", list(pf_cfg.keys()))
        else:
            logger.warning("--manage-port-forwards set but no port_forwards config found")

    logger.info(
        "record_episode: namespace=%s  services=%d  scenarios=%d  reps=%d  output=%s",
        namespace,
        len(canonical_services),
        len(scenarios),
        repetitions,
        output_root,
    )

    enable_prometheus = not args.no_prometheus
    enable_jaeger = not args.no_jaeger
    enable_loki = not args.no_loki

    try:
        for scenario_name in scenarios:
            if _shutdown_requested:
                logger.warning("shutdown requested — breaking before scenario=%s", scenario_name)
                break
            for rep in range(repetitions):
                if _shutdown_requested:
                    logger.warning("shutdown requested — breaking at (%s, rep=%d)", scenario_name, rep)
                    break

                if checkpoint.is_done(scenario_name, rep):
                    logger.info(
                        "skip (scenario=%s rep=%d) — already in checkpoint",
                        scenario_name, rep,
                    )
                    continue

                try:
                    episode_id, boundaries, scenario_info = _run_episode(
                        scenario_name=scenario_name,
                        rep=rep,
                        injector=injector,
                        baseline_s=baseline_s,
                        pre_s=pre_s,
                        recovery_s=recovery_s,
                        dry_run=args.dry_run,
                    )
                except Exception:
                    logger.exception(
                        "episode execution failed (scenario=%s rep=%d)",
                        scenario_name, rep,
                    )
                    continue

                # Fresh port-forwards before each dump: kills stale SPDY
                # tunnels and starts new ones.  The recorder also gets a
                # fresh requests.Session so there are no pooled connections
                # pointing at a dead tunnel.
                if pf_group is not None:
                    pf_group.restart_all()
                    recorder.refresh_session()

                tmp_dir = output_root / (episode_id + ".tmp")
                if tmp_dir.exists():
                    import shutil
                    shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True)

                manifest = _record_and_persist(
                    recorder=recorder,
                    episode_dir=tmp_dir,
                    services=canonical_services,
                    t_start=boundaries.baseline_start,
                    t_end=boundaries.recovery_end,
                    enable_prometheus=enable_prometheus,
                    enable_jaeger=enable_jaeger,
                    enable_loki=enable_loki,
                )

                # Kill port-forwards immediately after dump — no point
                # keeping them alive during cool-down sleep.
                if pf_group is not None:
                    pf_group.stop_all()

                _write_episode_json(
                    tmp_dir,
                    episode_id=episode_id,
                    scenario_info=scenario_info,
                    boundaries=boundaries,
                    services=canonical_services,
                    cluster_cfg=cluster_cfg,
                    collection_cfg=collection_cfg_out,
                    endpoint_mode=args.endpoint_mode,
                    recorder_params=recorder_params,
                )

                with (tmp_dir / "manifest.json").open("w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                final_dir = output_root / episode_id
                if final_dir.exists():
                    import shutil
                    shutil.rmtree(final_dir)
                os.rename(tmp_dir, final_dir)
                logger.info("[%s] saved -> %s", episode_id, final_dir)

                # Post-dump quality gate. Dry-runs get the usual empty-payload
                # pass since we deliberately skip the injection window.
                ok, reasons = (True, []) if args.dry_run else _check_episode_quality(
                    manifest,
                    enable_prometheus=enable_prometheus,
                    enable_jaeger=enable_jaeger,
                    enable_loki=enable_loki,
                )
                if ok:
                    consecutive_failures = 0
                    checkpoint.mark_done(scenario_name, rep, episode_id)
                else:
                    consecutive_failures += 1
                    (final_dir / ".quality_failed").write_text(
                        json.dumps({"reasons": reasons, "at": datetime.now(UTC).isoformat()}) + "\n",
                        encoding="utf-8",
                    )
                    logger.error(
                        "[%s] QUALITY GATE FAILED: %s (consecutive=%d / max=%d)",
                        episode_id, reasons, consecutive_failures, max_consecutive_failures,
                    )
                    if consecutive_failures >= max_consecutive_failures:
                        raise SystemExit(
                            f"aborting: {consecutive_failures} consecutive quality failures "
                            f">= --max-consecutive-failures={max_consecutive_failures}. "
                            "Inspect recent episodes and cluster state before relaunching.",
                        )

                if cool_down_s > 0 and not args.dry_run:
                    logger.info("  cool-down %.0fs", cool_down_s)
                    _sleep_with_status(cool_down_s, "cool_down", episode_id)
    finally:
        if pf_group is not None:
            pf_group.stop_all()


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EWAT Phase 1 — raw telemetry recorder")
    p.add_argument("--config", default="configs/collection.yaml")
    p.add_argument("--base-config", default="configs/default.yaml")
    p.add_argument(
        "--endpoint-mode",
        choices=["cluster", "local-portforward", "nodeport"],
        default="local-portforward",
    )
    p.add_argument("--scenarios", default="", help="comma-separated subset of scenarios")
    p.add_argument("--no-prometheus", action="store_true")
    p.add_argument("--no-jaeger", action="store_true")
    p.add_argument("--no-loki", action="store_true")
    p.add_argument(
        "--manage-port-forwards",
        action="store_true",
        help="Automatically start/stop kubectl port-forwards before each dump. "
        "Prevents SPDY tunnel degradation on long-running collections.",
    )
    p.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Wipe .checkpoint.jsonl before starting. Forces a full re-run.",
    )
    p.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Abort after this many consecutive quality-gate failures. "
        "Protects long campaigns from silently collecting garbage.",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main()
