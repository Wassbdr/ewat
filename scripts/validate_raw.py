"""EWAT — Phase 1 raw dataset validator.

Walks a ``data/raw`` directory (or a single episode) and reports, per
episode, whether the manifest contains non-empty payloads for each enabled
modality. Reads only ``manifest.json`` — no gzip decompression needed,
fast even on 300-episode campaigns.

Usage
-----

::

    python -m scripts.validate_raw --raw-root data/raw
    python -m scripts.validate_raw --episode data/raw/episode_crash_000_...
    python -m scripts.validate_raw --raw-root data/raw --strict  # exit 1 on any FAIL

Verdict per modality:
    prometheus : queries_ok non-empty
    jaeger     : n_traces_total > 0
    loki       : n_lines > 0
Plus a check that the 5 expected files exist and no ``.quality_failed`` marker.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_EXPECTED_FILES = (
    "episode.json",
    "manifest.json",
    "prometheus_range.json.gz",
    "jaeger_spans.json.gz",
    "loki_logs.json.gz",
)


@dataclass
class EpisodeVerdict:
    path: Path
    ok: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name


def _validate_episode(ep_dir: Path) -> EpisodeVerdict:
    reasons: list[str] = []

    manifest_path = ep_dir / "manifest.json"
    if not manifest_path.exists():
        return EpisodeVerdict(ep_dir, ok=False, reasons=["no-manifest"])

    for fname in _EXPECTED_FILES:
        if not (ep_dir / fname).exists():
            reasons.append(f"missing:{fname}")

    if (ep_dir / ".quality_failed").exists():
        reasons.append("quality_failed-marker")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return EpisodeVerdict(ep_dir, ok=False, reasons=[f"manifest-unreadable:{exc}"])

    sources = manifest.get("sources", {}) or {}

    prom = sources.get("prometheus", {}) or {}
    if not prom.get("skipped"):
        if not prom.get("queries_ok"):
            reasons.append("prometheus-no-queries")

    jae = sources.get("jaeger", {}) or {}
    if not jae.get("skipped"):
        if int(jae.get("n_traces_total", 0)) <= 0:
            reasons.append("jaeger-empty")

    loki = sources.get("loki", {}) or {}
    if not loki.get("skipped"):
        if int(loki.get("n_lines", 0)) <= 0:
            reasons.append("loki-empty")

    return EpisodeVerdict(ep_dir, ok=(len(reasons) == 0), reasons=reasons)


def _is_episode_dir(path: Path) -> bool:
    return path.is_dir() and (path / "manifest.json").exists()


def _find_episodes(root: Path) -> list[Path]:
    if _is_episode_dir(root):
        return [root]
    return sorted(
        child for child in root.iterdir()
        if child.is_dir() and child.name.startswith("episode_") and _is_episode_dir(child)
    )


def _print_table(verdicts: list[EpisodeVerdict]) -> None:
    if not verdicts:
        print("no episodes found")
        return
    width = max(len(v.name) for v in verdicts)
    fmt = f"{{:<{width}}}  {{:<6}}  {{}}"
    print(fmt.format("episode", "status", "reasons"))
    print("-" * (width + 20))
    for v in verdicts:
        status = "PASS" if v.ok else "FAIL"
        reasons = ",".join(v.reasons) if v.reasons else "-"
        print(fmt.format(v.name, status, reasons))


def main() -> int:
    parser = argparse.ArgumentParser(description="EWAT Phase 1 raw episode validator")
    parser.add_argument("--raw-root", default="data/raw", help="Directory of episode_* subdirs")
    parser.add_argument("--episode", default="", help="Validate a single episode directory")
    parser.add_argument("--strict", action="store_true", help="exit 1 if any episode fails")
    args = parser.parse_args()

    if args.episode:
        root = Path(args.episode)
    else:
        root = Path(args.raw_root)
    if not root.is_absolute():
        root = REPO_ROOT / root

    if not root.exists():
        print(f"path not found: {root}", file=sys.stderr)
        return 2

    episodes = _find_episodes(root)
    verdicts = [_validate_episode(ep) for ep in episodes]
    _print_table(verdicts)

    n_pass = sum(1 for v in verdicts if v.ok)
    n_total = len(verdicts)
    print(f"\nverdict: {n_pass}/{n_total} passed")

    if args.strict and n_pass < n_total:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
