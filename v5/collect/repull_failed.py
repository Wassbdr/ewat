"""EWAT v5 — repull les sources vides dans les épisodes .raw_failed.

Pour chaque épisode marqué .raw_failed, re-tire les sources (prometheus / jaeger / loki)
qui avaient échoué (prom=0, logs=0, traces=0) en utilisant t_start + durée depuis
episode_meta.json. Supprime .raw_failed si toutes les sources passent le gate QC.

Cas d'usage principal : incident 2026-06-06 (jnk2v NotReady → Loki Pending + prom=0).
Prometheus a une rétention de ~15j → données des épisodes juin 3-6 accessibles jusqu'au
~18-21 juin. À lancer dès que le cluster est stable (loki-0 Running).

Usage :
    cd ~/ewat/v5
    PYTHONPATH=../src python -m collect.repull_failed --raw-root ../data/raw_v5 --dry-run
    PYTHONPATH=../src python -m collect.repull_failed --raw-root ../data/raw_v5
    PYTHONPATH=../src python -m collect.repull_failed --raw-root ../data/raw_v5 --prom-only
    PYTHONPATH=../src python -m collect.repull_failed --raw-root ../data/raw_v5 --loki-only
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

from collect import probe


# Reps 0-9 → tt, 10-19 → tt-b, 20-29 → tt-c (fallback si namespace absent du meta).
_REP_TO_NS = {**{i: "tt" for i in range(10)},
              **{i: "tt-b" for i in range(10, 20)},
              **{i: "tt-c" for i in range(20, 30)}}


def _infer_ns(episode_id: str) -> str:
    """Infère le namespace depuis le rep numéro dans episode_id (ex. episode_cpu_007_...)."""
    parts = episode_id.split("_")
    for p in parts:
        if p.isdigit():
            return _REP_TO_NS.get(int(p), "tt")
    return "tt"


def _parse_raw_failed(content: str) -> dict[str, int]:
    """Parse 'traces=X logs=Y prom=Z' → {traces: X, logs: Y, prom: Z}."""
    result = {}
    for part in content.strip().split():
        k, _, v = part.partition("=")
        try:
            result[k] = int(v)
        except ValueError:
            pass
    return result


def _count_prom(data: dict) -> int:
    return sum(len(v) if isinstance(v, list) else 0 for v in data.values())


def _load_gz(ep_dir: Path, name: str) -> dict:
    p = ep_dir / f"{name}.json.gz"
    if not p.exists():
        return {}
    try:
        with gzip.open(p, "rt") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_gz(ep_dir: Path, name: str, data: dict) -> None:
    with gzip.open(ep_dir / f"{name}.json.gz", "wt") as f:
        json.dump(data, f)


def repull_episode(ep_dir: Path, dry_run: bool,
                   prom_only: bool, loki_only: bool) -> str:
    meta_path = ep_dir / "episode_meta.json"
    failed_path = ep_dir / ".raw_failed"

    if not meta_path.exists():
        return f"SKIP {ep_dir.name}: pas de episode_meta.json"

    meta = json.loads(meta_path.read_text())
    t_start: float = meta["t_start"]
    step: int = meta.get("step", 30)
    ns: str = meta.get("namespace") or _infer_ns(meta.get("episode_id", ep_dir.name))
    boundaries = meta.get("boundaries_rel", {})
    t_end = t_start + boundaries.get("recovery_end", 1800) + 60  # +60s buffer drainage

    failed_content = failed_path.read_text() if failed_path.exists() else "traces=0 logs=0 prom=0"
    failed_vals = _parse_raw_failed(failed_content)

    need_prom = not prom_only and not loki_only and failed_vals.get("prom", 0) == 0
    need_loki = not prom_only and not loki_only and failed_vals.get("logs", 0) == 0
    need_jaeger = not prom_only and not loki_only and failed_vals.get("traces", 0) == 0
    if prom_only:
        need_prom = failed_vals.get("prom", 0) == 0
        need_loki = need_jaeger = False
    if loki_only:
        need_loki = failed_vals.get("logs", 0) == 0
        need_prom = need_jaeger = False

    if not (need_prom or need_loki or need_jaeger):
        return f"SKIP {ep_dir.name}: rien à repull (prom={failed_vals.get('prom')} logs={failed_vals.get('logs')} traces={failed_vals.get('traces')})"

    tag = f"ns={ns} t_start={t_start:.0f} window={t_end - t_start:.0f}s"
    if dry_run:
        wants = " ".join(x for x, ok in [("prom", need_prom), ("loki", need_loki), ("jaeger", need_jaeger)] if ok)
        return f"DRY {ep_dir.name}: {tag} → repull {wants}"

    prom = _load_gz(ep_dir, "prometheus")
    jae = _load_gz(ep_dir, "jaeger")
    loki = _load_gz(ep_dir, "loki")
    actions = []

    if need_prom:
        try:
            prom = probe.pull_prometheus(t_start, t_end, step, ns)
            actions.append(f"prom={_count_prom(prom)}")
        except Exception as e:
            actions.append(f"prom=ERR({e})")

    if need_loki:
        try:
            loki = probe.pull_loki(t_start, t_end, step, ns)
            actions.append(f"loki={loki.get('n_lines', 0)}")
        except Exception as e:
            actions.append(f"loki=ERR({e})")

    if need_jaeger:
        try:
            jae = probe.pull_jaeger(t_start, t_end, 300, 1500, ns)
            actions.append(f"jaeger={jae.get('n_traces_total', 0)}")
        except Exception as e:
            actions.append(f"jaeger=ERR({e})")

    _write_gz(ep_dir, "prometheus", prom)
    _write_gz(ep_dir, "jaeger", jae)
    _write_gz(ep_dir, "loki", loki)

    n_traces = jae.get("n_traces_total", 0)
    n_logs = loki.get("n_lines", 0)
    n_prom = _count_prom(prom)
    ok = n_traces > 0 and n_logs > 0 and n_prom > 0

    if ok:
        failed_path.unlink(missing_ok=True)
        return f"OK {ep_dir.name}: {' '.join(actions)}"
    else:
        failed_path.write_text(f"traces={n_traces} logs={n_logs} prom={n_prom} (post-repull)")
        return f"FAIL {ep_dir.name}: {' '.join(actions)} → traces={n_traces} logs={n_logs} prom={n_prom}"


def main() -> None:
    p = argparse.ArgumentParser(description="EWAT v5 — repull sources vides dans .raw_failed")
    p.add_argument("--raw-root", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true", help="Affiche sans écrire")
    p.add_argument("--prom-only", action="store_true", help="Repull uniquement prom=0")
    p.add_argument("--loki-only", action="store_true", help="Repull uniquement logs=0")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallélisme (défaut=1 pour ne pas surcharger le cluster pendant la collecte)")
    args = p.parse_args()

    failed_dirs = sorted(d.parent for d in args.raw_root.glob("episode_*/.raw_failed"))
    print(f"Épisodes .raw_failed trouvés : {len(failed_dirs)}")
    if args.dry_run:
        print("[DRY RUN — aucune écriture]")

    recovered = 0
    skipped = 0
    still_fail = 0

    for i, ep_dir in enumerate(failed_dirs, 1):
        result = repull_episode(ep_dir, args.dry_run, args.prom_only, args.loki_only)
        print(f"[{i}/{len(failed_dirs)}] {result}", flush=True)
        if result.startswith("OK"):
            recovered += 1
        elif result.startswith("SKIP"):
            skipped += 1
        elif result.startswith("FAIL"):
            still_fail += 1
        # Pause courte pour ne pas saturer NodePort pendant la collecte active
        if not args.dry_run and not result.startswith("SKIP"):
            time.sleep(2)

    if not args.dry_run:
        print(f"\n=== Récupérés : {recovered}  Toujours échoués : {still_fail}  Skippés : {skipped} ===")


if __name__ == "__main__":
    main()
