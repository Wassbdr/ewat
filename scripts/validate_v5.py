"""EWAT v5 — porte de validation pour les épisodes/dataset Train Ticket.

Profil v5 (N=41, contrat v4-conforme, schéma feature v5.1 = 18 features). Vérifie par épisode :
  - shape signal (T, N, 18), adjacency (T, N, N, 3), mask cohérent
  - N == nombre de services canoniques (services.json)
  - 0 % NaN dans signal.npz (imputé) ; NaN brut (signal_raw) < seuil
  - G(t) non vide : ≥ min_graph_fraction des steps ont ≥ 1 arête
  - couverture trace : ≥ trace_floor services ont ≥ 1 span sur l'épisode
  - régimes ∈ {normal, injection, recovery}
Au niveau dataset (si --dataset) : aucune fuite held-out (scénarios/bugs
held_out_flag=True absents de train+val).

Usage :
    python scripts/validate_v5.py --features-root data/raw_v5
    python scripts/validate_v5.py --episode data/raw_v5/episode_xxx
    python scripts/validate_v5.py --dataset data/datasets/ewat_v5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VALID_REGIMES = {"normal", "injection", "recovery", "drift_anomaly"}
N_CANON = 41
N_FEATURES = 18


def _check_episode(ep: Path, max_raw_nan: float, trace_floor: int,
                   min_graph_fraction: float) -> dict:
    fails: list[str] = []
    sig = np.load(ep / "signal.npz")["signal"]
    raw = np.load(ep / "signal_raw.npz")["signal_raw"] if (ep / "signal_raw.npz").exists() else sig
    adj = np.load(ep / "adjacency.npz")["adjacency"]
    services = json.load(open(ep / "services.json"))
    labels = pd.read_parquet(ep / "labels.parquet")
    T, N, F = sig.shape

    if (N, F) != (N_CANON, N_FEATURES):
        fails.append(f"shape (N,F)=({N},{F}) != ({N_CANON},{N_FEATURES})")
    if adj.shape != (T, N, N, 3):
        fails.append(f"adjacency {adj.shape} != ({T},{N},{N},3)")
    if len(services) != N:
        fails.append(f"services.json={len(services)} != N={N}")

    nan_imp = float(np.isnan(sig).mean())
    if nan_imp > 0.0:
        fails.append(f"signal imputé NaN={nan_imp:.3%} (>0)")
    nan_raw = float(np.isnan(raw).mean())
    if nan_raw > max_raw_nan:
        fails.append(f"signal brut NaN={nan_raw:.1%} (> {max_raw_nan:.0%})")

    # couverture trace : services avec ≥1 span (latence non-NaN dans le brut)
    lat = raw[:, :, 2]  # latency_p99 (depuis traces)
    traced = int((~np.isnan(lat)).any(axis=0).sum())
    if traced < trace_floor:
        fails.append(f"couverture trace {traced}/{N} < plancher {trace_floor}")

    # G(t) non vide
    frac_edges = float(np.mean([(adj[t, :, :, 0] > 0).any() for t in range(T)]))
    if frac_edges < min_graph_fraction:
        fails.append(f"G(t) vide : {frac_edges:.0%} steps avec arête < {min_graph_fraction:.0%}")

    bad = set(labels["regime"].unique()) - VALID_REGIMES
    if bad:
        fails.append(f"régimes invalides : {bad}")
    if len(labels) != T:
        fails.append(f"labels rows={len(labels)} != T={T}")

    return {"episode": ep.name, "T": T, "traced": traced, "nan_raw": nan_raw,
            "g_edge_frac": frac_edges, "pass": not fails, "failures": fails}


def _check_heldout_leak(dataset: Path) -> list[str]:
    """Aucun épisode held_out_flag=True ne doit être en train/val."""
    fails = []
    split = json.load(open(dataset / "split.json"))
    train_val = set(split.get("train", [])) | set(split.get("val", []))
    eps_dir = dataset / "episodes"
    for eid in train_val:
        lab = eps_dir / eid / "labels.parquet"
        if lab.exists():
            df = pd.read_parquet(lab)
            if bool(df["held_out_flag"].any()):
                fails.append(f"FUITE held-out en train/val : {eid}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 validation gate")
    ap.add_argument("--features-root", type=Path)
    ap.add_argument("--episode", type=Path)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--max-raw-nan", type=float, default=0.50)
    # Plancher = détecter une collecte CASSÉE, pas exiger la couverture totale.
    # Réalité TT : 100% des services ont des métriques ; ~20-30/41 ont des traces
    # par épisode (le reste = services dépendant du cycle de commande ou
    # structurellement silencieux : ui-dashboard nginx, verification-code bypassé,
    # news, ticket-office). Imputés en activité-0. La diversité des scénarios
    # couvre tous les services au niveau dataset.
    ap.add_argument("--trace-floor", type=int, default=18)
    ap.add_argument("--min-graph-fraction", type=float, default=0.10)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    episodes: list[Path] = []
    if args.episode:
        episodes = [args.episode]
    elif args.features_root:
        episodes = [p for p in sorted(args.features_root.iterdir())
                    if (p / "signal.npz").exists()]
    elif args.dataset:
        episodes = [p for p in sorted((args.dataset / "episodes").iterdir())
                    if (p / "signal.npz").exists()]
    if not episodes:
        print("Aucun épisode featurisé trouvé.", file=sys.stderr)
        sys.exit(2)

    results = [_check_episode(p, args.max_raw_nan, args.trace_floor,
                              args.min_graph_fraction) for p in episodes]
    n = len(results)
    n_pass = sum(r["pass"] for r in results)
    print(f"EWAT v5 gate — {n} épisodes ({n_pass} pass, {n - n_pass} fail)")
    for r in results:
        tag = "[OK]" if r["pass"] else "[FAIL]"
        print(f"  {tag} {r['episode']}  T={r['T']} traced={r['traced']}/41 "
              f"nan_raw={r['nan_raw']:.1%} g_edges={r['g_edge_frac']:.0%}")
        for f in r["failures"]:
            print(f"        - {f}")

    leak_fails = []
    if args.dataset:
        leak_fails = _check_heldout_leak(args.dataset)
        print(f"\nFuite held-out : {'AUCUNE' if not leak_fails else len(leak_fails)}")
        for f in leak_fails:
            print(f"  - {f}")

    if args.output:
        json.dump({"n": n, "n_pass": n_pass, "results": results,
                   "heldout_leak": leak_fails}, open(args.output, "w"), indent=2)

    sys.exit(0 if (n_pass == n and not leak_fails) else 1)


if __name__ == "__main__":
    main()
