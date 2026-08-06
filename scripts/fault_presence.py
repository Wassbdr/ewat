"""EWAT v5 — moniteur de PRÉSENCE DE FAUTE.

Vérifie qu'une faute injectée s'est réellement manifestée dans le signal —
la couche de validation qui manquait (nuit du 2026-07-20 : ~450 épisodes
collectés sans faute réelle sont passés inaperçus car aucun contrôle ne
vérifiait la correspondance label↔signal) :
  - chaos-daemon absent des nœuds collectors → tt/tt-b n'injectaient rien ;
  - panne cAdvisor → métriques M(t) imputées.
Le raw gate (traces/logs/prom > 0) et validate_v5 (shape/NaN/régimes) ne
regardent PAS si la faute a eu lieu. Ce script comble ce trou.

Détecteur (calibré sur ewat_v5, fautes connues présentes) :
  score = max sur la phase HIGH d'injection (dernier tiers) / médiane baseline,
  pris sur le MEILLEUR de {semantic_anomaly, latency_p99} — deux features qui
  bougent sur *tous* les types de faute (×1.8–2.2 en calibration). ≥1.4 = faute
  visible. 82 % des épisodes sains passent ; un namespace cassé tombe à ~10 %.

Usage systémique (le bon) : comparer le TAUX par namespace/runner. Un lot très
en-dessous de la cohorte = défaillance d'injection systémique → recollecter.

    python scripts/fault_presence.py --features-root data/raw_v5_2
    python scripts/fault_presence.py --features-root data/raw_v5_2 --by-rep
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

SEMANTIC_ANOMALY = 16
LATENCY_P99 = 2
# scénarios AVEC faute attendue (on exclut normal/drift : pas de faute → score bas normal)
CHAOS_PREFIXES = ("cpu_", "memory_", "network_", "net_", "dns_", "pod_",
                  "container_", "time_", "held_", "compo_", "faulty_", "F")
NO_FAULT = ("normal_baseline", "rolling_deploy", "config_rollout", "autoscale_up")
PASS_THRESHOLD = 1.4


def _scenario(ep_name: str) -> str:
    return "_".join(ep_name.split("_")[1:-2])


def _rep(ep_name: str) -> int | None:
    try:
        return int(ep_name.split("_")[-2])
    except (ValueError, IndexError):
        return None


def _namespace_of_rep(rep: int) -> str:
    """Split 3-runner de la campagne : tt=0-9, tt-b=10-19, tt-c=20-29."""
    return "tt" if rep < 10 else ("tt-b" if rep < 20 else "tt-c")


def fault_score(ep: str) -> float | None:
    """max(semantic_anomaly, latency_p99) en phase high / médiane baseline."""
    try:
        sig = np.load(os.path.join(ep, "signal.npz"))["signal"]
        lab = pd.read_parquet(os.path.join(ep, "labels.parquet"))
    except Exception:
        return None
    reg = lab["regime"].values
    base = np.where(reg == "normal")[0]
    inj = np.where(np.isin(reg, ("injection", "drift_anomaly")))[0]
    if len(base) < 3 or len(inj) < 3:
        return None
    high = inj[len(inj) * 2 // 3:]  # dernier tiers = intensité haute
    if len(high) < 2:
        high = inj
    scores = []
    for fi in (SEMANTIC_ANOMALY, LATENCY_P99):
        svc_max = np.nanmax(sig[:, :, fi], axis=1)
        b = np.nanmedian(svc_max[base])
        h = np.nanmax(svc_max[high])
        if not np.isnan(b) and not np.isnan(h):
            scores.append(h / b if b > 1e-6 else (5.0 if h > 1e-6 else 1.0))
    return max(scores) if scores else None


def main() -> None:
    ap = argparse.ArgumentParser(description="EWAT v5 — moniteur de présence de faute")
    ap.add_argument("--features-root", required=True)
    ap.add_argument("--by-rep", action="store_true",
                    help="regrouper par namespace de campagne (tt/tt-b/tt-c via rep)")
    ap.add_argument("--threshold", type=float, default=PASS_THRESHOLD)
    ap.add_argument("--list-absent", action="store_true",
                    help="lister les épisodes sans faute détectée")
    args = ap.parse_args()

    groups: dict[str, list] = {}
    absent = []
    for ep in sorted(glob.glob(os.path.join(args.features_root, "episode_*"))):
        name = os.path.basename(ep)
        scen = _scenario(name)
        if scen in NO_FAULT or not scen.startswith(CHAOS_PREFIXES):
            continue
        s = fault_score(ep)
        if s is None:
            continue
        key = "all"
        if args.by_rep:
            rep = _rep(name)
            key = _namespace_of_rep(rep) if rep is not None else "?"
        groups.setdefault(key, []).append(s)
        if s < args.threshold:
            absent.append((name, s))

    print(f"Présence de faute — seuil ≥ {args.threshold} "
          f"(sain ≈ 80 % ; namespace cassé ≈ 10 %)")
    print(f"{'groupe':<8}{'n':>6}{'médiane':>10}{'% faute visible':>18}")
    rates = {}
    for key in sorted(groups):
        v = np.array(groups[key])
        rate = 100 * (v >= args.threshold).mean()
        rates[key] = rate
        print(f"{key:<8}{len(v):>6}{np.median(v):>10.2f}{rate:>16.0f} %")

    # alarme systémique : un groupe très en dessous de la cohorte
    if len(rates) > 1:
        cohort = np.median(list(rates.values()))
        for key, rate in rates.items():
            if rate < 0.5 * cohort:
                print(f"  ⚠ {key} = {rate:.0f} % << cohorte {cohort:.0f} % "
                      f"→ défaillance d'injection systémique probable, recollecter")

    if args.list_absent:
        print(f"\n{len(absent)} épisodes sans faute détectée :")
        for name, s in sorted(absent, key=lambda x: x[1])[:40]:
            print(f"  {s:.2f}  {name}")


if __name__ == "__main__":
    main()
