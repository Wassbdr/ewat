"""C-3 — End-to-end latency benchmark for the EWAT pipeline.

Question (Plan unifié — C-3)
----------------------------
La formalisation EWAT spécifie un budget de latence < 5 s :
  Étape 0 (drift) < 1 s, Étape 1 (encoder) < 2 s, Étape 3 (precursor) < 1 s.
Aucun benchmark concret n'avait été mesuré. Ce script instrumente la pipeline
complète à travers ``AlertAssembler.from_experiment_dirs`` et chronomètre
chaque étape sur ``n`` itérations.

Method
------
1. Charger le pipeline ewat_v3 (typage + encodeur + précurseurs + drift).
2. Pour chaque épisode test, exécuter la chaîne :
     S(t) → instance norm → encoder → siamois → drift → précurseurs → OpenMax
3. Chronométrer chaque étape avec ``time.perf_counter()`` (résolution µs).
4. Reporter mediane, p95, max sur ``n`` exécutions.

Sortie
------
- ``experiments/bench/results.json`` : timings raw + agrégés
- ``experiments/bench/results.md`` : tableau lisible par étape

Verdict
-------
- Vert : p95 < budget pour toutes les étapes
- Orange : p95 < 2× budget
- Rouge : p95 ≥ 2× budget

Usage
-----
    python -m experiments.bench.latency_e2e \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --precursor-dir experiments/precursor \\
        --features-root data/features/v3 \\
        --dataset data/datasets/ewat_v3 \\
        [--n-iterations 200] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ewat.alerts.assembler import AlertAssembler
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

class StepTimer:
    """Collects per-step timings across multiple invocations."""

    def __init__(self) -> None:
        self.records: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            self.records.setdefault(name, []).append(t1 - t0)

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for name, ts in self.records.items():
            if not ts:
                out[name] = {"n": 0}
                continue
            arr = np.array(ts) * 1000.0   # ms
            out[name] = {
                "n": len(ts),
                "median_ms": float(np.median(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
                "max_ms": float(arr.max()),
                "mean_ms": float(arr.mean()),
                "std_ms": float(arr.std()),
            }
        return out


# ---------------------------------------------------------------------------
# Per-stage instrumentation — wraps AlertAssembler.predict
# ---------------------------------------------------------------------------

@torch.no_grad()
def _instrumented_predict(
    asm: AlertAssembler,
    signal: np.ndarray,
    adjacency: np.ndarray,
    timer: StepTimer,
    episode_id: str = "",
) -> tuple[list, dict[str, float]]:
    """Replicates AlertAssembler.predict but timestamps each stage explicitly."""
    sub: dict[str, float] = {}

    # Reset drift detector if needed
    with timer.measure("drift_reset"):
        asm._maybe_reset_drift(episode_id)

    # Stage: normalize
    with timer.measure("normalize"):
        signal = signal.astype(np.float32)
        if asm.scaler is not None:
            t_len, n_nodes, d = signal.shape
            flat = signal.reshape(-1, d)
            nan_mask = np.isnan(flat)
            flat = np.where(nan_mask, asm.scaler.mean_, flat)
            flat = asm.scaler.transform(flat).astype(np.float32)
            signal = flat.reshape(t_len, n_nodes, d)
        else:
            signal = np.nan_to_num(signal, nan=0.0)
        adjacency = np.nan_to_num(adjacency.astype(np.float32), nan=0.0)

    # Stage 0: DriftDetector
    drift_flag = False
    with timer.measure("step0_drift"):
        if asm.drift_detector is not None:
            res = asm.drift_detector.update(signal[-1].astype(np.float64))
            drift_flag = res.flag

    if not asm.classifiers:
        return [], sub

    # Stage 1+2: Encoder + Siamois (grouped by k*)
    from collections import defaultdict
    groups: dict[int, list[int]] = defaultdict(list)
    for cluster_id in asm.classifiers:
        k = int(asm.k_optimal.get(cluster_id, 2))
        groups[k].append(cluster_id)

    n_nodes = signal.shape[1]
    feature_dim = signal.shape[2]
    adj_channels = adjacency.shape[-1]
    t_total = signal.shape[0]
    alerts = []

    for k, cluster_ids in groups.items():
        with timer.measure("step1_2_encode"):
            actual_k = min(k, t_total)
            sig_window = signal[-actual_k:]
            adj_window = adjacency[-actual_k:]
            if actual_k < k:
                pad_t = k - actual_k
                sig_window = np.concatenate([
                    np.zeros((pad_t, n_nodes, feature_dim), dtype=np.float32),
                    sig_window,
                ], axis=0)
                adj_window = np.concatenate([
                    np.zeros((pad_t, n_nodes, n_nodes, adj_channels), dtype=np.float32),
                    adj_window,
                ], axis=0)
            sig_t = torch.from_numpy(sig_window).float().unsqueeze(0).to(asm.device)
            adj_t = torch.from_numpy(adj_window).float().unsqueeze(0).to(asm.device)
            z = asm.typer.embed(sig_t, adj_t).cpu().numpy()

        # Stage 3: Precursors (one classifier per cluster_id in group)
        with timer.measure("step3_precursors"):
            for cluster_id in cluster_ids:
                clf = asm.classifiers[cluster_id]
                proba = clf.predict_proba(z)
                p_i = float(proba[0, cluster_id])
                if p_i >= asm.threshold:
                    from ewat.alerts.alert import Alert
                    alerts.append(Alert(
                        cluster_id=cluster_id, probability=p_i,
                        horizon_steps=k, horizon_seconds=k * asm.step_seconds,
                        fiche=asm.fiches.get(cluster_id, {}),
                        timestamp=0.0, episode_id=episode_id, drift_flag=drift_flag,
                    ))

    with timer.measure("end_to_end"):
        pass   # marker — we'll use timer total elsewhere
    return alerts, sub


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C-3 — End-to-end latency benchmark")
    p.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    p.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    p.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v3"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/bench"))
    p.add_argument("--n-iterations", type=int, default=200,
                   help="Number of predict() calls to time (default 200)")
    p.add_argument("--warmup", type=int, default=10,
                   help="Warmup iterations (not measured)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    return p


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.output.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    print("Loading pipeline via AlertAssembler.from_experiment_dirs …")
    try:
        asm = AlertAssembler.from_experiment_dirs(
            typing_dir=args.typing_dir,
            encoder_dir=args.encoder_dir,
            precursor_dir=args.precursor_dir,
            threshold=args.threshold,
            device=device,
        )
    except RuntimeError as e:
        # Try with use_layer_norm fallback (legacy checkpoints)
        print(f"Standard load failed ({e}); falling back to manual load…")
        from ewat.encoder.stgcn import STGCNEncoder
        from ewat.typing.siamese import SiameseTyper

        enc_ckpt = torch.load(
            args.encoder_dir / "checkpoints" / "best_encoder.pt",
            map_location="cpu", weights_only=False,
        )
        arch = enc_ckpt.get("arch") or {}
        has_norm = any(".norm.weight" in k for k in enc_ckpt["encoder_state"].keys()
                        if "tcn_blocks" in k)
        encoder = STGCNEncoder(
            d_feat=int(arch.get("d_feat", 17)),
            n_nodes=int(arch.get("n_nodes", 6)),
            d_hidden=int(arch.get("d_hidden", 64)),
            d_embed=int(arch.get("d_embed", 64)),
            use_layer_norm=has_norm,
        )
        encoder.load_state_dict(enc_ckpt["encoder_state"])
        typer_ckpt = torch.load(
            args.typing_dir / "checkpoints" / "best_siamese.pt",
            map_location="cpu", weights_only=False,
        )
        typer = SiameseTyper(encoder, d_proj=int(typer_ckpt.get("d_proj", 32)))
        typer.load_state_dict(typer_ckpt["typer_state"])
        results = json.loads((args.precursor_dir / "results.json").read_text())
        k_optimal = {int(k): int(v) for k, v in results["k_optimal"].items()}
        n_clusters = results["n_clusters"]
        from ewat.precursor.model import PrecursorClassifier
        classifiers = {}
        for c in range(n_clusters):
            k_opt = k_optimal[c]
            ckpt_path = args.precursor_dir / "checkpoints" / f"classifier_type{c}_k{k_opt}.pkl"
            if ckpt_path.exists():
                classifiers[c] = PrecursorClassifier.load(ckpt_path)
        scaler_path = args.encoder_dir / "scaler.pkl"
        scaler = None
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        from ewat.drift.detector import DriftDetector
        from ewat.drift.mmd import RFFKernel
        kernel = RFFKernel(rff_dim=256, seed=42)
        drift = DriftDetector(kernel=kernel, epsilon_drift=0.5226)
        asm = AlertAssembler(
            typer=typer, classifiers=classifiers, k_optimal=k_optimal,
            fiches={}, threshold=args.threshold, scaler=scaler,
            drift_detector=drift, device=device,
        )

    print(f"Loaded {len(asm.classifiers)} classifiers, k_optimal={asm.k_optimal}")

    # Load test episodes
    split = json.loads((args.dataset / "split.json").read_text())
    test_ids = split["test"]
    print(f"Test episodes: {len(test_ids)}")

    timer = StepTimer()

    # Warmup
    print(f"\nWarmup {args.warmup} iterations …")
    for i in range(args.warmup):
        ep_id = test_ids[i % len(test_ids)]
        ep_dir = args.features_root / ep_id
        if not ep_dir.exists():
            continue
        sig = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        adj = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
        _instrumented_predict(asm, sig, adj, StepTimer(), episode_id=ep_id)

    # Measure
    print(f"Measuring {args.n_iterations} iterations …")
    rng = np.random.default_rng(args.seed)
    total_times: list[float] = []
    for i in range(args.n_iterations):
        ep_id = test_ids[rng.integers(0, len(test_ids))]
        ep_dir = args.features_root / ep_id
        if not ep_dir.exists():
            continue
        sig = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        adj = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
        t0 = time.perf_counter()
        _instrumented_predict(asm, sig, adj, timer, episode_id=ep_id)
        total_times.append(time.perf_counter() - t0)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.n_iterations}")

    # Aggregate
    summary = timer.summary()
    total_arr = np.array(total_times) * 1000.0
    summary["TOTAL"] = {
        "n": len(total_times),
        "median_ms": float(np.median(total_arr)),
        "p95_ms": float(np.percentile(total_arr, 95)),
        "p99_ms": float(np.percentile(total_arr, 99)),
        "max_ms": float(total_arr.max()),
        "mean_ms": float(total_arr.mean()),
        "std_ms": float(total_arr.std()),
    }

    # Budget check
    budgets_ms = {
        "step0_drift": 1000,
        "step1_2_encode": 2000,
        "step3_precursors": 1000,
        "TOTAL": 5000,
    }
    verdicts = {}
    for stage, budget in budgets_ms.items():
        if stage not in summary or summary[stage].get("n", 0) == 0:
            verdicts[stage] = "N/A"
            continue
        p95 = summary[stage]["p95_ms"]
        if p95 < budget:
            verdicts[stage] = "GREEN"
        elif p95 < 2 * budget:
            verdicts[stage] = "ORANGE"
        else:
            verdicts[stage] = "RED"

    results_full = {
        "device": str(device),
        "n_iterations": args.n_iterations,
        "warmup": args.warmup,
        "n_classifiers": len(asm.classifiers),
        "k_optimal": asm.k_optimal,
        "summary_ms": summary,
        "budgets_ms": budgets_ms,
        "verdicts": verdicts,
    }
    (args.output / "results.json").write_text(json.dumps(results_full, indent=2))

    # Markdown
    lines = [
        "# C-3 — End-to-end latency benchmark",
        "",
        f"Device : `{device}` | iterations : {args.n_iterations} "
        f"(+ {args.warmup} warmup) | classifiers : {len(asm.classifiers)}",
        "",
        "## Per-stage latency (ms)",
        "",
        "| stage | median | p95 | p99 | max | mean ± std | budget | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for stage in ["normalize", "drift_reset", "step0_drift", "step1_2_encode",
                  "step3_precursors", "TOTAL"]:
        if stage not in summary or summary[stage].get("n", 0) == 0:
            continue
        s = summary[stage]
        b = budgets_ms.get(stage, "—")
        v = verdicts.get(stage, "—")
        v_str = {"GREEN": "🟢", "ORANGE": "🟡", "RED": "🔴", "N/A": "—"}.get(v, v)
        b_str = f"< {b} ms" if isinstance(b, int) else "—"
        lines.append(
            f"| {stage} | {s['median_ms']:.2f} | {s['p95_ms']:.2f} | "
            f"{s['p99_ms']:.2f} | {s['max_ms']:.2f} | "
            f"{s['mean_ms']:.2f} ± {s['std_ms']:.2f} | {b_str} | {v_str} |"
        )

    overall_v = verdicts.get("TOTAL", "N/A")
    lines += [
        "",
        f"## Verdict global : {overall_v}",
        "",
        "- Budget formel : Étape 0 < 1 s, Étape 1+2 < 2 s, Étape 3 < 1 s, Total < 5 s",
        f"- Mesuré (p95) : TOTAL = {summary.get('TOTAL', {}).get('p95_ms', 'N/A'):.2f} ms",
        "",
        "Lecture :",
        "- 🟢 GREEN : p95 < budget",
        "- 🟡 ORANGE : budget ≤ p95 < 2× budget",
        "- 🔴 RED : p95 ≥ 2× budget",
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")
    print(f"\nTOTAL median = {summary['TOTAL']['median_ms']:.2f} ms | "
          f"p95 = {summary['TOTAL']['p95_ms']:.2f} ms | "
          f"verdict = {overall_v}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
