"""C2-A1 — Distant-window stress test for the Chaos Mesh-trained model (C1).

Question (Phase C2)
-------------------
The Phase A A1 test was run on the EWAT-label encoder (circular target) and
showed Δ(far − near) ≈ 0. The B1 diagnostic on raw features + Chaos Mesh
showed Δ ≈ −0.06 on v4_strat (real dynamic signal).

Now that we have a Chaos Mesh-trained STGCN classifier (C1, test AUROC = 0.863
on v4_strat), we re-run the distant-window test: does the trained STGCN+head
exploit dynamics from the pre-injection window, or only the static signature?

Method
------
Load C1 checkpoint. For each test episode, build *truncated* inputs:
  - `near` : keep only the last  k steps of the normal regime (status quo);
  - `middle`: middle k of normal;
  - `first`: first k of normal (maximum distance from injection).
Pass each truncation through the model, compute macro-AUROC vs Chaos Mesh
labels.

Interpretation
--------------
- Δ(far − near) ≪ 0 → real dynamic signal exploited by the model
- Δ(far − near) ≈ 0 → still scenario signature even with Chaos Mesh target

Usage
-----
    python -m experiments.architecture_v2.distant_window_chaos_mesh \\
        --checkpoint experiments/architecture_v2/chaos_mesh/checkpoints/best_model.pt \\
        --dataset data/datasets/ewat_v4_strat \\
        --features-root data/features/v4 \\
        --output experiments/architecture_v2/distant_window_v4 \\
        [--k 6] [--n-bootstrap 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from experiments.architecture_v2.train_chaos_mesh import (
    STGCNClassifier,
    _macro_auroc,
    _scan_scenarios,
    _scenario_to_int,
)

# ---------------------------------------------------------------------------
# Truncated dataset
# ---------------------------------------------------------------------------

class TruncatedEpisodeDataset(Dataset):
    """Like EpisodeDataset but emits only k steps from the normal regime,
    selected by ``window_position``."""

    def __init__(
        self,
        episode_ids: list[str],
        features_root: Path,
        k: int,
        window_position: str = "last",
        scaler=None,
        instance_normalize: bool = True,
    ) -> None:
        if window_position not in ("last", "first", "middle"):
            raise ValueError(window_position)
        self.episode_ids = episode_ids
        self.features_root = Path(features_root)
        self.k = k
        self.window_position = window_position
        self.scaler = scaler
        self.instance_normalize = instance_normalize

    def __len__(self) -> int:
        return len(self.episode_ids)

    def __getitem__(self, idx: int) -> dict:
        ep_id = self.episode_ids[idx]
        ep_dir = self.features_root / ep_id
        signal = np.load(ep_dir / "signal.npz")["signal"].astype(np.float32)
        adjacency = np.load(ep_dir / "adjacency.npz")["adjacency"].astype(np.float32)
        labels_df = pd.read_parquet(ep_dir / "labels.parquet",
                                    columns=["regime", "scenario"])

        normal_mask = (labels_df["regime"] == "normal").values
        normal_idx = np.where(normal_mask)[0]
        if len(normal_idx) == 0:
            normal_idx = np.arange(min(self.k, signal.shape[0]))
        n = len(normal_idx)
        if self.window_position == "last":
            sel = normal_idx[-self.k:]
        elif self.window_position == "first":
            sel = normal_idx[: self.k]
        else:
            if n <= self.k:
                sel = normal_idx
            else:
                start = (n - self.k) // 2
                sel = normal_idx[start: start + self.k]

        win_sig = signal[sel]
        win_adj = adjacency[sel]
        actual = win_sig.shape[0]

        # Instance normalize using NORMAL stats from full episode
        if self.instance_normalize and normal_mask.sum() >= 2:
            ref = signal[normal_mask]
            mu = np.nanmean(ref, axis=(0, 1), keepdims=True)
            sd = np.nanstd(ref, axis=(0, 1), keepdims=True) + 1e-6
            win_sig = ((win_sig - mu) / sd).astype(np.float32)

        # Apply global scaler if provided
        if self.scaler is not None:
            T, N, d = win_sig.shape
            flat = win_sig.reshape(-1, d)
            nan_mask = np.isnan(flat)
            flat = np.where(nan_mask, self.scaler.mean_, flat)
            flat = self.scaler.transform(flat).astype(np.float32)
            win_sig = flat.reshape(T, N, d)
        else:
            win_sig = np.nan_to_num(win_sig, nan=0.0)
        win_adj = np.nan_to_num(win_adj, nan=0.0)

        # Left-pad to k
        if actual < self.k:
            pad = self.k - actual
            win_sig = np.concatenate(
                [np.zeros((pad, *win_sig.shape[1:]), dtype=np.float32), win_sig], axis=0
            )
            win_adj = np.concatenate(
                [np.zeros((pad, *win_adj.shape[1:]), dtype=np.float32), win_adj], axis=0
            )
        return {
            "signal": torch.from_numpy(win_sig),
            "adjacency": torch.from_numpy(win_adj),
            "scenario": labels_df["scenario"].iloc[0],
            "T": self.k,
        }


def _collate(batch):
    sigs = torch.stack([b["signal"] for b in batch])
    adjs = torch.stack([b["adjacency"] for b in batch])
    return {
        "signal": sigs, "adjacency": adjs,
        "scenario": [b["scenario"] for b in batch],
        "T": torch.tensor([b["T"] for b in batch], dtype=torch.long),
    }


@torch.no_grad()
def _eval_position(
    model: STGCNClassifier, episode_ids: list[str], features_root: Path,
    k: int, position: str, scaler, classes: list[str],
    batch_size: int, device: torch.device,
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    ds = TruncatedEpisodeDataset(
        episode_ids, features_root, k=k, window_position=position,
        scaler=scaler, instance_normalize=True,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)
    model.eval()
    all_probs, all_y = [], []
    for batch in loader:
        sig = batch["signal"].to(device)
        adj = batch["adjacency"].to(device)
        lengths = batch["T"].to(device)
        logits, _ = model(sig, adj, lengths)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        y = _scenario_to_int(batch["scenario"], classes)
        all_probs.append(probs)
        all_y.append(y)
    probs = np.concatenate(all_probs, axis=0)
    y = np.concatenate(all_y, axis=0)
    macro, per_class = _macro_auroc(y, probs, len(classes))
    return macro, per_class, probs, y


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="C2-A1 — Distant-window on Chaos Mesh model")
    p.add_argument("--checkpoint", type=Path,
                   default=Path("experiments/architecture_v2/chaos_mesh/checkpoints/best_model.pt"))
    p.add_argument("--scaler", type=Path,
                   default=Path("experiments/architecture_v2/chaos_mesh/checkpoints/scaler.pkl"))
    p.add_argument("--dataset", type=Path, default=Path("data/datasets/ewat_v4_strat"))
    p.add_argument("--features-root", type=Path, default=Path("data/features/v4"))
    p.add_argument("--output", type=Path,
                   default=Path("experiments/architecture_v2/distant_window_v4"))
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    classes = _scan_scenarios(args.dataset)
    n_classes = len(classes)
    print(f"Classes: {n_classes} | k={args.k}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = STGCNClassifier(n_classes=n_classes).to(device)
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.classifier.load_state_dict(ckpt["classifier_state"])
    print(f"Loaded model from {args.checkpoint}")

    with open(args.scaler, "rb") as f:
        scaler = pickle.load(f)

    # Get test episode list
    df = pd.read_parquet(args.dataset / "index.parquet")
    test_eps = df[df["split"] == "test"]["episode_id"].tolist()
    print(f"Test episodes: {len(test_eps)}")

    rng = np.random.default_rng(args.seed)
    results = {}
    for position in ["last", "middle", "first"]:
        print(f"\n--- position = {position} ---")
        macro, per_class, probs, y = _eval_position(
            model, test_eps, args.features_root, args.k, position, scaler,
            classes, args.batch_size, device,
        )
        # Bootstrap CI on macro
        n = len(y)
        boots = []
        for _ in range(args.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            m, _ = _macro_auroc(y[idx], probs[idx], n_classes)
            if not np.isnan(m):
                boots.append(m)
        boots = np.array(boots)
        ci_lo = float(np.percentile(boots, 2.5))
        ci_hi = float(np.percentile(boots, 97.5))
        print(f"  macro-AUROC = {macro:.3f}  95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]")
        results[position] = {
            "macro_auroc": macro, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "per_class": {classes[c]: float(per_class.get(c, float("nan")))
                          for c in range(n_classes)},
        }

    delta = results["first"]["macro_auroc"] - results["last"]["macro_auroc"]
    verdict = ("LEAK_CONFIRMED" if abs(delta) < 0.05
               else "GENUINE_DYNAMIC" if delta < -0.05
               else "AMBIGUOUS")

    summary = {
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "k": args.k,
        "n_test": len(test_eps),
        "n_classes": n_classes,
        "results": results,
        "delta_far_near_macro": delta,
        "verdict": verdict,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# C2-A1 — Distant-window on Chaos Mesh STGCN model",
        "",
        f"Checkpoint : `{args.checkpoint}` | dataset : `{args.dataset}` | k = {args.k}",
        "",
        "## Macro-AUROC by window position (test set)",
        "",
        "| position | macro-AUROC | 95% CI |",
        "|---|---|---|",
    ]
    for pos in ["last", "middle", "first"]:
        r = results[pos]
        lines.append(
            f"| {pos} | {r['macro_auroc']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |"
        )
    lines += [
        "",
        f"**Δ(far − near) = {delta:+.3f}**  →  **{verdict}**",
        "",
        "**Lecture** :",
        "- |Δ| < 0.05 → leak signature scénario (cohérent avec A1 sur labels EWAT)",
        "- Δ < −0.05 → vraie dynamique pré-injection exploitée",
        "- 0 < |Δ| < 0.05 → ambigu",
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nΔ(far − near) = {delta:+.3f}  →  {verdict}")
    print(f"Report: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
