"""A1 — Distant-window stress test.

Question
--------
The H3 precursor reports AUROC = 0.987 ± 0.011 on the *last* k timesteps of the
normal regime (right before injection). Is this signal genuine *precursion*, or
does the classifier just learn the *static scenario signature* (which service
is targeted, baseline load, etc.) that is already present from the very start
of the normal regime?

Method
------
Evaluate the same trained SiameseTyper + PrecursorClassifier on three window
positions within the normal regime:

  - ``near``   : ``normal_indices[-k:]``   (status quo, right before injection)
  - ``middle`` : k indices centered in the middle of the normal window
  - ``far``    : ``normal_indices[:k]``    (beginning of the normal regime)

Interpretation
--------------
If ``AUROC_far ≈ AUROC_near`` (within bootstrap CI) → the precursor relies on
*static scenario signatures*, not on dynamics that build up toward the
injection. The 0.987 headline is then *not* a measure of precursion power.

If ``AUROC_far ≪ AUROC_near`` → there is genuine pre-injection dynamics being
captured. Good news for H3.

Usage
-----
    python -m experiments.h3_robustness.distant_window \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --features-root data/features/v3 \\
        [--k 6] [--seed 42] [--n-bootstrap 1000] \\
        [--output experiments/h3_robustness/distant_window]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ewat.encoder.factory import build_encoder
from ewat.precursor.dataset import PrecursorDataset
from ewat.precursor.model import PrecursorClassifier
from ewat.typing.siamese import SiameseTyper
from ewat.utils.bootstrap import bootstrap_auroc_ci
from utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Embedding helper — mirrors experiments/precursor/train.py:_embed_dataset
# ---------------------------------------------------------------------------

@torch.no_grad()
def _embed_dataset(
    typer: SiameseTyper,
    dataset: PrecursorDataset,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    typer.eval()
    embeddings, labels = [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        sig = item["signal"].unsqueeze(0).to(device)
        adj = item["adjacency"].unsqueeze(0).to(device)
        z = typer.embed(sig, adj).cpu().numpy()
        embeddings.append(z[0])
        labels.append(item["cluster"])
    return np.stack(embeddings), np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A1 — Distant-window stress test")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=None,
                        help="Default: <typing-dir>.parent/encoder")
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/h3_robustness/distant_window"))
    parser.add_argument("--k", type=int, default=6,
                        help="Window length in timesteps (default: 6 = 180s)")
    parser.add_argument("--classifier-type", default="lr_tuned",
                        choices=["lr", "lr_tuned", "rf", "svc"])
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--reg-c", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _load_typer(
    typing_dir: Path,
    encoder_dir: Path,
    device: torch.device,
) -> tuple[SiameseTyper, Path]:
    enc_ckpt_path = encoder_dir / "checkpoints" / "best_encoder.pt"
    typer_ckpt_path = typing_dir / "checkpoints" / "best_siamese.pt"
    enc_ckpt = torch.load(enc_ckpt_path, map_location="cpu", weights_only=False)
    arch_meta = enc_ckpt.get("arch") or {}
    # Legacy checkpoints saved with use_layer_norm=True (now default False) carry
    # tcn_blocks.*.norm.{weight,bias} in the state_dict — detect and forward.
    sd_keys = enc_ckpt["encoder_state"].keys()
    has_tcn_layer_norm = any(".norm.weight" in k for k in sd_keys if "tcn_blocks" in k)
    encoder = build_encoder(
        arch_meta.get("architecture", "stgcn"),
        d_feat=int(arch_meta.get("d_feat", 17)),
        n_nodes=int(arch_meta.get("n_nodes", 6)),
        d_hidden=int(arch_meta.get("d_hidden", 64)),
        d_embed=int(arch_meta.get("d_embed", 64)),
        use_layer_norm=has_tcn_layer_norm,
    )
    encoder.load_state_dict(enc_ckpt["encoder_state"])
    typer_ckpt = torch.load(typer_ckpt_path, map_location="cpu", weights_only=False)
    d_proj = int(typer_ckpt.get("d_proj", 32))
    typer = SiameseTyper(encoder, d_proj=d_proj)
    typer.load_state_dict(typer_ckpt["typer_state"])
    typer = typer.to(device).eval()
    default_scaler = str(encoder_dir / "scaler.pkl")
    scaler_path = Path(enc_ckpt.get("scaler_path", default_scaler))
    return typer, scaler_path


def _evaluate_position(
    typer: SiameseTyper,
    cluster_manifest: dict,
    features_root: Path,
    k: int,
    scaler_path: Path,
    position: str,
    n_clusters: int,
    classifier_type: str,
    reg_c: float,
    max_iter: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    device: torch.device,
) -> dict:
    """Train + evaluate precursor classifier at a given window position."""
    ds_train = PrecursorDataset(
        cluster_manifest, features_root, k=k, split="train", window_position=position
    )
    ds_test = PrecursorDataset(
        cluster_manifest, features_root, k=k, split="test", window_position=position
    )
    if scaler_path.exists():
        ds_train.load_scaler(scaler_path)
        ds_test.load_scaler(scaler_path)

    z_train, y_train = _embed_dataset(typer, ds_train, device)
    z_test, y_test = _embed_dataset(typer, ds_test, device)

    clf = PrecursorClassifier(
        n_clusters=n_clusters, reg_c=reg_c, max_iter=max_iter,
        classifier_type=classifier_type,
    )
    clf.fit(z_train, y_train)
    auroc_test = clf.auroc_per_type(z_test, y_test)
    scores = clf.scores_per_type(z_test, y_test)

    per_cluster_ci = {}
    for c in range(n_clusters):
        if c in scores:
            y_true, y_score = scores[c]
            ci = bootstrap_auroc_ci(y_true, y_score, n=n_bootstrap, rng=rng)
            per_cluster_ci[c] = {**ci.as_dict(), "n_pos": int(np.sum(y_true == 1))}
        else:
            per_cluster_ci[c] = {
                "estimate": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n_pos": 0,
            }

    valid_aucs = [a for a in auroc_test.values() if not np.isnan(a)]
    macro_auroc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
    return {
        "macro_auroc": macro_auroc,
        "per_cluster": {str(c): float(auroc_test.get(c, float("nan")))
                        for c in range(n_clusters)},
        "per_cluster_ci": {str(c): v for c, v in per_cluster_ci.items()},
    }


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    encoder_dir = args.encoder_dir if args.encoder_dir is not None else (
        args.typing_dir.parent / "encoder"
    )

    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest = json.loads(manifest_path.read_text())
    n_clusters = max(int(v["cluster"]) for v in cluster_manifest.values()) + 1
    print(f"Manifest: {len(cluster_manifest)} episodes, {n_clusters} clusters")

    typer, scaler_path = _load_typer(args.typing_dir, encoder_dir, device)

    positions = ["last", "middle", "first"]   # near → middle → far
    results: dict[str, dict] = {}
    rng = np.random.default_rng(args.seed)
    for pos in positions:
        print(f"\n--- window_position = {pos!r} (k={args.k}) ---")
        results[pos] = _evaluate_position(
            typer, cluster_manifest, args.features_root, args.k,
            scaler_path, pos, n_clusters, args.classifier_type,
            args.reg_c, args.max_iter, args.n_bootstrap, rng, device,
        )
        print(f"  macro-AUROC test = {results[pos]['macro_auroc']:.3f}")

    # Diagnostic Δ
    delta_far_near = results["first"]["macro_auroc"] - results["last"]["macro_auroc"]
    verdict = "LEAK_CONFIRMED" if abs(delta_far_near) < 0.05 else "GENUINE_PRECURSION"
    print(f"\nΔ(far − near) = {delta_far_near:+.3f}  →  {verdict}")

    summary = {
        "k": args.k,
        "n_clusters": n_clusters,
        "classifier_type": args.classifier_type,
        "seed": args.seed,
        "n_bootstrap": args.n_bootstrap,
        "results": results,
        "delta_far_near_macro": delta_far_near,
        "verdict": verdict,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    lines = [
        "# A1 — Distant-window stress test",
        "",
        f"k = {args.k} steps ({args.k * 30}s) | classifier = {args.classifier_type} "
        f"| seed = {args.seed} | n_bootstrap = {args.n_bootstrap}",
        "",
        "## Macro-AUROC by window position (test set)",
        "",
        "| position | macro-AUROC test |",
        "|---|---|",
    ]
    for pos in positions:
        lines.append(f"| {pos} | {results[pos]['macro_auroc']:.3f} |")
    lines += [
        "",
        f"**Δ(far − near) = {delta_far_near:+.3f}**  →  **{verdict}**",
        "",
        "If `far` ≈ `near` (|Δ| < 0.05) → the classifier exploits static scenario "
        "signature, not pre-injection dynamics. The headline AUROC = 0.987 is then "
        "*not* a measure of precursion power, only of scenario discrimination.",
        "",
        "## Per-cluster AUROC × position",
        "",
        "| cluster | n_pos | near | middle | far |",
        "|---|---|---|---|---|",
    ]
    for c in range(n_clusters):
        n_pos = results["last"]["per_cluster_ci"][str(c)]["n_pos"]
        near = results["last"]["per_cluster"][str(c)]
        mid = results["middle"]["per_cluster"][str(c)]
        far = results["first"]["per_cluster"][str(c)]
        fmt = lambda x: f"{x:.3f}" if not np.isnan(x) else "NaN"
        lines.append(f"| C{c} | {n_pos} | {fmt(near)} | {fmt(mid)} | {fmt(far)} |")
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
