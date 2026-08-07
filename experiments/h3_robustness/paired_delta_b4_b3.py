"""A5 — Paired bootstrap CI on Δ(B4 - B3) = macro-AUROC gain from STGCN encoder.

Question
--------
STATUS reports Δ(B4 - B3) = 0.000 (B4 = STGCN on Chaos Mesh scenarios,
B3 = raw features). Is this Δ a meaningful "encoder neutral" result, or could
the wide confidence band absorb a non-trivial encoder effect?

Method
------
1. Reproduce B3 and B4 setups from
   ``experiments/baselines/scenario_baselines.py``.
2. Fit B3 and B4 multi-class LR-OvR once each.
3. For each bootstrap resample of the *test indices* (same indices used for
   both B3 and B4 — *paired*):
      - macro-AUROC_B3_b
      - macro-AUROC_B4_b
      - Δ_b = macro-AUROC_B4_b - macro-AUROC_B3_b
4. Report mean Δ and 95% bootstrap CI. If 0 ∈ CI → encoder neutrality
   defensible; if CI excludes 0 negatively → encoder *hurts*; positively →
   encoder helps.

Usage
-----
    python -m experiments.h3_robustness.paired_delta_b4_b3 \\
        --typing-dir experiments/typing \\
        --encoder-dir experiments/encoder \\
        --features-root data/features/v3 \\
        [--k 6] [--n-bootstrap 1000] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ewat.precursor.dataset import PrecursorDataset
from experiments.baselines.scenario_baselines import (
    _flatten_with_scenario,
    _scenario_to_int,
)
from experiments.h3_robustness.distant_window import _load_typer
from utils.seeding import seed_everything


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A5 — Paired bootstrap CI on Δ(B4-B3)")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=None)
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--output", type=Path,
                        default=Path("experiments/h3_robustness/paired_delta_b4_b3"))
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--reg-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


@torch.no_grad()
def _embed_z_e(
    dataset: PrecursorDataset,
    encoder: torch.nn.Module,
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """Forward through ENCODER ONLY (no siamese head)."""
    from src.ewat.encoder.dataset import EpisodeDataset  # noqa: F401
    encoder.eval()
    embeddings, scenarios = [], []
    for idx in range(len(dataset)):
        item = dataset[idx]
        sig = item["signal"].unsqueeze(0).to(device)
        adj = item["adjacency"].unsqueeze(0).to(device)
        z = encoder(sig, adj)
        if isinstance(z, tuple):
            z = z[0]
        # Pool over time dim if needed: encoder returns (B, T, N, d) or (B, d)
        if z.ndim == 4:
            z = z.mean(dim=(1, 2))   # → (B, d)
        elif z.ndim == 3:
            z = z.mean(dim=1)
        embeddings.append(z[0].cpu().numpy())
        scenarios.append(item["episode_id"])
    return np.stack(embeddings), scenarios


def _macro_auroc_from_probas(y: np.ndarray, p: np.ndarray, n_classes: int) -> float:
    aurocs = []
    for i in range(n_classes):
        y_bin = (y == i).astype(int)
        if y_bin.sum() < 1 or y_bin.sum() == len(y_bin):
            continue
        try:
            aurocs.append(float(roc_auc_score(y_bin, p[:, i])))
        except ValueError:
            continue
    return float(np.mean(aurocs)) if aurocs else float("nan")


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    encoder_dir = args.encoder_dir if args.encoder_dir is not None else (
        args.typing_dir.parent / "encoder"
    )

    manifest_path = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
    cluster_manifest = json.loads(manifest_path.read_text())

    typer, scaler_path = _load_typer(args.typing_dir, encoder_dir, device)
    encoder = typer.encoder

    # Build datasets
    ds_train = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="train")
    ds_test = PrecursorDataset(cluster_manifest, args.features_root, k=args.k, split="test")
    if scaler_path.exists():
        ds_train.load_scaler(scaler_path)
        ds_test.load_scaler(scaler_path)

    # B3 features (raw flattened S(t), averaged over k)
    x_train_raw, sc_train = _flatten_with_scenario(ds_train, cluster_manifest)
    x_test_raw, sc_test = _flatten_with_scenario(ds_test, cluster_manifest)

    # Map scenario strings → int (same mapping for train and test)
    y_train, classes = _scenario_to_int(np.array(sc_train))
    # Use same classes ordering for test
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y_test = np.array([cls_to_idx[s] for s in sc_test], dtype=int)
    n_classes = len(classes)

    # Fit B3 (raw features LR)
    print("Fitting B3 (raw features) …")
    clf_b3 = LogisticRegression(C=args.reg_c, max_iter=args.max_iter, solver="lbfgs")
    clf_b3.fit(x_train_raw, y_train)
    probas_b3 = clf_b3.predict_proba(x_test_raw)

    # Align column order to canonical class order
    def _align_probas(clf, probas, classes):
        idx = [list(clf.classes_).index(i) for i in range(len(classes))
               if i in clf.classes_]
        return probas[:, idx]
    probas_b3_full = np.zeros((len(y_test), n_classes), dtype=np.float64)
    for col_idx, cls_id in enumerate(clf_b3.classes_):
        probas_b3_full[:, int(cls_id)] = probas_b3[:, col_idx]

    # Fit B4 (encoder z_e LR)
    print("Embedding with encoder z_e (train + test) …")
    # Embed via encoder forward — use SiameseTyper's underlying encoder
    @torch.no_grad()
    def _encode(ds):
        outs = []
        for i in range(len(ds)):
            it = ds[i]
            sig = it["signal"].unsqueeze(0).to(device)
            adj = it["adjacency"].unsqueeze(0).to(device)
            z = encoder(sig, adj)
            if isinstance(z, tuple):
                z = z[0]
            if z.ndim == 4:
                z = z.mean(dim=(1, 2))
            elif z.ndim == 3:
                z = z.mean(dim=1)
            outs.append(z[0].cpu().numpy())
        return np.stack(outs)

    z_train = _encode(ds_train)
    z_test = _encode(ds_test)

    print("Fitting B4 (encoder z_e) …")
    clf_b4 = LogisticRegression(C=args.reg_c, max_iter=args.max_iter, solver="lbfgs")
    clf_b4.fit(z_train, y_train)
    probas_b4 = clf_b4.predict_proba(z_test)
    probas_b4_full = np.zeros((len(y_test), n_classes), dtype=np.float64)
    for col_idx, cls_id in enumerate(clf_b4.classes_):
        probas_b4_full[:, int(cls_id)] = probas_b4[:, col_idx]

    # Observed macro-AUROC
    obs_b3 = _macro_auroc_from_probas(y_test, probas_b3_full, n_classes)
    obs_b4 = _macro_auroc_from_probas(y_test, probas_b4_full, n_classes)
    obs_delta = obs_b4 - obs_b3
    print(f"\nObserved: B3 = {obs_b3:.4f} | B4 = {obs_b4:.4f} | "
          f"Δ(B4-B3) = {obs_delta:+.4f}")

    # Paired bootstrap
    rng = np.random.default_rng(args.seed)
    n_test = len(y_test)
    boot_b3, boot_b4, boot_delta = [], [], []
    for _ in range(args.n_bootstrap):
        idx = rng.integers(0, n_test, size=n_test)
        y_boot = y_test[idx]
        m3 = _macro_auroc_from_probas(y_boot, probas_b3_full[idx], n_classes)
        m4 = _macro_auroc_from_probas(y_boot, probas_b4_full[idx], n_classes)
        if np.isnan(m3) or np.isnan(m4):
            continue
        boot_b3.append(m3)
        boot_b4.append(m4)
        boot_delta.append(m4 - m3)

    boot_delta_arr = np.array(boot_delta)
    delta_ci_lo = float(np.percentile(boot_delta_arr, 2.5))
    delta_ci_hi = float(np.percentile(boot_delta_arr, 97.5))
    excludes_zero = (delta_ci_lo > 0) or (delta_ci_hi < 0)
    p_neutral_or_negative = float(np.mean(boot_delta_arr <= 0))

    summary = {
        "k": args.k,
        "n_test": n_test,
        "n_classes": n_classes,
        "n_bootstrap": args.n_bootstrap,
        "observed_b3": obs_b3,
        "observed_b4": obs_b4,
        "observed_delta": obs_delta,
        "delta_ci_lo": delta_ci_lo,
        "delta_ci_hi": delta_ci_hi,
        "delta_bootstrap_mean": float(np.mean(boot_delta_arr)),
        "delta_bootstrap_std": float(np.std(boot_delta_arr)),
        "ci_excludes_zero": bool(excludes_zero),
        "p_delta_leq_zero": p_neutral_or_negative,
    }
    (args.output / "results.json").write_text(json.dumps(summary, indent=2))

    print("\nPaired bootstrap on Δ(B4 - B3):")
    print(f"  Δ = {obs_delta:+.4f}   95% CI = [{delta_ci_lo:+.4f}, {delta_ci_hi:+.4f}]")
    print(f"  CI excludes 0: {excludes_zero}")
    print(f"  P(Δ ≤ 0) = {p_neutral_or_negative:.3f}")

    lines = [
        "# A5 — Paired bootstrap CI on Δ(B4 − B3)",
        "",
        f"k = {args.k} | n_test = {n_test} | n_bootstrap = {args.n_bootstrap}",
        "",
        "## Observed",
        "",
        f"- B3 (raw features) macro-AUROC = **{obs_b3:.4f}**",
        f"- B4 (STGCN z_e)    macro-AUROC = **{obs_b4:.4f}**",
        f"- **Δ(B4 − B3) = {obs_delta:+.4f}**",
        "",
        "## Paired bootstrap",
        "",
        f"- Δ 95% CI = [{delta_ci_lo:+.4f}, {delta_ci_hi:+.4f}]",
        f"- CI excludes 0: **{excludes_zero}**",
        f"- P(Δ ≤ 0) = {p_neutral_or_negative:.3f}",
        "",
        ("**Verdict** : 0 est dans l'IC → l'encodeur STGCN est **neutre** en "
         "discriminabilité agrégée sur les labels Chaos Mesh (n=45 trop petit "
         "pour distinguer Δ=0 d'un effet < ±|IC|). La valeur du STGCN n'est "
         "pas prédictive globale ; elle est géométrique (H1) et redistributive "
         "(table par scénario)."
         if not excludes_zero else
         f"**Verdict** : 0 hors de l'IC. L'encodeur "
         f"{'aide' if obs_delta > 0 else 'détériore'} la discriminabilité agrégée "
         f"vs features brutes (Δ={obs_delta:+.4f})."),
    ]
    (args.output / "results.md").write_text("\n".join(lines))
    print(f"\nReport: {args.output / 'results.md'}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
