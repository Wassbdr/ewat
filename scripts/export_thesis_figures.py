"""Regenerate thesis figures and copy them into docs/rapport/figures/.

Runs alert ROC/PR evaluation, cluster analysis heatmap, and cluster semantics table,
then copies PNG outputs for LaTeX inclusion.

Usage
-----
    python -m scripts.export_thesis_figures \\
        [--typing-dir experiments/typing] \\
        [--skip-eval]   # only copy existing PNGs
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES_OUT = REPO_ROOT / "docs" / "rapport" / "figures"

COPY_MAP = {
    "roc_pr_curve.png": "roc_pr_curve.png",
    "confusion_matrix.png": "confusion_matrix.png",
    "scenario_cluster_heatmap.png": "scenario_cluster_heatmap.png",
}


def _run(cmd: list[str], label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"Warning: {label} exited with code {result.returncode}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis figures to docs/rapport/figures")
    parser.add_argument("--typing-dir", type=Path, default=Path("experiments/typing"))
    parser.add_argument("--encoder-dir", type=Path, default=Path("experiments/encoder"))
    parser.add_argument("--precursor-dir", type=Path, default=Path("experiments/precursor"))
    parser.add_argument("--features-root", type=Path, default=Path("data/features/v3"))
    parser.add_argument("--alerts-dir", type=Path, default=Path("experiments/alerts"))
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip re-running eval (copy only)")
    parser.add_argument("--n-bootstrap", type=int, default=200,
                        help="Bootstrap samples for alert CIs (0=skip)")
    args = parser.parse_args()

    FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    py = sys.executable

    if not args.skip_eval:
        manifest = args.typing_dir / "cluster_artifacts" / "cluster_manifest.json"
        if not manifest.exists():
            print(f"Missing {manifest}. Train typing first.", file=sys.stderr)
            sys.exit(1)

        _run(
            [
                py, "-m", "experiments.alerts.eval",
                "--typing-dir", str(args.typing_dir),
                "--encoder-dir", str(args.encoder_dir),
                "--precursor-dir", str(args.precursor_dir),
                "--features-root", str(args.features_root),
                "--output", str(args.alerts_dir),
                "--roc-sweep",
                "--n-bootstrap", str(args.n_bootstrap),
            ],
            "Alert evaluation + ROC/PR",
        )

        _run(
            [
                py, "-m", "experiments.typing.analyze_clusters",
                "--typing-dir", str(args.typing_dir),
                "--features-root", str(args.features_root),
                "--output", str(args.typing_dir),
                "--n-perm-shap", "0",
            ],
            "Cluster scenario heatmap",
        )

        _run(
            [
                py, "-m", "scripts.build_cluster_semantics",
                "--typing-dir", str(args.typing_dir),
                "--output", str(args.typing_dir),
            ],
            "Cluster semantics table",
        )

    copied = 0
    for src_name, dst_name in COPY_MAP.items():
        for root in (args.alerts_dir, args.typing_dir):
            src = root / src_name
            if src.exists():
                dst = FIGURES_OUT / dst_name
                shutil.copy2(src, dst)
                print(f"Copied {src} → {dst}")
                copied += 1
                break

    semantics_md = args.typing_dir / "cluster_semantics.md"
    if semantics_md.exists():
        dst_md = REPO_ROOT / "docs" / "cluster_semantics.md"
        shutil.copy2(semantics_md, dst_md)
        print(f"Copied {semantics_md} → {dst_md}")

    if copied == 0:
        print(
            "No figures copied. Run training pipeline first or remove --skip-eval.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nDone. {copied} figure(s) in {FIGURES_OUT}")


if __name__ == "__main__":
    main()
