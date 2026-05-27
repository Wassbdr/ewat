"""CLI — generate composite synthetic episodes for the ontology pipeline.

Loads ewat_v3 episodes via :func:`ewat.ontology.synthesis.load_episode`,
samples cluster pairs from ``experiments/typing/cluster_artifacts/cluster_manifest.json``
prioritising the 22 temporal-significant pairs reported in
``experiments/ontology/results.md``, and writes the validated synthetic
episodes in canonical layout to the requested output directory.

Usage
-----
.. code-block:: bash

    python -m scripts.synthesize_composite_episodes \\
        --features-root data/features/v3 \\
        --manifest experiments/typing/cluster_artifacts/cluster_manifest.json \\
        --output data/features/v3_synthetic \\
        --n-per-pair 5

The script writes a ``synthesis_report.json`` summarising counts and the
corpus-level discriminator AUC.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import yaml

from ewat.ontology.synthesis import (
    EpisodeBundle,
    audit_realism_corpus,
    cascade_episodes,
    load_episode,
    overlay_episodes,
    realism_envelope,
    write_episode,
)


log = logging.getLogger("synth")


def _episodes_by_cluster(
    manifest_path: Path,
    features_root: Path,
) -> dict[int, list[Path]]:
    manifest = json.loads(manifest_path.read_text())
    out: dict[int, list[Path]] = defaultdict(list)
    for ep_id, info in manifest.items():
        ep_dir = features_root / ep_id
        if ep_dir.exists():
            out[int(info["cluster"])].append(ep_dir)
    return out


def _priority_pairs(default: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Default to the 12 temporal cross-cluster transitions reported in
    ``experiments/ontology/results.md`` (support >= 3)."""
    return default or [
        (0, 3), (4, 1), (1, 4), (3, 0),
        (7, 0), (7, 8), (0, 7), (8, 5),
        (4, 8), (5, 7), (2, 6), (1, 2),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("experiments/typing/cluster_artifacts/cluster_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ontology-config", type=Path,
        default=Path("configs/ontology.yaml"),
    )
    parser.add_argument("--n-per-pair", type=int, default=5)
    parser.add_argument("--envelope-corpus-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.ontology_config.read_text())["synthesis"]
    alphas = cfg["overlay"]["alphas"]
    gap_options = cfg["cascade"]["gap_steps_options"]

    rng = random.Random(args.seed)
    episodes_by_cluster = _episodes_by_cluster(args.manifest, args.features_root)
    log.info(
        "Loaded manifest: %d clusters, %d total episodes",
        len(episodes_by_cluster),
        sum(len(v) for v in episodes_by_cluster.values()),
    )

    # Envelope for clipping garde-fou: sample a corpus of real episodes.
    all_paths = [p for paths in episodes_by_cluster.values() for p in paths]
    rng.shuffle(all_paths)
    envelope_paths = all_paths[: args.envelope_corpus_size]
    envelope_corpus = [load_episode(p) for p in envelope_paths]
    p01, p99 = realism_envelope(envelope_corpus)
    log.info("Realism envelope computed on %d episodes", len(envelope_corpus))

    args.output.mkdir(parents=True, exist_ok=True)
    pairs = _priority_pairs(default=[])

    written: list[EpisodeBundle] = []
    rejected: list[dict] = []
    for cid_a, cid_b in pairs:
        if cid_a not in episodes_by_cluster or cid_b not in episodes_by_cluster:
            continue
        for i in range(args.n_per_pair):
            ep_a_path = rng.choice(episodes_by_cluster[cid_a])
            ep_b_path = rng.choice(episodes_by_cluster[cid_b])
            ep_a = load_episode(ep_a_path)
            ep_b = load_episode(ep_b_path)

            for alpha in alphas:
                bundle, check = overlay_episodes(
                    ep_a, ep_b, alpha=alpha,
                    p01_table=p01, p99_table=p99,
                )
                if check.passed:
                    bundle.episode_id = f"{bundle.episode_id}_p{cid_a}-{cid_b}_i{i}"
                    write_episode(bundle, args.output)
                    written.append(bundle)
                else:
                    rejected.append({
                        "kind": "overlay", "alpha": alpha,
                        "pair": (cid_a, cid_b), "reasons": check.reasons,
                    })

            for gap in gap_options:
                bundle, check = cascade_episodes(
                    ep_a, ep_b, gap_steps=gap,
                    p01_table=p01, p99_table=p99,
                )
                if check.passed:
                    bundle.episode_id = f"{bundle.episode_id}_p{cid_a}-{cid_b}_i{i}"
                    write_episode(bundle, args.output)
                    written.append(bundle)
                else:
                    rejected.append({
                        "kind": "cascade", "gap": gap,
                        "pair": (cid_a, cid_b), "reasons": check.reasons,
                    })

    log.info("Wrote %d synthetic episodes, rejected %d",
             len(written), len(rejected))

    # Corpus-level audit
    audit_auc = None
    if written:
        try:
            audit_auc = audit_realism_corpus(envelope_corpus, written)
            log.info("Discriminator AUC: %.3f (target < 0.75)", audit_auc)
        except Exception as e:
            log.warning("audit failed: %s", e)

    report = {
        "n_written": len(written),
        "n_rejected": len(rejected),
        "discriminator_auc": audit_auc,
        "alphas": alphas,
        "gap_options": gap_options,
        "pairs": pairs,
        "n_per_pair": args.n_per_pair,
        "seed": args.seed,
        "rejected_samples": rejected[:10],
    }
    (args.output / "synthesis_report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    log.info("Report written to %s/synthesis_report.json", args.output)


if __name__ == "__main__":
    main()
