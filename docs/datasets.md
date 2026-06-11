# EWAT — Nomenclature officielle des datasets

_Page de référence unique (E12, audit 2026-06). Avant elle, la réponse à
« quelle version fait foi ? » était dispersée entre STATUS.md et les results.md._

| Version | Chemin | T (steps) | N | Scénarios | Split | Statut | Fait foi pour |
|---|---|---|---|---|---|---|---|
| v1 / v1p / v2 | `data/features/v1*`, `v2` | ~21 | 6 | 15 | — | Historique | Rien (étapes intermédiaires de build/imputation) |
| **v3** | `data/datasets/ewat_v3` | ~21 | 6 | 15 | stratifié 209/45/45 | Figé | Résultats historiques (H1/H3 v3, ontologie Phase 8, synthèse) |
| v3_synthetic | `data/features/v3_synthetic` | ~50 | 6 | composites | — | Figé | Co-occurrences/causalité ontologie (épisodes synthétiques) |
| v4 | `data/datasets/ewat_v4` | 47–51 | 6 | 15 | temporel 262/56/57 | ❌ **Cassé** | Rien — 4 scénarios absents du train (AUROC=0.500 trivial). Conservé comme pièce à conviction D4 |
| **v4_strat** | `data/datasets/ewat_v4_strat` | 47–51 | 6 | 15 | stratifié 270/60/45 | **Courant** | **Tous les chiffres de soutenance** : B2=0.920, multiseed Phase H/J/K, correctifs audit 2026-06 |
| ewat_rcaeval | `data/datasets/ewat_rcaeval` | 48 | 6 | 30 fault types | — | Figé | Transfert zero-shot/few-shot (négatif documenté) |
| **v5** | (collecte en cours) | ~120 | 41 | 27 (19 + held-out A/B) | stratifié + held-out test-only | 🔄 Collecte VM | Travaux post-collecte — protocole figé dans [evaluation_protocol_v5.md](evaluation_protocol_v5.md) |

## Règles

1. **Reproduire la soutenance** = `ewat_v4_strat` exclusivement (les chiffres
   v3 sont historiques, conservés pour la traçabilité).
2. `ewat_v4` (temporel) ne doit plus être utilisé — c'est l'exemple qui a
   motivé le défaut stratifié de `assemble_dataset` (D4, audit 2026-06).
3. Depuis l'audit 2026-06, `assemble_dataset` est **stratifié par défaut**,
   route les épisodes `held_out_flag=True` en test-only (D5) et refuse un
   scénario absent du test (override `--allow-missing-test-scenarios`).
4. Le schéma de features est versionné dans `telemetry.feature_names`
   (v4 = 17, v5.1 = 18) ; `metadata.signal_feature_names` fait foi par épisode.
5. Les splits shuffled (`--split-mode shuffled`) servent UNIQUEMENT au
   protocole E5 (variance inter-split) — jamais pour entraîner un résultat
   headline.

## Variance inter-split (E5, audit 2026-06)

Mesurée sur v4_strat : B2 = 0.969 ± 0.011 sur 5 splits shuffled vs **0.920
sur le split temporel officiel** — le split temporel est systématiquement
plus dur (~−5 pp AUROC, PR-AUC 0.59 vs 0.80–0.91) : le décalage temporel de
l'infra pénalise surtout la précision. Le headline est donc *conservateur*.
Détails : `experiments/audit2026/split_variance/results.md`.
