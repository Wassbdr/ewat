# EWAT — État courant du projet

_Mis à jour : 2026-04-27_

## Dataset

| Phase | État | Détail |
|---|---|---|
| Phase 1 — record | 1 épisode collecté | `data/raw/run_20260416_112413/` |
| Phase 2 — build_features | Fait pour l'épisode existant | signal.npz, labels.parquet, adjacency.npz présents |
| Phase 3 — assemble | Non lancé | `data/processed/` vide |

Prochaine étape dataset : collecter plus d'épisodes (variété de scénarios chaos), puis assembler.

## Infrastructure code

| Module | État |
|---|---|
| `src/telemetry/` | Complet — collecteurs Prometheus + OTel + extracteurs fichier |
| `src/graph/` | Complet — builder, adjacency, serialization, validation |
| `src/ewat/drift/` | Dossier vide — **à implémenter** (Étape 0 : MMD-RFF) |
| `src/ewat/encoder/` | Dossier vide — **à implémenter** (Étape 1 : STGCN) |
| `src/ewat/typing/` | Dossier vide — **à implémenter** (Étape 2 : siamois) |
| `src/ewat/ontology/` | Dossier vide — **à implémenter** (Étape 2b : TE-KSG) |
| `src/ewat/precursor/` | Dossier vide — **à implémenter** (Étape 3) |
| `src/ewat/alerts/` | Dossier vide — **à implémenter** (sortie) |

## Scripts pipeline

| Script | État |
|---|---|
| `record_episode.py` | Opérationnel (graceful shutdown, checkpointing) |
| `build_features.py` | Opérationnel |
| `assemble_dataset.py` | Opérationnel |
| `validate_dataset.py` | Opérationnel |
| `chaos_injector.py` | Opérationnel |

## Cluster

- 1 épisode enregistré (run_20260416_112413)
- OTel Gateway déployé dans `ewat`, internalTrafficPolicy:Cluster appliqué
- Embeddings SentenceBERT : dossier présent, vérifier si modèle téléchargé

## Prochaine priorité

**Implémenter Étape 0 (MMD-RFF)** dans `src/ewat/drift/` :
- `mmd.py` — calcul MMD² via Random Fourier Features
- `detector.py` — fenêtre glissante, flag DRIFT, recalibration
- `calibration.py` — calibration de ε_drift par injection de drifts bénins
