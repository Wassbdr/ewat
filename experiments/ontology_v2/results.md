# EWAT — Validation de l'ontologie OWL/RDF

_Score : **8/10 critères atteints**_

## 1. Critères chiffrés

| # | Critère | Valeur | Cible | Statut |
|---|---------|--------|-------|--------|
| 1 | Coverage classes littérature → instances | 100% (15/15) | ≥ 80% | ✓ |
| 2 | Coverage clusters → classes ontologiques | 100% | ≥ 100% | ✓ |
| 3 | Relations causales inférées (composites) | 3 | ≥ 15 | ✗ |
| 4 | Relations co-occurrence inférées (composites) | 19 | ≥ 10 | ✓ |
| 5 | HermiT classification time | 0.61s | < 30s | ✓ |
| 6 | OWL consistency check | OK | OK | ✓ |
| 7 | Inférences matérialisées (class triples) | 0 | ≥ 30 | ✗ |
| 8 | Réalisme synthèse (AUC réel/synth) | 0.529 | < 0.75 | ✓ |
| 9 | Service propagation edges (post-filter) | 46 | ≥ 30 | ✓ |
| 10 | Queries SPARQL canoniques (parsing OK) | 5/5 | 5/5 | ✓ |

## 2. Comparaison avec l'ontologie originale

| Aspect | Original (`experiments/ontology/results.md`) | Nouvelle ontologie OWL |
|---|---|---|
| Relations causales | 0 | **3** |
| Relations de co-occurrence | 0 | **19** |
| Relations temporelles (cross-cluster) | 12 (support ≤ 4) | 12 (precedes) |
| Relations de propagation (services) | n/a | **46** (post-filtre spécificité) |
| Taxonomie formelle | Non | **29 classes ancrées littérature** |
| Raisonneur | Non | **HermiT** (consistant) |

## 3. Détail des entailments

| Métrique | Valeur |
|----------|--------|
| Class triples (rdf:type Anomaly + subclasses) | 6 |
| Causal pairs (ewat:causes) | 0 |
| Propagation triples (ewat:propagatesThrough) | 30 |
| Composite_Anomaly instances | anomaly_cluster_8 |

## 4. Queries SPARQL canoniques

| Query | Résultats |
|-------|-----------|
| `all_composites` | 1 |
| `downstream_of_memory_saturation` | 0 |
| `services_affected_by_cascading` | 0 |
| `signatures_sharing_heavy_features` | 21 |
| `fast_precursors_of_composite` | 0 |

## 5. Relations causales découvertes

| Source | Target | TE | p_adj |
|--------|--------|-----|-------|
| C4 | C1 | 0.1815 | 0.0149 |
| C6 | C5 | 0.0669 | 0.0149 |
| C4 | C8 | 0.1413 | 0.0299 |

## 6. Limitations connues

- **HermiT individual classification** : owlready2 ne matérialise pas les entailments d'instances dans `.is_a` pour les axiomes d'équivalence basés sur cardinalité. Les inférences sont accessibles via SPARQL.
- **TE multivariée** : T composite ~50 steps suffit pour KSG sur d=17 filtrées (réduction dynamique des features dégénérées).
- **Synthèse vs collecte réelle** : le corpus synthétique passe le test discriminatif (AUC=0.529). Validation finale recommandée sur épisodes multi-scénario réels (ewat_v4 multi).

## 7. Reproduction

```bash
# Phase 4 — synthèse
python -m scripts.synthesize_composite_episodes \
    --features-root data/features/v3 \
    --output data/features/v3_synthetic \
    --n-per-pair 5

# Phases 1-5 — build complet
python -m experiments.ontology_v2.build_owl \
    --synthetic-root data/features/v3_synthetic \
    --n-permutations 200

# Phase 6 — validation
python -m experiments.ontology_v2.validate_ontology
```