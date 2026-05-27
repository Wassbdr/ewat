# MLflow experiment registry

_Step 10 fix 10.6 (audit 2026-05-26): central registry of EWAT MLflow experiments + Model Registry stages._

## Tracked experiments

| Experiment name | Script | Logged params | Logged metrics | Logged artefacts |
|---|---|---|---|---|
| `encoder_train` | `experiments/encoder/train.py` | `lr`, `batch_size`, `epochs`, `d_hidden`, `d_embed`, `use_layer_norm`, `seed` | `val_loss`, `train_loss` per epoch | `best_encoder.pt`, `scaler.pkl` |
| `typing_train` | `experiments/typing/train.py` | `d_proj`, `margin`, `mining`, `clustering_linkage`, `clustering_metric`, `k_selection_method`, `seed` | `val_loss`, `sil_train/val/test`, `k_optimal` | `best_siamese.pt`, `cluster_manifest.json`, `centroids.npy` |
| `precursor_train` | `experiments/precursor/train.py` | `k_values`, `classifier_type`, `reg_c`, `ci_method`, `seed` | `auroc_val_kX`, `auroc_test_kX`, `h3_pass` | `classifier_type{C}_k{K}.pkl`, `results.json` |
| `ontology_build` | `experiments/ontology/build_service.py` | `te_method`, `k_knn`, `n_permutations`, `min_series_length`, `correction`, `regime`, `seed` | n_causal_relations, n_above_pmax | OWL/RDF files, results.md |
| `architecture_v2_chaos_mesh` | `experiments/architecture_v2/train_chaos_mesh.py` | `epochs`, `batch_size`, `lr`, `instance_norm`, `seed` | `val_macro_auroc`, `test_macro_auroc` | `best_model.pt`, `scaler.pkl` |
| `ewat_improvements` (legacy) | various sweeps | many | many | swept JSONs |

## Model Registry stages

Step 10 fix 10.6 — recommended workflow for production-grade versioning :

1. **`None`** : freshly logged run, not yet promoted.
2. **`Staging`** : passed full unit test suite + audit data quality checks.
   - Apply via `mlflow.register_model(...)` then `client.transition_model_version_stage(name, version, "Staging")`.
3. **`Production`** : passed stress tests A1-A5 + open-set evaluation.
   - Apply only after explicit human review.
4. **`Archived`** : superseded by a newer Production version.

To promote :

```bash
mlflow models transition --name ewat-precursor-v4-strat \
    --version 3 --stage Production --archive-existing-versions
```

## Storage layout

```
mlruns/
├── 0/                            # default experiment id (legacy)
├── 1/                            # ewat_improvements
│   ├── <run_id>/
│   │   ├── meta.yaml
│   │   ├── params/
│   │   ├── metrics/
│   │   ├── tags/
│   │   └── artifacts/
└── models/                       # Step 10 fix 10.6 — Model Registry
    └── ewat-precursor-v4-strat/
        ├── version-1/   # archived
        ├── version-2/   # staging
        └── version-3/   # production
```

## Reproducibility checklist

Before promoting a model to Production :

- [ ] `pytest tests/unit/` ≥ 668 tests pass
- [ ] `python -m experiments.bench.latency_e2e` p95 < 5 s
- [ ] `python -m experiments.bench.power_analysis` ≥ 5/10 clusters reportable
- [ ] `python -m experiments.h3_robustness.distant_window` Δ(far−near) documented
- [ ] Dataset audit at `experiments/data_quality/<dataset>/audit.md` reviewed
- [ ] Git commit SHA logged in MLflow tags
- [ ] Docker image built and tagged (`docker build -t ewat:<sha> .`)

## Querying the registry

```python
from mlflow.tracking import MlflowClient
client = MlflowClient(tracking_uri="file:./mlruns")
for mv in client.search_model_versions("name='ewat-precursor-v4-strat'"):
    print(f"v{mv.version}  stage={mv.current_stage}  run={mv.run_id}")
```
