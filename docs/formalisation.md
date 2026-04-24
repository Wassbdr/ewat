# EWAT — Formalisation mathématique

## Graphe de services

G(t) = (V, E(t), w_E(t))

V = Services et Deployments Kubernetes (pas les Pods). |V| = N constant.

Arêtes pondérées : w_E(t) : E(t) → ℝ³, e_ij(t) ↦ (volume_ij, latence_med_ij, taux_erreur_ij). Seuil de présence : volume > 0 sur la fenêtre glissante.

Agrégation intra-service (par composante) :
- Saturation (CPU, RAM, net_sat, disk_io) → max
- Taux (error_rate, warn_rate) → somme pondérée par volume
- Latence (P99, span_dur) → percentile 99 sur l'union des distributions (pas percentile de percentiles)
- Structurel (trace_depth, fan_out, lexical_entropy) → médiane

## Signal de télémétrie

S(t) ∈ ℝ^{N×17} = [M(t) | T(t) | L(t)]

M(t) ∈ ℝ^{N×7} — Métriques (sources : Prometheus existant + OTel Metrics) :
1. CPU utilisation
2. RAM utilisation
3. Latence P99
4. Taux d'erreur HTTP (4xx + 5xx)
5. Saturation réseau
6. Disk I/O (IOPS + throughput)
7. Longueur de file d'attente (pending requests, queue depth)

T(t) ∈ ℝ^{N×6} — Traces (source : OTel Collector, spans OTLP) :
1. Durée médiane des spans
2. Taux de spans anormaux
3. Profondeur de trace
4. Fan-out
5. Taux de retry (spans retentés / total)
6. Variance de latence (coefficient de variation)

L(t) ∈ ℝ^{N×4} — Logs (source : OTel Collector, logs OTLP) :
1. Taux d'erreurs (ERROR / total)
2. Taux de warnings
3. Anomalie sémantique : e(ℓ) = SentenceBERT(ℓ) ∈ ℝ^384, score = distance moyenne au centroïde normal μ_v
4. Entropie lexicale

## Régimes opérationnels

θ(t) ∈ {θ_normal, θ_drift, θ_anomaly, θ_{drift∩anomaly}}

Quatre régimes, pas trois. θ_{drift∩anomaly} modélise les déploiements défectueux (simultanément drift et anomalie). Traité par le mécanisme de look-through (étape 0).

S(t) ∼ D_{θ(t)}(G(t)) — le signal n'est pas une somme additive.

## Pipeline EWAT

**Étape 0 — Détection de drift (MMD-RFF, O(nD))**
MMD²(W_ref, W_cur) via Random Fourier Features, φ : ℝ^d → ℝ^D.
Filtrage avec look-through :
- MMD² < ε_drift → signal transmis tel quel
- MMD² ≥ ε_drift + test post-drift positif → signal transmis avec flag DRIFT
- MMD² ≥ ε_drift + test post-drift négatif → RECALIBRATE (W_ref ← W_cur)
ε_drift calibré par injection de drifts bénins via Chaos Mesh.

**Étape 1 — Encodeur STGCN**
z_e = Enc_θ(S̃_{[t-W, t+δ]}, G(t)) ∈ ℝ^{d_e}
Convolution avec matrice d'adjacence pondérée par w_E(t).

**Étape 2 — Typage contrastif**
Réseau siamois : d_φ(z_i, z_j) → 0 si même type Chaos Mesh, → 1 sinon.
Clustering hiérarchique agglomératif → C = {C_1, ..., C_K}.
Interprétabilité : SHAP → fiche par type.

**Étape 2b — Ontologie**
O = (C, R), trois types de relations :
- Temporelles : C_i →^{Δt,σ} C_j
- Causales : Transfer Entropy (estimateur KSG, Kraskov et al. 2004), n_min = 30, seuil par permutation. Pas de Granger.
- Co-occurrence : χ²

**Étape 3 — Précurseurs typés**
p̂_i(t) = f_i(S̃_{[t-k,t]}, G(t)) ∈ [0,1], k ∈ {2, 5, 10, 20, 30, 60} min
k*_i = argmax_k AUROC(f_i, k)

**Sortie** : Alert(t) = (C_i, p̂_i(t), k*_i, fiche_{C_i})

## Budget de latence
Étape 0 < 1s, Étape 1 < 2s, Étape 3 < 1s. Total < 5s. Étapes 2/2b offline.

## Hypothèses et falsification

**H1 — Structurabilité**
Silhouette < 0.3 en held-out sur 5 graines × 5 splits → falsifié.
Compléments : gap statistic, BIC/GMM.
Seuil justifié par Kaufman & Rousseeuw (1990).

**H2 — Séparabilité du drift**
Pas de réduction significative du FPR à rappel constant (p > 0.05, Student) → falsifié.

**H3 — Prédictibilité**
AUROC par type < baseline générique ∀k → falsifié.

**Ablation**
Par modalité (M, T, L, paires, triplet). Par feature (leave-one-out, Wilcoxon signé). Redondance : |ρ| > 0.9.

## Références
- Fu et al. (2025) — Survey RCA microservices, gap benchmark/production
- Myrtollari et al. (2025) — Concept drift-aware anomaly detection for K8s
- Hinder et al. (2024) — Concept drift in unsupervised data streams
- Kaufman & Rousseeuw (1990) — Justification seuil silhouette
- Kraskov et al. (2004) — Estimateur KSG pour Transfer Entropy
- Tibshirani et al. (2001) — Gap statistic
- Gregg (2013) — Méthodologie USE, queue depth comme leading indicator
