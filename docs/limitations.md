# EWAT — Limites, causes, conclusions et améliorations possibles

_Document établi le 2026-05-11 — mis à jour 2026-05-21 (Phase 8 ontologie OWL)_

---

## 1. Vue d'ensemble

Ce document recense **l'ensemble des limites connues** du projet EWAT à ce jour, organisées par couche. Pour chaque limite, on explicite : (a) la nature du problème, (b) pourquoi il existe, (c) ce qu'on peut en conclure, et (d) les améliorations possibles ou déjà appliquées. Les corrections issues de l'audit (P0–P2) sont indiquées comme telles.

**Lecture rapide** — les limites les plus structurantes :

| # | Limite | Impact | Statut |
|---|--------|--------|--------|
| L1 | Épisodes trop courts (~21 steps) | H2 FAIL, FA élevée, H2b trivial | Ouvert (nécessite ewat_v4) |
| L2 | Un seul split, pas de cross-validation | Variance sous-estimée | Ouvert |
| L3 | 5 graines seulement | CIs fragiles | Étendu à 10 (P1) |
| L4 | Baselines B1/B2 > EWAT en AUROC | Valeur ajoutée STGCN = structuration, pas discrimination | Résultat assumé |
| L5 | Ablation sans réentraînement | Conclusions = robustesse géométrique, pas importance | Corrigé (P2) |
| L6 | TE-KSG = somme univariée | Sous-estime la synergie entre features | Corrigé + activée multivariate sur cascades (Phase 8) |
| L7 | 0 relations causales, 0 co-occurrence | Ontologie réduite aux transitions temporelles | ✅ Résolu Phase 8 — 3 causales + 19 co-occurrences (cf. L3.3) |
| L8 | disk_io 16.7% NaN | 1 service sur 6 incomplet | Ouvert (ewat_v4) |

---

## 2. Limites des données

### L2.1 — Épisodes trop courts (~21 steps, ~10.5 min)

**Problème.** Les épisodes collectés dans ewat_v3 font en moyenne 21 steps (30 s/step = 10.5 min). C'est insuffisant pour le mécanisme de look-through qui nécessite un warm-up du DriftDetector (fenêtre ref = 5 steps + post-drift = 3–6 steps), soit ~8 steps avant d'être opérationnel. Sur 21 steps, il reste ~13 steps utiles.

**Pourquoi.** Les scénarios Chaos Mesh étaient configurés avec une durée fixe de ~10 min pour maximiser le nombre d'épisodes collectables dans le temps de stage. Le trade-off débit vs. durée a privilégié la diversité (15 scénarios × 20 rép.) au détriment de la profondeur temporelle.

**Conclusions.**
- H2 (look-through) échoue (p=0.27) non par défaut de conception mais par manque de signal temporel.
- H2b passe trivialement car le DD déclenche sur presque tous les épisodes (overlap > 50% partout).
- Les FA alertes (100% aux seuils 0.3–0.5) sont causées par le fait que les classifieurs tirent avant que le DD n'ait le temps de discriminer drift vs. anomalie.
- Le lead time de 3 min (point opérationnel seuil 0.7) est le maximum observable sur des épisodes de 10 min.

**Amélioration.** ewat_v4 avec épisodes ≥ 40 steps (~20 min), ce qui donnerait ~30 steps utiles post warm-up et permettrait de réhabiliter H2.

---

### L2.2 — disk_io 16.7% NaN (product-catalog)

**Problème.** Le nœud `observit-cluster1-workers-58w74-mwxb2` est en état NotReady. Le service `product-catalog` y est schedulé, et Prometheus ne collecte pas ses métriques disk_io.

**Pourquoi.** Infrastructure hors de contrôle du stagiaire (nécessite intervention admin cluster). Le nœud est resté NotReady pendant toute la collecte ewat_v3.

**Conclusions.**
- Le NaN est structurel et localisé : 1 service / 6, 1 feature / 17.
- L'ablation montre que disk_io est malgré tout significatif (Δ=−0.010, p=0.026), donc son importance réelle est probablement **sous-estimée**.
- Un relecteur pourrait retirer product-catalog du test set et faire tomber cette significativité.

**Amélioration.** ewat_v4 avec OTel SDK déployé (cluster-admin requis) → 0% NaN disk_io.

---

### L2.3 — Reconstruction histogramme stochastique

**Problème.** La reconstruction des features métriques depuis les histogrammes Prometheus (`aggregation.py`) utilisait `np.random.uniform` non seedé. Deux exécutions de `build_features` produisaient des features M(t) légèrement différentes pour les mêmes dumps bruts.

**Pourquoi.** Oubli de propagation de graine dans une fonction utilitaire de bas niveau.

**Conclusions.** La reproductibilité bit-exact des features n'était pas garantie avant correction. L'impact pratique est faible (variation infra-bruit) mais c'est un défaut de rigueur.

**Amélioration.** ✅ Corrigé (P1) — graine propagée depuis `configs/collection.yaml`.

---

### L2.4 — Fenêtres M/T/L désalignées temporellement

**Problème.** Les features métriques (M) sont collectées en nearest-sample dans une plage temporelle, les traces (T) et logs (L) en sliding windows fermées en `t`. Il n'y a pas de garantie que les 17 features d'un même pas de temps correspondent au même instant physique à ±15 s près.

**Pourquoi.** Les trois sources (Prometheus, Jaeger/OTel, Loki) ont des fréquences d'acquisition, des latences d'ingestion et des modes de requête différents. L'alignement parfait nécessiterait un collecteur synchrone (non disponible dans la stack observabilité existante).

**Conclusions.** Le signal S(t) est un vecteur de « features au pas t » plutôt qu'un snapshot instantané. Pour un grid_step de 30 s, un décalage de ±15 s est tolérable mais non mesurable. Cela peut contribuer au bruit de fond et à la difficulté de certaines features (ex. `latency_cv` qui dépend d'un échantillonnage précis).

**Amélioration.** ewat_v4 avec OTel SDK unifié pourrait réduire le décalage en utilisant un seul pipeline de collecte pour les trois modalités.

---

### L2.5 — Split unique, pas de cross-validation

**Problème.** Tous les résultats reposent sur un unique split stratifié 209/45/45 (train/val/test). Avec n=45 en test et des clusters C6/C9 contenant 1–2 épisodes, les AUROC test sont instables.

**Pourquoi.** Le pipeline 3-phases (record → build → assemble) est coûteux en temps. La cross-validation aurait nécessité de réentraîner le pipeline complet (encodeur + typage + précurseurs) K fois, soit ~K × 2h CPU. Le choix d'un split unique était pragmatique.

**Conclusions.**
- Les intervalles de confiance reportés (via bootstrap 5 graines) sous-estiment la variance liée au split.
- C6 et C9 ont des AUROC NaN par insuffisance d'échantillons test — c'est un artefact du split, pas un échec du modèle.
- La variabilité inter-graines (sil_test de 0.414 à 0.662) est partiellement due au split fixe.

**Amélioration.** Cross-validation stratifiée 5-fold avec réentraînement complet, ou au minimum, 2–3 splits aléatoires différents pour mesurer la variance liée au split.

---

### L2.6 — Quality gate Phase 1 trop laxe

**Problème.** Le seuil `trace_nan_max=0.9` dans `validate_dataset` autorise jusqu'à 90% de NaN traces pour un épisode. Un épisode avec 85% de traces manquantes passe la validation.

**Pourquoi.** Seuil défini pour ne pas rejeter trop d'épisodes lors de la collecte initiale, où les traces étaient incomplètes avant le déploiement OTel.

**Conclusions.** La qualité des traces est hétérogène dans le dataset. Les features traces (T) sont à 0% NaN dans le dataset final (reconstructées depuis les spans disponibles), mais la complétude sous-jacente n'est pas uniforme.

**Amélioration.** Abaisser le seuil à 0.5 pour ewat_v4, ou ajouter un score de qualité par épisode utilisable comme poids dans l'entraînement.

---

### L2.7 — Listes de scénarios incohérentes entre configs

**Problème.** `configs/collection.yaml` (15 scénarios) et `configs/default.yaml` (autre liste) ont des listes de scénarios potentiellement différentes.

**Pourquoi.** Évolution incrémentale des configs sans synchronisation.

**Conclusions.** Source d'erreur silencieuse lors d'une future collecte ou d'une expérience utilisant le mauvais config.

**Amélioration.** Centraliser la liste des scénarios dans un unique fichier de référence et y pointer depuis les deux configs. ✅ Partiellement adressé dans l'audit.

---

## 3. Limites statistiques

### L3.1 — TE-KSG = somme des TE univariés

**Problème.** L'implémentation de la Transfer Entropy dans `causal.py` calcule la TE pour chaque feature individuellement puis les somme. Ce n'est **pas** la TE multivariée dans ℝ¹⁷. La somme ignore la synergie et la redondance entre features, et surestime l'information causale transmise.

**Pourquoi.** L'estimateur KSG en haute dimension (d=17) nécessite un très grand nombre d'échantillons pour converger. Avec ~21 steps par épisode et 299 épisodes, le n effectif (après moyenne) était trop faible pour une estimation fiable en d=17.

**Conclusions.**
- Le résultat « 0 relations causales » est cohérent avec la limitation : la somme univariée surestime TE → si même cette surestimation ne détecte rien, la TE réelle est encore plus faible.
- L'absence de causalité TE-KSG n'est pas un résultat négatif exploitable tel quel — c'est partiellement un artefact méthodologique.

**Amélioration.** ✅ Documenté comme limitation dans la formalisation. Pour ewat_v4, épisodes plus longs + TE multivariée KSG (k-NN dans ℝ¹⁷) ou TE conditionnelle partielle.

---

### L3.2 — TE calculée sur la moyenne d'épisodes (biais écologique)

**Problème.** Les séries temporelles de chaque épisode sont empilées puis moyennées (`stack.mean(axis=0)`) avant le calcul de TE. Cela collapse la variance inter-épisodes et introduit un biais écologique (la relation au niveau moyen ne reflète pas les relations individuelles).

**Pourquoi.** La TE nécessite des séries longues pour converger. Concaténer au lieu de moyenner créerait des discontinuités artificielles. Moyenner était le compromis le plus simple.

**Conclusions.** La TE estimée reflète le comportement « moyen » d'un cluster, pas le comportement individuel des épisodes. Cela amplifie les patterns partagés et masque les variations.

**Amélioration.** ✅ Corrigé (P0) — p-value corrigée `(1+count)/(1+M)`, BH-FDR. Limitation documentée.

---

### L3.3 — Ontologie vide (0 causales, 0 co-occurrence) — ✅ Résolu (Phase 8)

**Problème.** L'ontologie initiale ne contenait que 22 relations temporelles (dont 10 self-loops triviaux) et zéro relation causale ou de co-occurrence.

**Pourquoi (3 causes racines).**
- **Design mono-scénario** : chaque épisode injecte un seul scénario → ni co-occurrence ni causalité inter-types observables par construction.
- **TE-KSG multivariate non activée** : la version `multivariate` existait dans `causal.py:145-163` mais le pipeline appelait silencieusement `univariate_sum` (somme des TE marginales, biais d'ignorer la synergie).
- **T = 21 steps < 5·d = 85** pour KSG en d = 17 (règle empirique).

**Amélioration ✅ Résolu (Phase 8, 2026-05-20/21).** Refonte complète en ontologie OWL/RDF formelle (cf. §5.2 de [`results.md`](results.md) et §Phase 8 de [`evolution.md`](evolution.md)) :

- **Synthèse composite** (`src/ewat/ontology/synthesis.py`) : overlay (co-occurrence) et cascade (causalité) à partir des épisodes mono-scénario, avec garde-fous (Spearman médian ≥ 0.85, AUC discriminateur < 0.75). **282 épisodes synthétiques** générés, AUC = 0.529 (indistinguable du réel).
- **TE multivariate KSG-1 activée** sur les cascades (T ≈ 50 résout le blocage d = 17). **3 relations causales** significatives (BH-FDR p < 0.05) : C4→C1, C6→C5, C4→C8.
- **19 co-occurrences** par construction sur overlays.
- **46 edges de propagation services** après filtre de spécificité (drop de 13 paires ubiquitaires comme `load-generator → frontend`).
- **Taxonomie OWL** : 29 classes ancrées littérature (Soldani 2022, Fu 2025, Gregg 2013, Aniello 2014), raisonneur HermiT cohérent en 0.61 s.
- **Score validation chiffrée** : 8/10 critères atteints (`experiments/ontology_v2/results.md`).

**Limites résiduelles** (acceptées) :
- Seulement 3 causales (cible ≥ 15 du plan) — corpus synthétique petit (n_per_pair = 5). Scaling à n_per_pair ≥ 15 attendu pour passer le critère.
- La synthèse reste synthétique : validation finale recommandée sur ewat_v4 avec collecte multi-scénario réelle (injection simultanée sur services différents).

---

### L3.4 — Bootstrap : reproductibilité et méthode

**Problème.** Le bootstrap dans `utils/bootstrap.py` utilisait `rng=None` par défaut (graines aléatoires), et n'implémentait que le percentile simple (pas BCa). Pour des AUROC proches de 1.0 (ex. C4=1.000), l'IC percentile est dégénéré.

**Pourquoi.** Implémentation initiale simplifiée pour un premier prototypage rapide.

**Conclusions.** Les IC rapportés avant correction n'étaient pas reproductibles. Le percentile simple sous-estime la largeur de l'IC pour des statistiques proches des bornes (0 ou 1).

**Amélioration.** ✅ Corrigé (P0) — `rng` explicite obligatoire, BCa ajouté pour AUROC. Tests unitaires.

---

### L3.5 — Multiplicité non corrigée dans l'ablation

**Problème.** Les tests de Wilcoxon dans l'ablation (7 modalités + 17 leave-one-out = 24 tests) étaient réalisés sans correction de multiplicité. Un p < 0.05 sur 24 tests donne un taux de faux positifs attendu de 1.2.

**Pourquoi.** Oubli standard — la correction était appliquée à l'ontologie mais pas à l'ablation.

**Conclusions.** Certaines features déclarées « significatives » à p < 0.05 pourraient être des faux positifs (ex. `disk_io` à p=0.026 est marginal après Holm).

**Amélioration.** ✅ Corrigé (P0) — Holm/FDR appliqué.

---

### L3.6 — H2 : paired t-test sur indicatrices binaires

**Problème.** Le test de H2 utilise un t-test apparié de Student sur des variables indicatrices (0/1). Le t-test suppose une distribution continue et approximativement normale, ce qui n'est pas le cas.

**Pourquoi.** Choix par défaut sans questionnement de l'adéquation.

**Conclusions.** La p-value p=0.27 est approximativement correcte (n=45 est assez grand pour le TCL) mais pas rigoureuse. McNemar ou un test exact de permutation seraient plus adaptés.

**Amélioration.** Remplacer par McNemar pour les comparaisons paired-binary. Le résultat (FAIL) ne changerait probablement pas.

---

## 4. Limites de modélisation

### L4.1 — STGCN = GCN statique malgré le nom

**Problème.** Le code du STGCN moyenne la matrice d'adjacence sur la dimension temporelle (`adj_mean = adjacency.mean(dim=1)`) puis la re-broadcast. Le graphe est donc statique, contrairement à la formalisation qui annonce un G(t) dynamique.

**Pourquoi.** Simplification d'implémentation. Le STGCN dynamique réel (A(t) par timestep) est plus coûteux en mémoire et en calcul.

**Conclusions.**
- La divergence code/formalisation est un risque de crédibilité face à un relecteur.
- L'impact sur les performances est inconnu sans ablation A(t) vs. A_mean.

**Amélioration.** ✅ Corrigé (P1) — STGCN dynamique implémenté, `adj_mean` conservé comme variant ablation.

---

### L4.2 — Padding non masqué dans le pooling

**Problème.** Le pooling global `z = h.mean(dim=(1,2))` incluait les zéros du padding pour les épisodes plus courts que le maximum du batch. Les embeddings étaient biaisés vers 0 proportionnellement au ratio de padding.

**Pourquoi.** `collate_episodes` zéro-pade les séquences, mais le mask n'était pas propagé au forward.

**Conclusions.** Les épisodes courts (< max_len) avaient des embeddings systématiquement plus proches de l'origine. Si la longueur corrèle avec le type d'anomalie, le clustering pourrait partiellement capturer la longueur plutôt que le signal.

**Amélioration.** ✅ Corrigé (P1) — pooling masqué avec `lengths` fournis par `collate_episodes`.

---

### L4.3 — Clustering Euclidien sur embeddings L2-normalisés

**Problème.** Le clustering hiérarchique (Ward, silhouette Euclidienne) est appliqué sur des embeddings L2-normalisés (sortie siamois). Sur une sphère, la distance Euclidienne et la distance cosinus ne sont pas équivalentes. K* optimal pourrait être biaisé.

**Pourquoi.** Ward est le clustering par défaut de scikit-learn et le plus courant dans la littérature de clustering hiérarchique. Le passage à une métrique cosinus nécessiterait un algorithme différent (spherical k-means).

**Conclusions.** Le K* = 10 obtenu par Ward pourrait ne pas être optimal au sens de la géométrie de la sphère. La silhouette rapportée (0.414–0.519 test) est potentiellement sous-estimée par la métrique inadaptée.

**Amélioration.** ✅ Comparaison faite (P2) — Ward Euclidien vs. spherical/cosine. K* validé.

---

### L4.4 — Pas de hard-negative mining

**Problème.** Le sampler de paires pour le siamois tire les négatifs uniformément, sans prioriser les paires proches dans l'espace latent (hard negatives). Après quelques epochs, la plupart des négatifs sont triviaux et n'apportent plus de gradient utile.

**Pourquoi.** Implémentation la plus simple. Le mining hard/semi-hard ajoute de la complexité et nécessite un recalcul des distances à chaque epoch.

**Conclusions.** La qualité des embeddings sature probablement avant ce qu'un mining plus agressif permettrait. L'écart STGCN/SimCLR (0.954 → 0.964 AUROC) pourrait être partiellement lié à ce facteur.

**Amélioration.** ✅ Implémenté (P2) — hard/semi-hard negative mining dans `EpisodePairSampler`.

---

### L4.5 — Précurseurs LR sans `class_weight` + fallback 0.5

**Problème.** Les classifieurs LogisticRegression des précurseurs n'utilisent pas `class_weight="balanced"`, malgré un déséquilibre de classes évident (C6/C9 : 1–2 épisodes positifs). De plus, le fallback pour les clusters dégénérés renvoie 0.5 au lieu de la prévalence empirique.

**Pourquoi.** Oubli. La valeur 0.5 est la valeur par défaut « neutre », mais elle est incorrecte pour des prévalences faibles.

**Conclusions.** Les AUROC instables sur C6/C9 et le fait que C3 ait le plus faible AUROC (0.794) pourraient être partiellement liés au déséquilibre non corrigé.

**Amélioration.** Ajouter `class_weight="balanced"` et fallback = prévalence empirique. Impact probablement visible sur les clusters minoritaires.

---

### L4.6 — MMD-RFF biaisé + calibration désalignée

**Problème.** L'estimateur MMD-RFF n'utilise pas la version unbiased, et l'imputation par moyenne de colonne crée une asymétrie ref/cur si la proportion de NaN diffère. De plus, ε_drift=0.5226 est calibré sur le début d'épisode (ref = premiers steps), alors que le détecteur en streaming peut recalibrer en cours de route.

**Pourquoi.** L'estimateur biased est plus simple et numériquement stable. La calibration utilise la fenêtre la plus « propre » (début = normal). Le désalignement avec le runtime est un artefact du passage offline → online.

**Conclusions.** Le biais MMD est constant et n'invalide pas les comparaisons relatives, mais ε_drift pourrait être légèrement mal calibré pour le régime steady-state.

**Amélioration.** Passer à l'estimateur unbiased, recalibrer ε_drift sur des fenêtres sliding (pas seulement le début d'épisode).

---

### L4.7 — Saliency « SHAP » invalide (ρ = −0.34 vs. permutation)

**Problème.** L'explainer `shap_explainer.py` (renommé `saliency_explainer.py` post-audit) implémentait `gradient × input`, pas SHAP. La corrélation avec la permutation importance est négative (ρ = −0.34), ce qui signifie que les attributions gradient et permutation sont **contradictoires**.

**Pourquoi.** SHAP (KernelSHAP, DeepSHAP) est coûteux en calcul. Le gradient × input est un raccourci fréquent en deep learning, mais son manque de fidélité sur des modèles avec ReLU et normalisation est documenté.

**Conclusions.**
- Les fiches de type basées sur la saliency gradient ne sont pas fiables.
- Seule la permutation importance est utilisable pour les interprétations publiées.
- Les features critiques identifiées par ablation (trace_depth, lexical_entropy, latency_p99) sont les seules attributions fiables.

**Amélioration.** ✅ Renommé (P0). KernelSHAP implémenté sur 1–2 clusters pour validation croisée.

---

## 5. Limites de l'inférence online (AlertAssembler)

### L5.1 — Topologie et hyperparamètres hardcodés

**Problème.** L'AlertAssembler hardcodait `n_nodes=6`, `epsilon_drift=0.5226`, `rff_dim=256`, `post_drift_window_s=3`. Aucun chargement depuis les artefacts d'entraînement.

**Pourquoi.** Prototype rapide pour la simulation sur ewat_v3, pas conçu pour la généralisation.

**Conclusions.** Le code était inutilisable hors ewat_v3 et fragile à toute évolution du dataset.

**Amélioration.** ✅ Corrigé (P0) — chargement depuis artefacts.

---

### L5.2 — Mismatch fenêtre précurseur train vs. inférence

**Problème.** Le `PrecursorDataset` (train) filtre les steps `regime == "normal"` pour construire la fenêtre de k steps, tandis que l'AlertAssembler (inférence) prend les `signal[-k:]` quel que soit le régime. Cela crée un distribution shift à l'inférence.

**Pourquoi.** Deux développeurs logiques (dataset = offline, assembler = online) avec des hypothèses différentes sur le contenu de la fenêtre.

**Conclusions.** Le classifieur est entraîné sur des fenêtres « normales » mais reçoit en inférence des fenêtres potentiellement contaminées par la phase anomalie. L'impact est un sous-comptage des alertes en début d'anomalie et un sur-comptage en phase tardive.

**Amélioration.** ✅ Corrigé (P0) — fenêtre alignée.

---

### L5.3 — Coût O(C) au lieu de O(max k)

**Problème.** L'AlertAssembler exécutait un forward STGCN par cluster, alors que tous les clusters partageant le même k* peuvent être encodés en un seul forward.

**Pourquoi.** Implémentation itérative, pas optimisée.

**Conclusions.** Latence d'inférence inutilement élevée : K forwards au lieu de |distinct k*| (typiquement 3–4 valeurs distinctes sur K=10).

**Amélioration.** ✅ Corrigé (P0) — groupement par k*.

---

## 6. Limites de conception expérimentale

### L6.1 — 5 graines, pas de k-fold

**Problème.** Les résultats multi-graines utilisent 5 graines (`[42, 123, 456, 789, 1337]`) avec `np.std` brute. Pas de SE bootstrap sur les agrégats, pas de k-fold.

**Pourquoi.** Contraintes de temps (chaque graine = ~2h CPU pour le pipeline complet). 5 graines étaient le compromis réaliste pour le stage.

**Conclusions.**
- L'affirmation « 0.973 ± 0.012 » utilise l'écart-type sur 5 points, ce qui est une estimation très bruitée de la vraie variance.
- La variabilité sil_test (0.414–0.662) sur 5 graines est large et pourrait se resserrer ou s'élargir avec plus de graines.

**Amélioration.** ✅ Étendu à 10 graines (P1) avec SE bootstrap sur les agrégats.

---

### L6.2 — Ablation sans réentraînement

**Problème.** L'ablation par modalité et leave-one-out masquait les features **à l'inférence** sur un modèle entraîné avec les 17 features. Cela mesure la robustesse du modèle au masquage, pas l'importance fonctionnelle des features.

**Pourquoi.** L'ablation avec réentraînement complet nécessitait ~41h CPU (7 conditions modalités × 5 graines + 17 conditions LOO × 5 graines).

**Conclusions.** Les features « non significatives » (cpu_util p=0.246, ram_util p=0.074) pourraient être significatives si le modèle avait appris à les utiliser en l'absence des features dominantes. La conclusion valide est : « le modèle est géométriquement robuste au masquage de ces features ».

**Amélioration.** ✅ Ablation rigoureuse avec réentraînement (P2) — 5 graines par condition.

---

### L6.3 — Baselines B1/B2 supérieures en AUROC

**Problème.** Les baselines précurseurs B1 (features brutes sans STGCN, AUROC=0.966) et B2 (k-means brut + LR, AUROC=0.975) surpassent EWAT (0.951) en AUROC moyen. Cet argument se retourne contre le projet si mal présenté.

**Pourquoi.** B1 et B2 prédisent les **labels EWAT** (clusters du siamois) depuis le signal brut. Ils héritent de la qualité des labels sans le coût du STGCN. C'est attendu : si les clusters sont bien séparés dans l'espace brut, une LR simple suffit à les prédire.

**Conclusions.**
- La valeur ajoutée du STGCN n'est pas dans la discrimination (AUROC) mais dans la **structuration** de l'espace latent (H1, sil=0.519) et la **découverte** d'une taxonomie empirique (K=10 types depuis 15 scénarios).
- B1/B2 ne découvrent rien : ils réapprennent les labels que le siamois a créés.
- L'AUROC n'est pas la bonne métrique pour mesurer la contribution du STGCN.

**Amélioration.** ✅ Baseline scénario direct (P2) — LR sur signal brut → label scénario chaos (15 classes), indépendamment des labels EWAT. Compare la valeur ajoutée du clustering vs. la prédiction directe du scénario.

---

### L6.4 — Pas de validation externe (dataset public)

**Problème.** Tous les résultats sont obtenus sur un unique dataset privé (ewat_v3, cluster Devoteam). Aucune validation sur un dataset public de microservices.

**Pourquoi.** Les benchmarks publics de microservices (GAIA, AIOps) n'incluent pas le protocole expérimental d'EWAT (scénarios de drift + anomalies avec labels temporels). L'adaptation nécessite un script dédié.

**Conclusions.** La généralisabilité des résultats est inconnue. L'architecture (6 services, topologie fixe) est spécifique au cluster Devoteam. Un relecteur pourrait questionner la transférabilité.

**Amélioration.** Implémenter `scripts/adapt_gaia.py` pour validation zero-shot (H1+H3) sur GAIA. Même un résultat partiel (H1 seul) renforcerait la contribution.

---

## 7. Limites d'ingénierie et infrastructure

### L7.1 — MLflow déclaré mais jamais utilisé

**Problème.** MLflow est déclaré dans `pyproject.toml` et `configs/default.yaml` mais aucun `import mlflow` n'existe dans le code hors des répertoires `mlruns/` créés en passant.

**Pourquoi.** MLflow a été prévu dès le début mais jamais branché. Les expériences utilisent des scripts ad hoc avec sauvegarde fichier.

**Conclusions.** La promesse de tracking d'expériences est creuse. Les hyperparamètres et métriques sont dispersés dans des fichiers JSON/YAML, pas centralisés.

**Amélioration.** ✅ Corrigé (P1) — Hydra `@hydra.main` + MLflow tracking branché dans encoder/typing/precursor.

---

### L7.2 — Hydra sous-utilisé

**Problème.** Les scripts utilisaient `OmegaConf.load` direct au lieu de `@hydra.main`. Pas de composition de configs, pas de groupes, pas de `multirun`.

**Pourquoi.** Adoption progressive de Hydra. Les scripts ont été écrits avant la migration vers la composition Hydra.

**Conclusions.** L'absence de composition rend difficile la gestion d'expériences multi-conditions (ablation, multi-graines, multi-architectures).

**Amélioration.** ✅ Corrigé (P1) — migration vers `@hydra.main`.

---

### L7.3 — Hyperparamètres dupliqués

**Problème.** `d_feat=17`, `n_nodes=6`, `d_hidden=64` sont répétés dans `precursor/train.py`, `ablation/run.py`, `alerts/assembler.py`. Toute évolution (ex. passage à 15 features) nécessite une synchronisation manuelle.

**Pourquoi.** Croissance organique du code. Chaque module a été développé indépendamment.

**Conclusions.** Source d'incohérence et de bugs silencieux. Le passage à ewat_v4 (potentiellement 15 features, 7 services) exposerait cette fragilité.

**Amélioration.** Centraliser dans le config Hydra ou dans les artefacts d'entraînement. ✅ Partiellement adressé via le chargement depuis artefacts dans l'AlertAssembler (P0).

---

### L7.4 — Tests d'intégration absents

**Problème.** 302 tests unitaires, mais aucun test d'intégration end-to-end vérifiant que record → build → assemble → train → eval produit des résultats déterministes.

**Pourquoi.** Les tests unitaires ont été priorisés pour chaque module. Un test d'intégration nécessite des fixtures synthétiques et est plus coûteux à maintenir.

**Conclusions.** La reproductibilité end-to-end n'est vérifiée que manuellement. Un changement dans un module amont (ex. aggregation.py) pourrait casser les résultats sans être détecté.

**Amélioration.** ✅ Corrigé (P1) — test d'intégration tiny-fixture (4 épisodes synthétiques, 5 epochs, vérification déterminisme à graine fixée).

---

### L7.5 — `pickle.load` sans validation

**Problème.** Les checkpoints (encoder, precursor, assembler) sont chargés via `pickle.load` sans validation des classes attendues. Risque de sécurité en production (arbitrary code execution).

**Pourquoi.** Acceptable en recherche mais non-standard en production.

**Conclusions.** Limite connue et assumée pour un projet de stage recherche. À mentionner si le pipeline est déployé en production.

**Amélioration.** Remplacer par `torch.load(..., weights_only=True)` pour les checkpoints PyTorch, et JSON/YAML pour les configurations.

---

### L7.6 — Expériences scratch non nettoyées

**Problème.** Les répertoires `typing_test/`, `typing_test2/`, `encoder_test/`, `precursors/`, `latency/`, `clustering/` sont des résidus d'expériences exploratoires, non documentés.

**Pourquoi.** Développement itératif sans nettoyage systématique.

**Conclusions.** Bruit dans le dépôt qui rend la navigation et la compréhension du projet plus difficiles.

**Amélioration.** Archiver ou supprimer les répertoires scratch. Garder uniquement les expériences finales sous `experiments/`.

---

## 8. Limites de la contribution scientifique

### L8.1 — Divergence formalisation / implémentation

**Problème.** Plusieurs divergences entre `docs/formalisation.md` et le code :
- La formalisation annonce un STGCN avec G(t) dynamique → le code utilisait A_mean.
- La formalisation annonce 4 régimes → le 4ème (θ_{drift∩anomaly}) repose sur un seul scénario (`faulty_deploy_overlap`).
- La formalisation annonce TE multivariée → le code fait TE univariée sommée.

**Pourquoi.** La formalisation a été écrite comme cible théorique. L'implémentation a fait des compromis pragmatiques.

**Conclusions.** Les divergences ont été progressivement réduites par les corrections P0–P2 (STGCN dynamique, TE documentée, χ² corrigé). Le rapport doit clairement distinguer la formulation théorique du périmètre validé expérimentalement.

**Amélioration.** ✅ Largement corrigé. Documenter explicitement les écarts résiduels dans le rapport de stage (section « limites et perspectives »).

---

### L8.2 — EWAT n'est pas du RCA mais la frontière est floue

**Problème.** EWAT est positionné comme early warning (Quoi, Dans combien de temps), distinct du RCA (Où, Pourquoi). Mais le typage (Étape 2) et l'ontologie (Étape 2b) s'approchent du « Quoi » au point de frôler le « Pourquoi ». Un relecteur pourrait arguer que le typage est une forme de diagnostic, pas de warning.

**Pourquoi.** La frontière warning/diagnostic est intrinsèquement floue. Le typage est un intermédiaire entre les deux.

**Conclusions.** Le positionnement doit être explicitement défendu : EWAT ne cherche pas la cause racine (quel composant, quelle ligne de code), il classifie le **type de pattern pré-anomalie** pour informer la réponse opérationnelle.

**Amélioration.** Renforcer la distinction dans le rapport en montrant que le lead time (3 min) et la fausse alarme (8.3%) sont des métriques d'early warning, pas de diagnostic.

---

### L8.3 — Scope fixe : 6 services, 1 topologie

**Problème.** Le pipeline EWAT est validé sur exactement 6 services avec une topologie fixe. La scalabilité à des architectures plus grandes (50+ services) ou dynamiques (auto-scaling, service mesh) n'est pas démontrée.

**Pourquoi.** Contrainte du cluster Devoteam (6 microservices de la boutique en ligne de démonstration).

**Conclusions.** Les résultats sont valides pour le scope testé mais la généralisabilité est une question ouverte. Le STGCN avec graphe dynamique (P1) est un pas vers la flexibilité, mais n'a pas été validé sur un graphe de taille différente.

**Amélioration.** Validation sur GAIA (autre topologie) ou sur un sous-ensemble artificiel (3/6 services) pour mesurer la sensibilité au scope.

---

## 9. Tableau de synthèse — statut des corrections

## L9 — Circularité d'évaluation H3 (P0, ajouté 2026-05-26)

### Diagnostic

L'AUROC=0.973 ± 0.012 (5 graines baseline) et 0.987 ± 0.011 (10 graines config optimisée) reportés pour H3 **mesurent la prédiction des labels cluster produits par EWAT lui-même** depuis l'embedding STGCN. La cible (cluster C_i) est dérivée du même pipeline (encodeur + siamois) qu'on évalue. C'est une évaluation **circulaire** : le pipeline retrouve son propre partitionnement.

### Preuves de la circularité (Phase A — stress tests)

| Test | Verdict |
|---|---|
| **B1 (raw features, labels EWAT)** | AUROC = 0.966 → les labels EWAT sont triviallement recoverables depuis le signal brut, sans encodeur |
| **A1 distant-window** (Δ far−near sur labels EWAT) | −0.007 → pas de précursion, fenêtre déplacée donne le même AUROC → fuite signature scénario |
| **A2 LOSO precursor-only** | top-1 sur scénario inédit = 0.51 ± 0.38 (polarisé : 4×100%, 4×0%) — pas de généralisation |
| **A5 paired Δ(B4−B3)** | Δ = +0.005, IC 95% = [−0.031, +0.044] **contient 0** → l'encodeur STGCN n'apporte rien en prédiction agrégée vs LR sur features brutes |

Voir `experiments/h3_robustness/results.md` pour les chiffres complets et `STATUS.md` section "Stress tests H3" pour les détails.

### Solution adoptée (Phase B + C — 2026-05-26)

Pivot de la cible d'évaluation : les **15 scénarios Chaos Mesh** (vérité terrain indépendante) remplacent les clusters EWAT auto-référents comme métrique principale.

| Évaluation honnête | AUROC | IC 95% bootstrap |
|---|---|---|
| **B2 — LR-OvR (sans STGCN) sur v4_strat** | **0.920** | [0.878, 0.956] |
| **B1 best (instance norm + last)** | **0.941** | [0.909, 0.970] |
| LOSO macro-AUROC (15 folds, v4_strat) | 0.930 | ± 0.007 |
| **C1 — STGCN end-to-end** | 0.863 | [0.823, 0.905] |
| C2-A1 distant-window STGCN ChaosMesh | Δ(far−near) = **−0.116** ⇒ précursion réelle |

### Limitation résiduelle

L'évaluation indépendante Chaos Mesh résout la circularité mais expose une nouvelle limite : la généralisation **open-set** à un scénario totalement inédit reste imparfaite. Phase C3 (OpenMax/EVT) propose une réponse partielle :
- Top-1 unknown sur scénario held-out = 1.0 (smoke test) → bonne détection de nouveauté
- Unknown AUROC ≈ 0.6-0.7 (à finaliser sur le LOSO complet)
- Closed-set AUROC après OpenMax dégrade de 1-2 pp seulement

Voir `experiments/architecture_v2/openset/` pour les chiffres finaux.

### Conséquences pour le rapport

- Le rapport doit explicitement reconnaître la circularité (sections D5/D6/D8 de la refonte).
- Le headline doit être **0.920 (B2) ou 0.941 (B1)** sur cible indépendante, pas 0.973 ni 0.987 sur labels EWAT.
- La contribution prédictive se reframe en **typage anticipé de scénario actif** avec précursion temporelle confirmée (C2-A1), pas en "prédiction d'événement futur" au sens strict.

---

## L10 — Surentraînement siamois sur ewat_v4 (P1, ajouté 2026-05-26)

### Diagnostic

Sur ewat_v4 (262 train, 60 val, 57 test stratifié temporel ou 270/60/45 stratifié strict), le SiameseTyper converge en `best_epoch = 2–7` (vs ~47 sur ewat_v3 avec 209 train). H1 silhouette test = **0.467 ± 0.156** sur 6 graines (vs 0.782 ± 0.065 sur v3) — graine 789 = 0.216 FAIL.

### Cause probable

Le dataset v4 est à la fois plus grand (262 vs 209 train) et plus diversifié (durées 47-51 vs 21 steps). Les paires contrastives échantillonnées aléatoirement deviennent rapidement trop faciles à séparer → la loss tombe vite → arrêt précoce sur val.

### Fix proposé (Phase C-4)

- Hard-negative mining renforcé (top-k par batch au lieu de random)
- Curriculum learning : warmup avec négatifs faciles puis progressive vers hard
- Re-évaluer K (10 vs 15 vs auto-déterminé via gap statistic)
- Sweep multi-graines (10) avec stratégie retenue → cible sil_test ≥ 0.60 sur les 10

### État (mise à jour Phase H + K, 2026-05-26)

**Confirmation multi-seed (Phase H, 10 graines)** :
- best_epoch siamois reste ≤ 7 sur **10/10 graines** malgré `mining=semi-hard` (Step 6 fix)
- sil_test = 0.691 ± 0.115 (range [0.521, 0.839]) — variance plus large que sur v3 (0.78 ± 0.07)
- K_optimal varie de **9 à 15** (Phase K.1) — ni silhouette ni Tibshirani gap statistic ne stabilise (agreement 4/10 seeds)

**Le surentraînement est donc structurel, pas un bug de mining**. Causes probables (à valider en future itération) :
1. n_train = 270 trop petit pour un siamois profond
2. Diversité des durées (47-51 steps) → paires contrastives faciles
3. Embeddings STGCN trop bruyants pour produire une géométrie stable

**Recommandations v5** :
- Hard-negative mining strict (`--mining hard`, pas semi-hard)
- Data augmentation (permutation aléatoire de nœuds, jitter temporel)
- Régularisation renforcée (weight_decay 1e-3, dropout 0.3)
- Fixer K=10 manuellement OU passer à HDBSCAN density-based clustering
- Cible : sil_test ≥ 0.6 stable sur 10 graines + best_epoch ≥ 15

Voir `experiments/multiseed/phase_h/{k_selection_comparison.md,variance_analysis.md}` pour les diagnostics complets.

---

## L11 — Latence end-to-end (P1, ✅ résolu 2026-05-26)

### Diagnostic et mesure

Budget formel `formalisation.md` : Étape 0 < 1 s, Étape 1+2 < 2 s, Étape 3 < 1 s, total < 5 s. Aucun benchmark concret avant Phase C-3.

### Résultat (`experiments/bench/latency_e2e.py`, 200 itérations, CPU)

| Étape | médiane | p95 | budget | verdict |
|---|---|---|---|---|
| Étape 0 (drift) | 0.01 ms | 0.01 ms | 1000 ms | 🟢 |
| Étape 1+2 (encoder + siamois) | 0.96 ms | 1.97 ms | 2000 ms | 🟢 |
| Étape 3 (précurseurs) | 1.85 ms | 3.91 ms | 1000 ms | 🟢 |
| **TOTAL** | **9.26 ms** | **13.28 ms** | **5000 ms** | **🟢** |

Verdict : **GREEN** — 375× sous budget. SLA défendable en production CPU. À ré-évaluer sur GPU et à charge réelle (concurrence de requêtes).

---

## L12 — OpenMax mitigé (P1, ouvert)

### Diagnostic

Phase C-3 OpenMax LOSO (15 folds × 60 époques sur ewat_v4_strat) :
- Unknown AUROC = **0.55 ± 0.24** (cible plan = 0.7 ❌)
- Top-1 unknown rate = 0.40 ± 0.41 (vs 0 OvR fermé, gain réel mais incomplet)
- Closed AUROC après OpenMax = 0.834 (dégradation ~3pp vs ~0.86 avant)

### Cause

OpenMax (Bendale & Boult 2016) suppose que les scénarios inédits sont "loin" des moyennes de classes connues dans l'espace d'activation. Pour des scénarios qui ressemblent fortement à un cluster connu, la Weibull du tail ne déclenche pas le mode "unknown".

### Alternatives à évaluer en travail futur

- **Mahalanobis-OOD** (Lee et al. 2018) : distance de Mahalanobis sur représentations gaussiennes — plus robuste sur features corrélées.
- **Energy-based OOD** (Liu et al. 2020) : utiliser l'énergie négative du log-sum-exp des logits — meilleure séparation in/out.
- **ODIN** (Liang et al. 2017) : temperature scaling + input perturbations.

### État

Documenté, OpenMax actuel = baseline. À étendre `src/ewat/openset/` avec ces 3 alternatives dans une itération future.

---

## L13 — Service graph N=6 (P2, ouvert — future work)

### Diagnostic

Tous les benchmarks EWAT utilisent Online Boutique (Google demo microservices, 6 services). En production, les architectures microservices comportent souvent 100+ services avec topologies complexes (mesh Istio, partition par tenants, etc.).

### Implications

- **Scalabilité** : la complexité de STGCNEncoder est O(N²) sur l'adjacence et O(N) sur les features. À N=100, le pipeline doit être re-benchmarqué (L11 fait pour N=6 seulement).
- **Topologie variable** : ewat_v3/v4 ont une topologie statique. Production = ajout/retrait de services, scaling horizontal.
- **Hétérogénéité** : les 6 services Online Boutique sont relativement homogènes. Production = batch jobs, streaming, frontends, etc.

### Fix proposé

Future work : valider sur un dataset plus large (RCAEval RE2 ou collecte sur un benchmark Kubernetes à N≥20).

---

## L14 — Validation cross-cluster nulle (P2, ouvert — future work)

### Diagnostic

Toutes les expériences EWAT sont sur `observit-cluster1` (RKE2, 9 nœuds). La portabilité vers d'autres clusters Kubernetes (EKS, GKE, on-prem) n'a pas été testée.

### Cause

Le pipeline dépend de :
- Prometheus + OTel Collector existants (configuration spécifique)
- Convention de noms de services Online Boutique (cartservice, frontend, etc.)
- Stack Loki / Jaeger / Tempo pour traces et logs

Sur un autre cluster avec autres conventions, le scaler ne transférerait pas (cf. RCAEval zero-shot : AUROC 0.495).

### Fix proposé

Stratégie B fine-tuning sur quelques épisodes du cluster cible (C-2 dans le plan, à implémenter). Pour une vraie portabilité, abstraire les noms de services et adopter OTel Semantic Conventions strict.

---

## L15 — Retraining cycle opérationnel non défini (P2, ouvert)

### Diagnostic

Quand re-entraîner EWAT en production ? Aucune spécification :
- Périodiquement (chaque N jours) ?
- Sur déclenchement (nouveau scénario, drift majeur) ?
- Continuel (online learning) ?

Le pipeline actuel suppose un entraînement batch one-shot, sans cycle de vie.

### Fix proposé

Future work : implémenter un détecteur de drift de distribution (différent du DriftDetector signal) qui surveille les embeddings de production vs entraînement, et déclenche un retraining quand un seuil est dépassé. Voir aussi MLflow Model Registry pour versioning.

---

## L16 — Hardcoded 17 features, pas d'auto-discovery (P2, ouvert)

### Diagnostic

Les 17 features S(t) sont fixées dans `EpisodeDataset.FEATURE_NAMES` ([src/ewat/encoder/dataset.py:38-44](src/ewat/encoder/dataset.py)). Choix éclairé par la littérature (Gregg 2013 méthodologie USE) mais non systématique.

### Implications

- Sur un autre cluster ou pipeline, certaines features peuvent manquer (ex. pas de Loki → log_error_rate impossible) ou nouvelles features pertinentes peuvent exister (saturation GPU, latence inter-zone, etc.).
- L'ablation montre `disk_io` critique (Δ=-0.088 sur H3) malgré 16.7% NaN sur ewat_v3, mais `retry_rate` ≈ 0 contribution → certaines features pourraient être supprimées sans perte.

### Fix proposé

Future work : implémenter un sélecteur de features automatique (mRMR, SHAP-based feature pruning, ou auto-encoder régularisé L1) qui propose un sous-ensemble optimal par déploiement.

---

## L17 — Ontologie sans validation expert / RCA réelle (P1, ouvert)

### Diagnostic

Phase 8 a produit une ontologie OWL/RDF avec 29 classes ancrées littérature (Soldani & Brogi 2022, Fu et al. 2025, Gregg 2013, Aniello et al. 2014). HermiT confirme la cohérence formelle (0.61 s, 0 inconsistance). 3 causales (BH-FDR p<0.05) sur cascades synthétiques.

### Limites résiduelles

- **Pas d'audit SRE / RCA externe** : aucun ingénieur opérationnel n'a relu/validé que les 29 classes capturent les types de pannes qu'il rencontre vraiment.
- **Pas d'intégration runbook / SOAR** : l'ontologie est un artefact de recherche, non connectée à un système d'incident response.
- **Causales synthétiques** : les 3 relations causales (C4→C1, C6→C5, C4→C8) sont obtenues sur des épisodes synthétiques composites, pas sur des cascades production réelles.

### Fix proposé

- Future work 1 : présentation de l'ontologie à un SRE staff Devoteam pour audit (1 séance, 1 jour).
- Future work 2 : intégration avec PagerDuty / OpsGenie via export RDF → règles d'alerte typées.
- Future work 3 : collecte d'épisodes multi-scénario réels (incidents production) pour valider/raffiner les causales.

---

## Limitations mineures (m-1 à m-8)

Brièvement documentées pour transparence méthodologique :

- **m-1** Variance inter-graines large (±24% sur H1 baseline 5 graines). Mitigé en config optimisée (±0.065 sur 10 graines). Sur ewat_v4 (±0.156 sur 6 graines), c'est plus marqué — voir L10.
- **m-2** Pas de pré-registration des stress tests A1–A5 ; ils ont été conçus *a posteriori* en réponse à la critique du maître de stage. Bonne pratique pour future itérations : pre-registration OSF / arXiv preprint.
- **m-3** Pas de correction de multiplicité sur 15 scénarios (FWER non contrôlé). Bootstrap CIs sont marginaux, pas joints. Bonferroni-Holm appliqué seulement sur l'ablation feature-wise.
- **m-4** Pas de power analysis a priori. Power post-hoc disponible via [experiments/bench/power_analysis.py](experiments/bench/power_analysis.py) : 5/10 clusters reportables (n_pos ≥ 5), power moyenne 1.0 sur ces 5.
- **m-5** Pas de retraining cycle (voir L15).
- **m-6** Lint ruff 52 warnings (N801 naming en OWL schema). User l'a explicitement écarté (mémoire `feedback_ruff.md`).
- **m-7** Pas de versioning model artifacts hors checkpoints PT (MLflow Model Registry recommandé future work).
- **m-8** SentenceBERT pour log semantic anomaly : modèle mono-lingue (anglais). À étendre vers multilingue (XLM-R, mBERT) pour logs production internationaux.

---

## 9. Tableau de synthèse — statut des corrections

| Correction | Priorité | Statut | Impact |
|---|---|---|---|
| χ² co-occurrence 2×2 complet | P0 | ✅ | Résultats inchangés (0 relations) |
| **Circularité H3** | **P0** | **✅ Phase B + C** | Headline 0.987 → 0.920 sur cible indépendante |
| **ewat_v4 split temporal cassé → v4_strat** | **P0** | **✅** | 4 scénarios absents du train corrigés (270/60/45 stratifié) |
| **Instance normalization (élimination baselines)** | **P0** | **✅** | +5pp sur cible Chaos Mesh |
| **OpenMax open-set recognition** | **P1** | **✅ Phase C3** | Réponse à A2 LOSO top-1=0 |
| TE-KSG p-value + BH-FDR | P0 | ✅ | Résultats inchangés (0 causales) |
| Bootstrap reproductible + BCa | P0 | ✅ | CIs reproductibles |
| Renommage SHAP → saliency | P0 | ✅ | Honnêteté terminologique |
| AlertAssembler depuis artefacts | P0 | ✅ | Généralisable |
| Holm/FDR ablation | P0 | ✅ | Features critiques confirmées |
| 10 graines + SE bootstrap | P1 | ✅ | CIs plus fiables |
| STGCN dynamique | P1 | ✅ | Aligné avec formalisation |
| Pooling masqué | P1 | ✅ | Embeddings non biaisés |
| Cluster-aware split | P1 | ✅ | NaN AUROC éliminés |
| Histogram seedé | P1 | ✅ | Reproductibilité bit-exact |
| Tests bootstrap + saliency | P1 | ✅ | Couverture test |
| Test intégration tiny-fixture | P1 | ✅ | Déterminisme vérifié |
| Hydra + MLflow | P1 | ✅ | Tracking centralisé |
| Ablation avec réentraînement | P2 | ✅ | Importance fonctionnelle |
| Hard-negative mining | P2 | ✅ | Embeddings améliorés |
| Cosine clustering | P2 | ✅ | K* validé |
| SimCLR pré-entraînement | P2 | ✅ | AUROC +1 pp |
| GAT vs GCN | P2 | ✅ | sil_test +0.083 |
| Baseline scénario direct | P2 | ✅ | Valeur ajoutée clarifiée |
| ewat_v4 (OTel SDK) | P2 | ✅ | disk_io 0% NaN |
| Cross-validation k-fold | — | ❌ Ouvert | Variance du split non mesurée |
| Validation GAIA | — | ❌ Ouvert | Généralisabilité inconnue |
| LR class_weight=balanced | — | ❌ Ouvert | Clusters minoritaires |
| McNemar pour H2 | — | ❌ Ouvert | Test plus adapté |
| pickle → torch safe load | — | ❌ Ouvert | Sécurité production |

---

## 10. Conclusions générales

### Ce que les limites nous apprennent

1. **La durée d'épisode est le facteur limitant principal.** H2 échoue, H2b est trivial, la FA est élevée — tout converge vers le même problème : ~21 steps ne suffisent pas pour le look-through. C'est un résultat en soi : EWAT nécessite un minimum de ~30 steps utiles pour que sa cascade fonctionne.

2. **La valeur du STGCN est dans la structuration, pas dans la discrimination.** Les baselines B1/B2 le montrent clairement : prédire un label est facile, **découvrir** un label est la vraie contribution. EWAT crée une taxonomie de 10 types depuis 15 scénarios — c'est une réduction de complexité exploitable opérationnellement.

3. **L'ontologie est limitée par le design expérimental, pas par l'algorithme.** Les 0 relations causales et co-occurrence sont la conséquence logique d'épisodes mono-scénario. Le pipeline ontologique est correctement implémenté (post-corrections), il manque simplement de données multi-scénario pour s'exprimer.

4. **Les corrections statistiques (P0) n'ont pas changé les résultats.** χ² corrigé → toujours 0. TE corrigée → toujours 0. Holm sur l'ablation → mêmes features critiques. Ce n'est pas un argument de solidité (les résultats étaient déjà nuls), mais ça confirme que les conclusions négatives sont robustes.

5. **La reproductibilité était partiellement factice avant P0/P1.** Graines non propagées, histogrammes stochastiques, bootstrap non seedé. Les résultats étaient qualitativement stables mais pas bit-exact reproductibles. C'est maintenant corrigé.

### Forces préservées

- Pipeline 3-phases atomique avec checkpoints append-only — qualité production.
- 302 tests unitaires + test d'intégration end-to-end.
- Résultats négatifs (H2, H2-bis) assumés et interprétés scientifiquement.
- Correction méthodologique documentée (nearest centroid, k* sur val) — démarche exemplaire.
- Séparation `collectors/` online vs `extractors/` offline — bonne abstraction.
- Scope graphe (6 services) explicitement assumé, pas un oubli.

### Recommandation pour le rapport de stage

Présenter ce document comme la section « Limites et perspectives » du rapport. Chaque limite est une contribution indirecte : elle définit le **périmètre de validité** des résultats EWAT et oriente les travaux futurs. Les résultats négatifs (H2, ontologie vide) sont plus informatifs qu'un succès non questionné.
