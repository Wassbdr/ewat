# EWAT — Limites, causes, conclusions et améliorations possibles

_Document établi le 2026-05-11 — état post-audit complet et corrections P0–P2_

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
| L6 | TE-KSG = somme univariée | Sous-estime la synergie entre features | Corrigé + documenté (P0) |
| L7 | 0 relations causales, 0 co-occurrence | Ontologie réduite aux transitions temporelles | Ouvert |
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

### L3.3 — Ontologie vide (0 causales, 0 co-occurrence)

**Problème.** Après 100 permutations avec corrections de multiplicité, l'ontologie ne contient que 22 relations temporelles (transitions) et zéro relation causale ou de co-occurrence.

**Pourquoi.**
- **Causalité** : chaque épisode injecte un seul scénario → pas de co-causalité entre types différents dans un même épisode. Les 2 relations du dry-run (20 perm.) étaient des faux positifs.
- **Co-occurrence** : même raison — un épisode = un scénario, donc les clusters ne co-occurrent pas dans la même fenêtre temporelle.
- La conception expérimentale (scénarios isolés) empêche structurellement d'observer des relations inter-types.

**Conclusions.**
- L'ontologie est fondamentalement limitée par le protocole de collecte mono-scénario.
- Les relations temporelles (auto-transitions + 12 transitions cross-cluster) sont le seul apport exploitable.
- Ce n'est pas un défaut du pipeline ontologique mais du design expérimental.

**Amélioration.** Collecte ewat_v4 avec scénarios composés (ex. cascade : `memory_pressure` → `crash`) pour observer de vraies co-occurrences et causalités. Alternative : injection multi-scénario simultanée sur différents services.

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

| Correction | Priorité | Statut | Impact |
|---|---|---|---|
| χ² co-occurrence 2×2 complet | P0 | ✅ | Résultats inchangés (0 relations) |
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
