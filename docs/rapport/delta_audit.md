# Memo delta — ce que l'audit 2026-06 change dans le rapport de stage

_2026-06-11. Mode d'emploi : chaque entrée dit QUOI changer dans
`rapport_stage.md`, avec le chiffre avant/après et la formulation suggérée.
Les références pointent vers les artefacts (`experiments/audit2026/*`,
`experiments/sota/usad/`, STATUS §Phase L, limitations §9)._

> ⚠ Phase H-bis (retrain 10 graines avec les correctifs) en cours — les
> entrées marquées 🕐 seront complétées à l'agrégation. Tout le reste est
> final.

---

## 1. Headline — à reformuler (obligatoire)

**Avant** : « B2 = 0.920 [0.878, 0.956] sur cible Chaos Mesh indépendante. »

**Après** : « B2 = 0.920 [0.878, 0.956] en macro-AUROC et **0.587 en
macro-PR-AUC** sur le split temporel officiel (3 positifs/scénario en test :
l'AUROC seule est optimiste). Le split temporel est *conservateur* : sur 5
ré-échantillonnages stratifiés, B2 atteint 0.969 ± 0.011 (PR-AUC 0.80–0.91) —
l'écart mesure le décalage temporel d'infrastructure, qui pénalise d'abord la
précision. »

Réfs : `experiments/audit2026/{oracle,split_variance}/results.md`.

## 2. Positionnement — nouvelle sous-section « Comparaison à l'état de l'art »

Le rapport n'avait AUCUNE baseline publiée. Ajouter :

| Méthode | macro-AUROC | macro-PR-AUC |
|---|---|---|
| **EWAT B2** (LR-OvR fenêtres brutes) | **0.920** | **0.587** |
| USAD (Audibert et al., KDD 2020), latents + LR | 0.878 | 0.505 |

Formulation : « À protocole et featurisation strictement identiques (fenêtres
pré-injection instance-normalisées), la représentation non supervisée d'USAD
porte moins d'information de type que les features brutes — B2 reste devant
de 4,2 pp AUROC / 8,2 pp PR-AUC. USAD étant un détecteur, ce volet mesure sa
représentation, pas sa tâche native (limite documentée). »
Réf : `experiments/sota/usad/results.md`.

## 3. Bornes — nouvelle figure/paragraphe « lecture du gap »

Oracle fit-on-all = 1.000 ; borne légale train+val = 0.962 [0.929, 0.992].
Formulation : « Le gap 0.920 → 1.0 est intégralement rattrapable par cette
classe de modèles (l'oracle sature) : il reflète un budget de données, pas un
chevauchement irréductible entre scénarios. » Réf : `audit2026/oracle/`.

## 4. Robustesse production — nouveau paragraphe + figure

Courbe `audit2026/nan_robustness/nan_robustness.png` : −1,3 pp AUROC à 20 %
de NaN injectés, −8,3 pp à 50 %. Formulation : « le pipeline tolère des trous
de collecte massifs sans recalibration — la dégradation reste graduelle
jusqu'à 50 % de cellules manquantes. »

## 5. Calibration — correction OBLIGATOIRE de la section alerting

L'ancienne table seuil (0.7 → FA 8,3 %, lead 3,0 min, ewat_v3) **ne doit plus
être présentée comme un point opérationnel valide** :

1. Les probabilités des précurseurs ne sont pas calibrées : ECE = 0.120
   (> 0.10). La recalibration isotonique (fittée sur val) ramène à 0.021.
2. Sur le pipeline retrainé v4_strat, FA drift = 100 % à tous les seuils et
   lead ≈ 14,5 min ≈ toute la phase pré-injection : l'« alerte » identifie le
   scénario dès les premiers steps (cohérent avec la fuite A1), elle ne
   précède pas l'injection.

Formulation suggérée : « les scores de précurseurs doivent être recalibrés
(isotonique sur val, ECE 0.120 → 0.021) avant toute interprétation en
probabilité ; la table opérationnelle sera re-déduite sur ewat_v5 avec ce
prérequis (protocole §6). » Réfs : `audit2026/calibration/`,
`audit2026/alerts_v4strat/`, limitations L9.2.

## 6. Drift (H2a) — renforcer le résultat négatif

Ajouter aux deux falsifications existantes : « la recalibration sur v4_strat
donne AUC = 0.315 (fenêtres 10/10) et 0.284 (5/5) — *pire que le hasard* :
l'amplitude du MMD² est plus déplacée par les anomalies que par les drifts
bénins. H2a n'est pas un problème de réglage de ε ; le mécanisme de
séparation par seuil est structurellement falsifié (3e falsification,
critère de sortie défini pour v5). » Réf : `audit2026/drift_calibration_v4strat*/`.

## 7. Typage — mise à jour des chiffres et du récit

- 🕐 H1 multi-graines : remplacer 0.691 ± 0.115 (Phase H) par l'agrégat
  Phase H-bis (graine 42 : sil_test **0.880**, K=10 fixe). K n'est plus
  sélectionné (instabilité Phase K actée) : « K = 10 fixé par design ».
- **Nouveau constat à assumer (L9.3)** : avec la sélection de checkpoint sur
  la silhouette val, le meilleur checkpoint est **l'époque 1** quelle que
  soit la marge (sweep 1.5/1.8/2.0 : 0.819/0.669/0.880). Formulation : « le
  fine-tuning contrastif n'améliore pas la géométrie héritée de l'encodeur
  pré-entraîné — il l'érode après la première époque. La valeur géométrique
  réside dans l'encodeur (+ self-loops), le siamois est au mieux un
  ajustement d'une époque. » C'est une explication PLUS forte que le
  « surentraînement » de L10.
- Silhouette désormais accompagnée d'un modèle nul (M9) : reporter
  Δ(sil − sil_null) et p empirique au lieu du seul seuil 0.3.

## 8. Limites — remplacer/compléter

- Section limites : pointer vers limitations.md §9 (L9.1–L9.6) : PR-AUC du
  headline, calibration, contrastif époque-1, drift falsifié-aux-3-niveaux,
  couverture clusters test (4/10 vides avec K=10), missingness sans gain (D3,
  négatif propre).
- Mettre à jour L10 (surentraînement) avec le récit L9.3.

## 9. Méthodologie — phrases à ajouter

- « Tous les AUROC sont désormais accompagnés de PR-AUC (E3) et filtrés par
  n_pos ≥ 5 (E8) ; les tests statistiques suivent le cadre unifié du
  protocole (McNemar/Wilcoxon/Fisher/BCa/permutation, correction Holm/BH). »
- « Le pipeline v5 est protégé par un registre de schéma versionné, un split
  stratifié par défaut avec routage held-out test-only, et un test golden
  bout-en-bout ; le protocole d'évaluation v5 a été figé avant l'arrivée des
  données (evaluation_protocol_v5.md). »

## 10. Chiffres d'infrastructure à rafraîchir

- 401 tests unitaires → **739** (suite complète verte).
- Mentionner la branche `audit-fixes-2026-06` (12+ commits) comme cycle de
  durcissement post-audit.

## 🕐 11. À compléter après Phase H-bis (agrégation 10 graines)

- Tableau multi-graines H1/H3/A1 (mean ± std, K=10 constant par design).
- Comparaison Phase H vs Phase H-bis (impact agrégé des fixes).
- Si sil_test moyen ≥ 0.75 : « les correctifs d'audit (self-loops,
  sélection sur silhouette, K fixe) déplacent la moyenne multi-graines de
  +X pp et suppriment la variance due à K ».
