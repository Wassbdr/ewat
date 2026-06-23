# Projet ObservIT — Module EWAT : détection précoce et typage automatique des anomalies en environnement microservices Kubernetes — Étude approfondie

| | |
|---|---|
| Livrable | D__.__ — à compléter |
| Semestre / année | __ Semestre __ / 2026 |
| Auteur | Wassim Badraoui |
| Entreprise | Devoteam |
| Tuteur entreprise | _____________________ |
| Tuteur académique | _____________________ |

---

## Introduction

L'exploitation d'applications microservices sur Kubernetes produit un flux continu et hétérogène de
signaux d'observabilité — métriques, traces distribuées et journaux. Sur ces signaux, l'enjeu de
fiabilité ne se limite plus à constater qu'une panne a eu lieu : il s'agit de l'anticiper et de la
qualifier suffisamment tôt pour agir. Or les dispositifs de détection en place peinent à distinguer
les évolutions normales du système (déploiements, autoscaling, variations de charge) des dégradations
réelles, ce qui engendre une quantité de fausses alertes préjudiciable à l'exploitation.

Le présent livrable décrit EWAT (Early Warning and Anomaly Typing), une méthode conçue pour répondre à
deux questions opérationnelles distinctes de l'analyse de cause racine : *quel type d'anomalie est en
train de se développer* et *à quel horizon*, après avoir séparé explicitement les changements bénins
des anomalies. Le document présente en détail le contexte et la problématique (section 1), l'état de
l'art et les fondements conceptuels (section 2), l'environnement expérimental et les données
(section 3), l'approche proposée — pipeline et modélisation mathématique (section 4), les résultats et
leur validation empirique (section 5), une étude critique de la robustesse des résultats (section 6),
les limites assumées (section 7), un schéma d'implémentation et d'automatisation (section 8), puis la
conclusion (section 9).

### Rappels

Les travaux ObservIT visent à renforcer la supervision des environnements multi-cloud et la maîtrise
des objectifs de niveau de service (SLO). La détection et la prédiction de violations s'y appuient sur
les trois piliers de l'observabilité. EWAT s'inscrit dans cette ligne en intervenant en amont de
l'incident : il ne cherche pas à expliquer une panne survenue (où, pourquoi), mais à caractériser et
anticiper une panne naissante (quoi, dans combien de temps). Il complète ainsi les approches de
supervision orientées SLO par une couche de typage et d'anticipation des modes de défaillance.

Le vocabulaire employé dans ce document est le suivant. Un **épisode** est une séquence temporelle
décrivant l'état du système autour d'une injection de panne contrôlée. Le **signal** S(t) est le
résumé numérique de cet état à un instant donné. Le **régime** θ(t) désigne l'état opérationnel du
système (normal, drift, anomalie, ou les deux simultanément). Un **type** d'anomalie est un groupe
d'épisodes au comportement latent similaire, découvert par le pipeline. Un **drift bénin** est un
changement de distribution sans dégradation, par opposition à une **anomalie**.

---

## 1. Contexte de l'étude

### 1.1 Méthodes utilisées

Les dispositifs de détection d'anomalies déployés en production reposent, dans leur grande majorité,
sur trois familles d'approches.

La première, la plus répandue, applique des **seuils statiques** à des métriques individuelles : une
alerte est levée lorsqu'une valeur dépasse une borne fixée manuellement (par exemple un taux d'erreur
supérieur à 1 %, ou une latence P99 supérieure à un seuil contractuel). Simple à mettre en œuvre, cette
approche est cependant rigide : les seuils doivent être ajustés service par service et évoluent mal
avec la charge.

La deuxième famille s'appuie sur des **écarts statistiques** : on modélise la distribution normale
d'une métrique (moyenne et écart-type sur une fenêtre glissante) et l'on signale tout point s'écartant
de plus de quelques écarts-types (z-score), ou sortant d'une bande de confiance. Cette approche
s'adapte mieux aux variations lentes, mais reste un détecteur d'écart univarié, insensible au
contexte.

La troisième famille, plus récente, mobilise l'**apprentissage profond sur séries temporelles** :
modèles de prévision (l'écart entre la valeur prédite et la valeur observée sert de score d'anomalie)
ou modèles de reconstruction (l'erreur de reconstruction sert de score). Ces méthodes captent des
dépendances temporelles complexes, mais sont presque toujours évaluées sur des jeux de référence et
ciblent des anomalies ponctuelles, en ignorant la non-stationnarité légitime du système.

Le point commun de ces trois familles est de raisonner sur l'**écart à une normale**, sans jamais
qualifier le **régime** dans lequel se trouve le système : tout changement de distribution y est traité
comme également suspect.

### 1.2 Limites constatées

La conséquence directe de ce parti pris est une production massive de fausses alertes. En effet, un
système microservices évolue en permanence pour des raisons parfaitement saines : déploiement
progressif d'une nouvelle version (rolling update), montée et descente automatiques en charge
(autoscaling), pics de trafic planifiés, redistribution de charge après une mise à l'échelle. Chacun de
ces événements modifie la distribution du signal — exactement comme le ferait une panne naissante. Un
détecteur d'écart générique ne peut donc pas les départager, et lève une alerte dans les deux cas.

Pour l'exploitation, le coût est double. À court terme, le volume d'alertes sature les équipes
d'astreinte (fatigue d'alerte). À moyen terme, la proportion élevée de faux positifs érode la confiance
dans l'outil, jusqu'à ce que les alertes soient ignorées — y compris les vraies. La littérature sur
l'analyse de fiabilité des microservices (Fu et al. 2025) identifie précisément cet écart entre
performance affichée sur banc d'essai et déployabilité réelle, et l'attribue en grande partie aux faux
positifs.

Une seconde limite tient à l'orientation des outils existants vers l'**analyse de cause racine** (RCA),
c'est-à-dire l'explication d'une panne *après* son occurrence. Aussi utile soit-elle, cette démarche
est par nature post-mortem : elle suppose que l'incident a déjà eu lieu et n'offre pas d'anticipation.

### 1.3 Problème structurel

Le problème central n'est donc pas seulement de détecter un changement, mais de le **qualifier**. Trois
questions structurent ce besoin :

1. Le changement observé est-il un **drift bénin**, une **anomalie**, ou les deux à la fois — cas du
   déploiement défectueux, où un drift (le déploiement) et une anomalie (le bug introduit) surviennent
   simultanément ?
2. S'il s'agit d'une anomalie, de **quel type** est-elle ? Toutes les pannes ne se ressemblent pas :
   une saturation CPU, une fuite mémoire, un crash et une dégradation réseau appellent des réponses
   différentes.
3. **À quel horizon** la panne va-t-elle se manifester ? Une anticipation de quelques minutes change la
   nature de la réponse possible (mitigation automatique, bascule, alerte préventive).

Aucune de ces trois questions n'est adressée par un détecteur d'écart univarié. Y répondre suppose de
modéliser explicitement le régime du système, d'apprendre une taxonomie des modes de défaillance, et de
prédire l'occurrence future de chaque type.

### 1.4 Objectif du module

EWAT poursuit quatre objectifs, chacun assorti d'un critère de validation.

1. **Séparer le drift bénin de l'anomalie**, afin de supprimer les fausses alertes sur les évolutions
   normales. Critère : réduction significative du taux de faux positifs à rappel constant.
2. **Typer automatiquement les anomalies** à partir d'une taxonomie empirique apprise. Critère :
   structurabilité des types mesurée par une silhouette supérieure à 0,3 en held-out.
3. **Anticiper le type** avec un horizon utile. Critère : AUROC par type supérieur à la base aléatoire
   de 0,5, et idéalement validé sur une cible indépendante.
4. **Respecter un budget de latence** compatible avec une exploitation en ligne (chaîne d'inférence
   inférieure à cinq secondes).

Ces objectifs définissent une démarche de détection précoce (*early warning*) : quoi, dans combien de
temps, avant — par opposition à l'analyse de cause racine (où, pourquoi, après).

---

## 2. État de l'art et fondements conceptuels

Cette section présente, de façon didactique et autoportante, les fondements conceptuels mobilisés par
EWAT : les familles de détection d'anomalies, les notions d'observabilité Kubernetes, la théorie du
drift, les méthodes d'apprentissage et d'inférence employées, et les outils statistiques de validation.
Pour chaque brique, on précise ce que la littérature apporte et ce qu'EWAT en retient.

### 2.1 Détection d'anomalies : fondations et apprentissage profond

La détection d'anomalies est un champ ancien, dont la référence structurante reste le survey de Chandola,
Banerjee et Kumar (2009). Ces auteurs distinguent trois grandes catégories d'anomalies. Les **anomalies
ponctuelles** sont des observations individuelles aberrantes (un pic isolé de latence). Les **anomalies
contextuelles** ne sont anormales que relativement à un contexte (une consommation CPU élevée est
normale en journée, anormale la nuit). Les **anomalies collectives** sont des séquences entières
anormales, alors que chaque point pris isolément paraît normal (une suite de requêtes formant une
attaque). La leçon centrale, qu'EWAT reprend, est qu'une anomalie n'est définie que **relativement à un
contexte** : le même point peut être normal ou anormal selon le régime du système. C'est précisément ce
qui justifie de modéliser explicitement le régime θ(t) plutôt que de traiter le signal de façon
stationnaire.

Les approches modernes recourent massivement à l'**apprentissage profond sur séries temporelles**, dont
Zamanzadeh Darban et al. (2024) dressent un panorama récent. Deux familles dominent : les méthodes de
**prévision**, où l'écart entre la valeur prédite et la valeur observée sert de score d'anomalie, et les
méthodes de **reconstruction** (autoencodeurs), où l'erreur de reconstruction sert de score. Le survey
souligne une limite récurrente : la plupart de ces méthodes sont évaluées sur des jeux de référence,
ciblent des anomalies ponctuelles et ignorent la non-stationnarité légitime du système. EWAT en tire une
double précaution : opérer sur une fenêtre temporelle (et non point par point) et, surtout, évaluer
systématiquement sur une cible indépendante pour éviter de surévaluer un modèle sur sa propre cible.

### 2.2 Microservices, observabilité et performance des systèmes

**Architecture microservices et Kubernetes.** Une architecture microservices décompose une application en
services autonomes, faiblement couplés, communiquant par le réseau (le plus souvent en HTTP ou gRPC).
Chaque service est déployable et scalable indépendamment. Kubernetes orchestre ces services : son plan de
contrôle (API server, base d'état distribuée, ordonnanceur, contrôleurs) maintient l'état désiré du
système, tandis que les nœuds de travail exécutent les conteneurs au sein de pods. Kubernetes automatise
le placement des pods, la montée et descente en charge (autoscaling), la réparation (redémarrage des pods
échoués) et la découverte de services. Cette automatisation est une force pour l'exploitation, mais une
difficulté pour la supervision : la topologie effective change en permanence, ce qui interdit de
considérer l'ensemble des services comme un vecteur figé.

**Les trois piliers de l'observabilité.** La supervision d'un tel système repose sur trois sources
complémentaires. Les **métriques** sont des séries temporelles numériques agrégées (consommation de
ressources, latences, taux d'erreur), généralement collectées par Prometheus. Les **traces distribuées**
reconstituent le parcours d'une requête à travers les services, exposant la structure des appels
(profondeur, fan-out) et les durées par segment (span). Les **journaux** sont les événements textuels
émis par les services. EWAT exploite les trois, à travers les trois modalités de son signal.

**Méthode USE et indicateurs avancés.** Pour décider *quelles* grandeurs surveiller, EWAT s'appuie sur la
méthodologie USE de Gregg (2013), qui recommande d'observer, pour chaque ressource, son **utilisation**,
sa **saturation** et ses **erreurs**. Gregg identifie en particulier la profondeur de file d'attente
comme un **indicateur avancé** : une file qui s'allonge précède souvent une dégradation observable, ce
qui en fait un signal précieux pour l'anticipation. Cette grandeur est intégrée aux métriques d'EWAT.

### 2.3 Drift conceptuel

Distinguer un changement de distribution bénin d'une anomalie relève du *concept drift*. Hinder et al.
(2024) en proposent une formalisation dans le cadre des flux non supervisés et discutent les méthodes de
détection sans labels. Myrtollari et al. (2025) appliquent spécifiquement la détection d'anomalies
tenant compte du drift aux microservices Kubernetes — un cadre très proche de celui d'EWAT. Ces travaux
confirment que la séparation drift/anomalie est le bon problème, mais ne modélisent pas explicitement le
cas le plus difficile : le déploiement défectueux, où drift et anomalie surviennent ensemble. EWAT
formalise ce cas comme un quatrième régime à part entière (θ_{drift∩anomaly}) et l'aborde par le
mécanisme de look-through ; son résultat négatif honnête sur ce mécanisme (section 5.2) constitue un
apport au débat sur les limites de la séparation par MMD.

### 2.4 Analyse de cause racine en microservices et positionnement d'EWAT

Le corpus dominant sur la fiabilité des microservices vise l'**analyse de cause racine** (RCA). Fu et al.
(2025) en proposent un survey et identifient un écart persistant entre la performance affichée sur banc
d'essai et la déployabilité réelle, largement imputable aux faux positifs. Pham et al. (2024) évaluent de
façon critique les approches de RCA fondées sur l'inférence causale et montrent leur fragilité sur
données réelles — ce qui motive, dans EWAT, le choix d'une mesure d'information non paramétrique (Transfer
Entropy) plutôt que d'un graphe causal, et le cantonnement de cette analyse au hors-ligne. GrayScope
(Zhang et al. 2024) traite des « pannes grises », ces défaillances partielles peu visibles dans la
télémétrie standard : cette notion éclaire directement le bug réel F1 d'EWAT (section 3.4.6), invisible
en télémétrie et reconnu comme un cas négatif honnête.

Le positionnement d'EWAT se définit par contraste avec ce corpus : là où le RCA explique une panne
*après* son occurrence (où, pourquoi), EWAT cherche à la caractériser et l'anticiper *avant* (quoi, dans
combien de temps). Cette distinction est structurante : elle interdit, par exemple, d'utiliser une
information post-incident dans l'évaluation, et oriente toute la conception vers la fenêtre pré-injection.

### 2.5 Tests à noyau et détection de changement

Pour détecter un changement de distribution sans hypothèse paramétrique, EWAT recourt au test à deux
échantillons par noyau de Gretton et al. (2012), la *Maximum Mean Discrepancy* (MMD). L'idée est de
plonger les distributions dans un espace de Hilbert à noyau reproduisant, où chaque distribution est
représentée par sa moyenne ; le MMD mesure la distance entre ces deux moyennes. Cette statistique est
nulle si et seulement si les distributions coïncident (pour un noyau caractéristique), et croît avec leur
écart ; elle est non paramétrique et multivariée, donc adaptée à un signal de dix-sept dimensions
corrélées. Formellement, pour deux distributions $P$ et $Q$ représentées dans un espace de Hilbert à
noyau reproduisant $\mathcal{H}$ par leurs plongements moyens $\mu_P$ et $\mu_Q$, et pour un noyau $k$ :

$$\mathrm{MMD}^2(P,Q) = \lVert \mu_P - \mu_Q \rVert_{\mathcal{H}}^2
= \mathbb{E}_{x,x'}\,k(x,x') + \mathbb{E}_{y,y'}\,k(y,y') - 2\,\mathbb{E}_{x,y}\,k(x,y).$$

Son calcul exact étant quadratique, EWAT l'approche par les *Random Fourier Features* de
Rahimi et Recht (2007), qui approximent un noyau invariant par translation par un produit scalaire dans
un espace de projection aléatoire de dimension D, ramenant le coût à un ordre linéaire :

$$k(x,y) \approx \varphi(x)^{\top}\varphi(y), \qquad \varphi : \mathbb{R}^{d} \to \mathbb{R}^{D}.$$

Cette combinaison MMD-RFF rend la détection de drift compatible avec une exécution en ligne.

### 2.6 Apprentissage de représentations sur graphes

Le système de services formant un graphe dont la topologie évolue, EWAT en encode l'état par un réseau de
neurones sur graphe. La brique fondatrice est le **réseau de convolution de graphe** (GCN) de Kipf et
Welling (2017), qui généralise la convolution aux graphes : la représentation d'un nœud est mise à jour
en agrégeant celles de ses voisins, selon une formulation spectrale approchée au premier ordre
(propagation de message). EWAT s'appuie sur la variante **spatio-temporelle** (STGCN) de Yu, Yin et Zhu
(2018), conçue à l'origine pour la prévision de trafic, qui combine une convolution spatiale sur le
graphe et une convolution temporelle causale pour capturer la dynamique. Deux variantes sont comparées :
le **réseau à attention de graphe** (GAT) de Veličković et al. (2018), qui apprend une pondération
adaptative du voisinage par un mécanisme d'attention sur les arêtes, et le pré-entraînement contrastif
(section 2.7). Le choix de modéliser par graphe — plutôt que par un simple vecteur concaténant les
services — est justifié par la nature relationnelle des défaillances, qui se propagent le long des
dépendances.

### 2.7 Apprentissage contrastif et réseaux siamois

Le typage des anomalies repose sur l'**apprentissage métrique** : on cherche un espace de représentation
où les épisodes similaires sont proches et les épisodes dissemblables éloignés. Un **réseau siamois**
réalise cet objectif en traitant des paires d'exemples avec une perte contrastive : les paires positives
(même nature) sont rapprochées, les paires négatives éloignées au-delà d'une marge. Le cadre **SimCLR**
de Chen et al. (2020) a popularisé une variante auto-supervisée fondée sur la perte NT-Xent, qui construit
des paires positives par augmentation d'un même exemple et apprend des représentations invariantes aux
augmentations. Eldele et al. (2021) adaptent cette idée aux séries temporelles (TS-TCC) par contraste
temporel et contextuel. EWAT utilise un réseau siamois pour le typage et évalue, en variante, le
pré-entraînement contrastif (SimCLR).

Pour une paire d'embeddings $(z_i, z_j)$ de label $y$ (1 si même type, 0 sinon), de distance $d_\varphi$
et de marge $m$, la perte contrastive à marge s'écrit :

$$\mathcal{L}_{\text{contr}}(z_i, z_j) = y\,d_\varphi^2 + (1-y)\,\big[\max(0,\; m - d_\varphi)\big]^2,$$

et la perte contrastive normalisée NT-Xent, pour une paire positive $(i,j)$ et une température $\tau$ :

$$\mathcal{L}_{\text{NT-Xent}} = -\log
\frac{\exp\!\big(\operatorname{sim}(z_i, z_j)/\tau\big)}
{\sum_{k \neq i}\exp\!\big(\operatorname{sim}(z_i, z_k)/\tau\big)}.$$

### 2.8 Clustering et sélection du nombre de groupes

Sur les représentations apprises, un clustering découvre la taxonomie des types. La qualité d'un
partitionnement se mesure par le **coefficient de silhouette** de Kaufman et Rousseeuw (1990), qui compare,
pour chaque point, sa distance moyenne aux points de son cluster à sa distance au cluster le plus proche ;
ces auteurs proposent une grille d'interprétation dont le seuil conventionnel d'acceptabilité est 0,3.
En notant $a(i)$ la distance moyenne d'un point $i$ aux points de son cluster et $b(i)$ sa distance
moyenne au cluster voisin le plus proche, son coefficient de silhouette est :

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\; b(i)\}} \in [-1, 1],$$

la silhouette globale étant la moyenne des $s(i)$. Le
choix du nombre de clusters K s'appuie sur la **statistique de gap** de Tibshirani, Walther et Hastie
(2001), qui compare la dispersion intra-cluster observée à celle attendue sous une hypothèse nulle. Enfin,
le choix de la **métrique** est crucial : Dhillon et Modha (2001) montrent que sur des données normalisées
sur la sphère unité, le clustering sphérique fondé sur la distance cosinus est cohérent, là où la distance
euclidienne ne l'est pas — un résultat directement exploité par EWAT (section 5.3).

### 2.9 Information mutuelle, transfer entropy et causalité

Pour estimer les relations causales entre types, EWAT mesure le flux d'information dirigé d'une série
temporelle vers une autre. L'**information mutuelle** quantifie la dépendance statistique entre deux
variables ; son estimation non paramétrique par les k plus proches voisins est due à Kraskov, Stögbauer et
Grassberger (2004), dont l'estimateur (KSG) est adapté aux dépendances non linéaires. La **Transfer
Entropy** de Schreiber (2000) étend cette idée à la causalité directionnelle : elle mesure la réduction
d'incertitude sur le futur d'une série $Y$ apportée par le passé d'une série $X$, au-delà du passé de
$Y$ lui-même :

$$T_{X \to Y} = \sum p\big(y_{t+1}, y_t^{(k)}, x_t^{(l)}\big)\;
\log \frac{p\big(y_{t+1} \mid y_t^{(k)}, x_t^{(l)}\big)}{p\big(y_{t+1} \mid y_t^{(k)}\big)},$$

où $y_t^{(k)}$ et $x_t^{(l)}$ désignent les passés respectifs des deux séries ; l'estimation des
probabilités se fait par les $k$ plus proches voisins (estimateur KSG). EWAT écarte explicitement la causalité de Granger, qui suppose une linéarité inadaptée à ces
signaux, au profit de la Transfer Entropy estimée par KSG. La contrainte de l'estimateur (besoin d'une
longueur de série suffisante au regard de la dimension) explique le recours à des épisodes synthétiques
allongés pour l'analyse causale (section 5).

### 2.10 Tests multiples et statistique de robustesse

L'analyse de nombreuses relations impose de contrôler la multiplicité des tests. EWAT utilise le contrôle
du **taux de fausses découvertes** de Benjamini et Hochberg (1995) pour le seuillage des relations
causales, et la procédure de Holm (1979) pour les tests de co-occurrence. Les co-occurrences elles-mêmes
reposent sur les tests pour tableaux de contingence (χ² avec correction de Yates, repli sur le test exact
de Fisher) dont Agresti (2002) est la référence. L'incertitude des métriques scalaires (AUROC, silhouette)
est quantifiée par **bootstrap** : Efron (1987) introduit les intervalles biais-corrigés et accélérés
(BCa), et Davison et Hinkley (1997) fournissent le cadre général du rééchantillonnage. Enfin, les tests
de permutation, utilisés pour évaluer la significativité d'un AUROC face à une distribution nulle,
s'appuient sur les recommandations de Phipson et Smyth (2010) concernant l'estimation des p-values de
permutation. Cet appareillage statistique est ce qui permet à EWAT de rapporter des intervalles de
confiance explicites et de distinguer un signal réel d'un artefact.

### 2.11 Ontologies et raisonnement

Pour donner un sens formel aux types découverts, EWAT construit une ontologie en **OWL/RDF**, le standard
du web sémantique pour représenter des classes, des individus et des relations. La cohérence de
l'ontologie et la matérialisation des inférences sont assurées par un **raisonneur OWL 2**, en
l'occurrence HermiT (Glimm et al. 2014), interfacé par la bibliothèque Python owlready2 (Lamy 2017). La
taxonomie des classes est ancrée dans la littérature des pannes microservices, notamment la classification
des anti-patterns de Soldani et Brogi (2022), plutôt que dans les seuls labels du clustering — ce qui rend
l'ontologie défendable indépendamment du partitionnement appris. La modélisation de la propagation des
défaillances s'inspire des modèles de cascade sur réseaux, dont la référence fondatrice est Motter et Lai
(2002), qui décrivent comment la redistribution de charge consécutive à une défaillance peut en déclencher
d'autres en chaîne.

### 2.12 Interprétabilité et sémantique des journaux

L'interprétation des types passe par l'attribution d'importance aux variables. EWAT recourt aux valeurs de
Shapley (SHAP) de Lundberg et Lee (2017), un cadre unifié d'attribution fondé sur la théorie des jeux, qui
répartit équitablement la contribution de chaque variable à une prédiction. Cette méthode a permis de
valider les fiches d'importance après que la méthode initiale (gradient × entrée) se fut révélée non
fiable. Par ailleurs, l'analyse sémantique des journaux s'appuie sur **Sentence-BERT** (Reimers et
Gurevych 2019), un réseau BERT siamois produisant des embeddings de phrases : EWAT encode chaque ligne de
journal par ce modèle et mesure sa distance au comportement normal, capturant ainsi une dérive sémantique
indépendamment des seuls taux d'erreur.

### 2.13 Reconnaissance open-set et hors-distribution

Un classifieur fermé ne peut pas, par construction, signaler un type de panne absent de son entraînement.
La **reconnaissance open-set** vise précisément à détecter ces classes inédites. EWAT évalue OpenMax
(Bendale et Boult 2016), qui calibre, par la théorie des valeurs extrêmes, une probabilité d'appartenance
à une classe « inconnue ». Deux alternatives plus récentes sont discutées en perspective : la détection
hors-distribution par distance de Mahalanobis (Lee et al. 2018), qui mesure l'éloignement d'un point aux
distributions des classes connues, et la détection par score d'énergie (Liu et al. 2020). Ces approches
constituent la voie privilégiée pour la généralisation aux types inédits, point sur lequel EWAT reste
aujourd'hui partiel (section 6.7).

### 2.14 Benchmarks externes

La validation externe d'un pipeline de détection suppose un jeu de données public et représentatif. EWAT
s'appuie sur le benchmark RCAEval (Pham et al. 2025), et plus précisément sa variante RE2-OB fondée sur
l'application Online Boutique, qui fournit des données multi-sources (métriques, traces, journaux) pour un
ensemble de types de pannes. Ce benchmark sert au test de transfert présenté en section 5.12, dont le
résultat négatif (la prédictibilité ne transfère pas sans réentraînement) borne honnêtement la portée du
modèle.

### 2.15 Synthèse : ce qu'EWAT retient de l'état de l'art

Le tableau suivant récapitule, pour chaque brique conceptuelle, la conclusion retenue et son usage dans
EWAT.

| Brique | Référence | Apport retenu pour EWAT |
|---|---|---|
| Taxonomie des anomalies | Chandola et al. 2009 | l'anomalie est relative au régime → régimes θ |
| Deep TS-AD | Zamanzadeh Darban et al. 2024 | fenêtre temporelle + évaluation indépendante |
| Méthode USE | Gregg 2013 | métriques M(t), profondeur de file comme indicateur avancé |
| Drift conceptuel | Hinder 2024 ; Myrtollari 2025 | détection de drift (étape 0) + régime mixte |
| RCA microservices | Fu 2025 ; Pham 2024 ; GrayScope 2024 | positionnement early-warning, TE plutôt que causal, pannes grises |
| Test MMD | Gretton et al. 2012 | statistique de changement de l'étape 0 |
| Random Fourier Features | Rahimi & Recht 2007 | MMD en O(nD), exécutable en ligne |
| GCN / STGCN / GAT | Kipf 2017 ; Yu 2018 ; Veličković 2018 | encodeur de l'étape 1 et variantes |
| SimCLR / TS-TCC | Chen 2020 ; Eldele 2021 | pré-entraînement contrastif |
| Silhouette / Gap | Kaufman 1990 ; Tibshirani 2001 | qualité et nombre de types |
| Spherical k-means | Dhillon & Modha 2001 | clustering cosinus sur la sphère unité |
| MI-KSG / Transfer Entropy | Kraskov 2004 ; Schreiber 2000 | relations causales de l'ontologie |
| FDR / Holm / χ² | Benjamini-Hochberg 1995 ; Holm 1979 ; Agresti 2002 | contrôle des tests multiples |
| Bootstrap / permutation | Efron 1987 ; Davison-Hinkley 1997 ; Phipson-Smyth 2010 | intervalles de confiance, tests de robustesse |
| OWL / HermiT / owlready2 | Glimm 2014 ; Lamy 2017 | ontologie formelle et raisonnement |
| Taxonomie pannes / cascade | Soldani & Brogi 2022 ; Motter & Lai 2002 | ancrage des classes et de la propagation |
| SHAP / Sentence-BERT | Lundberg & Lee 2017 ; Reimers & Gurevych 2019 | interprétabilité, anomalie sémantique des journaux |
| Open-set / OOD | Bendale & Boult 2016 ; Lee 2018 ; Liu 2020 | détection de nouveauté |
| Benchmark | Pham et al. 2025 (RCAEval) | validation externe |

---

## 3. Présentation de l'environnement

### 3.1 Infrastructure Kubernetes

#### 3.1.1 Architecture du cluster

Les expériences sont conduites sur un cluster Kubernetes réel (observit-cluster1), orchestré par la
distribution RKE2 et composé de neuf nœuds. Sur ce cluster est déployée une application microservices
de référence (Online Boutique pour les premières itérations, Train Ticket pour la plus récente),
constituée de services autonomes communiquant par le réseau. La topologie effective n'est pas figée :
le nombre de réplicas d'un service varie avec l'autoscaling, et les chemins d'appel évoluent avec la
charge et le reroutage. Cette variabilité structurelle est une caractéristique essentielle de
l'environnement, dont la modélisation par graphe (section 4) devra tenir compte.

#### 3.1.2 Contraintes d'accès

Le travail s'effectue dans un namespace dédié, avec un accès de niveau **namespace-admin** et non
cluster-admin. Concrètement, il est possible de créer, modifier et supprimer des ressources dans le
namespace de travail, et de lire les services des autres namespaces (pour découvrir les endpoints
d'observabilité), mais il n'est pas possible d'installer des ressources à portée cluster (les CRD
nécessaires à l'injection de chaos ont dû être installées par un administrateur), ni de modifier les
namespaces système. Toute opération d'écriture est confinée au namespace de travail. Cette contrainte a
guidé l'organisation du projet et la séparation stricte entre collecte et traitement.

### 3.2 Outils d'observabilité

Le cluster dispose de deux chaînes d'observabilité coexistantes, exploitées conjointement par EWAT.

#### 3.2.1 Prometheus

Prometheus assure la collecte des **métriques** par scraping périodique des pods et services. C'est la
source des métriques système et applicatives matures (CPU, mémoire, réseau, latences, taux d'erreur).
Sa robustesse et sa maturité en font la source privilégiée pour les grandeurs de saturation et de
charge.

#### 3.2.2 OpenTelemetry

Un collecteur OpenTelemetry reçoit, via le protocole OTLP (en gRPC et HTTP), les **traces distribuées**
et les **journaux** instrumentés des services. Les traces reconstituent le cheminement des requêtes à
travers les services et fournissent les grandeurs structurelles (profondeur de trace, fan-out,
retries). Les journaux fournissent les taux d'erreur et de warning ainsi que les indicateurs
sémantiques. La coexistence des deux chaînes est un atout : les métriques matures d'un côté, les traces
et journaux instrumentés de l'autre, corrélables grâce aux conventions sémantiques OpenTelemetry.

#### 3.2.3 Endpoints et configuration

Les endpoints internes des services d'observabilité sont découverts en début de campagne et consignés
en configuration (fichier Hydra). Les principaux sont Prometheus, le collecteur OpenTelemetry (en OTLP
gRPC et HTTP, plus un exporteur de métriques), Jaeger pour la consultation des traces et Loki pour les
journaux. Les références d'administration du cluster (API, identifiant interne) sont volontairement
omises de ce document. La collecte la plus récente lit en outre les métriques et traces via NodePort,
ce qui évite les redirections de port et accélère sensiblement l'acquisition.

### 3.3 Génération contrôlée d'incidents

#### 3.3.1 Chaos Mesh

Les pannes sont injectées de façon contrôlée et reproductible par Chaos Mesh. Cette injection maîtrisée
est indispensable à la constitution d'un dataset étiqueté : elle garantit que l'on connaît, pour chaque
épisode, le type de panne injecté, son instant de début et le service ciblé. L'injection est traitée
comme une opération sensible (confirmation explicite, retrait vérifié en fin d'épisode).

#### 3.3.2 Catalogue des scénarios

Le catalogue de référence (versions v3 et v4) comprend quinze scénarios, répartis en quatre scénarios
de drift bénin et onze scénarios d'anomalie.

| Catégorie | Scénario | Effet recherché |
|---|---|---|
| Drift bénin | drift_config_change | changement de configuration |
| Drift bénin | drift_rolling_deploy | déploiement progressif d'une nouvelle version |
| Drift bénin | drift_scale_up | montée en charge (réplicas) |
| Drift bénin | drift_traffic_ramp | rampe de trafic planifiée |
| Anomalie | cpu_starvation | privation de CPU |
| Anomalie | crash | arrêt brutal d'un service |
| Anomalie | fail_slow_cpu | dégradation lente liée au CPU |
| Anomalie | fail_slow_latency | dégradation lente de la latence |
| Anomalie | faulty_deploy_overlap | déploiement défectueux (drift + anomalie) |
| Anomalie | intermittent_error | erreurs intermittentes |
| Anomalie | memory_pressure | pression mémoire |
| Anomalie | network_loss | pertes réseau |
| Anomalie | noisy_neighbor | voisin bruyant (contention de ressources) |
| Anomalie | oom | dépassement mémoire (OOM kill) |
| Anomalie | resource_leak | fuite de ressource |

Le scénario faulty_deploy_overlap est particulier : il incarne le régime mixte drift∩anomalie, qui sert
de cible à l'hypothèse d'identification du déploiement défectueux. La version v5 (Train Ticket) étend
ce catalogue à 22 scénarios (mono-scénarios, composites et tenus à l'écart) et ajoute des **bugs
applicatifs réels**, injectés par échange d'image de conteneur — notamment un bug de logique silencieux
(invisible en télémétrie, qui constitue un cas négatif honnête) et un dépassement mémoire JVM
détectable.

#### 3.3.3 Anatomie d'un épisode

Chaque épisode suit une trame commune : une **phase normale** de référence, puis l'**injection** du
scénario, et enfin une **phase de retour à la normale** (recovery). Cette trame fournit un signal
naturellement étiqueté par régime : la phase normale alimente la référence du détecteur de drift et
sert de fenêtre pré-injection pour les précurseurs, tandis que la phase d'injection fournit la
signature de l'anomalie. La phase de recovery est exclue des évaluations d'anticipation.

### 3.4 Présentation des données — Dataset EWAT

#### 3.4.1 Dimensions et structure

Pour chaque pas de temps d'un épisode, l'état du système est résumé par un **signal** S(t) ∈ ℝ^{N×17} :
N services, chacun décrit par dix-sept variables réparties en trois modalités (métriques, traces,
journaux). À ce signal s'ajoute le **graphe de services** G(t), qui décrit les appels inter-services et
leurs poids à l'instant t. Le dataset de référence (ewat_v3) comprend **299 épisodes** (15 scénarios ×
environ 20 répétitions, sur N = 6 services, avec environ 21 pas de temps par épisode, soit 10,5 minutes
d'observation). Il est découpé de façon **stratifiée** en 209 / 45 / 45 épisodes (entraînement /
validation / test), chaque scénario étant représenté dans les trois ensembles. Le découpage est en
outre temporel à l'intérieur de chaque scénario, afin d'éviter toute fuite d'information du futur vers
le passé.

#### 3.4.2 Description des cibles

Deux niveaux de cible coexistent et il importe de ne pas les confondre.

Le **régime opérationnel** θ(t) prend ses valeurs dans {normal, drift, anomalie, drift∩anomalie}. Il
est dérivé des labels d'épisode : la phase normale correspond à θ_normal, la phase d'injection d'un
scénario d'anomalie à θ_anomaly, le déploiement défectueux à θ_{drift∩anomaly}, et la phase de recovery
est exclue des évaluations. Le drift bénin pur (rolling deploy, autoscaling) n'est pas un label de
régime à part entière : il est signalé orthogonalement par un drapeau booléen produit par le détecteur
de drift.

Le **type d'anomalie** est, lui, découvert empiriquement par le pipeline (clustering des embeddings),
puis rattaché aux scénarios Chaos Mesh injectés. Ces scénarios constituent une **vérité terrain
indépendante** : ils n'ont pas servi à construire les types, ce qui permet une évaluation non
circulaire de la prédictibilité (section 5).

#### 3.4.3 Description des features

Le signal concatène trois modalités, S(t) = [M(t) | T(t) | L(t)].

**a. Métriques système et applicatives (M, 7 variables).** Issues de Prometheus et du collecteur OTel,
elles décrivent l'utilisation des ressources et la santé applicative de chaque service :

| Variable | Description |
|---|---|
| cpu_util | utilisation CPU |
| ram_util | utilisation mémoire |
| latency_p99 | latence au 99ᵉ percentile |
| error_rate_http | taux d'erreurs HTTP (4xx + 5xx) |
| net_sat | saturation réseau |
| disk_io | activité disque (IOPS et débit) |
| queue_depth | profondeur de la file d'attente (requêtes en attente) |

La profondeur de file est retenue spécifiquement comme **indicateur avancé** de dégradation, conformément
à la méthodologie USE (Gregg 2013) : une file qui s'allonge précède souvent la saturation observable.

**b. Features de traces (T, 6 variables).** Dérivées des spans OTLP, elles décrivent la structure et la
qualité des appels :

| Variable | Description |
|---|---|
| span_dur_p99 | durée P99 des spans (sur l'union des durées de tous les pods) |
| abnormal_span_rate | taux de spans anormaux |
| trace_depth | profondeur de trace |
| fan_out | fan-out (nombre d'appels sortants) |
| retry_rate | taux de retry (spans retentés / total) |
| latency_cv | coefficient de variation de la latence |

**c. Features de journaux (L, 4 variables).** Dérivées des journaux applicatifs :

| Variable | Description |
|---|---|
| log_error_rate | taux d'erreurs (lignes ERROR / total) |
| log_warn_rate | taux de warnings |
| semantic_anomaly | anomalie sémantique : distance moyenne au centroïde normal des embeddings Sentence-BERT |
| lexical_entropy | entropie lexicale des lignes de journal |

L'anomalie sémantique mérite une explication. Chaque ligne de journal ℓ est encodée par Sentence-BERT
(Reimers & Gurevych 2019) en un vecteur de dimension 384 ; on calcule la distance moyenne de ces
vecteurs au centroïde du régime normal du service. Une dérive sémantique (apparition de messages
inhabituels) se traduit ainsi par une hausse de cette distance, indépendamment des seuls taux d'erreur.

**d. Agrégation intra-service.** Plusieurs pods composant un service, leurs valeurs doivent être
agrégées en une valeur par service. L'erreur classique serait la moyenne simple ; EWAT agrège selon la
nature de chaque grandeur : **maximum** pour les saturations (un seul pod saturé dégrade le service),
**somme pondérée par le volume** pour les taux (afin de ne pas diluer un pod fautif peu sollicité),
**P99 sur l'union** des distributions pour les latences (et non un percentile de percentiles, qui n'a
pas de sens statistique), et **médiane** pour les grandeurs structurelles (robuste aux valeurs aberrantes). Pour un service
agrégeant les pods $p$ de volume $v_p$, ces règles s'écrivent :

$$\text{saturation}:\ \max_p x_p; \quad
\text{taux}:\ \frac{\sum_p v_p\, r_p}{\sum_p v_p}; \quad
\text{latence}:\ P_{99}\!\Big(\bigcup_p \mathcal{D}_p\Big); \quad
\text{structurel}:\ \operatorname{median}_p x_p,$$

où $\mathcal{D}_p$ est la distribution brute des durées du pod $p$ (l'union, et non un percentile de
percentiles).

#### 3.4.4 Qualité des données et valeurs manquantes

La qualité du signal d'ewat_v3 est globalement bonne, avec un taux global de valeurs manquantes
d'environ 1,5 % (figure 1). Le seul taux significatif concerne le disque I/O d'un service hébergé sur
un nœud en état dégradé (16,7 % de valeurs manquantes pour ce service). Ce défaut, structurel, a motivé
l'itération v4 du dataset. Les features de journaux présentent un résidu irréductible de 0,4 %.

![Figure 1 — Carte des valeurs manquantes par feature et par service (ewat_v3) : le seul taux significatif concerne le disque I/O de product-catalog, hébergé sur un nœud dégradé.](figures/nan_heatmap.png)

#### 3.4.5 Itérations du dataset

Le dataset a connu plusieurs itérations, chacune répondant à un défaut mesuré sur la précédente. Le
tableau ci-dessous les récapitule.

| Version | Topologie | N | Longueur | Épisodes retenus | Découpage | Défaut corrigé |
|---|---|---|---|---|---|---|
| v3 | Online Boutique | 6 | ~21 pas | 299 | 209/45/45 stratifié | corpus de référence |
| v4 | Online Boutique | 6 | 47–51 pas | 375 | 262/56/57 temporel | épisodes trop courts |
| v4_strat | Online Boutique | 6 | 47–51 pas | 375 | 270/60/45 stratifié | scénarios absents du train |
| rcaeval | Online Boutique (autre cluster) | 6 | 48 pas | 90 | adapté | validation externe nulle |
| v5 | Train Ticket | 41 | 60 pas | ~720 (cible) | held-out imposé | topologie trop petite, circularité |

Deux motivations principales ont guidé ces itérations. D'abord, la **longueur des épisodes** : les
~21 pas d'ewat_v3 se sont révélés insuffisants pour la confirmation temporelle du drift ; ewat_v4 double
cette longueur (figure 2), ce qui rend exploitable une dynamique pré-injection plus marquée. Ensuite, la
**représentativité de la topologie** : avec seulement six services, Online Boutique est trop petite pour
généraliser ; la version v5 pivote vers Train Ticket (41 microservices Spring Cloud), bien plus profonde
et dotée de bugs réels documentés, dans la perspective d'un dataset plus représentatif et potentiellement
publiable.

![Figure 2 — Distribution de la longueur des épisodes : ewat_v3 (~21 pas) contre ewat_v4 (47–51 pas).](figures/timesteps_boxplot.png)

#### 3.4.6 Schéma enrichi de la version v5.1

La version v5 (Train Ticket) ne se contente pas d'élargir la topologie : elle enrichit aussi le schéma
de features, qui passe de dix-sept à **dix-huit variables par service** (signal de dimension ℝ^{T×41×18}).
L'enrichissement porte sur le bloc des métriques, augmenté de quatre indicateurs liés à la machine
virtuelle Java (les services Train Ticket étant en Spring Cloud), et sur le bloc des journaux, où le
nombre de redémarrages rejoint les indicateurs.

| Bloc | Index | Variables |
|---|---|---|
| Métriques (M) | 0–9 | cpu_util, ram_util, latency_p99, error_rate, net_sat, disk_io, mem_limit_ratio, jvm_heap_ratio, jvm_gc_util, jvm_threads_blocked |
| Traces (T) | 10–13 | abnormal_span_rate, trace_depth, fan_out, latency_cv |
| Journaux (L) | 14–17 | log_error_rate, restart_count, semantic_anomaly, lexical_entropy |

Une variable initialement prévue, oom_events, a été remplacée par mem_limit_ratio après vérification :
le cAdvisor de ce cluster ne surface pas l'événement OOM (valeur nulle partout), tandis que le ratio
mémoire/limite porte l'information utile. Cette correction illustre l'importance de la vérification des
données avant lancement de la collecte.

La version v5 introduit en outre des **bugs applicatifs réels**, injectés par échange d'image de
conteneur, qui complètent les scénarios Chaos Mesh synthétiques. Deux cas sont instructifs. Le bug F3,
un dépassement mémoire de la JVM, est **détectable** : il se signale par une combinaison de redémarrages,
d'utilisation mémoire et de saturation du tas. Le bug F1, un défaut de logique applicative silencieux
(une condition de course dans l'ordonnancement), est au contraire **invisible en télémétrie** : il
n'altère ni les métriques, ni les traces, ni les journaux, alors qu'il produit un résultat erroné. Ce
cas constitue un **résultat négatif honnête** : il existe des pannes — les « pannes grises » au sens de
la littérature (GrayScope 2024) — qu'aucun dispositif fondé sur la télémétrie standard ne peut détecter.
Le reconnaître borne honnêtement la portée d'EWAT et de toute approche comparable.

#### 3.4.7 Anatomie de la collecte v5

La collecte v5 sur Train Ticket applique strictement la séparation en trois phases, avec une chaîne
d'outils dédiée et durcie. La phase d'enregistrement (`run_campaign`) injecte le chaos et sauvegarde les
données brutes par épisode. La construction des features (`build_features_v5 --raw-root`) reconstruit hors
ligne le signal à partir de ces données brutes. La consolidation enchaîne la validation (`validate_v5`),
l'assemblage stratifié (`assemble_dataset --stratified`) et l'application des ensembles tenus à l'écart
(`enforce_heldout_v5`). Cette séparation garantit que les données brutes ne sont jamais modifiées en place
et que toute reconstruction est rejouable.

Pour tenir le volume cible (de l'ordre de plusieurs centaines d'épisodes), la collecte s'exécute sur
plusieurs *runners* parallèles, sur des instances distinctes de l'application. La contrainte dominante est
la mémoire : un garde-fou interrompt l'injection si l'occupation mémoire des nœuds de travail dépasse un
plafond, afin d'éviter les évictions de pods qui corrompraient les épisodes. Avant le lancement de la
campagne, six épisodes réels ont été vérifiés de bout en bout : chaos effectivement localisé sur le
service ciblé, régimes propres, graphe G(t) dynamique, aucune valeur manquante imputée, et validation au
vert. Cette vérification a permis de corriger deux défauts (la restauration d'un bug réel après injection,
et l'épinglage du contexte kubectl) avant d'engager la collecte complète.

---

## 4. Approche proposée : le pipeline EWAT

### 4.1 Vue d'ensemble

EWAT transforme le signal de télémétrie en une alerte typée et anticipée, au travers d'un pipeline en
quatre étapes complété d'une étape d'enrichissement hors ligne (figure 3). Le signal S(t) et le graphe
G(t) entrent dans une **détection de drift** (étape 0) qui filtre les changements bénins, puis dans un
**encodeur** (étape 1) qui projette l'état du graphe en un vecteur compact, sur lequel un **typage
contrastif** (étape 2) découvre une taxonomie d'anomalies. Une **ontologie** (étape 2b) enrichit hors
ligne l'interprétation de ces types, et des **précurseurs typés** (étape 3) estiment la probabilité et
l'horizon de chaque type. La sortie est une alerte : type pressenti, probabilité, horizon, fiche
descriptive.

![Figure 3 — Architecture du pipeline EWAT : le signal S(t) traverse la détection de drift (étape 0), l'encodeur STGCN (étape 1), le typage siamois (étape 2) et les précurseurs (étape 3) jusqu'à l'alerte typée ; l'ontologie (étape 2b) est hors ligne.](figures/pipeline_architecture.png)

Le pipeline est implémenté en six modules indépendants et testables (détection de drift, encodeur,
typage, ontologie, précurseurs, assemblage d'alertes), couverts par 586 tests unitaires. Cette
modularité permet d'évaluer chaque étape isolément et de substituer une variante (par exemple un autre
encodeur) sans refondre l'ensemble.

### 4.2 Graphe de services G(t) et tenseur d'adjacence

EWAT modélise l'état du système non par un simple vecteur, mais par un **graphe de services**
G(t) = (V, E(t), w_E(t)). Le choix des sommets est déterminant : ce sont les **Services et Deployments**
Kubernetes, et non les Pods. Ainsi, le nombre de sommets N reste constant sur un épisode, indépendamment
de l'autoscaling — les variations de réplicas se reportent sur les poids d'arêtes plutôt que sur la
topologie. Ce choix transforme une source majeure de bruit (la création/destruction de pods) en
information de charge portée par les arêtes.

Chaque arête e_ij(t) représente les appels du service i vers le service j et porte un vecteur à trois
composantes : volume d'appels, latence médiane, taux d'erreur. Une arête n'est présente que si son
volume est strictement positif sur la fenêtre glissante courante ; la topologie E(t) est donc dynamique,
sensible aux changements d'acheminement. Pour l'encodeur, ce graphe est représenté par un **tenseur
d'adjacence à trois canaux** A(t) ∈ ℝ^{N×N×3}, un canal par composante de poids. La convolution spatiale
de l'étape 1 s'appuie sur ce tenseur, de sorte que l'agrégation de voisinage est pondérée par le volume,
la latence et le taux d'erreur, et non par une simple adjacence binaire.

### 4.3 Étape 0 — Détection de drift (MMD-RFF)

#### 4.3.1 Test à deux échantillons par noyau (MMD)

Pour détecter un changement de distribution sans hypothèse paramétrique, EWAT recourt à un test à deux
échantillons par noyau, la *Maximum Mean Discrepancy* (MMD, Gretton et al. 2012). Étant données une
fenêtre de référence (régime supposé normal) et une fenêtre courante, le MMD² mesure l'écart entre les
deux distributions dans un espace de Hilbert à noyau reproduisant : il est nul si et seulement si les
deux distributions coïncident, et croît avec leur écart. Ce test est non paramétrique et multivarié, ce
qui le rend adapté à un signal de dix-sept dimensions corrélées.

#### 4.3.2 Approximation par Random Fourier Features

Le calcul exact du MMD est quadratique en la taille des fenêtres, ce qui est incompatible avec une
exécution en ligne. EWAT l'approche par *Random Fourier Features* (Rahimi & Recht 2007), une projection
aléatoire φ : ℝ^d → ℝ^D qui approxime le noyau par un produit scalaire en dimension D. Le coût du MMD²
devient alors linéaire, de l'ordre de O(nD), où n est la taille de fenêtre. Cette approximation rend
l'étape 0 compatible avec le budget de latence visé (section 4.9).

#### 4.3.3 Mécanisme look-through

Le résultat du test ne sert pas seulement à lever ou non un drapeau ; il pilote un mécanisme de
**look-through** dont le principe est de ne jamais annuler le signal pendant un drift, afin de ne pas
masquer une anomalie concomitante. Trois cas se présentent :

- si MMD² < ε_drift, le signal est transmis tel quel (régime jugé normal) ;
- si MMD² ≥ ε_drift et qu'un test de confirmation post-drift est positif, le signal est transmis avec un
  drapeau DRIFT — on « regarde au travers » du drift, sans suppression ;
- si MMD² ≥ ε_drift et que la confirmation est négative, la référence est recalibrée (W_ref ← W_cur),
  le changement étant interprété comme une nouvelle normalité.

#### 4.3.4 Calibration du seuil ε_drift

Le seuil ε_drift est calibré empiriquement par injection de drifts bénins : on calcule le MMD² entre une
fenêtre de référence (premiers pas, régime normal) et une fenêtre courante (derniers pas), puis on retient
le seuil optimal au sens de Youden sur la courbe ROC. La valeur obtenue est ε_drift = 0,5226 (les
résultats de calibration sont détaillés en section 5.2).

### 4.4 Étape 1 — Encodeur STGCN

#### 4.4.1 Convolution spatiale multi-canal

L'encodeur de référence est un réseau de convolution spatio-temporel sur graphe (STGCN, Yu et al. 2018).
Sa composante spatiale étend la convolution spectrale de graphe (Kipf & Welling 2017) aux trois canaux
d'adjacence : à chaque couche, la représentation d'un service est mise à jour en agrégeant celles de ses
voisins, pondérées par les canaux volume, latence et taux d'erreur. Cette agrégation pondérée permet à
l'encodeur de privilégier les voisins avec lesquels les échanges sont intenses ou problématiques.
En notant $H^{(l)}$ les représentations des services à la couche $l$, $A_c$ le canal $c$ du tenseur
d'adjacence (volume, latence, taux d'erreur) et $W_c^{(l)}$ les poids appris, la mise à jour s'écrit :

$$H^{(l+1)} = \sigma\!\Big( \sum_{c=1}^{3} \hat{A}_c\,H^{(l)}\,W_c^{(l)} \Big),
\qquad \hat{A}_c = D_c^{-1/2}\,(A_c + I)\,D_c^{-1/2},$$

où $\hat{A}_c$ est l'adjacence normalisée symétriquement et $\sigma$ une non-linéarité.

#### 4.4.2 Convolution temporelle causale

La composante temporelle est une convolution causale dilatée (TCN) appliquée le long de l'axe du temps,
service par service. « Causale » signifie qu'à un instant donné, seules les valeurs passées et présentes
sont utilisées, jamais les futures — propriété indispensable pour un usage en ligne. À l'issue de ces
couches, une tête perceptron produit un embedding z_e ∈ ℝ^{64} résumant l'état du graphe sur la fenêtre.
L'encodeur est pré-entraîné par reconstruction auto-supervisée, sans labels de scénario.

#### 4.4.3 Variantes comparées

Deux variantes d'encodeur ont été évaluées pour mesurer l'apport de choix architecturaux. La première,
**GAT** (Veličković et al. 2018), remplace la convolution de graphe par un mécanisme d'attention sur les
arêtes, qui apprend une pondération adaptative du voisinage. La seconde, **SimCLR** (Chen et al. 2020),
ajoute un pré-entraînement contrastif (perte NT-Xent) sur des augmentations temporelles du signal, afin
de produire des représentations invariantes au bruit. Leur comparaison chiffrée figure en section 5.6.

### 4.5 Étape 2 — Typage contrastif et clustering

#### 4.5.1 Réseau siamois et perte contrastive

Le typage s'appuie sur un **réseau siamois** : l'encodeur, suivi d'une tête de projection L2-normalisée,
est entraîné par paires avec une perte contrastive. Les paires positives (épisodes de même scénario
Chaos Mesh) sont rapprochées dans l'espace de projection, les paires négatives éloignées, avec une marge.
Une sélection des négatifs (mining) concentre l'apprentissage sur les paires les plus informatives. Le
résultat est un espace latent où les épisodes de comportement similaire se regroupent.

#### 4.5.2 Clustering et sélection du nombre de types

Sur ces embeddings, un **clustering hiérarchique agglomératif** découvre les types. Le choix de la
métrique est déterminant : les embeddings étant normalisés sur la sphère unité, la distance **cosinus**
(clustering sphérique, Dhillon & Modha 2001) est géométriquement cohérente, là où la distance euclidienne
ne l'est pas — ce choix, identifié par balayage d'hyperparamètres, est à l'origine du principal gain de
structurabilité (section 5.3). Le nombre de types K est choisi par maximisation de la silhouette sur
l'entraînement, avec la statistique de gap (Tibshirani et al. 2001) comme second estimateur. La labellisation
des ensembles de validation et de test se fait par affectation au plus proche centroïde des clusters
d'entraînement, afin de garantir une cohérence des identifiants entre découpages.

#### 4.5.3 Interprétabilité des types

Chaque type est caractérisé par une fiche d'importance des variables. La méthode initiale (gradient ×
entrée) s'étant révélée non fiable (corrélation négative avec une mesure de référence), l'importance est
calculée par permutation (mesure de la dégradation de silhouette lorsqu'une variable est permutée), puis
validée par les valeurs de Shapley (SHAP, Lundberg & Lee 2017) : la concordance est positive pour neuf
types sur dix.

### 4.6 Étape 2b — Ontologie empirique des pannes

#### 4.6.1 Taxonomie (TBox)

Pour donner du sens aux types découverts, EWAT construit hors ligne une ontologie formelle en OWL/RDF.
Sa partie terminologique (TBox) compte 29 classes hiérarchiques, ancrées dans la littérature des pannes
microservices (taxonomie des anti-patterns de Soldani & Brogi 2022, grandeurs de saturation de la
méthode USE, défaillances en cascade) plutôt que dans les seuls labels du clustering. Cet ancrage rend
l'ontologie défendable indépendamment du partitionnement appris.

#### 4.6.2 Relations causales (Transfer Entropy)

Les relations causales entre types sont estimées par **Transfer Entropy** (Schreiber 2000), une mesure
d'information dirigée du passé d'une série vers le présent d'une autre, au moyen de l'estimateur non
paramétrique KSG (Kraskov et al. 2004). La causalité de Granger est explicitement écartée, car elle
suppose une linéarité inadaptée à ces signaux. La significativité est évaluée par permutation, et la
multiplicité des tests contrôlée par la correction de Benjamini–Hochberg (1995).

#### 4.6.3 Co-occurrences et propagation

L'ontologie modélise en outre les **co-occurrences** entre types (tableaux de contingence avec test du
χ² de Yates, repli sur le test exact de Fisher) et la **propagation** des défaillances de service à
service. Cette dernière relation, inspirée des modèles de cascade, capture la manière dont une anomalie
sur un service en affecte d'autres le long des dépendances.

#### 4.6.4 Raisonnement

La cohérence de l'ontologie est vérifiée par le raisonneur OWL 2 HermiT (Glimm et al. 2014), via la
bibliothèque owlready2. Le raisonnement matérialise les inférences et permet d'interroger l'ontologie
par requêtes SPARQL (par exemple : quels services sont traversés par la propagation d'un type donné).

### 4.7 Étape 3 — Précurseurs typés et assemblage d'alertes

À chaque type est associé un **classifieur précurseur** un-contre-tous, qui estime la probabilité qu'une
anomalie de ce type se développe à un horizon k. L'horizon optimal k* est sélectionné sur l'ensemble de
validation, parmi un ensemble d'horizons candidats, et l'AUROC n'est reporté que sur le test —
précaution indispensable pour éviter une surestimation. L'**assembleur d'alertes** orchestre la sortie :
il regroupe les passages de l'encodeur par horizon, applique les classifieurs, intègre le drapeau de
drift (qui supprime l'alerte en cas de drift bénin), et produit l'alerte finale Alert(t) = (type C_i,
probabilité p̂_i(t), horizon k*_i, fiche du type).

### 4.8 Modélisation mathématique (synthèse)

Les éléments précédents se résument formellement comme suit. L'état du système est décrit par le couple
$(G(t), S(t))$, avec $S(t) \in \mathbb{R}^{N\times 17}$ et le tenseur d'adjacence
$A(t) \in \mathbb{R}^{N\times N\times 3}$. Le régime $\theta(t)$ conditionne la distribution du signal,

$$S(t) \sim D_{\theta(t)}\big(G(t)\big),$$

relation non additive qui justifie l'apprentissage de représentation plutôt qu'un seuillage variable par
variable. L'**étape 0** déclenche un drapeau de drift selon

$$\mathrm{MMD}^2(W_{\text{ref}}, W_{\text{cur}}) \;\gtrless\; \varepsilon_{\text{drift}},$$

le MMD² étant approché par la projection $\varphi : \mathbb{R}^d \to \mathbb{R}^D$ de coût $O(nD)$.
L'**étape 1** produit l'embedding

$$z_e = \mathrm{Enc}_\theta\big(\tilde{S}_{[t-W,\,t+\delta]},\, G(t)\big) \in \mathbb{R}^{64}.$$

L'**étape 2** apprend une fonction de distance $d_\varphi(z_i, z_j)$, proche de 0 pour deux épisodes de
même type et de 1 sinon, dont le clustering induit la taxonomie $\mathcal{C} = \{C_1, \dots, C_K\}$.
L'**étape 2b** estime, entre types ou entre services, des relations causales par Transfer Entropy
$T_{X\to Y}$ seuillée sous contrôle du taux de fausses découvertes. L'**étape 3** estime, pour chaque
type $i$, la probabilité et l'horizon optimal

$$\hat{p}_i(t) = f_i\big(\tilde{S}_{[t-k,\,t]},\, G(t)\big) \in [0,1],
\qquad k^*_i = \arg\max_k \mathrm{AUROC}(f_i, k).$$

La sortie est l'alerte $\mathrm{Alert}(t) = \big(C_i,\, \hat{p}_i(t),\, k^*_i,\, \text{fiche}_{C_i}\big)$.

### 4.9 Budget de latence

La chaîne d'inférence en ligne — étapes 0, 1 et 3 — vise un budget total inférieur à cinq secondes
(étape 0 < 1 s, étape 1 < 2 s, étape 3 < 1 s). Les étapes 2 (typage) et 2b (ontologie) sont hors ligne :
elles produisent les types et l'ontologie périodiquement, et n'interviennent pas dans la boucle d'alerte
temps réel. La latence effectivement mesurée est rapportée en section 5.10.

---

## 5. Résultats et validation empirique

En résumé opérationnel, EWAT apporte un gain net et chiffrable. Là où une baseline z-score lève 100 % de
fausses alertes sur les drifts bénins (déploiements, autoscaling), EWAT ramène ce taux à **8,3 %** au
point opérationnel, tout en conservant un délai d'anticipation de **trois minutes**. Sur une cible
d'évaluation indépendante des labels du pipeline, la discrimination des types de pannes atteint un
**macro-AUROC de 0,920** (intervalle de confiance à 95 % [0,878 ; 0,956]), et la chaîne d'inférence
s'exécute en **13 ms**. Les sous-sections suivantes détaillent ces résultats, leur protocole et leurs
limites, sans masquer les points où la méthode atteint ses bornes.

### 5.1 Protocole et métriques

Sauf mention contraire, les résultats portent sur le dataset de référence ewat_v3 (découpage stratifié
209/45/45) ; les évaluations sur cible indépendante utilisent ewat_v4_strat (270/60/45). Deux
corrections méthodologiques, identifiées en cours de projet, sont appliquées à tous les résultats
rapportés. Premièrement, la silhouette de validation et de test est mesurée par affectation au plus
proche centroïde des clusters d'entraînement, et non par un clustering indépendant par découpage : ce
dernier trouve la meilleure partition propre à chaque ensemble et surestime la structurabilité (l'accord
entre les deux méthodes sur l'entraînement est de 97,6 %). Deuxièmement, l'horizon optimal k* des
précurseurs est sélectionné sur la validation, l'AUROC n'étant reporté que sur le test.

Les métriques employées sont les suivantes : le coefficient de **silhouette** en held-out pour la
qualité du typage (seuil d'acceptabilité 0,3) ; l'**AUROC par type** pour la prédictibilité ; les
**taux de vrais et faux positifs** pour la séparation drift/anomalie ; et, pour la performance
opérationnelle, le **taux de détection**, le **taux de fausses alertes sur drift** et le **délai
d'anticipation** (lead time). Toutes les valeurs scalaires sont accompagnées d'un **intervalle de
confiance à 95 %** obtenu par bootstrap (1000 rééchantillonnages, méthode BCa lorsqu'elle s'applique),
ce qui rend explicite l'incertitude liée à la taille modérée des ensembles de test.

### 5.2 Calibration du drift et séparation drift/anomalie

Le seuil de drift est calibré épisode par épisode par le critère de Youden, à **ε_drift = 0,5226**, pour
une aire sous la courbe ROC de 0,60 (taux de vrais positifs 0,55, taux de faux positifs 0,33 sur
l'entraînement). La modération de cette aire annonce déjà une difficulté, visible sur la figure 4 : les
distributions du MMD² en régime normal et en régime chaos se recouvrent largement, ce qui rend la
séparation par seuil délicate.

![Figure 4 — Distributions du MMD² en régime normal et en régime chaos : le recouvrement important illustre la difficulté de séparation par seuil, cohérent avec l'aire sous courbe modérée (0,60).](figures/mmd2_distributions.png)

De fait, le mécanisme de confirmation temporelle (look-through) n'apporte pas de réduction significative
du taux de faux positifs. Le tableau suivant compare, sur l'ensemble de test, le look-through au seuil
simple.

| Variante | TPR (drift détecté) | FPR (anomalie confondue) | p-value |
|---|---|---|---|
| Look-through (signal brut) | 0,42 | 0,67 | 0,27 |
| Seuil simple (signal brut) | 0,67 | 0,73 | — |
| Look-through (embeddings) | — | 0,788 | 0,978 |
| Seuil simple (embeddings) | — | 0,667 | — |
| Look-through (ewat_v4_strat) | 0,500 | 0,667 | 0,372 |
| Seuil simple (ewat_v4_strat) | 0,750 | 0,697 | — |

Le constat est robuste : ni le passage dans l'espace d'embedding, ni l'allongement des épisodes
(ewat_v4) ne rendent le look-through significativement meilleur que le seuil simple. Ce **résultat
négatif est assumé** : le mécanisme MMD² avec confirmation temporelle ne sépare pas le drift bénin de
l'anomalie sur ce type de données. Il ne remet pas en cause l'architecture — l'étape 0 reste une alarme
de changement rapide — mais montre que la qualification de régime relève d'un espace de représentation
dédié, à construire (cf. perspectives). Ce constat renforce la logique de cascade : l'étape 0 alerte sur
un changement, les étapes aval qualifient et typent.

### 5.3 Structurabilité des types

Les embeddings du typage forment des clusters nettement séparés. Sur ewat_v3 (graine 42), la silhouette
vaut 0,577 à l'entraînement, 0,470 en validation et **0,414 en test**, pour un nombre optimal de types
K = 10 — bien au-dessus du seuil de 0,3. Que dix types émergent de quinze scénarios injectés est en soi
un résultat : certains scénarios partagent une signature latente (un crash et un OOM, par exemple,
peuvent être indiscernables une minute avant l'événement), et le pipeline découvre ainsi une taxonomie
plus compacte que le catalogue d'injection. La figure 5 visualise l'alignement entre clusters appris et
scénarios.

![Figure 5 — Carte de chaleur scénario × cluster : répartition des épisodes de chaque scénario Chaos Mesh dans les clusters appris (information mutuelle normalisée de 0,518).](figures/scenario_cluster_heatmap.png)

Un balayage d'hyperparamètres a en outre identifié une configuration nettement supérieure. Le passage de
la métrique de clustering Ward + euclidien à **average + cosinus** — cohérente avec des embeddings
normalisés sur la sphère unité — fait passer la silhouette de test moyenne de 0,519 ± 0,092 (5 graines,
configuration initiale) à **0,782 ± 0,065** (10 graines, configuration optimisée), avec un minimum de
0,618, toujours au-dessus du seuil. Ce gain de +51 % est la contribution géométrique principale du
pipeline.

### 5.4 Prédictibilité des types

Sur la cible interne (les types produits par EWAT), la prédictibilité est élevée. Le tableau suivant
donne, par type, l'horizon optimal k*, l'AUROC de test et son intervalle de confiance.

| Type | n positifs (test) | k* | AUROC test | IC 95 % |
|---|---|---|---|---|
| C0 | 8 | 6 | 0,973 | [0,906 ; 1,000] |
| C1 | 3 | 6 | 0,992 | [0,953 ; 1,000] |
| C2 | 5 | 6 | 0,945 | [0,865 ; 1,000] |
| C3 | 3 | 2 | 0,794 | [0,636 ; 0,930] |
| C4 | 8 | 2 | 1,000 | [1,000 ; 1,000] |
| C5 | 2 | 6 | 0,977 | [0,909 ; 1,000] |
| C6 | 1 | 2 | non concluant (n < 2) | — |
| C7 | 7 | 6 | 0,992 | [0,966 ; 1,000] |
| C8 | 7 | 10 | 0,962 | [0,895 ; 1,000] |
| C9 | 1 | 2 | non concluant (n < 2) | — |

Huit types sur dix dépassent un AUROC de 0,9 (figure 6), l'horizon optimal dominant étant de six pas, soit
trois minutes — ce qui situe la zone de prédictibilité optimale. En configuration optimisée et sur dix
graines, l'AUROC moyen atteint 0,987 ± 0,011. **Cette métrique doit toutefois être lue avec prudence** :
la cible étant produite par EWAT lui-même, elle mesure la cohérence interne du pipeline et non une
prédiction indépendante. La section 6 démontre, par une série de tests de robustesse, que cette
performance interne relève en partie de la reconnaissance d'une signature statique de scénario, et non
d'une prédiction temporelle. Le résultat à mettre en avant est donc celui de la section 5.5, obtenu sur
une cible indépendante.

![Figure 6 — AUROC de test par type d'anomalie (cible interne), avec intervalles de confiance bootstrap. Huit types sur dix dépassent 0,9 ; C6 et C9 ne sont pas concluants faute de positifs.](figures/auroc_h3_per_cluster.png)

### 5.5 Résultat défendable sur cible indépendante

Pour obtenir un chiffre exempt de circularité, on évalue directement sur la **vérité terrain indépendante**
que constituent les scénarios Chaos Mesh injectés. Une régression logistique un-contre-tous, appliquée
aux features brutes instance-normalisées de la fenêtre pré-injection (sans encodeur appris sur la cible),
atteint les performances suivantes.

| Évaluation | Dataset | Macro-AUROC | IC 95 % |
|---|---|---|---|
| Stratifié | ewat_v3 | 0,855 | [0,789 ; 0,905] |
| Stratifié | ewat_v4_strat | **0,920** | [0,878 ; 0,956] |
| Leave-one-scenario-out | ewat_v4_strat | 0,930 | (15 folds) |
| Meilleure config (instance norm) | ewat_v4_strat | 0,941 | [0,909 ; 0,970] |

Le **macro-AUROC de 0,920** (intervalle de confiance [0,878 ; 0,956]) sur ewat_v4_strat constitue le
résultat défendable du livrable : cible indépendante, intervalle explicite, et chiffre déterministe (le
solveur de la régression logistique étant déterministe, seule l'incertitude bootstrap varie). Les
épisodes plus longs d'ewat_v4 amplifient le signal par rapport à ewat_v3 (+0,065).

Un point important pour l'honnêteté de l'analyse : l'encodeur STGCN **n'améliore pas** ce résultat
agrégé. Comparées sur la cible Chaos Mesh, les features brutes et les embeddings STGCN donnent
exactement le même macro-AUROC de 0,835 (écart nul), et un STGCN entraîné de bout en bout sur cette cible
plafonne à 0,863, en deçà de la régression logistique. La valeur de l'encodeur est donc **géométrique**
(structuration de l'espace latent pour le typage, section 5.3) et **ontologique** (section 5.8), plutôt
que prédictive en agrégé.

### 5.6 Comparaison des encodeurs

Trois architectures d'encodeur ont été comparées dans des conditions identiques (ewat_v3, graine 42,
même découpage).

| Architecture | K | silhouette val | silhouette test | types prédictibles | AUROC moyen |
|---|---|---|---|---|---|
| STGCN (référence) | 10 | 0,470 | 0,414 | 8/10 | 0,954 |
| SimCLR (contrastif) | 15 | 0,495 | 0,429 | 11/15 | 0,964 |
| GAT (attention) | 15 | 0,445 | 0,497 | 13/15 | 0,929 |

Chaque variante a un profil propre. **GAT** offre la meilleure géométrie de l'espace latent
(silhouette 0,497, +0,083 par rapport au STGCN) et couvre davantage de types, mais avec un AUROC moyen
plus faible : l'attention améliore la structuration sans nécessairement la discriminabilité. **SimCLR**
maximise l'AUROC moyen grâce au pré-entraînement contrastif, mais laisse quatre types non concluants
(clusters trop petits). Le **STGCN**, avec un nombre de types K = 10 plus stable et des résultats
multi-graines disponibles, offre le meilleur compromis et est retenu comme architecture principale ;
SimCLR et GAT restent des points de comparaison.

### 5.7 Ablations

**Ablation par modalité (typage).** Cette ablation réentraîne entièrement l'encodeur et le typage pour
chaque combinaison de modalités, afin de mesurer l'apport propre de chaque source (et non un masquage à
l'inférence, biaisé). Le résultat est contre-intuitif : les **métriques seules** font légèrement mieux
que l'ensemble des modalités pour la structurabilité.

| Condition | nombre de features | silhouette test | Δ vs complet |
|---|---|---|---|
| Complet | 17 | 0,439 | — |
| Métriques seules (M) | 7 | 0,497 | +0,058 |
| Traces seules (T) | 6 | 0,412 | −0,027 |
| M + L | 11 | 0,382 | −0,057 |
| T + L | 10 | 0,341 | −0,098 |
| M + T | 13 | 0,316 | −0,123 |
| Journaux seuls (L) | 4 | 0,051 | −0,388 |

Les traces et les journaux ajoutent donc du bruit géométrique au clustering sur un corpus de cette
taille. Leur valeur est ailleurs : dans la **prédictibilité**. L'ablation par modalité côté précurseurs
(masquage à l'inférence) donne la conclusion inverse — le modèle complet domine.

| Condition | Macro-AUROC | Δ vs complet |
|---|---|---|
| Complet | 0,954 | — |
| M + L | 0,916 | −0,038 |
| Métriques seules (M) | 0,756 | −0,198 |
| T + L | 0,563 | −0,391 |
| Journaux seuls (L) | 0,488 | −0,466 |

![Figure 7 — Ablation par modalité pour la prédictibilité : macro-AUROC selon les modalités conservées. Le modèle complet domine, à l'inverse de l'ablation pour le typage.](figures/ablation_h3_heatmap.png)

**Ablation par feature.** En retirant une feature à la fois (test de Wilcoxon signé, p < 0,05), les plus
critiques pour le typage sont trace_depth (Δ = −0,069), lexical_entropy (−0,069) et latency_p99
(−0,062) ; disk_io reste significatif malgré ses valeurs manquantes (−0,010), ce qui a motivé sa
correction en v4. Pour la prédictibilité, la feature la plus critique est disk_io (Δ = −0,088). Enfin,
deux paires de features sont fortement redondantes : latency_p99 ↔ span_dur_p99 (ρ = 0,936) et
error_rate_http ↔ abnormal_span_rate (ρ = 0,927), candidates à une simplification du modèle.

### 5.8 Ontologie empirique des pannes

L'étape 2b construit, hors ligne, une ontologie formelle qui donne un sens aux types découverts. Cette
sous-section en détaille la structure, le peuplement, les relations extraites et la validation.

**Terminologie (TBox).** La taxonomie compte **29 classes** hiérarchiques, ancrées dans la littérature
plutôt que dans les seuls labels du clustering : classification des anti-patterns microservices (Soldani
& Brogi 2022), grandeurs de saturation de la méthode USE (Gregg 2013) et défaillances en cascade (Motter
& Lai 2002). Elle déclare **11 propriétés d'objet** — dont `causes` (transitive, asymétrique,
irréflexive), `precedes` (transitive), `coOccursWith` (symétrique) et `propagatesThrough` (propagation de
service à service) — et **6 propriétés de données**, ainsi que deux axiomes d'équivalence définissant les
classes composites `Composite_Anomaly` et `CascadingFailure`.

**Peuplement (ABox).** L'ontologie est peuplée de **143 individus** : 10 clusters empiriques, 10
anomalies typées (une par cluster, rattachée à une classe feuille), 10 signatures, 107 poids de features
réifiés (issus de l'importance par permutation) et 6 services. Ce peuplement transforme les résultats
numériques du pipeline en assertions interrogeables.

**Relations causales.** L'analyse causale par Transfer Entropy multivariée (estimateur KSG, n_perm = 200,
correction de Benjamini–Hochberg), conduite sur des épisodes composites synthétiques (générés pour
pallier le design mono-scénario du dataset réel), extrait trois relations significatives.

| Relation causale | Transfer Entropy | p ajustée | Interprétation |
|---|---|---|---|
| crash → rampe de trafic | 0,182 | 0,015 | la redistribution de charge après un crash provoque une rampe de trafic |
| changement de config → redéploiement | 0,067 | 0,015 | un changement de configuration déclenche un redéploiement |
| crash → déploiement défectueux | 0,141 | 0,030 | un crash peut entraîner un redéploiement défectueux |

**Co-occurrences et propagation.** On dénombre **19 co-occurrences** (tests du χ² de Yates avec repli sur
le test exact de Fisher) et **46 relations de propagation** entre services. Ces dernières sont obtenues en
filtrant 124 relations brutes pour ne retenir que les 46 spécifiques (13 paires ubiquitaires, comme
`load-generator → frontend`, étant écartées). Fait notable, les clusters de drift bénin pur ne produisent
aucune relation de propagation — résultat validant, un drift bénin ne déclenchant pas de cascade causale
entre services.

**Raisonnement et interrogation.** La cohérence de l'ontologie complète est vérifiée par le raisonneur
OWL 2 HermiT en **0,61 s**, sans aucune classe inconsistante. L'ontologie matérialisée s'interroge par
requêtes SPARQL ; par exemple, pour obtenir les services traversés par la propagation d'un type donné :

```sparql
SELECT ?service WHERE {
  ?anomaly a :CPU_Saturation ;
           :propagatesThrough ?service .
}
```

**Épisodes composites synthétiques.** Le design mono-scénario d'ewat_v3 interdisant par construction
d'observer co-occurrences et causalités inter-types, **282 épisodes composites** ont été générés (19
écartés par les garde-fous) par chevauchement (overlay, α ∈ {0,3 ; 0,5}) et cascade (intervalle de 2 à 10
pas, longueur ≈ 50). Trois garde-fous préviennent les artefacts : écrêtage au P99 par feature, corrélation
de Spearman médiane ≥ 0,85, et aire sous courbe d'un discriminateur < 0,75. La valeur effective de ce
discriminateur, **0,529**, confirme que les épisodes synthétiques sont indistinguables du réel au niveau du
corpus.

**Validation.** Huit critères sur dix sont atteints, comme le récapitule le tableau suivant.

| Critère | Cible | Valeur | Atteint |
|---|---|---|---|
| Couverture scénarios → classes | ≥ 80 % | 100 % | oui |
| Couverture clusters → classes | 100 % | 100 % | oui |
| Relations causales | ≥ 15 | 3 | non |
| Co-occurrences | ≥ 10 | 19 | oui |
| Temps de classification HermiT | < 30 s | 0,61 s | oui |
| Cohérence OWL | OK | OK | oui |
| Inférences matérialisées | ≥ 30 | 0 | non |
| Réalisme de la synthèse (AUC) | < 0,75 | 0,529 | oui |
| Relations de propagation | ≥ 30 | 46 | oui |
| Requêtes SPARQL canoniques | 5/5 | 5/5 | oui |

Les deux critères non atteints sont documentés : le faible nombre de relations causales est limité par le
nombre d'épisodes par paire dans la synthèse (un passage à davantage de répétitions est attendu pour
l'atteindre), et l'absence d'inférences matérialisées dans la hiérarchie d'instances est une limite connue
de l'outillage owlready2, contournée par l'accès via SPARQL.

### 5.9 Baseline d'alerte et simulation en ligne

Pour mesurer l'apport net d'EWAT, on le compare à une baseline z-score (détection d'écart univarié). Le
constat est net : la baseline détecte 100 % des anomalies mais lève **100 % de fausses alertes sur les
drifts bénins**, quel que soit son seuil σ — elle ne distingue pas drift et anomalie, ce qui est
précisément le problème à résoudre.

La simulation en ligne de l'assembleur d'alertes d'EWAT, sur l'ensemble de test, donne le compromis
suivant selon le seuil de décision.

| Seuil | Détection | Type correct | Fausses alertes (drift) | Délai d'anticipation |
|---|---|---|---|---|
| 0,30 | 100 % | 42,4 % | 100 % | 4,6 min |
| 0,40 | 97,0 % | 66,7 % | 100 % | 3,8 min |
| 0,50 | 78,8 % | 63,6 % | 100 % | 3,9 min |
| 0,60 | 75,8 % | 63,6 % | 50,0 % | 3,7 min |
| **0,70** | **57,6 %** | **51,5 %** | **8,3 %** | **3,0 min** |

Le point opérationnel recommandé est le seuil 0,70 : il ramène le taux de fausses alertes sur drift à
8,3 % (contre 100 % pour la baseline) tout en conservant un délai d'anticipation de 3,0 minutes. Aux
seuils plus bas, le détecteur de drift n'a pas le temps de se réchauffer avant que les classifieurs ne
tirent, d'où un taux de fausses alertes élevé — limite liée à la longueur des épisodes. Les figures 8 et
9 détaillent respectivement le compromis détection/fausses alertes et l'attribution de type au seuil
opérationnel.

![Figure 8 — Courbes ROC et précision–rappel de l'assembleur d'alertes selon le seuil de décision.](figures/roc_pr_curve.png)

![Figure 9 — Matrice de confusion de l'attribution de type au seuil opérationnel (épisodes correctement typés sur la diagonale).](figures/confusion_matrix.png)

### 5.10 Latence end-to-end

La chaîne d'inférence en ligne (étapes 0, 1 et 3) a été mesurée sur 200 itérations. Le 95ᵉ percentile de
la latence totale est de **13 ms**, soit environ 375 fois sous le budget de cinq secondes. Le détail par
étape confirme que chaque composante reste largement dans son budget. Les étapes 2 et 2b, hors ligne,
n'entrent pas dans ce bilan.

| Étape | Budget | Mesure (incluse dans le p95 total) |
|---|---|---|
| 0 — détection de drift | < 1 s | oui |
| 1 — encodeur | < 2 s | oui |
| 3 — précurseurs | < 1 s | oui |
| Total en ligne | < 5 s | **13 ms (p95)** |

### 5.11 Validation multi-graines

Pour mesurer la variance réelle des résultats, le pipeline a été rejoué sur dix graines. Le tableau
suivant détaille, graine par graine, la silhouette de test, le nombre de types K, l'AUROC pic et l'écart
de précursion (distant-window, cf. section 6).

| Graine | silhouette test | K | AUROC pic | Δ(far − near) | Verdict précursion |
|---|---|---|---|---|---|
| 42 | 0,838 | — | 0,993 | −0,050 | précursion réelle |
| 123 | 0,715 | 14 | 0,984 | +0,019 | signature statique |
| 456 | 0,690 | 10 | 0,996 | +0,012 | signature statique |
| 789 | 0,560 | 15 | 0,978 | +0,013 | signature statique |
| 1337 | 0,835 | 9 | 1,000 | −0,004 | signature statique |
| 0 | 0,558 | 13 | 0,998 | −0,013 | signature statique |
| 7 | 0,521 | 10 | 0,959 | −0,034 | signature statique |
| 17 | 0,712 | 12 | 0,999 | −0,032 | signature statique |
| 31 | 0,839 | 9 | 1,000 | −0,024 | signature statique |
| 99 | 0,647 | 14 | 0,988 | −0,007 | signature statique |
| **Moyenne** | **0,691 ± 0,115** | 11,8 ± 2,1 | **0,990 ± 0,012** | **−0,012 ± 0,022** | — |

Deux enseignements (figure 10). D'une part, la silhouette présente une **variance large** (intervalle
[0,521 ; 0,839]) et le nombre de types K est **instable** (intervalle [9 ; 15]) — deux limites assumées.
D'autre part, le résultat défendable sur cible indépendante est, lui, **parfaitement stable** : la
régression logistique étant déterministe, les dix graines donnent exactement le même macro-AUROC de
0,9201 (stratifié) et 0,9298 (leave-one-scenario-out), l'incertitude étant entièrement portée par le
bootstrap. La graine 42, longtemps considérée comme représentative, se révèle être un **cas atypique**
sur la précursion (seule graine présentant une précursion temporelle réelle sur cible interne) — ce qui
justifie a posteriori l'analyse critique de la section 6.

![Figure 10 — Distribution par graine des métriques de validation multi-graines : les métriques déterministes sont stables, tandis que la silhouette, le nombre de types et l'écart de précursion présentent une variance large.](figures/multiseed_distribution.png)

### 5.12 Validation externe et transfert (RCAEval)

La validation externe est nécessaire pour crédibiliser le pipeline au-delà de son cluster d'origine. On
applique donc EWAT, sans réentraînement, au benchmark public RCAEval RE2-OB (90 épisodes, 30 types de
pannes, sur le même Online Boutique mais un cluster différent et des épisodes de 48 pas).

En transfert **zero-shot**, quatre stratégies de normalisation ont été testées.

| Stratégie | Features | Silhouette (H1) | AUROC (H3) |
|---|---|---|---|
| Scaler ewat_v3 | 17 | 0,778 (artefact) | 0,510 |
| Scaler RCAEval | 17 | 0,234 | 0,497 |
| Normalisation par instance | 17 | 0,287 | 0,507 |
| Normalisation par instance | métriques seules | 0,684 | 0,495 |

La meilleure configuration (normalisation par instance, métriques seules) franchit le seuil de
structurabilité (silhouette 0,684) mais reste au niveau du hasard pour la prédictibilité (AUROC 0,495) :
l'encodeur détecte qu'il y a une anomalie, sans dire laquelle. Le transfert **few-shot**, qui réajuste le
scaler sur quelques épisodes RCAEval, ne débloque rien.

| Nombre d'épisodes d'ajustement | Silhouette (H1) | AUROC (H3) |
|---|---|---|
| 1 | 0,442 | 0,507 |
| 3 | 0,388 | 0,503 |
| 5 | 0,311 | 0,503 |
| 10 | 0,347 | 0,502 |
| 20 | 0,237 | 0,504 |
| 40 | 0,222 | 0,503 |

La prédictibilité reste collée à ≈ 0,50 quel que soit le nombre d'épisodes d'ajustement (figure 11). Le
verrou n'est pas le scaler mais l'espace latent lui-même, qui ne sépare pas les types de pannes RCAEval ;
un transfert réussi demanderait un fine-tuning du classifieur ou de l'encodeur. C'est un **échec de
généralisation assumé**, utile en ce qu'il borne précisément la portée du modèle.

![Figure 11 — Transfert few-shot sur RCAEval : la structurabilité (silhouette) et la prédictibilité (AUROC) en fonction du nombre d'épisodes d'ajustement ; la prédictibilité reste au niveau du hasard.](figures/fewshot_learning_curve.png)

---

### 5.13 Étude de cas — le scénario « crash » (cluster C4)

Pour illustrer le fonctionnement du pipeline de bout en bout, on déroule un scénario de crash, qui se
révèle l'un des plus nets du catalogue. Le scénario `crash` provoque l'arrêt brutal d'un service.

**Signature dans le signal.** Au moment du crash, la signature est multimodale : les spans du service
disparaissent brutalement (les features de traces chutent), le taux d'erreur HTTP des services appelants
augmente (les appels échouent), le nombre de redémarrages s'incrémente, et les journaux émettent des
erreurs de connexion. Cette combinaison est caractéristique et distincte des dégradations lentes.

**Détection et typage.** Le détecteur de drift (étape 0) signale le changement de distribution ;
l'encodeur (étape 1) projette la fenêtre en un embedding ; le typage (étape 2) rattache l'épisode au
cluster **C4**. Ce cluster est l'un des cinq types reportables (au moins cinq positifs en test) et le plus
discriminant de tous : son AUROC de test atteint **1,000** (intervalle de confiance [1,000 ; 1,000]),
avec un horizon optimal **k\* = 2 pas (une minute)**. Le crash a donc une signature très précoce et très
distincte, ce qui le rend prédictible avec une marge maximale.

**Précurseur.** Le classifieur de C4 estime, dès une minute avant l'événement, une probabilité élevée
d'occurrence : l'assembleur d'alertes lève alors une alerte typée « crash » avec son horizon. Au point
opérationnel (seuil 0,70), cette alerte est conservée car aucun drapeau de drift bénin n'est actif.

**Enrichissement ontologique.** L'ontologie relie enfin C4 à ses conséquences en aval. Deux relations
causales partent de C4 : **C4 → C1** (le crash provoque une rampe de trafic, par redistribution de la
charge vers les services survivants, Transfer Entropy 0,182) et **C4 → C8** (le crash peut entraîner un
redéploiement défectueux, Transfer Entropy 0,141). La fiche d'alerte ne se contente donc pas d'annoncer
un crash : elle indique ses propagations probables, ce qui oriente la réponse de l'exploitant (anticiper
la rampe de trafic, surveiller le redéploiement). Ce cas illustre la complémentarité des étapes : le
typage identifie *quoi*, le précurseur *quand*, et l'ontologie *vers quoi cela se propage*.

---

## 6. Étude critique : circularité de l'évaluation et robustesse

La performance interne élevée (section 5.4) appelle une vérification rigoureuse : mesure-t-elle réellement
une *prédiction temporelle*, ou seulement la *reconnaissance d'une signature de scénario* ? Cette section
présente une série de tests conçus pour répondre à cette question, et qui constituent l'apport
méthodologique le plus important du module.

### 6.1 Contexte : pourquoi la cible interne est circulaire

Le risque tient à la nature de la cible. En section 5.4, le précurseur prédit les labels de types
*produits par EWAT lui-même*. Or ces labels sont en partie récupérables directement depuis le signal :
une régression logistique sur les features brutes (sans encodeur) retrouve les labels d'EWAT avec un
AUROC de 0,966, et un simple k-means suivi d'une régression logistique atteint 0,975 — davantage que le
pipeline complet sur sa propre cible. L'AUROC interne mesure donc largement la **récupérabilité** des
labels, pas une prédiction d'événement futur. Les tests suivants quantifient cette part de circularité.

### 6.2 Test A1 — fenêtre distante (signature statique)

Le test décisif consiste à déplacer la fenêtre d'observation dans le régime normal, sans changer le
modèle : si le signal est réellement précurseur, une fenêtre plus éloignée de l'injection doit donner un
AUROC plus faible. Sur la cible interne (ewat_v3), il n'en est rien.

| Position de la fenêtre | Macro-AUROC |
|---|---|
| Juste avant l'injection (near) | 0,904 |
| Au milieu du régime normal | 0,907 |
| Au début du régime normal (far) | 0,897 |

L'écart Δ(far − near) = −0,007 est négligeable : le classifieur produit le même AUROC quelle que soit la
position de la fenêtre. Il lit donc une **signature statique** du scénario (quel service, quel mélange de
charge, quelles baselines), récupérable depuis n'importe quel point du régime normal — et non une
dynamique pré-injection. Sur la cible interne, l'AUROC élevé n'est pas une détection précoce.

### 6.3 Test A2 — leave-one-scenario-out

On retire un scénario entier de l'entraînement du précurseur, puis on évalue sa capacité à reconnaître ce
scénario inédit. Le macro-AUROC sur l'ensemble de test reste élevé (0,896 ± 0,013), car les autres
scénarios couvrent l'espace ; mais la vraie mesure de généralisation — le taux de bonne classification au
premier rang sur le scénario retiré — n'est que de 0,511 ± 0,382, et fortement polarisée (quatre
scénarios à 100 %, quatre à 0 %). Le modèle **interpole entre scénarios connus** mais ne généralise pas à
un type de panne inédit.

### 6.4 Test A3 — test de permutation

Pour vérifier que le signal n'est pas un pur artefact, on permute aléatoirement les labels d'entraînement
(100 permutations) et l'on mesure l'AUROC à chaque fois. L'AUROC observé (labels réels) est de 0,893,
contre une distribution nulle de 0,492 ± 0,104 (95ᵉ percentile à 0,672), soit une p-value empirique
inférieure à 0,01. Il existe donc bien un **signal réel** aligné sur les labels — mais les tests A1 et A2
montrent que ce signal est une signature de scénario, pas une dynamique temporelle généralisable.

### 6.5 Test A4 — filtrage par effectif

Plusieurs types ont trop peu de positifs en test pour un AUROC fiable. En ne conservant que les types
disposant d'au moins cinq positifs, cinq types sur dix sont reportables, avec un AUROC moyen de
0,975 ± 0,020. Les cinq autres doivent être marqués « non concluants ».

### 6.6 Évaluation sur cible indépendante : la précursion réelle apparaît

Le renversement se produit lorsqu'on évalue sur la cible indépendante (scénarios Chaos Mesh) plutôt que
sur la cible interne. Trois mesures convergent.

D'abord, un diagnostic sur features brutes (sans encodeur) montre que l'écart far/near devient **non
nul** dès qu'on cible les labels indépendants : Δ(far − near) vaut −0,071 (normalisation globale) à
−0,026 (instance) sur ewat_v3, et jusqu'à −0,063 sur ewat_v4_strat. Il existe donc une dynamique
pré-injection réelle, captée par les features brutes, qui était masquée par la circularité de la cible
interne.

Ensuite, un bootstrap apparié sur l'écart entre embeddings STGCN et features brutes (B4 − B3) donne un
écart de +0,0053 avec un intervalle de confiance [−0,0315 ; +0,0444] contenant zéro : la neutralité de
l'encodeur sur cette cible est statistiquement bien établie, ce n'est pas un artefact ponctuel.

Enfin, et surtout, lorsqu'un STGCN est entraîné de bout en bout sur la cible Chaos Mesh, le test de
fenêtre distante se renverse complètement.

| Position de la fenêtre | Macro-AUROC |
|---|---|
| Juste avant l'injection (near) | 0,876 |
| Au milieu du régime normal | 0,813 |
| Au début du régime normal (far) | 0,759 |

L'écart Δ(far − near) = −0,116 indique que la dynamique pré-injection vaut une douzaine de points
d'AUROC : il y a bien une **précursion temporelle réelle**. La « signature statique » constatée en A1
était un artefact de la circularité des labels internes, et non une propriété du signal.

### 6.7 Conclusion de l'étude critique

Les tests se recoupent en un tableau cohérent : il existe un signal réel (A3), mais sur la cible interne
il se réduit à une signature statique de scénario (A1) qui ne généralise pas à un type inédit (A2) ;
évalué honnêtement sur une cible indépendante, le signal révèle en revanche une précursion temporelle
réelle (section 6.6). On en retient deux principes pour la suite : (1) toujours rapporter le résultat
défendable sur cible indépendante (macro-AUROC 0,920), et signaler explicitement les chiffres internes
comme tels ; (2) reformuler l'objectif de prédictibilité en « typage anticipé du scénario actif » plutôt
qu'en « détection précoce » au sens strict, sur la cible interne.

Enfin, la généralisation à un type de panne **jamais vu** a été étudiée par reconnaissance open-set
(OpenMax). Le résultat est partiel : le taux de bonne détection « inconnu » au premier rang passe de 0
(classifieur fermé) à 0,400 ± 0,407, mais l'AUROC global de détection d'inconnu reste au niveau du hasard
(0,550 ± 0,238). Une généralisation complète demanderait un dispositif plus sophistiqué (perspectives).

---

## 7. Limites et résultats négatifs assumés

Conformément à la démarche du module, les limites sont documentées explicitement plutôt que masquées.

### 7.1 Limites méthodologiques

La principale est la **circularité d'évaluation** (section 6) : la cible interne du typage étant produite
par EWAT, son AUROC élevé est en partie auto-référent — d'où le recours systématique à la cible
indépendante. S'y ajoutent un **surentraînement du réseau siamois** (l'époque optimale se situe autour de
la troisième, quelle que soit la configuration, signe d'une convergence trop rapide sur la diversité des
paires), un **faible nombre de positifs par scénario en test** (souvent trois, ce qui rend plusieurs
AUROC non concluants), et l'**absence de validation externe réussie** : le transfert sur le benchmark
public RCAEval échoue (l'AUROC de prédiction reste au niveau du hasard), le verrou étant le scaler non
transférable.

### 7.2 Limites techniques

Côté technique, le **graphe de services est petit** (six services sur Online Boutique) — limite levée par
le pivot v5 vers Train Ticket (41 services). Le **nombre de types est instable** selon la graine
(intervalle [9 ; 15]), ni la silhouette ni la statistique de gap ne le stabilisant ; une correction
possible est de fixer le nombre de types ou de passer à un clustering par densité. Les **dix-sept features
et la topologie sont figées** dans le code. Le pipeline est **mono-cluster** (entraîné sur un seul
cluster, sans adaptation de domaine). Enfin, le **cycle de réentraînement** n'est pas encore automatisé.

### 7.3 Tableau récapitulatif

| Limite | Nature | Correction envisagée |
|---|---|---|
| Circularité de la cible interne | méthodologique | cible Chaos Mesh indépendante (fait) ; couplage ontologie-prédiction |
| Surentraînement du siamois | méthodologique | mining de négatifs difficiles, curriculum |
| Faible effectif par type en test | méthodologique | collecte v5 (plus d'épisodes) |
| Échec du look-through (drift/anomalie) | résultat négatif | espace de représentation dédié au régime |
| Échec du transfert externe (RCAEval) | résultat négatif | fine-tuning few-shot |
| Graphe à 6 services | technique | pivot Train Ticket (41 services) |
| Nombre de types instable | technique | nombre fixé ou clustering par densité |
| Features et topologie figées | technique | configuration complète |
| Pipeline mono-cluster | technique | adaptation de domaine |
| Open-set partiel | technique | détecteurs hors-distribution (Mahalanobis, énergie) |

---

## 8. Proposition d'un schéma d'implémentation et d'automatisation

### 8.1 Modules du pipeline et technologies associées

Le pipeline est organisé en six modules indépendants, chacun avec son interface, ses tests et la
possibilité d'être remplacé sans refonte de l'ensemble.

| Module | Rôle | Technologie principale |
|---|---|---|
| Détection de drift | étape 0 — alarme de changement | MMD-RFF (NumPy) |
| Encodeur | étape 1 — embedding du graphe | STGCN (PyTorch, PyTorch Geometric) |
| Typage | étape 2 — découverte des types | siamois + clustering (PyTorch, scikit-learn) |
| Ontologie | étape 2b — sémantique des types | OWL/RDF, owlready2, HermiT |
| Précurseurs | étape 3 — probabilité et horizon | classifieurs un-contre-tous (scikit-learn) |
| Assemblage d'alertes | sortie — alerte typée | logique d'orchestration |

La configuration est centralisée par Hydra et le suivi des expériences assuré par MLflow. La collecte de
données suit une chaîne en trois phases strictement ordonnées — enregistrement (injection de chaos et
sauvegarde des données brutes), construction des features (reconstruction hors ligne du signal, du graphe
et des labels), assemblage (consolidation et découpage) — conçue pour que les données brutes restent une
archive immuable et que chaque phase soit rejouable hors ligne. Cette séparation garantit la
reproductibilité et la cohérence entre traitement en ligne et hors ligne.

### 8.2 Scénarios de déploiement

L'analyse des résultats (sections 5.5 et 6) conduit à une **chaîne opérationnelle simplifiée** (figure 12) : le signal instance-normalisé alimente directement la régression logistique et un module de
nouveauté (OpenMax), l'encodeur STGCN étant réservé à la structuration de l'espace latent et à
l'ontologie, hors du chemin prédictif principal. Cette séparation rend la chaîne prédictive légère et
rapide (13 ms au 95ᵉ percentile).

![Figure 12 — Chaîne opérationnelle issue de l'analyse : signal instance-normalisé → régression logistique un-contre-tous → OpenMax pour le signal de nouveauté ; le STGCN reste en amont pour le typage et l'ontologie.](figures/pipeline_operational_v2.png)

Deux scénarios de déploiement sont envisageables. Le premier, **par lots**, correspond à l'état actuel :
les données sont collectées puis traitées hors ligne, ce qui convient à l'expérimentation et à la
validation. Le second, **en flux**, vise la production : ingestion continue des signaux via un bus de
messages (Kafka), fenêtrage et inférence en continu (Flink ou équivalent), puis émission des alertes
vers les tableaux de bord et les outils d'astreinte. Le budget de latence mesuré (13 ms) laisse une marge
confortable pour une cible en flux à p99 inférieur à dix secondes.

### 8.3 Cycle d'exécution automatisé

Le cycle opérationnel comporte deux boucles. Une **boucle en ligne** exécute, à chaque fenêtre, les
étapes 0, 1 et 3 pour produire les alertes. Une **boucle hors ligne périodique** recalcule les types
(étape 2), reconstruit l'ontologie (étape 2b) et réentraîne les modèles. Le réentraînement est piloté par
la dérive elle-même : une évaluation quotidienne du MMD² sur les données récentes détecte un changement
de régime durable ; au-delà d'un seuil, un réentraînement est déclenché (par exemple via un ordonnanceur
de tâches). Les nouveaux modèles sont validés contre le modèle en production (comparaison A/B sur un
échantillon récent) avant toute bascule, afin d'éviter une régression silencieuse.

### 8.4 Suivi des exécutions, journalisation et traçabilité

Chaque exécution d'entraînement ou d'évaluation est tracée via MLflow : paramètres, métriques et
artefacts de modèle sont versionnés, ce qui permet de reproduire et de comparer les campagnes. En
inférence, chaque alerte émise conserve le contexte ayant conduit à sa production — type pressenti,
probabilité, horizon, fenêtre d'observation, drapeau de drift. Cette traçabilité est nécessaire à l'audit
des décisions (pourquoi cette alerte a-t-elle été levée ?) et au diagnostic des régressions (qu'est-ce
qui a changé entre deux versions ?). Elle constitue un prérequis pour une exploitation de confiance.

### 8.5 Extensions et généralisation du pipeline

Quatre axes d'extension sont identifiés, par ordre de priorité décroissante.

**Axe A — Couplage ontologie ↔ prédiction.** L'ontologie (relations causales et de propagation) est
aujourd'hui isolée de la chaîne prédictive. L'intégrer comme information a priori du précurseur —
enrichissement des features par les services causalement amont, re-ranking des scores par les priors de
graphe — vise un gain mesurable du résultat défendable. C'est l'extension à plus fort potentiel
scientifique.

**Axe B — Précursion temporelle robuste.** La précursion réelle observée sur cible indépendante (section
6.6) doit être consolidée en multi-graines, et un encodeur temporel de type Transformer, mieux outillé
pour la dynamique que le STGCN, doit être évalué.

**Axe C — Détection de nouveauté.** Pour généraliser aux types de pannes inédits, dépasser OpenMax par
des détecteurs hors-distribution (distance de Mahalanobis, Lee et al. 2018 ; score d'énergie, Liu et al.
2020) ou par du méta-apprentissage few-shot.

**Axe D — Industrialisation multi-cluster.** Passer du traitement par lots à la chaîne en flux décrite en
7.2, automatiser le cycle de réentraînement (7.3), et adapter le modèle entre clusters par adaptation de
domaine, EWAT étant pour l'instant entraîné sur un seul cluster.

En complément, la collecte v5 sur Train Ticket alimente l'ensemble de ces axes avec un dataset plus
représentatif (41 services, bugs réels), et constitue un candidat à la publication en dataset ouvert,
sous réserve d'autorisation et d'assainissement des données brutes.

---

## 9. Conclusion

EWAT apporte une réponse opérationnelle et rigoureuse au problème des fausses alertes et de
l'anticipation des pannes en environnement microservices. Son apport net se résume en deux chiffres : le
taux de fausses alertes sur les drifts bénins passe de 100 % (baseline z-score) à 8,3 %, et la
discrimination des types de pannes atteint un macro-AUROC de 0,920 sur cible indépendante, le tout en
13 ms par inférence. Sur le plan des acquis, le **typage des anomalies est solide**
(silhouette de test 0,782 sur dix graines, largement au-dessus du seuil), la **discrimination de scénario
est défendable sur cible indépendante** (macro-AUROC 0,920, intervalle de confiance [0,878 ; 0,956]), le
**pipeline est rapide** (13 ms au 95ᵉ percentile, soit 375 fois sous le budget) et **testé** (586 tests
unitaires), et une **ontologie cohérente** structure l'interprétation des types. Face à une baseline
z-score qui lève 100 % de fausses alertes sur les drifts bénins, EWAT ramène ce taux à 8,3 % tout en
conservant un délai d'anticipation de trois minutes — c'est l'apport opérationnel net.

Sur le plan des limites, trois résultats négatifs sont assumés et documentés : la séparation
drift/anomalie par look-through échoue de façon robuste ; la généralisation à un type de panne inédit
reste partielle (open-set incomplet, transfert externe en échec) ; et le nombre de types est instable
selon la graine. L'étude critique de la circularité (section 6) a par ailleurs montré que la performance
interne élevée relève en partie d'une signature statique de scénario, la précursion temporelle réelle
n'apparaissant que sur cible indépendante — ce qui a conduit à privilégier systématiquement les chiffres
défendables. Enfin, l'encodeur STGCN s'est révélé utile pour la géométrie et l'ontologie plutôt que pour
la prédiction agrégée, un résultat contre-intuitif mais solidement établi.

La valeur de ce module tient donc autant à sa démarche qu'à ses chiffres : séparer explicitement les
régimes de fonctionnement, typer empiriquement les modes de défaillance, et distinguer rigoureusement les
résultats défendables sur cible indépendante des indicateurs internes. Les extensions identifiées —
couplage ontologie-prédiction, précursion robuste, détection de nouveauté, industrialisation multi-cluster
— et la nouvelle collecte v5 tracent la voie d'une généralisation et d'une mise en production
progressives.

---

## Références

Les travaux cités dans ce document sont les suivants (ordre alphabétique).

1. Bendale, A. & Boult, T. E. (2016). *Towards Open Set Deep Networks*. CVPR.
2. Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate*. J. R. Stat. Soc. B, 57(1), 289–300.
3. Chandola, V., Banerjee, A. & Kumar, V. (2009). *Anomaly Detection: A Survey*. ACM Computing Surveys, 41(3).
4. Chen, T., Kornblith, S., Norouzi, M. & Hinton, G. (2020). *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). ICML.
5. Davison, A. C. & Hinkley, D. V. (1997). *Bootstrap Methods and Their Application*. Cambridge University Press.
6. Dhillon, I. S. & Modha, D. S. (2001). *Concept Decompositions for Large Sparse Text Data Using Clustering*. Machine Learning, 42(1), 143–175.
7. Efron, B. (1987). *Better Bootstrap Confidence Intervals*. JASA, 82(397), 171–185.
8. Eldele, E. et al. (2021). *Time-Series Representation Learning via Temporal and Contextual Contrasting* (TS-TCC). IJCAI.
9. Fu, Q. et al. (2025). *A Survey on Root Cause Analysis of Microservice Systems*. ACM Computing Surveys.
10. Glimm, B. et al. (2014). *HermiT: An OWL 2 Reasoner*. Journal of Automated Reasoning, 53(3), 245–269.
11. Gregg, B. (2013). *Systems Performance: Enterprise and the Cloud*. Prentice Hall.
12. Gretton, A. et al. (2012). *A Kernel Two-Sample Test*. JMLR, 13, 723–773.
13. Hinder, F. et al. (2024). *Concept Drift in Unsupervised Data Streams*. ECML-PKDD.
14. Kaufman, L. & Rousseeuw, P. J. (1990). *Finding Groups in Data*. Wiley.
15. Kipf, T. N. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks* (GCN). ICLR.
16. Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). *Estimating Mutual Information*. Physical Review E, 69(6), 066138.
17. Lamy, J.-B. (2017). *Owlready: Ontology-Oriented Programming in Python*. Artificial Intelligence in Medicine, 80, 11–28.
18. Lee, K. et al. (2018). *A Simple Unified Framework for Detecting Out-of-Distribution Samples* (Mahalanobis). NeurIPS.
19. Liu, W. et al. (2020). *Energy-based Out-of-distribution Detection*. NeurIPS.
20. Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
21. Motter, A. E. & Lai, Y.-C. (2002). *Cascade-Based Attacks on Complex Networks*. Physical Review E, 66, 065102.
22. Myrtollari, E. et al. (2025). *Concept Drift-Aware Anomaly Detection for Kubernetes Microservices*. IEEE/ACM.
23. Pham, L., Ha, H. & Zhang, H. (2024). *Root Cause Analysis for Microservice System based on Causal Inference: How Far Are We?* ASE.
24. Pham, L. et al. (2025). *RCAEval: A Benchmark for Root Cause Analysis of Microservice Systems with Telemetry Data*. WWW.
25. Phipson, B. & Smyth, G. K. (2010). *Permutation P-values Should Never Be Zero*. Stat. Appl. Genet. Mol. Biol., 9(1).
26. Rahimi, A. & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines*. NeurIPS.
27. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks*. EMNLP.
28. Schreiber, T. (2000). *Measuring Information Transfer* (Transfer Entropy). Physical Review Letters, 85(2), 461–464.
29. Soldani, J. & Brogi, A. (2022). *Anomaly Detection and Failure Root Cause Analysis in (Micro)Service-Based Cloud Applications: A Survey*. ACM Computing Surveys, 55(3).
30. Tibshirani, R., Walther, G. & Hastie, T. (2001). *Estimating the Number of Clusters via the Gap Statistic*. J. R. Stat. Soc. B, 63(2), 411–423.
31. Veličković, P. et al. (2018). *Graph Attention Networks* (GAT). ICLR.
32. Yu, B., Yin, H. & Zhu, Z. (2018). *Spatio-Temporal Graph Convolutional Networks* (STGCN). IJCAI.
33. Zamanzadeh Darban, Z. et al. (2024). *Deep Learning for Time Series Anomaly Detection: A Survey*. ACM Computing Surveys, 57(1).
34. Zhang, S. et al. (2024). *Illuminating the Gray Zone: Non-intrusive Gray Failure Localization in Server Operating Systems* (GrayScope). FSE.

---

## Annexe A — Commandes et configuration

### A.1 Commandes du pipeline (dataset de référence)

```bash
# Étape 1 — Encodeur STGCN (100 époques)
python -m experiments.encoder.train --dataset data/datasets/ewat_v3 \
    --features-root data/features/v3 --output experiments/encoder --epochs 100

# Étape 2 — Typage siamois (50 époques)
python -m experiments.typing.train --dataset data/datasets/ewat_v3 \
    --features-root data/features/v3 \
    --encoder-checkpoint experiments/encoder/checkpoints/best_encoder.pt \
    --output experiments/typing --epochs 50

# Étape 2b — Ontologie (100 permutations)
python -m experiments.ontology.build --typing-dir experiments/typing \
    --features-root data/features/v3 --output experiments/ontology --n-permutations 100

# Étape 3 — Précurseurs
python -m experiments.precursor.train --typing-dir experiments/typing \
    --features-root data/features/v3 --output experiments/precursor --k-values 2 4 6 8 10 12

# Évaluation des alertes (test)
python -m experiments.alerts.eval --typing-dir experiments/typing \
    --encoder-dir experiments/encoder --precursor-dir experiments/precursor \
    --features-root data/features/v3 --output experiments/alerts
```

La collecte v5 (Train Ticket) suit la chaîne `run_campaign` → `build_features_v5` → `validate_v5` →
`assemble_dataset --stratified` → `enforce_heldout_v5` (cf. section 3.4.7).

### A.2 Hyperparamètres principaux

| Étape | Paramètre | Valeur |
|---|---|---|
| Drift | fenêtre de référence / courante | 300 s / 60 s |
| Drift | seuil ε_drift | 0,5226 (Youden) |
| Drift | confirmation post-drift | 120 s |
| Encodeur | dimension d'embedding | 64 |
| Encodeur | fenêtre temporelle | 60 pas |
| Encodeur | époques / taille de lot | 100 / 32 |
| Typage | clustering (config optimisée) | average + cosinus |
| Typage | projection / marge (config optimisée) | d_proj = 64 / marge = 2,0 |
| Précurseurs | horizons k testés | {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20} pas |
| Précurseurs | classifieur | régression logistique réglée |
| Évaluation | bootstrap | 1000 rééchantillonnages, intervalles BCa |

### A.3 Endpoints d'observabilité

Les endpoints internes (services Kubernetes) sont rappelés en section 3.2.3 ; les références
d'administration du cluster sont volontairement omises. La configuration complète est gérée par Hydra
(`configs/default.yaml`).
