# EWAT — Point de projet

---

## Le projet

Le projet s'appelle EWAT, pour Early Warning and Anomaly Typing. L'idée de départ est simple : dans une application moderne découpée en microservices qui tourne sur Kubernetes, quand quelque chose se passe mal, les outils classiques de monitoring sonnent l'alarme. Mais ils sonnent aussi pour plein de choses qui ne sont pas vraiment des pannes — un déploiement en cours, une montée en charge prévue, un changement de configuration. Résultat : les équipes ops se retrouvent noyées sous les fausses alertes et finissent par les ignorer.

EWAT essaie de répondre à une question différente : est-ce qu'on peut détecter qu'une vraie panne est en train de se préparer, avant qu'elle arrive, et en plus lui donner un nom — un type — pour que l'équipe sache déjà à quoi s'attendre ?

C'est important de préciser ce que ce n'est pas. EWAT n'est pas du RCA, du Root Cause Analysis — ce truc qui, après la panne, te dit "c'était le service machin qui a planté à cause de ça". Ça c'est du post-mortem. EWAT c'est du pré-mortem : on veut lever une alerte deux à cinq minutes avant que ça casse, en disant "ce type de panne arrive, et d'habitude il arrive dans 3 minutes".

---

## L'architecture

Pour faire ça, on observe en continu l'état du cluster à travers ce qu'on appelle un signal : pour chaque service Kubernetes, on collecte 17 valeurs toutes les 30 secondes. Ces 17 valeurs viennent de trois sources :

- **Les métriques** (7 valeurs) : CPU, RAM, latence P99, taux d'erreurs HTTP, saturation réseau, disque, et longueur de file d'attente. Ce sont les chiffres classiques qu'on scrape depuis Prometheus.
- **Les traces** (6 valeurs) : durée des spans, taux de spans anormaux, profondeur des traces, fan-out, taux de retry, variance de latence. Ces données viennent des traces distribuées OpenTelemetry — elles donnent une vision de comment les requêtes circulent entre services.
- **Les logs** (4 valeurs) : taux d'erreurs, taux de warnings, une mesure d'anomalie sémantique (on encode les logs avec un modèle de langage et on mesure leur distance au comportement normal), et l'entropie lexicale — est-ce que le vocabulaire des logs change ?

Ce signal passe ensuite dans un pipeline en quatre étapes.

**Étape 0 — Détection de drift.** Avant d'analyser quoi que ce soit, on se pose la question : est-ce que l'état actuel du système est juste une variation normale, ou est-ce qu'il y a eu un vrai changement de régime ? Un déploiement, par exemple, va faire changer plein de métriques — mais c'est attendu, prévu, bénin. On utilise une méthode statistique appelée MMD (Maximum Mean Discrepancy) pour comparer la distribution actuelle du signal avec une fenêtre de référence "normale". Si le signal a drifté, on le laisse passer avec un flag DRIFT — c'est ce qu'on appelle le look-through — plutôt que de le bloquer ou de l'ignorer.

**Étape 1 — L'encodeur.** Le signal passe dans un réseau de neurones appelé STGCN — Spatio-Temporal Graph Convolutional Network. L'idée c'est que les services ne sont pas indépendants : si le service A appelle le service B, une latence sur B va se propager sur A. Le STGCN modélise ça comme un graphe pondéré, où chaque arête représente le volume d'appels, la latence moyenne et le taux d'erreurs entre deux services. Le réseau lit l'historique du signal sur une fenêtre temporelle, prend en compte la topologie du graphe, et produit un vecteur de 64 dimensions qui résume "ce qui se passe en ce moment" dans le cluster.

**Étape 2 — Le typage.** Ces vecteurs sont entraînés avec un réseau siamois : on dit au modèle "ces deux épisodes sont du même type de panne, rapproche leurs représentations ; ces deux-là sont différents, éloigne-les". On obtient un espace où les pannes similaires se regroupent naturellement. On fait ensuite un clustering hiérarchique pour partitionner cet espace en K types de pannes — ce sont les clusters. Sur notre dataset, on en a trouvé 10.

**Étape 2b — L'ontologie.** Une fois les types identifiés, on construit une sorte de carte des relations entre eux. Est-ce que le type A précède souvent le type B dans le temps ? Est-ce qu'il y a une relation causale entre certains types ? Ces relations sont calculées statistiquement et forment une ontologie des pannes.

**Étape 3 — Les précurseurs.** Pour chaque type de panne, on entraîne un classifieur qui prédit la probabilité que ce type survienne dans k minutes, en regardant seulement le signal actuel. On cherche le k optimal par type : pour certains types, le signal devient prédictible 1 minute à l'avance ; pour d'autres, 5 minutes. La sortie finale est une alerte qui donne le type de panne probable, la probabilité, et le délai typique.

---

## Le dataset

Pour entraîner tout ça, on a besoin d'exemples labellisés — des épisodes où on sait exactement ce qui s'est passé et quand. On a donc construit un dispositif de collecte expérimentale sur un vrai cluster Kubernetes.

Le principe : on démarre l'application (une version de Online Boutique, un site e-commerce fictif avec plusieurs microservices), on laisse le cluster se stabiliser, puis on injecte une panne précise avec un outil qui s'appelle Chaos Mesh. Chaos Mesh permet de simuler toutes sortes de défaillances de manière contrôlée et reproductible : crash de pod, saturation mémoire, perte de paquets réseau, CPU starvé, déploiement défectueux...

Un épisode se décompose en quatre phases :

1. **Baseline** : on laisse le système tourner normalement pour établir une référence. Le signal capturé ici représente θ_normal.
2. **Pré-injection** : on continue à observer sans rien faire. Cette phase est cruciale pour les précurseurs — les signaux précurseurs doivent apparaître ici, avant la panne.
3. **Injection** : Chaos Mesh déclenche la panne. Le signal change brutalement ou progressivement selon le type.
4. **Recovery** : on arrête la panne et on observe comment le système récupère.

On a 15 scénarios différents, répartis en quatre grandes familles :
- Les **drifts bénins** : scale-up, déploiement progressif, changement de config, montée en charge.
- Les **anomalies dures** : crash de pod, OOM, perte réseau.
- Les **anomalies grises** : erreurs intermittentes, latence lente, CPU lent — difficiles à détecter car graduelles.
- Les **contentions** : voisin bruyant, fuite mémoire, CPU starvé.
- Et une catégorie hybride : un déploiement défectueux, qui est simultanément un drift et une anomalie.

Pour le premier dataset (v3), on a collecté 20 répétitions par scénario, soit 300 épisodes au total, chacun d'environ 21 pas de 30 secondes.

---

## État actuel du projet

Le pipeline est entièrement implémenté et évalué sur ce premier dataset. Voici où on en est sur les trois hypothèses de recherche.

**H1 — Est-ce que les pannes forment des groupes stables et séparables dans l'espace latent ?**

Oui. Le score de silhouette — une mesure de qualité du clustering, qui vaut 1 si les groupes sont parfaits et 0 s'ils se chevauchent totalement — est de 0.519 en moyenne sur 5 exécutions différentes, avec un minimum de 0.414. Le seuil qu'on s'était fixé pour valider est 0.3. On a 10 types de pannes identifiés, stables quelle que soit la graine d'initialisation du modèle.

**H2 — Est-ce que le mécanisme de look-through améliore la discrimination entre drift et anomalie ?**

Non, et c'est un résultat négatif qu'on assume. Le test statistique donne p = 0.27 — pas significatif. La cause est identifiée : les épisodes v3 font ~21 pas, soit environ 10 minutes. Le look-through nécessite une fenêtre de confirmation temporelle de 3 à 6 pas après la détection du drift pour distinguer "c'est un drift bénin" de "c'est une anomalie". Avec seulement 21 pas par épisode, cette confirmation arrive trop tard ou n'arrive pas du tout. Le mécanisme n'est pas faux — il n'a juste pas les conditions pour fonctionner.

**H3 — Est-ce qu'on peut prédire les pannes plusieurs minutes à l'avance, de façon typée ?**

Oui. Pour 8 types de pannes sur 10, l'AUROC est de 0.97 en moyenne — sur 5 graines. Ça veut dire que le classifieur distingue très bien les moments qui précèdent une panne de ceux qui n'en précèdent pas. Les 2 types manquants (C6 et C9) souffrent simplement d'un manque d'exemples dans le jeu de test — pas d'un problème de modèle.

Sur le système d'alerte en conditions réelles : au seuil 0.7, on détecte 48.5% des anomalies avec seulement 8.3% de fausses alertes sur les drifts bénins, et un délai moyen de 2.9 minutes. Le z-score classique, lui, génère 100% de fausses alertes sur les drifts — il ne distingue pas.

---

## Ce qu'on a ajouté en cours de route, et pourquoi

Au départ, les résultats semblaient encore meilleurs — silhouette à 0.61 sur le test set, AUROC parfait sur presque tous les types. En creusant, on a trouvé deux problèmes méthodologiques.

**Premier problème** : pour évaluer le clustering sur des données nouvelles, on recalculait les centroïdes du clustering indépendamment sur chaque split. Ça fait fuiter de l'information — le clustering du test "sait" déjà quelle forme il doit prendre pour bien séparer les données. La correction : on calcule les centroïdes une seule fois sur le train, et on assigne les nouveaux points au centroïde le plus proche. C'est ça qui a fait tomber la silhouette de 0.61 à 0.41 — et c'est la valeur honnête.

**Deuxième problème** : le k* optimal, le délai de prédiction choisi pour chaque type de précurseur, était sélectionné sur le jeu de test. Ça correspond à regarder les réponses avant de répondre. Correction : k* est maintenant sélectionné sur la validation uniquement, et on rapporte les résultats sur le test.

Ensuite, pour solidifier les résultats en vue d'une publication, on a ajouté :
- **Les intervalles de confiance par bootstrap** — pour chaque métrique clé, on sait maintenant que le résultat n'est pas un accident de la graine.
- **L'évaluation multi-graines** — 5 graines, 5 entraînements indépendants, pour montrer que H1 et H3 tiennent.
- **Des baselines comparatives** — un z-score pour l'alerte (pour montrer qu'on fait mieux), et des classifieurs sur features brutes pour les précurseurs (pour montrer ce que le STGCN apporte réellement : pas de l'AUROC brut, mais une structure latente interprétable).
- **L'analyse des clusters** : NMI, pureté par cluster, heatmap scénario × cluster, et validation de la méthode d'interprétabilité SHAP — qui s'avère peu fiable (ρ = −0.34 entre gradient et permutation importance), ce qu'on déclare comme limitation.

---

## Pourquoi on refait une collecte

La conclusion de H2 indique clairement le problème : 21 pas par épisode, c'est trop court. C'est trop court pour le look-through, trop court pour que les précurseurs aient un vrai signal à exploiter avant l'injection, et trop court pour capturer proprement la récupération post-panne.

Le nouveau dataset v4 corrige ça :

- La baseline passe de 5 à 8 minutes — le DriftDetector a le temps de se stabiliser.
- La pré-injection passe de 1 à 7 minutes — les précurseurs opèrent enfin dans leurs conditions nominales : un précurseur à 6 minutes (k*=12 steps) a maintenant de la marge.
- La récupération passe de 2 à 5 minutes — on capture le retour à la normale.
- On passe à 25 répétitions par scénario au lieu de 20 — pour corriger les types C6 et C9 qui manquent d'exemples.

Un épisode v4 fait ~45 pas au lieu de 21. Le coût est réel : environ 6 jours de collecte. Mais c'est la condition pour que H2 puisse être retestée honnêtement.

La collecte v4 est en cours. Elle a eu un premier arrêt : le timeout HTTP vers Loki était fixé à 30 secondes, ce qui était suffisant pour les épisodes courts de v3 mais pas pour v4 dont les fenêtres de logs sont deux fois plus grandes. Corrigé — le timeout est maintenant à 90 secondes. Le mécanisme de checkpoint garantit qu'on ne repasse pas les épisodes déjà validés.

---

## Ce qui reste à faire

Sur les résultats actuels (v3), quelques éléments manquent encore pour une publication propre :

- **Les courbes ROC et précision-rappel** pour le système d'alerte — un sweep complet sur les seuils pour visualiser le compromis détection / fausse alarme.
- **La matrice de confusion des clusters** — pas juste un taux agrégé, mais la matrice complète de qui prédit quoi, pour voir les confusions entre types.
- **Le nommage sémantique des clusters** — donner un nom lisible à chaque type (ex : "crash réseau", "déploiement défectueux", "voisin bruyant") à partir des scénarios dominants et des features les plus importantes.

Sur v4, une fois la collecte terminée :

- Réentraîner le pipeline complet et retester H2 dans ses conditions nominales.
- Faire une ablation rigoureuse : pas juste masquer des features à l'inférence, mais réentraîner le modèle complet pour chaque condition — c'est le seul moyen de mesurer l'impact réel de chaque modalité.
- Explorer des améliorations d'architecture : un pré-entraînement contrastif (SimCLR-style) au lieu de la reconstruction, et une comparaison entre le GCN actuel et une Graph Attention Network.
