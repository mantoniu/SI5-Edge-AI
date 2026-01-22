= Critères de benchmark

Afin de qualifier la pertinence des différentes architectures matérielles et logicielles pour notre système de suivi, nous avons défini trois axes d'évaluation distincts. Ces critères visent à garantir que la solution retenue respecte les contraintes de temps réel, d'autonomie énergétique et de fiabilité de détection.

== Performance temporelle

Dans notre cas d'usage, l'objectif est de minimiser la latence pour rester synchronisé avec la réalité, tout en assurant une fréquence (FPS) suffisante pour un suivi fluide.

Pour évaluer cette rapidité, nous nous basons sur le temps d'inférence. Celui-ci correspond au délai introduit spécifiquement par le réseau de neurones, calculé en mesurant la durée écoulée entre l'entrée et la sortie du modèle.

== Efficacité énergétique

Dans un environnement embarqué contraint par l'autonomie de la batterie, l'optimisation énergétique est primordiale. Notre objectif est d'isoler la consommation dynamique imputable au traitement IA, indépendamment de la consommation statique du matériel. Le protocole de mesure consiste à évaluer le différentiel entre la consommation du système en charge (durant l'inférence) et sa consommation basale (au repos/idle).

== Qualité de prédiction

Enfin, l'optimisation de la vitesse et de la consommation ne doit pas se faire au détriment de la perception du robot. L'objectif est de maximiser la précision pour garantir que les personnes sont correctement détectées et segmentées.

La fiabilité du modèle est quantifiée par la mAP (mean Average Precision), calculée selon le standard COCO. Cette métrique se construit en trois étapes :

D'abord, pour qualifier une prédiction, il faut mesurer sa cohérence géométrique avec la réalité. Nous utilisons l'IoU (Intersection over Union), qui représente le rapport entre la surface commune (intersection) et la surface totale (union) des deux zones (prédiction et vérité terrain).

$ "IoU"=frac("Area of Overlap"​, "Area of Union") $

Un seuil d'IoU est défini pour classer les détections :
- Vrai Positif (TP) : L'IoU dépasse le seuil (détection correcte).
- Faux Positif (FP) : L'IoU est sous le seuil (mauvaise localisation ou détection d'un objet inexistant).
- Faux Négatif (FN) : Le modèle a manqué une personne présente.

#pagebreak()
Ensuite, deux indicateurs sont calculés :
- La Précision (P) : La fiabilité des détections positives. $"Précision"=frac("TP", "TP"+"FP")​$
- Le Rappel (R) : La capacité à trouver toutes les occurrences. $"Rappel"=frac("TP", "TP"+"FN")​​$

Enfin, pour synthétiser le compromis entre ces deux valeurs, nous calculons l'Average Precision (AP), l'aire sous la courbe Précision-Rappel.

Cependant, un seuil d'IoU unique (ex: 0.5) est insuffisant pour la précision requise par un robot suiveur. Conformément au protocole COCO, la mAP finale est la moyenne des AP calculées pour 10 seuils d'IoU différents (de 0.5 à 0.95 par pas de 0.05).
$ "mAP"=frac(1,10) sum_("IoU"=0.5)^0.95 "AP"_"IoU" $

Cette méthode pénalise les modèles qui détectent la bonne classe (personne) mais avec un positionnement approximatif (IoU faible).

Le modèle utilisé étant de type YOLOv11n-seg, l'évaluation se fait sur deux niveaux distincts:
- mAP Box : Évalue la précision des cadres englobants (bounding boxes). C'est un indicateur de la capacité du robot à localiser grossièrement la cible.
- mAP Mask : Évalue la précision de la segmentation pixel par pixel (le masque de la silhouette).

Nos résultats présenteront ces deux valeurs afin de vérifier si les optimisations (comme la quantification ou le pruning) dégradent la finesse du masque ou la détection globale (Box).
