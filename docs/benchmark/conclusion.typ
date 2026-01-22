= Conclusion

#let good = rgb("#64b052").lighten(30%).saturate(20%)
#let low_good = rgb("#9bcb69").lighten(30%).saturate(20%)
#let mid = rgb("#f5d268").lighten(30%).saturate(20%)
#let bad = rgb("#f29f28").lighten(30%).saturate(20%)
#let really_bad = rgb("#d6191b").lighten(30%).saturate(20%)

Grâce à l'ensemble des métriques obtenues durant les benchmarks et à l'étude des résultats réalisée précédemment, nous pouvons déterminer la solution optimale en combinant la plateforme et le modèle les plus adaptés.

En effet, pour notre cas d’usage initial (un robot suiveur de personne) le meilleur choix de plateforme est sans conteste la caméra Luxonis Oak-D Pro V2. Elle offre des résultats de précision (mAP) similaires aux autres plateformes pour tous les modèles testés, qu'il s'agisse des boîtes de détection ou des masques de segmentation.

De plus, son temps d'inférence moyen, compris entre 229 ms et 263 ms selon les modèles, est tout à fait acceptable. Cette latence permet de mettre à jour la position de la cible environ 3 fois par seconde, ce qui est suffisant pour assurer un suivi fluide en temps réel.

Enfin, pour un robot fonctionnant sur une batterie à capacité limitée, la sobriété énergétique est un enjeu crucial. À cet égard, la caméra s'impose comme la plateforme la plus efficiente, affichant la consommation électrique la plus faible, que ce soit durant les phases d'inférence ou au repos.


#box(figure(
  caption: [Synthèse comparative des plateformes],
  table(
    columns: 4,
    table.header([], [*Consommation moyenne*], [*Temps d'inférence*], [*Efficacité\ énergétique*]),
    
    [*PC (GPU)*], 
    table.cell(fill:bad)[Élevée], 
    table.cell(fill:good)[Très Rapide], 
    table.cell(fill:good)[Excellente],
    
    [*OAK-D Pro*], 
    table.cell(fill:good)[Très Faible], 
    table.cell(fill:bad)[Moyen / Lent], 
    table.cell(fill:good)[Excellente],
    
    [*PC (CPU)*], 
    table.cell(fill:really_bad)[Très élevée], 
    table.cell(fill:mid)[Rapide], 
    table.cell(fill:bad)[Faible],
    
    [*Raspberry Pi 4*], 
    table.cell(fill:good)[Très Faible], 
    table.cell(fill:really_bad)[Très Lent], 
    table.cell(fill:really_bad)[Très Faible]
  )
))


L'ensemble de ces facteurs confirme que la caméra constitue le choix de plateforme le plus rationnel. Toutefois, cette décision doit être couplée à la sélection d'un modèle dont l'optimisation est en parfaite adéquation avec les capacités du matériel.

À cet égard, le modèle de base quantifié en FP16 (format requis pour l'exécution sur le VPU de la caméra) apparaît comme la solution la plus performante, car il offre les meilleurs scores mAP. Par ailleurs, bien qu'un modèle élagué (pruned) permettrait de réduire le temps d'inférence de quelques dizaines de millisecondes, cette option nécessiterait un ré-entraînement prolongé pour garantir une précision équivalente au modèle FP16. Étant donné que la latence actuelle du modèle FP16 n'est pas rédhibitoire pour notre application, la priorité est accordée au maintien de la précision maximale plutôt qu'à un gain de temps marginal.

#box(figure(
  caption: [Synthèse comparative des modèles],
  table(
    columns: (auto, 1fr, 1fr),
    align: horizon,
    table.header([], [*Précision*], [*Temps d'inférence*]), 
    
    [*Modèle fortement élagué\ (Pruned 75%)*], 
    table.cell(fill:really_bad)[Très faible], 
    table.cell(fill:good)[Très rapide], 
    
    [*Modèle faiblement élagué\ (Pruned 5%)*], 
    table.cell(fill:bad)[Faible],
    table.cell(fill:good)[Rapide],
    
    [*Modèle fortement quantifié\ (INT8)*],
    table.cell(fill:low_good)[Élevée], 
    table.cell(fill:bad)[Moyen / Lent (fallback)], 
    
    [*Modèle faiblement quantifié\ (INT16)*], 
    table.cell(fill:good)[Très élevée],
    table.cell(fill:mid)[Moyen],
    
    [*Modèle standard\ (FP32)*],
    table.cell(fill:good)[Très élevée],
    table.cell(fill:mid)[Moyen], 
  )
))

En conclusion, le binôme formé par la caméra OAK-D Pro V2 et le modèle YOLOv11n quantifié en FP16 constitue le choix optimal pour notre projet. Cette décision est validée par l'ensemble des benchmarks réalisés et l'analyse rigoureuse des métriques capturées lors de cette étude.
