= Les modèles et les optimisations

Dans sa configuration native, le modèle _YOLOv11 Nano Segmentation_ est encodé en FP32 (Floating Point 32-bit) et totalise 2,9 millions de paramètres. 
Nous avons mis en œuvre plusieurs types d'optimisations pour ensuite comparer leurs effets sur les performances selon nos critères d'évaluation.

== Optimisations générales

=== Quantification

La quantification est une technique incontournable pour le déploiement sur cibles matérielles contraintes.

Elle consiste en la réduction de la précision numérique des poids et des activations du modèle. En passant d'une représentation à haute précision (généralement 32 bits) à une représentation plus compacte, elle vise trois objectifs majeurs :
- Accélérer l'inférence : Les opérations sur des entiers ou des flottants de taille réduite sont traitées plus rapidement par les unités de calcul.
- Réduire la consommation énergétique : Moins de calculs et de transferts de données impliquent une baisse de la puissance requise.
- Diminuer l'empreinte mémoire : Le modèle occupe moins d'espace de stockage et de RAM, facilitant son intégration sur des dispositifs limités.

Il existe deux approches principales pour appliquer la quantification, chacune présentant des compromis différents :

#figure(caption: [Comparatif des stratégies de quantification], table(
  columns: 3,
  table.header([], [*Quantification\ Dynamique*], [*Quantification\ Statique*]),
  [*Quantification*], [Hybride\ #text(size: .8em)[Poids quantifiés hors-ligne,\ mais activations quantifiées à la volée]], [Hors-ligne\ #text(size: .8em)[Tout est calculé et fixé avant le déploiement]],
  [*Prérequis*], [Aucun], [Jeu de données de calibration\ #text(size: .8em)[pour fixer les échelles]],
  [*Vitesse\ d'inférence*], [Gains modérés\ #text(size: .8em)[Le calcul des facteurs d'échelle\ à l'exécution crée une surcharge]], [Maximale\ #text(size: .8em)[Opérations purement entières,\ exploitant pleinement\ les accélérateurs matériels]],
  [*Empreinte\ mémoire*], [Réduite\ #text(size: .8em)[Car nécessite de la mémoire tampon pour les conversions dynamiques]], [Minimale\ #text(size: .8em)[Le modèle est chargé et exécuté\ directement en format compressé]]
))


Dans un souci de simplification du protocole expérimental, nous avons choisi de nous concentrer exclusivement sur la quantification statique dans cette étude.

Nous avons décliné et évalué le modèle YOLOv11 selon trois formats de représentation des données :
- FP32 (Floating Point 32-bit) : Le format standard en simple précision, offrant la plus grande fidélité mais nécessitant plus de ressources.
- FP16 (Floating Point 16-bit) : Une version en demi-précision, réduisant la taille du modèle tout en conservant des poids en virgule flottante.
- INT8 (Integer 8-bit) : Une version compressée utilisant des nombres entiers sur 8 bits, maximisant la vitesse et réduisant l'empreinte mémoire.

La prise en charge de ces formats varie selon le matériel utilisé. Si les architectures classiques (CPU, GPU) et le Raspberry Pi supportent l'ensemble des formats, la caméra OAK-D Pro V2 est limitée exclusivement au format FP16. À noter que cette limitation a été levée sur la version 4 (OAK-D Pro V4), qui supporte davantage de formats.
    
#figure(caption: [Compatibilité des formats par plateforme], table(
 columns: 4,
 table.header([*Plateforme*], [*FP32*], [*FP16*], [*INT8*]),
 [CPU         ], [#sym.checkmark], [#sym.checkmark], [#sym.checkmark],
 [GPU         ], [#sym.checkmark], [#sym.checkmark], [#sym.checkmark],
 [Raspberry Pi], [#sym.checkmark], [#sym.checkmark], [#sym.checkmark],
 [OAK-D Pro V2], [              ], [#sym.checkmark], [              ],
))

=== Pruning structuré

Contrairement au pruning classique (non-structuré) qui se contente de fixer la valeur de certains poids à zéro sans modifier l'architecture, le pruning structuré supprime physiquement des canaux ou des couches entières du réseau.

Cette méthode présente deux avantages majeurs :  
- Le fichier du modèle est allégé de manière effective.
- L'inférence est réellement accélérée, car le matériel de calcul n'a pas besoin de traiter des matrices "creuses" (sparse), mais simplement des matrices plus petites.

Pour mettre en œuvre cette opération complexe, nous nous sommes appuyés sur le projet #link("https://github.com/heyongxin233/YOLO-Pruning-RKNN")[YOLO-Pruning-RKNN] de _heyongxin233_. 
Nous avons généré plusieurs variantes du modèle (voir @pruning) en faisant varier le taux de suppression des poids.

#figure(
  caption: [Impact du pruning sur le nombre de paramètres],
  box(table(
    columns: 8,
    [*Taux de suppression*], [5%], [10%], [15%], [20%], [25%], [50%], [75%],
    [*Paramètres restants (M)*], [2.755], [2.610], [2.465], [2.320], [2.175], [1.450], [0.725],
  ))
)<pruning>

== Optimisations spécifiques aux plateformes

Afin de tirer parti des architectures spécifiques de chaque plateforme, nous avons adapté les stratégies d'exécution (Providers) et les paramètres d'inférence pour chacune d'entre elles.

Sur la OAK-D Pro, l'optimisation se joue au niveau de l'allocation des ressources du VPU Myriad X. Nous avons configuré l'utilisation de 8 SHAVEs (cœurs vectoriels dédiés au traitement neuronal) sur les 16 disponibles. Selon la documentation officielle, cette répartition constitue le compromis technique idéal : elle maximise la vitesse d'inférence tout en réservant la puissance de calcul nécessaire aux tâches parallèles critiques, telles que la génération de la carte de profondeur et l'encodage du flux vidéo.

Pour l'environnement PC, nous avons distingué deux cas de figure en utilisant des moteurs d'exécution (providers) différents. Pour l'évaluation du processeur seul, nous avons utilisé le _CPUExecutionProvider_ standard. En revanche, pour bénéficier de l'accélération matérielle, nous avons sélectionné le _CUDAExecutionProvider_. Ce dernier permet de déporter les calculs sur la carte graphique NVIDIA, tirant ainsi parti du parallélisme massif des cœurs CUDA pour accélérer les opérations matricielles et libérer le processeur central.

Enfin, sur Raspberry Pi 4, bien que l'utilisation du _ACLExecutionProvider_ (basé sur l'Arm Compute Library) aurait été préférable pour exploiter les instructions NEON propres à l'architecture ARM, l'indisponibilité d'une version pré-compilée nous a contraints à utiliser le _CPUExecutionProvider_.
Pour compenser l'absence d'accélération matérielle dédiée, nous avons activé plusieurs optimisations : 
  - Graph Optimization : Fusion des couches pour réduire le nombre d'opérations.
  - Multi-threading : Exploitation des 4 cœurs physiques pour paralléliser les calculs.
  - Gestion Mémoire : Activation de drapeaux spécifiques pour prévenir la saturation de la RAM.
  - Mode Séquentiel : Imposition d'un ordre d'exécution strict pour stabiliser la charge CPU.
