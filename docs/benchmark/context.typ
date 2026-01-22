= Contexte

Le projet global vise à concevoir un robot capable de suivre une personne et de reconnaître des gestes afin de contrôler ses déplacements (le stopper, lui dire de reprendre le suivi, etc.). Pour répondre à ces besoins, il est nécessaire de disposer d’un modèle de vision capable de détecter les personnes en temps réel sur un système embarqué.

Dans ce contexte, nous avons sélectionné le modèle #link("https://docs.ultralytics.com/fr/models/yolo11/")[YOLOv11 nano] avec segmentation d’instances. Ce choix est motivé par la nécessité d'estimer la distance entre la personne et le robot. Cela nécessite de segmenter l'instance de la personne sur l'image de profondeur afin de réaliser ce calcul. De plus, ce choix repose sur son architecture améliorée qui offre une efficacité supérieure à celle de ses prédécesseurs.

Comparativement à YOLOv8, YOLOv11 réduit le nombre de paramètres d'environ 22%, allégeant la charge sur notre système embarqué tout en augmentant la précision (mAP). Face à YOLOv10, nous avons privilégié la v11 pour son support natif de la segmentation d'instance, élément critique pour notre calcul de profondeur. Cette précision accrue nous permet de fiabiliser l'estimation de la distance personne-robot sans compromettre la fluidité de la navigation en temps réel.

#figure(caption: [Différences entre les versions de YOLO], table(
 columns: 4,
 align: horizon,
 table.header([], [*YOLOv8*], [*YOLOv10*], [*YOLOv11*]),
 [*Tâches*                       ], [Segmentation,  \ détection,\ ...], [Détection], [Segmentation, \ détection,\ ...],
 [*Temps d'inference CPU ONNX\ (ms)*], [96.1], [-],    [65.8],
 [*Nombre de paramètres (M)*       ], [3.4],  [2.3],  [2.9],
 [*mAP Box 50%-95% (%)*            ], [36.7], [39.5], [38.9],
 [*mAP Mask 50%-95% (%)*           ], [30.5], [-],    [32.0],
))

L’objectif spécifique de ce rapport est de comparer différentes méthodes d’optimisation du modèle adaptées aux contraintes de l’edge computing, tout en quantifiant l'impact de l'accélération matérielle (NPU/VPU) face à un CPU généraliste. 

Cette étude s’appuie sur la mise en place d’une pipeline de benchmarking permettant d’évaluer ces approches selon trois axes principaux, afin de dégager des compromis pertinents entre temps d’inférence, performances et consommation de ressources
