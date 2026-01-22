= Protocole du benchmark

== Objectif et conditions expérimentales

Le protocole de benchmark a pour objectif de mesurer de manière quantitative le temps d’inférence ainsi que la consommation énergétique associés à l’exécution du modèle de segmentation de personnes sur différentes plateformes matérielles. L’ensemble du protocole a été conçu de manière à garantir des conditions d’évaluation identiques et reproductibles, afin de permettre une comparaison équitable entre les différents environnements testés.

Pour chaque appareil, le benchmark exécute l’ensemble des modèles successivement, chaque modèle étant évalué sur un ensemble de 100 images. Les images utilisées sont identiques pour tous les tests et proviennent du dataset COCO, en se limitant à la classe person. Entre l’exécution de deux modèles différents, un temps d’attente de 10 secondes est systématiquement appliqué afin de laisser au système le temps de revenir à un état de consommation au repos. Cette pause permet à la fois de stabiliser le système et de mesurer la consommation énergétique hors inférence, qui servira de référence lors de l’analyse des résultats.

== Architecture de la pipeline d’inférence

Afin de comparer équitablement les performances de différents environnements d’exécution, une abstraction de l’inférence a été mise en place à travers la classe InferenceBackend. Cette classe abstraite définit une interface commune pour l’exécution de l’inférence, indépendamment du matériel ou du framework utilisé. Elle impose notamment une méthode predict() que toutes les implémentations concrètes doivent fournir, ce qui permet d’uniformiser la manière dont les modèles sont appelés au sein de la pipeline de benchmark et de rendre celle-ci indépendante de toute implémentation spécifique.

Deux implémentations concrètes de cette abstraction ont été développées. La première, OnnxBackend, permet d’exécuter l’inférence d’un modèle YOLO exporté au format ONNX à l’aide d’ONNX Runtime, aussi bien sur CPU que sur GPU. La seconde, OakDBackend, exécute l’inférence directement sur une caméra OAK-D Pro en exploitant son NPU intégré via la librairie DepthAI. Bien que ces deux backends reposent sur des mécanismes d’exécution très différents, ils exposent exactement la même interface côté pipeline, ce qui garantit une utilisation totalement interchangeable au sein du benchmark.

#box[
L’un des objectifs majeurs de cette abstraction est la standardisation du pré-traitement et du post-traitement des données. Le pré-traitement inclut le redimensionnement des images ainsi que l’application du letterboxing afin de conserver le ratio d’aspect. Le post-traitement regroupe quant à lui le décodage des sorties du modèle YOLO, l’application des seuils de confiance et du Non-Maximum Suppression (NMS), la génération des bounding boxes et des masques de segmentation, ainsi que la suppression du letterboxing pour reprojeter correctement les résultats sur l’image originale. Cette standardisation garantit que les différences observées lors du benchmark proviennent uniquement du backend d’inférence et du matériel sous-jacent, et non de variations dans le traitement des données.
]

Grâce à cette approche, la pipeline de benchmark est totalement découplée des implémentations spécifiques d’inférence. Elle peut ainsi être utilisée pour évaluer différents matériels ou frameworks sans nécessiter de modifications structurelles. L’ajout d’un nouveau backend se limite à l’implémentation de la classe abstraite InferenceBackend, ce qui améliore la lisibilité, la maintenabilité et l’extensibilité du code, tout en assurant des résultats comparables, reproductibles et méthodologiquement rigoureux.

== Mesure de la consommation

Pour la caméra OAK-D Pro et le Raspberry Pi, un dongle USB-C est utilisé afin de mesurer la consommation énergétique des appareils. Ce dongle est intercalé entre la source d’alimentation et le dispositif à mesurer. Les données de consommation sont récupérées en filmant l’affichage du dongle pendant toute la durée des benchmarks, puis en traitant la vidéo à l’aide d’un script Python permettant d’extraire un ensemble de valeurs à une fréquence d’une mesure par seconde.

== Scénarios

Sur la base de ce protocole commun, plusieurs scénarios expérimentaux ont été définis afin d’évaluer le comportement du pipeline d’inférence sur différentes plateformes matérielles.

=== Scénario 1 : Caméra OAK-D Pro

Dans ce scénario, la caméra OAK-D Pro est connectée à un ordinateur hôte avec un dongle de mesure de la consommation électrique. L'inférence est exécutée directement sur le VPU embarqué de la caméra avec les images envoyées à la caméra également via le pipeline créé afin de lancer le modèle sur le VPU de la caméra.

=== Scénario 2 : Raspberry Pi 4

#box[Ce scénario vise à évaluer l’exécution du modèle sur une plateforme embarquée sans accélération matérielle dédiée. 

Le dongle de mesure est placé entre l'alimentation et le Raspberry Pi pour capturer la consommation globale du système. Seul le benchmark est exécuté sur l'appareil, avec un système d'exploitation épuré de toute autre application active. Il est primordial que la consommation hors inférence reste stable et constante tout au long des tests pour garantir la fiabilité des mesures.]

=== Scénario 3 : Ordinateur portable

Dans ce scénario, un ordinateur portable est utilisé pour exécuter le pipeline d’inférence. La consommation des ressources matérielles est suivie à l’aide d’outils logiciels tels que NVML ou HWiNFO. Ces deux outils se basent sur des résistances intégrées au matériel, un driver peut ensuite récupérer la tension aux bornes de ces résistances pour en déduire l'intensité du courant et donc la puissance globale consommée. Comme pour les autres scénarios, la consommation au repos sera prise en compte afin d’isoler l’impact du traitement du modèle.
