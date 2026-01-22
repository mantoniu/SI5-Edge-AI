= Plateformes visées

Pour valider la portabilité et l'efficacité du modèle de vision, l'évaluation expérimentale s'appuie sur trois plateformes aux profils matériels distincts. 
L'objectif est de balayer un large spectre de puissance de calcul, allant d'un système embarqué fortement contraint (Raspberry Pi) à un environnement de référence non limité (Ordinateur portable) mais aussi une caméra 3D (Luxonis OAK-D Pro) pouvant supporter des réseaux de neurones.
Cette segmentation nous permettra non seulement de mesurer les temps d'inférence bruts, mais aussi d'analyser l'impact de l'accélération matérielle et des optimisations logicielles sur la consommation énergétique globale.

== Raspberry Pi 4 modèle B

Le Raspberry Pi 4 modèle B constitue la plateforme embarqué à ressources limitées de notre étude. Il est équipé d’un processeur ARM Cortex-A72 quad-cœur cadencé à 1,5 GHz et ne dispose pas d’accélération matérielle dédiée pour l’inférence de réseaux de neurones.

Cette plateforme représente un cas d’usage réaliste pour un robot mobile fonctionnant sur batterie, où l’inférence est effectuée exclusivement sur CPU. Elle permet d’évaluer la faisabilité du modèle dans un environnement fortement contraint, ainsi que l’impact des optimisations logicielles (quantification, pruning) sur les performances temporelles et énergétiques.

== Caméra Luxonis OAK-D Pro V2

La Luxonis OAK-D Pro V2 est une caméra 3D intégrant un VPU Intel Myriad X, capable de délivrer jusqu’à 4 TOPS _(Trillions Operations Per Second)_ avec ses 16 cœurs SHAVEs _(Streaming Hybrid Architecture Vector Engine)_ cadencés à environ 700 MHz. À la différence du Raspberry Pi, le calcul d'inférence est ici délesté directement sur le matériel de la caméra.

Cette architecture permet de tirer parti d'une accélération matérielle dédiée tout en libérant les ressources du système hôte. L’OAK-D Pro sert de point de comparaison pour illustrer la plus-value d’un accélérateur spécialisé face à une exécution CPU conventionnelle.

#pagebreak()

== Ordinateur portable

La plateforme de référence haute performance retenue pour cette étude est un ordinateur portable. Sa configuration matérielle s'articule autour d'un processeur AMD Ryzen 5 8645HS (6 cœurs / 12 threads à 4,3 GHz) et de 32 Go de mémoire vive DDR5. La partie graphique est assurée par une carte NVIDIA RTX 4060 Laptop, dotée de 3072 cœurs CUDA à 2,1 GHz et de 8 Go de VRAM GDDR6.

Les tests y sont conduits selon deux configurations : exécution sur CPU seul et exécution accélérée par GPU (avec basculement CPU si nécessaire). Cette plateforme offre des conditions d'exécution nettement moins restrictives, bénéficiant d'une puissance de calcul supérieure à celle des systèmes embarqués.

Cet environnement sert principalement d'étalon (baseline) pour :
- Comparer les temps d’inférence face aux matériels embarqués.
- Analyser les écarts de consommation énergétique.
- Isoler l’impact des optimisations du modèle dans un contexte non contraint.

Enfin, cette plateforme permet de valider l'intégrité du pipeline d’inférence, garantissant que les limitations observées sur les autres supports découlent bien des contraintes matérielles et non de l'architecture du modèle elle-même.
