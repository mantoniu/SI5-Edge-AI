= Benchmark

Nous avons réalisé les benchmarks sur chaque plateforme et obtenu les résultats illustrés dans les figures suivantes.

Pour commencer, la @sheet_consumption_over_time présente l'évolution de la consommation électrique sur chaque plateforme au cours du benchmark. Afin de permettre une comparaison équitable entre les systèmes, la consommation statique (au repos) a été soustraite de la consommation totale pour ne conserver que la part due à l'inférence des modèles.\
On peut observer que, pour la caméra et le Raspberry Pi, la consommation est non seulement très faible, mais surtout très similaire d'un modèle à l'autre. Ainsi, ces graphiques permettent principalement de visualiser la variation de la durée des benchmarks selon le modèle, et par extension la variation des temps d’inférence (décrite plus en détail avec la @sheet_mean_inference_time).\
On remarque également que les temps d'inférence de la caméra et du Raspberry Pi sont bien supérieurs à ceux du processeur (CPU) et de la carte graphique (GPU) du PC. Enfin, l'examen du graphique du GPU montre que le modèle converti en INT8 requiert beaucoup plus de temps et d'énergie que les autres. À l'inverse, la durée du benchmark pour le modèle élagué à 75 % sur GPU a été nettement plus courte que pour tous les autres modèles.

#figure(
  caption: [Graphiques représentant la consommation au cours du temps en fonction fonction des plateformes],
  grid(columns: 2,
    image("./results/courbes_CPU.png"), 
    image("./results/courbes_GPU.png"), 
    image("./results/courbes_OAK-D_Pro_V2.png"), 
    image("./results/courbes_Raspberry_Pi_4.png")
  )
)<sheet_consumption_over_time>

#box[
La @sheet_mean_consumption présente la consommation moyenne de chaque modèle par plateforme. On y observe des résultats cohérents avec la @sheet_consumption_over_time : la caméra et le Raspberry Pi consomment moins que la plateforme PC, que ce soit sur CPU ou GPU. On note également qu'en moyenne, le GPU (avec ses potentielles fallback sur CPU) consomme moins que le CPU seul. Cela s'explique par le fait que le GPU est le principal composant s'occupant des calculs et qu'il est optimisé pour les calculs massivement parallèles, ce qui correspond parfaitement à l'architecture du modèle YOLOv11.\
Cependant, pour le modèle quantifié en INT8, cette tendance s'inverse : la consommation du CPU pour les fallbacks depuis le GPU devient très élevée tandis que celle du GPU seul est la plus faible de tous les modèles. Ce phénomène s'explique par le fait que la quantification INT8 est réalisée dynamiquement ; des opérations supplémentaires sont alors ajoutées. Ces dernières n'étant pas compatibles avec le GPU, elles entraînent de nombreux renvois vers le CPU (fallback), ce qui augmente significativement la consommation électrique globale.

#figure(
  caption: [Graphique représentant la consommation moyenne en watt de chaque plateforme en fonction des modèles], 
  image(width: 60%, "comparaison_conso_fusionnee (1).png")
)<sheet_mean_consumption>
]

La @sheet_mean_inference_time présente le temps d'inférence moyen pour chaque modèle sur chaque plateforme. Comme suggéré par la @sheet_consumption_over_time, le GPU et le CPU sont les plateformes affichant les temps d'inférence les plus courts, tandis que le Raspberry Pi présente les plus longs. Cette différence structurelle s'explique par l'optimisation du GPU pour ce type de calculs. Le processeur du PC, bien que non spécifiquement optimisé, compense par une fréquence d'horloge très élevée, ce qui lui permet d'obtenir des temps d'inférence inférieurs à ceux des deux autres plateformes. En effet, bien que la caméra dispose d'un VPU (Vision Processing Unit) propice aux calculs parallèles, sa fréquence de fonctionnement reste relativement basse. Enfin, le Raspberry Pi, équipé d'un simple processeur ARM, n'est pas optimisé pour ces tâches intensives.\
Une fois de plus, on remarque que pour le modèle quantifié en INT8, le GPU affiche un temps d'inférence plus élevé que le CPU, ce qui peut paraître contre-intuitif. Comme mentionné précédemment, cela est dû au fait que certaines opérations ne sont pas prises en charge par le GPU et sont renvoyées au CPU. Cela impose des transferts de données récurrents entre le CPU et le GPU durant l'inférence, ralentissant ainsi l'ensemble du processus.

#figure(
  caption: [Graphique représentant le temps d'inférence moyen en millisecondes de chaque plateforme en fonction des modèles],
  image(width: 60%, "./results/comparaison_temps_inference.png")
)<sheet_mean_inference_time>

Enfin, la @sheet_energy_consumption_summary résume l'ensemble des métriques capturées en mettant en relation la mAP (Mean Average Precision) avec l'énergie consommée. L'énergie est ici calculée en multipliant le temps d'inférence moyen (converti en secondes) par la consommation moyenne, permettant d'obtenir un résultat en Joules. Deux graphiques sont présentés : l'un corrèle la mAP des boîtes de détection et l'autre la mAP des masques de segmentation.\
Sur ces deux graphiques, on observe deux groupes distincts concernant le score mAP : d'un côté les modèles élagués (pruned) et de l'autre les modèles quantifiés. Cet écart vient du fait que les modèles élagués nécessitent un ré-entraînement. Nous avons limité celui-ci à une quinzaine de minutes mais un temps d'entraînement plus long aurait sans doute permis d'obtenir de meilleurs résultats.\
Si l'on exclut les modèles quantifiés en INT8 et le modèle en FP32, l'analyse de la consommation énergétique permet de distinguer deux groupes. Le premier, constitué de la caméra et du GPU, se montre le plus efficace : la caméra bénéficie d'un temps d'inférence très court, tandis que le GPU offre un bon compromis entre consommation et rapidité. Le second groupe, réunissant le Raspberry Pi et le CPU, affiche une consommation par inférence nettement plus élevée. Cela s'explique par un calcul séquentiel qui s'avère soit très coûteux en puissance (pour le CPU), soit excessif en temps d'exécution (pour le Raspberry Pi).

#figure(
  caption: [Graphiques représentant la consommation énergétique en joule de chaque plateforme en fonction du score mAP évalué sur les boites de détection et les masques de segmentation],
  grid(columns: 2, 
    image("./results/comparaison_efficacite_box.png"), 
    image("./results/comparaison_efficacite_mask.png")
  )
)<sheet_energy_consumption_summary>
