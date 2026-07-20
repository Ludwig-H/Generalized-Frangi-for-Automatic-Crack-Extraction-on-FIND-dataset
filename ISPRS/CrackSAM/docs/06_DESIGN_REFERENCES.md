# Références qui motivent l'architecture

Cette sélection ne prétend pas montrer que FrangiGraph-Residual fonctionnera.
Elle identifie les briques déjà étayées, puis les hypothèses propres au projet
qui doivent rester des ablations.

| Travail | Résultat transférable | Conséquence ici |
|---|---|---|
| [CrackSAM](https://arxiv.org/abs/2312.04233) | SAM 1 peut être adapté aux fissures par PEFT et évalué hors domaine | reproduire fidèlement son périmètre entraînable avant d'attribuer un écart à SAM 2 |
| [ViT-Adapter](https://arxiv.org/abs/2205.08534) | une branche légère peut apporter des biais spatiaux à un Transformer pour la prédiction dense | encoder les cartes Frangi comme features, pas comme pseudo-masque |
| [HQ-SAM](https://arxiv.org/abs/2306.01567) | la fusion de features précoces et finales améliore les détails de masques complexes | exploiter les features haute résolution pour les fissures minces |
| [SAM-Road](https://arxiv.org/abs/2403.16051) | des embeddings SAM et un Graph Transformer léger peuvent vérifier les arêtes d'un réseau | vérifier les nœuds/arêtes Frangi avec les features SAM, après le MVP raster |
| [clDice](https://arxiv.org/abs/2003.07311) | une loss différentiable cible la connectivité des structures tubulaires | ajouter une supervision topologique sans remplacer IoU/Dice |
| [Skeleton Recall Loss](https://arxiv.org/abs/2404.03010) | perte efficace pour structures minces, incluant les fissures de béton | ablation topologique moins coûteuse sur G4 |
| [Segment Any Crack](https://arxiv.org/abs/2504.14138) | l'ajustement sélectif des normalisations peut dépasser plusieurs PEFT sur ses jeux | inclure « normalisations + decoder » dans les baselines, sans en faire un résultat acquis pour Hiera |

## Pourquoi SAM 2.1 avant SAM 3

Le [dépôt officiel SAM 2](https://github.com/facebookresearch/sam2) distribue
les checkpoints et configurations SAM 2.1. Ce changement peut être isolé dans
une baseline contrôlée sans remplacer l'architecture complète.

[SAM 3](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
ajoute des prompts de concept par texte ou exemplaire et une tête de présence.
Cela en fait un vérificateur sémantique plausible pour une composante Frangi,
mais pas une solution directe à la résolution, à la connectivité ou à la
calibration du graphe. Le pilote doit donc réutiliser la même tête résiduelle et
mesurer l'apport du backbone à protocole constant.

[SAM 3.1](https://ai.meta.com/blog/segment-anything-model-3/) optimise surtout
le suivi vidéo multi-objet. Pour des images statiques de fissures, ce n'est pas
une raison suffisante de remplacer la baseline.

## Hypothèses encore propres à ce projet

Les points suivants ne sont pas démontrés par la littérature et constituent le
cœur des ablations :

- la similarité Frangi contient un signal conditionnel utile à poids SAM fixes ;
- des features vallée/marche et de stabilité multi-échelle distinguent assez
  les fissures des frontières d'ombre ;
- la qualité d'une composante peut être prédite sans consulter le GT ;
- le graphe explicite apporte davantage que sa rasterisation multicanal ;
- une gate out-of-fold réduit réellement la queue des pertes.

La [feuille de route](04_IMPLEMENTATION_ROADMAP.md) est ordonnée pour pouvoir
invalider chacune de ces hypothèses avant l'étape GPU suivante.
