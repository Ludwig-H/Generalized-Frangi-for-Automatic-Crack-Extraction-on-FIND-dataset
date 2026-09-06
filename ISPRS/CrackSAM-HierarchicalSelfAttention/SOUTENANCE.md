# Perspective : apprendre à lire la hiérarchie géométrique

## Une diapositive dans le prolongement du LiDAR 3D

**Titre : « Perspective — apprendre à quelle échelle lire la géométrie »**

![Principe du lecteur](figures/lecteur_polyedres.png)

À afficher :

- L’encodeur gelé fournit les représentations.
- Nos éléments conservent leur géométrie, leurs incidences et leurs niveaux de fusion.
- Conserver ce que les enfants partagent et ce qui les distingue.
- Un petit lecteur apprend l’importance de ces différences selon les regroupements.
- Les appartenances aux points restent multiples jusqu’à la décision finale.

À dire, environ 45 secondes :

> Chaque élément reçoit une description fournie par un encodeur gelé. Lorsqu’un groupe se forme, nous conservons sa description moyenne et les différences de ses enfants. Un petit lecteur apprend ensuite, selon la géométrie et le niveau de fusion, quels détails préserver ou renforcer. Nous pouvons ainsi tester l’utilité des regroupements intermédiaires sans réentraîner le grand modèle. L’hypothèse est que cette organisation aide à interpréter la scène, notamment quand les observations deviennent moins denses.

**Référence principale :** [Robert, Raguet et Landrieu, Superpoint Transformer, ICCV 2023](https://arxiv.org/abs/2306.08045), pour une hiérarchie géométrique organisant un Transformer de segmentation 3D. Notre lecteur prolongerait ce principe aux événements et aux recouvrements de la thèse.

**Pour le mécanisme minimal :** [Saito, Schonsheck et Shvarts, 2024](https://link.springer.com/article/10.1007/s43670-023-00076-4), pour des représentations multirésolution de signaux sur arêtes et faces. Cette référence concerne la représentation ; le Transformer reste une variante du lecteur.

## Une ouverture courte pour EUVIP

**Titre : « Perspective — lire la hiérarchie Frangi avec des représentations gelées »**

> SAM fournit les caractéristiques visuelles. La hiérarchie Frangi organise leurs regroupements. Un petit lecteur pourrait apprendre à quelle échelle conserver les différences ou partager le contexte, tout en gardant SAM gelé.

Le graphe EUVIP est le cas K = 1. Le passage à des atomes d’ordre supérieur serait une extension, pas un résultat du papier.

## Précisions pour les questions du jury

**« En quoi l’arbre ne suffit-il pas ? »** Il décrit l’emboîtement. Il faut aussi conserver les éléments et les connecteurs qui portent la géométrie. Partager un point et fusionner par un triangle sont deux relations différentes, comme l’illustre [Cell Attention Networks](https://arxiv.org/abs/2209.08179).

**« Aucun apprentissage ? »** L’encodeur est gelé ; le petit lecteur est appris sur ses caractéristiques sauvegardées. Aucun gradient ne traverse le grand modèle.

**« Les surfaces sont-elles déjà disponibles ? »** K = 2 travaille sur des arêtes. K = 3 fournit des atomes triangulaires ; leur interprétation surfacique et l’export vers le réseau restent à qualifier.

La [note approfondie](VOIE_POLYEDRES.md) distingue les acquis, l’architecture proposée et les contrôles.

La [lecture multirésolution](LECTURE_MULTIECHELLE.md) précise le premier mécanisme et un contrôle essentiel : avec des gains constants, les niveaux intermédiaires disparaissent du calcul.
