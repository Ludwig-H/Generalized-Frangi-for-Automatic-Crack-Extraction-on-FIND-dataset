# Perspective : apprendre à lire la hiérarchie géométrique

## Une diapositive dans le prolongement du LiDAR 3D

**Titre : « Des éléments géométriques à une grammaire de leurs regroupements »**

![Principe du lecteur](figures/lecteur_polyedres.png)

À afficher :

- L’encodeur gelé fournit les représentations.
- Nos éléments conservent leur géométrie, leurs incidences et leurs niveaux de fusion.
- Un petit lecteur apprend à combiner détail local et contexte hiérarchique.
- Les appartenances aux points restent multiples jusqu’à la décision finale.

À dire, environ 45 secondes :

> L’idée est de faire apprendre au modèle comment les éléments géométriques s’assemblent. Chaque élément conserve sa description ; lorsqu’un groupe se forme, le lecteur combine les informations de ses enfants, puis leur rend le contexte de l’ensemble. Les niveaux de fusion participent au calcul. Nous pouvons tester cette grammaire avec un petit module sur des caractéristiques déjà calculées, sans réentraîner un grand modèle. Il faudra vérifier que les événements de la hiérarchie apportent davantage que la seule géométrie locale.

**Référence principale :** [Robert, Raguet et Landrieu, Superpoint Transformer, ICCV 2023](https://arxiv.org/abs/2306.08045), pour une hiérarchie géométrique organisant un Transformer de segmentation 3D. Notre lecteur prolongerait ce principe aux événements et aux recouvrements de la thèse.

## Une ouverture courte pour EUVIP

**Titre : « Perspective — lire la hiérarchie Frangi avec des représentations gelées »**

> SAM fournit les caractéristiques visuelles. Le graphe Frangi fournit une succession de regroupements. Un petit lecteur pourrait apprendre à exploiter leurs relations pour améliorer la prédiction, tout en gardant SAM gelé. Ce serait un premier terrain 2D pour explorer le principe d’une grammaire géométrique.

Le graphe EUVIP est le cas K = 1. Le passage à des atomes d’ordre supérieur serait une extension, pas un résultat du papier.

## Précisions pour les questions du jury

**« En quoi l’arbre ne suffit-il pas ? »** Il décrit l’emboîtement. Il faut aussi conserver les éléments et les connecteurs qui portent la géométrie. Partager un point et fusionner par un triangle sont deux relations différentes, comme l’illustre [Cell Attention Networks](https://arxiv.org/abs/2209.08179).

**« Aucun apprentissage ? »** L’encodeur est gelé ; le petit lecteur est appris sur ses caractéristiques sauvegardées. Aucun gradient ne traverse le grand modèle.

**« Les surfaces sont-elles déjà disponibles ? »** K = 2 travaille sur des arêtes. K = 3 fournit des atomes triangulaires ; leur interprétation surfacique et l’export vers le réseau restent à qualifier.

La [note approfondie](VOIE_POLYEDRES.md) distingue les acquis, l’architecture proposée et les contrôles.
