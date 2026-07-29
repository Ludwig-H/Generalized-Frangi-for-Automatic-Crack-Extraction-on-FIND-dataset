# Adapter le graphe de Frangi à SAM 2

> Date : 29 juillet 2026
>
> Statut : proposition courte, sans nouveau résultat expérimental
>
> Sources : [conclusion du papier soumis au European Workshop on Visual
> Information Processing](../../../EUVIP/LaTeX/main.tex#L619) et
> [dernier audit de CrackSAM 2](08_AUDIT_CRACKSAM2_FRANGIGRAPH_LORA.md)

Dans ce rapport, **SAM 2** désigne *Segment Anything Model 2*, **LoRA** désigne
l'adaptation par matrices de faible rang et **GNN** désigne un réseau neuronal
de graphes. Ce sont les trois seuls sigles employés.

## Recommandation

La suite la plus simple et pertinente du papier est :

> **adapter SAM 2 aux fissures par LoRA, utiliser le graphe de Frangi pour
> proposer quelques points, puis faire vérifier ces points par un petit GNN
> avant de les transmettre à SAM 2.**

Le graphe ne devient ni un masque, ni une probabilité de fissure. Il propose des
positions plausibles ; SAM 2 réalise la segmentation.

Ce choix répond directement à l'audit. Le modèle de référence atteint `0,5675`
d'intersection sur union moyenne, contre `0,5563` lorsque Frangi est injecté
comme masque dense. La géométrie alignée contient de l'information, mais
l'entrée de masque dense appelée `mask_input` lui donne la mauvaise
signification.

```text
Image ──► encodeur visuel adapté par LoRA ──► segmentation de référence
  │                     │
  │                     └── caractéristiques visuelles internes
  │
  └──► graphe de Frangi ──► GNN ──► huit points fiables au maximum
                                          │
                                          ▼
                              décodeur de masque guidé par points

Résultat final :
- segmentation de référence si le graphe est rejeté ;
- segmentation guidée près du graphe accepté ;
- segmentation de référence partout ailleurs.
```

## Les trois composants

### SAM 2 adapté aux fissures

Deux LoRA sont séparées :

1. une LoRA adapte l'encodeur et le décodeur aux fissures et produit la
   segmentation de référence sans point ;
2. une seconde LoRA, limitée au décodeur de masque, apprend à exploiter les
   points du graphe. Sa dimension interne peut être fixée à quatre.

L'encodeur et la première LoRA sont gelés pendant la seconde étape. Une seule
analyse visuelle de l'image est donc nécessaire.

### GNN vérificateur

Le GNN comporte deux couches. Chaque nœud combine ses informations avec celles
de ses voisins. Il reçoit :

- courbure, échelle, orientation et similarité de Frangi ;
- nombre de voisins, centralité et taille de la composante ;
- longueur et accord d'orientation des arêtes ;
- caractéristiques internes de SAM 2 et score de la segmentation de référence
  à la position du nœud ;
- accord entre intensité et profondeur lorsque les deux modalités existent ;
- profil de luminosité de part et d'autre de la ligne.

Le dernier point traite le principal risque d'ombre :

- une fissure sombre ressemble généralement à une vallée, avec un centre
  sombre et deux côtés plus clairs ;
- une frontière d'ombre ressemble davantage à une marche, avec un côté clair
  et l'autre sombre.

Le GNN apprend cette différence à plusieurs distances. Ce n'est jamais une
règle absolue, car une fissure peut traverser une ombre. La couleur reste un
indice facultatif : elle ne doit pas remplacer la luminosité, car le test local
existant est moins bon avec la chrominance dans cinq des six cas étudiés.

Le GNN produit la plausibilité d'une fissure et le risque de faux positif pour
chaque nœud, puis pour chaque composante. Il peut rejeter tout le graphe.

### Guidage par points et retour sûr

Pour chaque composante acceptée :

1. retenir les extrémités et les intersections fiables ;
2. compléter avec les meilleurs nœuds encore éloignés des points choisis ;
3. s'arrêter à huit points positifs ;
4. transmettre leurs coordonnées à l'interface native de SAM 2.

Les composantes reconnues comme des ombres sont simplement omises. Les
transformer en points négatifs restera une expérience ultérieure, car un point
négatif mal placé pourrait supprimer une vraie fissure.

Si aucun point n'est accepté, le programme retourne directement la segmentation
de référence. Sinon, la segmentation guidée n'est utilisée que dans une bande
autour du graphe accepté, dont la largeur dépend de l'échelle Frangi. Un point
erroné ne peut donc pas créer une grande région parasite ailleurs.

## Apprentissage

L'apprentissage se déroule en trois temps :

1. adapter SAM 2 aux fissures, puis figer ce modèle ;
2. apprendre au GNN quels nœuds et arêtes sont proches du squelette vrai ;
3. apprendre la seconde LoRA avec une pénalité de recouvrement du masque et une
   faible pénalité de continuité du squelette.

Les ombres synthétiques sont ajoutées avant de recalculer Frangi. Chaque image
propre, sa version ombrée et leurs découpes restent dans le même groupe. Les
arêtes créées sur la frontière d'ombre sont des exemples négatifs difficiles.
Une pénalité demande au GNN de classer la composante de fissure au-dessus de la
composante créée par l'ombre.

## Expérience décisive

| Variante | Question |
|---|---|
| Graphe de Frangi seul | Que vaut la méthode sans apprentissage ? |
| SAM 2 gelé avec points Frangi bruts | Les points natifs suffisent-ils ? |
| SAM 2 adapté par LoRA, sans graphe | Quel est le gain de l'adaptation seule ? |
| Modèle adapté avec points Frangi bruts | La vérification est-elle nécessaire ? |
| Classificateur indépendant par nœud | Les informations locales suffisent-elles ? |
| GNN avec vraies arêtes | La topologie apporte-t-elle un gain ? |
| Même GNN avec arêtes mélangées | Le gain vient-il vraiment des connexions ? |

Il faut mesurer l'intersection sur union, le coefficient de Dice, la continuité
du squelette, les indices de Jaccard et de Tversky, la distance de Wasserstein,
les faux positifs près des frontières d'ombre et le rappel lorsqu'une ombre
traverse une fissure.

Le GNN n'est conservé que si les vraies arêtes font mieux que le classificateur
indépendant et que les arêtes mélangées.

## Mise en œuvre minimale

Le premier prototype demande seulement :

1. d'étendre
   [`decode_features`](../cracksam2/model.py#L278) pour accepter les coordonnées
   et les étiquettes des points ;
2. de réutiliser les profils vallée contre marche déjà présents dans
   [`evidence_selection.py`](../cracksam2/evidence_selection.py) ;
3. d'ajouter `graph_points.py` pour le GNN, la sélection des points et le retour
   exact au modèle de référence.

Pour ce premier test, les arêtes peuvent relier les pixels voisins du squelette
déjà précalculé. Si le résultat est positif, il faudra conserver directement
les nœuds et les arêtes actuellement calculés puis abandonnés dans
[`graph_extraction.py`](../../src/graph_extraction.py).

Cette première version représente environ trois à quatre cents lignes de code,
tests compris. Elle ne modifie ni l'encodeur visuel, ni l'entrée de masque
dense.

## Conclusion

Cette solution répond à la perspective du papier avec peu de modifications :
le graphe de Frangi propose, le GNN vérifie, LoRA adapte et SAM 2 segmente. Le
chemin de référence est retrouvé exactement lorsque le graphe est incertain.

Une fusion plus profonde du graphe dans SAM 2 pourra être étudiée seulement si
ce guidage simple démontre que les vraies connexions de Frangi apportent une
information complémentaire.

## Références

1. Hauseux et collaborateurs,
   [papier sur l'extraction de fissures sans apprentissage par graphe de Frangi](../../../EUVIP/EUVIP_2026_Generalized_Frangi_Multimodality.pdf),
   2026.
2. Ravi et collaborateurs,
   [article présentant SAM 2](https://arxiv.org/abs/2408.00714), 2024.
3. Ge et collaborateurs,
   [article adaptant le premier SAM aux fissures](https://arxiv.org/abs/2312.04233),
   2023.
4. Hetang et collaborateurs,
   [article combinant SAM et GNN pour les routes](https://arxiv.org/abs/2403.16051),
   2024.
5. Fan et collaborateurs,
   [article sur les fissures de chaussée couplées aux ombres](https://www.ieee-jas.net/article/doi/10.1109/JAS.2023.123447),
   2023.
6. Hu et collaborateurs,
   [article présentant LoRA](https://arxiv.org/abs/2106.09685), 2021.
