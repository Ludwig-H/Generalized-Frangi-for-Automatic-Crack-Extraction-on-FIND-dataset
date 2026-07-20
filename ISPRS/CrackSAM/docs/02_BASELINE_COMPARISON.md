# Baseline SAM 2 et comparaison avec CrackSAM

## Résumé

La baseline locale est une adaptation de l'idée CrackSAM à SAM 2, pas une
reproduction exacte de CrackSAM. Elle constitue néanmoins la référence correcte
pour évaluer les futures contributions ajoutées dans ce dépôt.

## Résultats

| Modèle | Original | Bruit 1 | Bruit 2 | Road420 | Facade390 | Concrete3k | Macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline SAM 2, époque 20 | 0,6238 | 0,5678 | 0,5133 | 0,4836 | 0,5164 | 0,6998 | **0,5675** |
| Prompt Frangi dense, époque 25 | 0,6230 | 0,5705 | 0,4841 | 0,4701 | 0,4999 | 0,6901 | 0,5563 |
| CrackSAM LoRA q/v r=4 publié, SAM 1 | 0,6416 | 0,5782 | 0,4915 | 0,6222 | 0,4544 | 0,6798 | 0,5780 |

Les valeurs CrackSAM publiées donnent un niveau de référence externe. Elles ne
constituent pas une ablation contrôlée contre la baseline locale.

## Pourquoi le port n'est pas fidèle

| Élément | CrackSAM publié | Baseline locale |
|---|---|---|
| Backbone | SAM 1 ViT-H | SAM 2 Hiera-L |
| Paramètres gelés | image encoder seulement | tout SAM 2 avant insertion LoRA |
| Paramètres adaptés | LoRA image encoder + prompt encoder et mask decoder complets | LoRA q/v Hiera + attentions du mask decoder |
| Sortie | deux classes, CE + Dice | un logit, BCE + Dice |
| Prétraitement | pipeline SAM 1 à 448 | source 448, entrée Hiera à 1024 normalisée ImageNet |
| Horizon documenté | 140 époques | 70 époques, best époque 20 |

Il est donc incorrect d'expliquer l'écart Road420 uniquement par SAM 2. Une
future baseline « fidèle » devra entraîner au minimum le prompt encoder et le
mask decoder en plus des LoRA Hiera, tout en conservant la baseline historique
comme point de comparaison.

## Conclusion permise par l'expérience Frangi

Le résultat apparié négatif permet la conclusion suivante :

> Sous le protocole 2026-07, l'entraînement séparé avec un pseudo-masque Frangi
> dense toujours actif est inférieur à l'entraînement sans prompt.

Il ne permet pas de conclure que :

- la similarité Frangi est intrinsèquement nuisible ;
- le MST ou la centralité sont inutiles ;
- les ombres expliquent la majorité des pertes ;
- un sélecteur pourrait reproduire l'oracle calculé entre deux checkpoints
  entraînés séparément.

## Contrôles causaux prioritaires

Avant tout nouvel entraînement lourd, exécuter sur validation la matrice :

| Poids chargés | `masks=None` | prompt Frangi historique |
|---|---:|---:|
| checkpoint baseline | B→∅ | B→F |
| checkpoint Frangi | F→∅ | F→F |

Ajouter :

- tenseur nul versus `masks=None` ;
- prompt d'une autre image ;
- prompt spatialement décalé ;
- seuil `0,5` versus seuil fixé uniquement sur validation.

Cette matrice sépare l'effet instantané du prompt de l'effet des poids appris.
Elle doit être exécutée à l'époque 20 commune, puis sur les deux checkpoints
sélectionnés.

## Protocole statistique futur

- unité de bootstrap : image physique, avec les trois conditions Khanhha dans
  le même cluster ;
- macro principal : quatre familles indépendantes — Khanhha, Road, Facade,
  Concrete ;
- strates : masque vide, clairsemé, substantiel, ombre naturelle, fissure
  traversant une ombre ;
- métriques : IoU, Dice, clDice/clIoU, P05 du delta, pertes `< −0,05`, temps et
  mémoire ;
- au moins trois graines pour les modèles finalistes ;
- nouveau holdout dédupliqué pour la conclusion confirmatoire.

Les données complètes, intervalles et cas qualitatifs sont conservés dans le
[rapport 2026-07](../results/frangi_milestone_report/RAPPORT_FRANGI_MILESTONES.md).
