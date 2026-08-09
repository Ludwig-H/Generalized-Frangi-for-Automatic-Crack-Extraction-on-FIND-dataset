# CrackSAM-GeoLoRA — présentation du 9 août 2026

Dix planches Beamer sur le gabarit Inria, copié dans [`theme/`](theme/). Rapport
complet : [`../../RAPPORT.md`](../../RAPPORT.md).

**Le message, en deux lignes.** Le guidage géométrique n'aide pas : évidence
alignée et évidence permutée sont indiscernables (`|Δ| < 0,001` partout, le
contrôle devant 5 fois sur 6). La dilatation, si : une perte tolérante à 3 px
gagne `+0,0035` en IoU stricte et `+0,0179` à 1 px, sans un paramètre de plus.

Les planches : les deux résultats · les 11 canaux ajoutés à la Frangi-similarité
· où ils entrent dans SAM 2 · les six barreaux et le contrôle · la géométrie ne
fait rien · la galerie de gains trompe · pourquoi dilater · ce que la dilatation
gagne · la suite.

## Compilation

```sh
make          # images puis main.pdf, deux passes lualatex
make figures  # uniquement les images
```

`tools/make_figures.py` ne produit **aucun graphique**. Il découpe les planches
par cas versionnées dans [`../../figures/generated/`](../../figures/generated/)
en panneaux individuels — localisés par leurs gouttières blanches, pour que
titres et scores soient composés par Beamer — et rend comme images les cinq cas
synthétiques, dont il recalcule les scores avec
[`../../scripts/05_tolerant_iou.py`](../../scripts/05_tolerant_iou.py) lui-même.

Les scores des vignettes de cas sont ceux inscrits dans les planches d'origine,
donc issus des exécutions du 8–9 août, sans recalcul.

## Une précision par rapport au rapport

Le rapport écrit qu'une rupture reste pénalisée « à toutes les tolérances ». La
vérification refaite ici précise la règle : la tolérance étant une distance
d'appariement, elle efface aussi une rupture, mais seulement plus courte que
`2k`. Une coupure de 4 px est pardonnée dès `k=2` (`0,917 → 1,000`) ; une
coupure de 20 px reste pénalisée jusqu'à `k=8` (`0,583 → 0,917`).
