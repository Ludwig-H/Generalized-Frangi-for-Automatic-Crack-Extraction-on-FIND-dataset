# CrackSAM-GeoLoRA — présentation du 9 août 2026

Présentation Beamer sur le gabarit Inria (copié dans [`theme/`](theme/)), pour la
réunion Inria–Cerema. Elle rend compte de la quatrième itération de la ligne
CrackSAM : la géométrie de Frangi **apprise dans** SAM 2 plutôt qu'appliquée en
correction après coup.

Le rapport complet, avec les artefacts et les tableaux par image, reste
[`../../RAPPORT.md`](../../RAPPORT.md).

## Le fil de la présentation

1. **Contexte** — les quatre itérations, et pourquoi les trois premières ne
   pouvaient rien conclure : elles corrigeaient un modèle gelé.
2. **Évidence** — les onze canaux ajoutés à la Frangi-similarité, leur rôle, et
   les échelles dérivées de la largeur mesurée plutôt qu'héritées.
3. **Intégration** — l'injection additive à trois résolutions, initialisée à
   zéro, sans jamais passer par `mask_input` ; et le point selle
   d'initialisation qui a figé l'adapter.
4. **Protocole** — les six exigences qui rendent la conclusion possible :
   budget égal, un facteur par barreau, contrôle apparié à évidence permutée,
   test de nécessité d'entrée, deltas appariés, métrique choisie d'avance.
5. **Mesurer** — pourquoi l'IoU à 3 pixels, ce que les deux conventions de
   tolérance ne classent pas pareil, et ce que la tolérance ne pardonne pas.
6. **Résultats** — l'échelle d'ablations, les six tolérances, et le contrôle
   permuté qui tranche.
7. **Cas** — galeries de réussites *et* d'échecs, dont les deux plus grands
   gains de la géométrie, qui sont exactement les deux plus grandes pertes de
   la perte tolérante.
8. **Bilan** — trois conclusions fausses corrigées par la mesure, les limites,
   la suite.

## Compilation

```sh
make          # figures puis main.pdf, en deux passes lualatex
make figures  # uniquement la préparation des images
```

`tools/make_figures.py` ne produit **aucune courbe**. Il fait deux choses :

- il **découpe** les planches par cas versionnées dans
  [`../../figures/generated/`](../../figures/generated/) en panneaux
  individuels, localisés par leurs gouttières blanches, pour que les titres et
  les scores soient composés par Beamer et non par matplotlib ;
- il **rend comme images** les cinq cas synthétiques qui valident la métrique
  tolérante, et recalcule leurs scores avec
  [`../../scripts/05_tolerant_iou.py`](../../scripts/05_tolerant_iou.py)
  lui-même, puis les imprime pour recopie dans `main.tex`.

Les scores affichés dans les vignettes de cas sont ceux inscrits dans les
planches d'origine ; ils viennent donc des exécutions du 8–9 août, sans
recalcul.

## Une précision par rapport au rapport

Le rapport annonce qu'une rupture reste pénalisée « à toutes les tolérances ».
La vérification synthétique refaite pour cette présentation précise la règle :
la tolérance étant une **distance d'appariement**, elle efface aussi une
rupture, mais seulement tant que celle-ci est plus courte que `2k`. Une coupure
de 4 px est pardonnée dès `k=2` (`0,917 → 1,000`) ; une coupure de 20 px reste
pénalisée à toutes les tolérances mesurées (`0,583 → 0,917` de `k=0` à `k=8`).
La conclusion du rapport ne change pas — la métrique distingue bien placement
et topologie — mais l'énoncé exact figure sur la planche.
