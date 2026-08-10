# CrackSAM-GeoLoRA — présentation du 9 août 2026

Douze planches Beamer sur le gabarit Inria, copié dans [`theme/`](theme/).
Rapport complet : [`../../RAPPORT.md`](../../RAPPORT.md).

**La question.** Une fissure a une forme : fine, longue, continue. Le Frangi
généralisé la décrit, SAM 2 non. Peut-on la lui donner, et sous quelle forme ?

**La réponse : non.** On donne au modèle la géométrie de son image, puis celle
d'une autre image. Les deux se valent : `|Δ| < 0,001` partout, et la mauvaise
gagne cinq fois sur six. Le modèle ne voit pas la différence.

Les planches : d'où l'on part (CrackSAM) · sa force, hors domaine et sous bruit
· la question et la réponse · les 11 canaux ajoutés à la Frangi-similarité · où
ils entrent dans SAM 2 · les six barreaux et le contrôle · la géométrie ne fait
rien · la galerie de gains trompe · la suite. Puis, **en digression**, ce que
l'IoU stricte mesure vraiment et ce qu'on gagne à tolérer 3 pixels — un
résultat annexe, sans rapport avec le guidage géométrique.

## Compilation

```sh
make          # images puis main.pdf, deux passes lualatex
make figures  # uniquement les images
```

`tools/make_figures.py` ne produit **aucun graphique**. Il fait trois choses :

* il découpe les planches par cas versionnées dans
  [`../../figures/generated/`](../../figures/generated/) en panneaux
  individuels — localisés par leurs gouttières blanches, pour que titres et
  scores soient composés par Beamer ;
* il rend comme images les cinq cas synthétiques, dont il recalcule les scores
  avec [`../../scripts/05_tolerant_iou.py`](../../scripts/05_tolerant_iou.py)
  lui-même ;
* il découpe de la même façon deux figures du papier CrackSAM, archivées avec
  la présentation de juillet sous
  `ISPRS/CrackSAM/reference/presentations/2026-07-10-inria-cerema/source/imgs/`
  — la comparaison zéro-shot sur `Road420` et la courbe de robustesse au flou.

Les scores des vignettes de cas sont ceux inscrits dans les planches d'origine,
donc issus des exécutions du 8–9 août, sans recalcul.

## Provenance des chiffres CrackSAM

Les deux planches d'ouverture ne citent que des valeurs publiées, relevées dans
le tableau 6 de `ISPRS/CrackSAM/reference/papers/CrackSAM.pdf` (Ge *et al.*,
*Construction and Building Materials* 431:136573, 2024). La ligne « SegFormer,
le meilleur des 12 autres » est exacte : SegFormer domine les onze autres
concurrents dans les six colonnes du tableau.

La ligne comparée est `CrackSAM_LoRA` (q/v, `r=4`), celle dont notre `baseline`
descend — et non `CrackSAM_adapter`, qui fait mieux sur `Facade390` (0,4718).

Les effectifs du corpus viennent des listes officielles versionnées sous
`ISPRS/CrackSAM/protocol/cracksam_paper/lists/lists_khanhha/` : 9 121 / 481 /
1 695, onze sous-ensembles, 12,5 % d'images sans fissure.

## Une précision par rapport au rapport

Le rapport écrit qu'une rupture reste pénalisée « à toutes les tolérances ». La
vérification refaite ici précise la règle : la tolérance étant une distance
d'appariement, elle efface aussi une rupture, mais seulement plus courte que
`2k`. Une coupure de 4 px est pardonnée dès `k=2` (`0,917 → 1,000`) ; une
coupure de 20 px reste pénalisée jusqu'à `k=8` (`0,583 → 0,917`).
