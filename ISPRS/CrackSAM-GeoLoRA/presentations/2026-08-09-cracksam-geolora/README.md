# CrackSAM-GeoLoRA — présentation du 9 août 2026

Dix-sept planches Beamer sur le gabarit Inria, copié dans [`theme/`](theme/).
Rapport complet : [`../../RAPPORT.md`](../../RAPPORT.md).

**La question.** Une fissure a une forme : fine, longue, continue. Le Frangi
généralisé la décrit, SAM 2 non. Peut-on la lui donner, et sous quelle forme ?

**La réponse : non.** On donne au modèle la géométrie de son image, puis celle
d'une autre image. Les deux se valent : `|Δ| < 0,001` partout, et la mauvaise
gagne cinq fois sur six. Le modèle ne voit pas la différence.

Le fil : le corpus Khánh Hà, illustré par sous-corpus · la force de CrackSAM,
le bruit et le hors-domaine · ce qu'est une LoRA de rang 4 · pourquoi seulement
sur `q` et `v` · la première tentative, la géométrie en `mask_input` · pourquoi
elle échoue, une frontière d'ombre étant localement une vallée · l'idée, séparer
la partie paire de la partie impaire d'un profil · les onze canaux · l'architecture,
`geo` contre `baseline` · la question et la réponse · les six barreaux et le
contrôle · la géométrie ne fait rien · la galerie de gains trompe · la suite.

Puis, **en clôture**, le seul gain de l'étude, sans rapport avec le guidage
géométrique : ce qu'est la tolérance de 3 px, et ce qu'elle fait gagner à
`tol3` contre la `baseline`.

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

La planche d'ouverture ne cite que des valeurs publiées, relevées dans
`ISPRS/CrackSAM/reference/papers/CrackSAM.pdf` (Ge *et al.*, *Construction and
Building Materials* 431:136573, 2024). Le modèle décrit est `CrackSAM_LoRA`
(q/v, `r=4`), celui dont notre `baseline` descend — et non `CrackSAM_adapter`,
qui fait mieux sur `Facade390` (0,4718).

Les effectifs du corpus viennent des listes officielles versionnées sous
`ISPRS/CrackSAM/protocol/cracksam_paper/lists/lists_khanhha/` : 9 121 / 481 /
1 695, onze sous-ensembles, 12,5 % d'images sans fissure. Le détail compté dans
`test_vol.txt`, qui somme bien à 1 695 : Rissbilder 573, CRACK500 505,
noncrack 212, Volker 148, DeepCrack 78, GAPS384 76, cracktree200 31, Sylvie 28,
forest 18, CFD 18, Eugen 8. La planche n'affiche pas ces effectifs — voir plus
bas.

## Les vignettes du corpus

Le corpus n'est pas versionné, et il n'est pas monté sur toutes les machines.
Chaque vignette `kh_*` est donc le **panneau d'entrée d'une planche par cas déjà
archivée**, et non une image tirée du jeu de données. Deux sources : les planches
de cette étude sous `figures/generated/`, et celles de l'étude anti-ombre du
8 août — seul endroit versionné où figurent Volker, DeepCrack, Sylvie et les
images sans fissure.

`forest` (18 images de test) et `Eugen` (8) ne figurent dans aucune planche
archivée, et le corpus n'est monté sur aucune machine de développement : ces
deux sous-corpus **ne peuvent pas être illustrés** en l'état. La planche montre
donc neuf vignettes sur onze, et n'affiche aucun effectif par sous-corpus — des
nombres qui ne sommeraient pas à 1 695 induiraient en erreur.

Les trois panneaux `ombre_*` viennent de `Sylvie_Chambon_319`, qui porte à la
fois une vraie fissure et une ombre portée franche. Le panneau de droite est
`node_sim_max` — exactement la carte qui était injectée dans `mask_input`.

## Provenance des chiffres de la LoRA

Les deux planches d'explication ne portent aucun résultat : elles décrivent
l'adaptation qu'injecte `inject_lora_qv`, dans
[`ISPRS/CrackSAM/cracksam2/model.py`](../../../CrackSAM/cracksam2/model.py).

* `453 248` est le nombre de paramètres entraînables inscrit dans
  [`../../tables/generated/baseline_training.json`](../../tables/generated/baseline_training.json).
  Il se décompose en 429 696 pour les 48 blocs de Hiera-L et 23 552 pour les
  sept modules d'attention du décodeur de masque.
* `224,9 M` est le total de SAM 2 Hiera-L, soit 224 883 378 paramètres. La part
  apprise vaut donc 0,202 %, et 1,8 Mo en `float32`.
* `4 608` et `331 776` valent pour **une** projection `576 × 576`, la largeur
  des 35 blocs du troisième étage : `A` et `B` y pèsent `2 × 4 × 576`.

Le `qkv` de Hiera étant une projection fusionnée, la correction ne peut pas y
être posée projection par projection : `LoRAQKV` écrit les deux résidus sur les
tranches `q` et `v`, et laisse celle de `k` nulle.

## Une précision par rapport au rapport

Le rapport écrit qu'une rupture reste pénalisée « à toutes les tolérances ». La
vérification refaite ici précise la règle : la tolérance étant une distance
d'appariement, elle efface aussi une rupture, mais seulement plus courte que
`2k`. Une coupure de 4 px est pardonnée dès `k=2` (`0,917 → 1,000`) ; une
coupure de 20 px reste pénalisée jusqu'à `k=8` (`0,583 → 0,917`).
