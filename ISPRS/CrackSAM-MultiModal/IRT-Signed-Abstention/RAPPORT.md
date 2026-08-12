# CrackSAM-IRT — la thermique d'IRT-Crack n'aide pas, et on sait pourquoi

**Cinquième itération de la ligne CrackSAM, première multimodale, exécutée.**
Campagne du 12 août 2026 sur `cracksam-frangigraph-g4-spot-ew8c` (RTX PRO 6000
Blackwell), 19 exécutions, 448 paires IRT-Crack, split officiel 358/90.

<div align="center">

| Question | Réponse | Sur quoi elle repose |
|:--:|:--:|:--:|
| l'évidence Frangi **thermique** corrige-t-elle un CrackSAM RGB gelé ? | **non** | A2 − A1 = `−0,0032` d'IoU stricte, `+0,0020` à 3 px avec IC95 incluant zéro |
| est-ce un échec de la géométrie, ou de la donnée ? | **de la donnée** | la thermique distribuée est décalée de `10,1 px` du RGB, mesuré avec contrôle |
| que retient-on de positif ? | **la recalibration de domaine** | 20 835 paramètres sur logits gelés : `+0,045` d'IoU à 3 px, IC95 excluant zéro |

</div>

> [!IMPORTANT]
> **Le critère de succès pré-enregistré n'est pas atteint, sur aucune métrique.**
> Il exigeait `A2 > A1` avec IC95 excluant zéro **et** `A2 > A3`. En IoU stricte
> `A2 − A1` est *négatif* et son IC95 exclut zéro — ajouter la thermique **coûte**.
> À 3 px, `A2 − A1 = +0,0020` avec IC95 `[−0,0004 ; +0,0045]`, soit un effet
> inférieur au plancher de détection `±0,0025` — et vingt fois plus petit que le
> gain de la simple recalibration.

---

## Sommaire

1. [Ce que la campagne a mesuré](#1-ce-que-la-campagne-a-mesuré)
2. [Les trois portes franchies avant d'entraîner](#2-les-trois-portes-franchies-avant-dentraîner)
3. [Résultats](#3-résultats)
4. [Pourquoi la thermique n'aide pas](#4-pourquoi-la-thermique-naide-pas)
5. [Ce que le correcteur fait réellement](#5-ce-que-le-correcteur-fait-réellement)
6. [Incidents](#6-incidents)
7. [Limites](#7-limites)
8. [Suite](#8-suite)

---

## 1. Ce que la campagne a mesuré

CrackSAM 2 + LoRA `tol3`, entraîné sur Khánh Hà, est **gelé** et ses logits sont
cachés une fois sur les 448 images d'IRT-Crack. Un correcteur de
**20 835 paramètres** lit ces logits et quatre canaux d'évidence, et choisit par
pixel entre renforcer, supprimer et s'abstenir. Sept bras, trois graines.

| Bras | Ce que reçoit le correcteur | Ce qu'il isole |
|:--|:--|:--|
| **A0** | rien — les logits cachés, tels quels | le transfert Khánh Hà → IRT-Crack |
| **A1** | `z₀`, `p₀`, `H(p₀)` ; les 4 canaux thermiques à **zéro** | la recalibration de domaine seule |
| **A2** | + évidence Frangi thermique **alignée** | la méthode |
| **A3** | + évidence Frangi **d'une autre image** | la causalité spatiale |
| **A4** | + thermique **brute**, même capacité | modalité contre abstraction géométrique |
| **A5** | A2 sans l'action « s'abstenir » | la valeur propre de l'abstention |
| **A6** | renforcement seul, `Δz = δ σ(q) S_max` | la valeur propre du signe |

Les sept partagent le même bloc d'hyperparamètres, **identique à l'octet près**
— un test le vérifie — et A1 à A4 ont exactement le même nombre de paramètres.

## 2. Les trois portes franchies avant d'entraîner

Aucun bras n'a tourné avant que ces trois mesures ne soient faites. Deux ont
changé la conception ; la troisième a changé la lecture du résultat.

### 2.1 Le décodage thermique — le piège du dépôt était réel

Les 448 thermiques sont en **fausses couleurs**, sans exception. La conversion
standard en niveaux de gris s'écarte de `0,2017` du décodage correct par
recherche de palette : sur un dégradé JET, elle est non monotone et le vert
médian y dépasse le rouge maximal. Le notebook `IRT-Crack Segmentation` du dépôt
les chargeait en `IMREAD_GRAYSCALE` — il corrompait donc bien la Hessienne, comme
`implementation_notes.md` le soupçonnait.

Le résidu de palette vaut `0,0948` contre `jet` et `0,0788` contre `turbo`, et il
est **identique en PNG et en JPEG** (`0,0939` contre `0,0942`) : ce n'est pas la
compression, c'est une palette FLIR propriétaire qu'aucune des deux ne reproduit.
Les décodages `jet` et `turbo` corrèlent à `0,987`, donc le choix ne change
presque rien à une détection de crêtes — mais il faut le dire.

### 2.2 Le plafond d'une correction bornée — `delta_max = 4` aurait handicapé la campagne

La correction vaut `Δz = δ_max(π⁺ − π⁻)`, donc un pixel n'est corrigeable que si
`|z₀| < δ_max`. Mesuré sur la validation, avec la baseline `tol3` gelée :

| `\|z₀\|` | p50 | p90 | p95 | p99 | p99,9 |
|:--|--:|--:|--:|--:|--:|
| CrackSAM `tol3` sur IRT-Crack | **12,27** | 14,83 | 15,59 | 16,90 | 18,15 |

| `δ_max` | oracle borné (IoU@3px) | marge sur la baseline | erreurs hors de portée |
|--:|--:|--:|--:|
| 4 *(valeur de la spécification)* | 0,9577 | +0,1211 | **18,9 %** |
| **12** *(retenue)* | 1,0000 | **+0,1634** | **0,0 %** |

La valeur recommandée par la spécification laissait près d'une erreur sur cinq
hors d'atteinte, et son `clip(z₀, −10, 10)` saturait **plus d'un pixel sur deux**,
privant le correcteur de l'information de confiance là où elle décide. `δ_max`
est passé à `12` et `logit_clip` à `20` dans les sept configurations, **avant** le
premier entraînement.

### 2.3 Le recalage — la porte a dit « rejeté », et c'est le résultat le plus utile

Le premier estimateur écrit pour cette étude — corrélation croisée des gradients
RGB/thermique — saturait au bord de sa fenêtre dans `63 %` des cas : il mesurait
sa fenêtre. Il a été remplacé par un estimateur **ancré sur la vérité terrain qui
porte son propre contrôle** : on cherche le décalage entier qui maximise le
contraste fissure/fond, et on applique le même estimateur au RGB.

| Champ testé | décalage médian | images à 0 px | images sous 2 px | saturation |
|:--|--:|--:|--:|--:|
| **RGB (contrôle)**, 448 images | **0,0 px** | **76,7 %** | **100 %** | 0 % |
| thermique décodée, 448 images | **10,1 px** | 0 % | **5,0 %** | 58 % |
| fusion FLIR 50/50, 60 images | 0,0 px | 63 % | 92 % | — |

La thermique sature au bord de la fenêtre de recherche dans `58 %` des cas :
son décalage réel est donc `≥ 10 px` plus souvent que la médiane ne le dit. Le
contrôle RGB, lui, ne sature jamais et tombe sous 2 px sur **la totalité** des
images.

C'est le contrôle qui rend le chiffre concluant : le même estimateur, sur les
mêmes masques, trouve le RGB aligné et la thermique décalée. L'annotation a donc
été tracée sur le visible, et **l'infrarouge brut distribué n'est pas
co-recalé** — contrairement à ce que laissait entendre la mention IR-Fusion™.
La fusion, elle, l'est ; mais elle contient déjà 50 % du visible, donc elle n'est
pas une modalité cachée pour SAM.

Aucune correction sans fuite d'étiquette n'est disponible : le décalage n'est pas
constant (écart-type `8,0` px en `dy`, `7,4` px en `dx`), et reconstruire l'IR
recalé par `2 × fusion − visible` échoue également (`10,2 px`).

**La campagne a été lancée malgré la porte fermée**, en déclarant la prédiction à
l'avance : si l'évidence thermique est décalée de `10 px`, `A2` doit être
indiscernable de `A3`. C'est ce qui s'est produit.

## 3. Résultats

Test officiel, 90 images, moyennes sur trois graines, `±` = écart-type
inter-graines.

| Bras | IoU stricte | IoU tolérante 3 px | clDice | Composantes |
|:--|--:|--:|--:|--:|
| **A0** baseline gelée | 0,6674 | 0,8405 | 0,9000 | **1,70** |
| **A1** recalibration RGB | **0,7096** ± 0,0014 | 0,8856 ± 0,0015 | **0,9109** | 4,09 |
| **A2** Frangi signé + abstention | 0,7064 ± 0,0009 | **0,8876** ± 0,0007 | 0,8952 | 13,47 |
| **A3** Frangi permuté | 0,7064 ± 0,0014 | 0,8856 ± 0,0011 | 0,8921 | 18,43 |
| **A4** thermique brute | 0,7049 ± 0,0014 | 0,8861 ± 0,0013 | 0,9016 | 13,01 |
| **A5** sans abstention | 0,7051 ± 0,0012 | 0,8863 ± 0,0009 | 0,8989 | 11,49 |
| **A6** renforcement seul | 0,6908 ± 0,0018 | 0,8733 ± 0,0020 | 0,8959 | 9,00 |

### 3.1 Les comparaisons qui décident

Deltas **appariés par image**, bootstrap 10 000 tirages, IC95 par percentiles.

| Comparaison | IoU stricte | IoU tolérante 3 px | Verdict |
|:--|:--|:--|:--|
| **A1 − A0** — la recalibration | `+0,0422` `[+0,0349 ; +0,0502]` | `+0,0451` `[+0,0360 ; +0,0550]` | **favorable** |
| **A2 − A1** — *la thermique aide-t-elle ?* | `−0,0032` `[−0,0054 ; −0,0009]` | `+0,0020` `[−0,0004 ; +0,0045]` | **non** |
| **A2 − A3** — *est-ce causal ?* | `−0,0000` `[−0,0022 ; +0,0021]` | `+0,0020` `[−0,0004 ; +0,0044]` | **indiscernable** |
| **A2 − A4** — *la géométrie au-delà de la modalité ?* | `+0,0015` `[−0,0004 ; +0,0035]` | `+0,0015` `[−0,0005 ; +0,0036]` | **indiscernable** |
| **A2 − A5** — l'abstention | `+0,0013` `[−0,0007 ; +0,0037]` | `+0,0013` `[−0,0006 ; +0,0035]` | indiscernable |
| **A2 − A6** — le signe | `+0,0156` `[+0,0119 ; +0,0195]` | `+0,0143` `[+0,0102 ; +0,0188]` | **favorable** |
| **A4 − A1** — la modalité brute | `−0,0047` `[−0,0065 ; −0,0029]` | `+0,0005` `[−0,0013 ; +0,0023]` | **défavorable** |

> [!CAUTION]
> **`A2 − A1` est négatif en IoU stricte, avec un IC95 qui exclut zéro.** Les
> quatre canaux thermiques ne sont pas neutres : ils coûtent. C'est la répétition
> exacte de ce que GeoLoRA avait mesuré sur les `290 801` paramètres de son
> adaptateur géométrique — une capacité ajoutée qui coûte un peu sans rien
> rapporter.

À 3 px, `A2 − A3 = +0,0020` a le **même signe sur les trois graines**
(`+0,0017`, `+0,0018`, `+0,0025`) mais son IC95 par image inclut zéro et son
amplitude est sous le plancher de détection `±0,0024`. La lecture littérale du
critère (« écart cohérent sur les graines ») est donc satisfaite là où la lecture
stricte (IC95) ne l'est pas — les deux sont rapportées, et **aucune** ne suffit
puisque `A2 > A1` échoue de toute façon.

### 3.2 Le rapport d'échelle, qui tranche

`A2 − A1` vaut `+4,5 %` du gain `A1 − A0` sur la métrique tolérante, et `−7,5 %`
sur l'IoU stricte. Autrement dit : **tout ce que ce dispositif gagne, il le gagne
sans la thermique.** Le multimodal contribue, au mieux, un vingtième de l'effet.

## 4. Pourquoi la thermique n'aide pas

Trois explications sont compatibles avec les mesures, et elles ne s'excluent pas.

**Le décalage de 10 px.** Le champ réceptif du correcteur vaut `15 px`
(trois convolutions dilatées 1-2-4). Une évidence décalée de `10 px` tombe donc
au bord de ce que le correcteur peut relier à un pixel — et une évidence
**permutée** tombe exactement aussi mal. C'est la lecture la plus simple de
`A2 ≈ A3`.

**La baseline est déjà bonne là où la thermique pourrait aider.** A0 atteint
`0,8405` d'IoU à 3 px en transfert pur, et A1 `0,8856`. La marge restante que
l'oracle borné autorise est `+0,1634` : il y a de la place, mais elle se trouve
sur des fissures que le visible voit mal — précisément là où un décalage de
`10 px` est fatal.

**Le canal ajoute plus de bruit que d'information.** Le fait le plus net du
tableau : la baseline prédit `1,70` composante connexe par image, A1 en prédit
`4,09`, et **tous les bras qui reçoivent quatre canaux supplémentaires en
prédisent 9 à 18** — alignés (`13,47`) comme permutés (`18,43`), Frangi comme
bruts (`13,01`). Le clDice suit : `0,9109` pour A1, `0,8921` à `0,9016` pour les
autres. Ces canaux **fragmentent la prédiction**, et ils la fragmentent
indépendamment de leur contenu.

## 5. Ce que le correcteur fait réellement

| Bras | renforcer | supprimer | s'abstenir | `\|Δz\|` moyen | p99 | FN récupérés | FP retirés |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **A1** — sans thermique | 0,001 | 0,000 | 0,999 | **0,377** | 3,14 | **0,320** | **0,159** |
| A2 | 0,001 | 0,000 | **0,999** | 0,337 | 2,79 | 0,310 | 0,136 |
| A3 | 0,001 | 0,000 | 0,999 | 0,387 | 2,85 | 0,314 | 0,152 |
| A4 | 0,001 | 0,000 | 0,999 | 0,375 | 2,69 | 0,308 | 0,121 |
| A5 | 0,728 | 0,272 | — | 0,554 | 3,15 | 0,293 | 0,165 |
| A6 | 0,023 | 0,000 | 0,977 | 0,104 | 2,25 | 0,215 | 0,000 |

La première ligne est la plus parlante : **A1, qui ne voit aucune thermique,
corrige plus fort (`0,377` contre `0,337`) et récupère plus de faux négatifs
(`32,0 %` contre `31,0 %`) que A2.** Les quatre canaux thermiques ne rendent pas
le correcteur meilleur, ils le rendent plus timide.

Deux enseignements de mécanisme, indépendants de la question multimodale.

**L'abstention est massive et elle est le bon régime.** A2 s'abstient sur
`99,9 %` des pixels au sens de l'`argmax`, et l'amplitude effective de sa
correction (`|Δz|` moyen `0,337`, p99 `2,79`) reste très en dessous de la borne
`δ_max = 12` — la porte de plafond était donc nécessaire pour ne pas *brider*, pas
pour être atteinte. Le correcteur agit peu, près de la frontière de décision, et
cela suffit à récupérer `31 %` des faux négatifs de la baseline. Forcer l'action
(A5 : `0 %` d'abstention, `72,8 %` de renforcement, `|Δz|` moyen `0,554`) ne
rapporte rien de plus.

**Le signe compte, l'abstention explicite non.** A6, qui ne peut que renforcer,
perd `0,0143` d'IoU à 3 px avec un IC95 qui exclut zéro — il ne retire **aucun**
faux positif (`0,000` contre `0,136`) et récupère `21,5 %` des faux négatifs au
lieu de `31,0 %`. A5, qui perd l'abstention mais garde le signe, est
indiscernable de A2. Ce qui porte le gain, c'est donc la capacité de
**soustraire**, pas celle de déclarer qu'on s'abstient.

## 6. Incidents

**Deux campagnes ont tourné en parallèle.** Un premier lancement `nohup` qu'on
croyait mort a continué et a écrit dans les mêmes dossiers qu'un second : l'index
est passé de 17 à 19 entrées après coup, et les deltas appariés changeaient d'une
lecture du rapport à l'autre — `A4 − A1` donnait `133/137` puis `119/151` gains.
Le symptôme est discret et l'artefact indétectable de l'intérieur.

La campagne a été **entièrement relancée** dans un dossier neuf, processus
unique, et le rapport vérifié reproductible sur deux lectures consécutives. Tous
les chiffres de ce document viennent de cette exécution-là. Un verrou exclusif
(`campaign.lock`) interdit désormais la situation.

**Le verdict automatique basculait selon la métrique.** La logique initiale
n'exigeait pas d'IC95 excluant zéro sur `A2 − A3`, et rendait donc « vrai » sur
la métrique tolérante et « faux » sur l'IoU stricte. Elle a été remplacée par un
verdict qui rapporte explicitement les deux lectures, plus l'amplitude relative
au gain de recalibration.

**Le checkpoint `tol3` était introuvable localement** — perdu avec le disque Spot
d'août, il n'en restait qu'un chemin absolu figé dans `eval_tol3.json`. Il a été
retrouvé **sur le disque de la VM**, intact. Les 36 checkpoints de cette
campagne ont été copiés dans un `vm_backup_20260812T1619Z/` avec leurs SHA-256
avant l'arrêt.

## 7. Limites

- **La porte de recalage était fermée.** Le protocole pré-enregistrait un seuil
  d'exclusion à `3 px` ; la mesure donne `10,1 px`. La campagne a été menée
  quand même, en déclarant la prédiction avant de l'observer. Un résultat
  négatif obtenu sur une donnée qui viole son propre critère d'admission ne
  réfute pas l'idée — il réfute *cette instance* de l'idée.
- **Une seule baseline.** Tous les bras corrigent les mêmes logits `tol3`. Les
  conclusions ne portent pas sur d'autres segmentateurs.
- **90 images de test.** Le plancher de détection vaut `±0,0024` à 3 px : cette
  campagne ne peut pas trancher un effet de `0,002`, et c'est précisément
  l'ordre de grandeur de `A2 − A3`.
- **La palette n'est pas identifiée.** Ni `jet` ni `turbo` ne reproduisent les
  couleurs FLIR. L'écart est une déformation monotone partagée par tous les
  bras, donc il ne peut pas fabriquer une différence `A2 − A3` — mais il affaiblit
  l'évidence de manière non quantifiée.
- **Le graphe n'a pas été testé.** MST, composantes et centralité restent hors
  périmètre, comme le protocole l'exige tant que la similarité dense n'a pas
  produit de signal causal. Elle n'en a pas produit.

## 8. Suite

Le protocole interdit de passer au Frangi-graphe tant que `A2 > A1` et
`A2 > A3` ne sont pas établis. Ils ne le sont pas. Trois directions, par ordre de
valeur :

1. **FIND devient le seul ancrage propre.** Son range laser est co-recalé *par
   construction du capteur* — même balayage, pas deux objectifs — et l'écart
   training-free du papier EUVIP (41 % → 63 % de Jaccard) prouve qu'il porte du
   signal fissure. Toute la chaîne écrite ici s'y transpose : seul le chargeur de
   modalité change.
2. **Mesurer avant d'espérer.** Les trois portes de cette campagne — décodage,
   plafond d'amplitude, recalage avec contrôle — coûtent quelques minutes de GPU
   et auraient chacune pu arrêter le projet. Elles doivent précéder toute
   nouvelle campagne, sur n'importe quel jeu.
3. **Le résultat positif mérite d'être poussé pour lui-même.** `+0,045` d'IoU à
   3 px en transfert de domaine, avec `20 835` paramètres et sans jamais toucher
   au segmentateur, est un résultat utile et bon marché. Il ne demande aucune
   seconde modalité.

---

*Données : [`results/2026-08-12_campagne_irt_crack/`](results/2026-08-12_campagne_irt_crack/) —
métriques par image des 19 exécutions, audit du jeu, plafond de correction,
split officiel. Spécification et ses corrections :
[`README.md`](README.md), [`ERRATA.md`](ERRATA.md),
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md).*
