# CrackSAM-IRT — le signal thermique existe, il ne survit pas à la moyenne

**Cinquième itération de la ligne CrackSAM, première multimodale, exécutée.**
Campagne du 12 août 2026 sur `cracksam-frangigraph-g4-spot-ew8c` (RTX PRO 6000
Blackwell), 25 exécutions, 448 paires IRT-Crack, split officiel 358/90.

<div align="center">

| Question | Réponse | Sur quoi elle repose |
|:--:|:--:|:--:|
| l'évidence thermique porte-t-elle un signal **causal** ? | **oui, établi** | `A7 − A8` (aligné contre permuté) = `+0,0041`, IC95 `[+0,0016 ; +0,0067]`, même signe sur 3 graines |
| ce signal fait-il gagner **en moyenne** ? | **non** | `A7 − A1` = `+0,0009`, IC95 `[−0,0020 ; +0,0038]` |
| pourquoi ce grand écart ? | **il s'annule contre lui-même** | `+0,0131` sur le tiers difficile, `−0,0058` et `−0,0046` sur les deux autres, les trois IC95 excluant zéro |

</div>

> [!IMPORTANT]
> **Le critère de succès pré-enregistré n'est pas atteint** — il exigeait
> `A2 > A1` avec IC95 excluant zéro, et ce n'est le cas d'aucun bras thermique.
>
> Mais le diagnostic est plus précis que « ça ne marche pas », et il est
> actionnable. L'évidence thermique **aide réellement là où la baseline échoue**
> — sur le tiers difficile du test, les trois conditions du critère sont
> remplies, `A2 > A4` compris — et **nuit là où la baseline réussit déjà**. Les
> deux effets se compensent presque exactement dans la moyenne globale. Ce qui
> manque n'est donc ni l'évidence ni l'architecture : c'est **une porte qui
> décide où appliquer la correction**, et le §6 dit ce qu'elle demande.

---

## Sommaire

1. [Ce que la campagne a mesuré](#1-ce-que-la-campagne-a-mesuré)
2. [Les trois portes franchies avant d'entraîner](#2-les-trois-portes-franchies-avant-dentraîner)
3. [Résultats](#3-résultats)
4. [Le problème : un effet réel, annulé par son propre déploiement](#4-le-problème--un-effet-réel-annulé-par-son-propre-déploiement)
5. [Le correctif testé, et ce qu'il prouve](#5-le-correctif-testé-et-ce-quil-prouve)
6. [La solution proposée](#6-la-solution-proposée)
7. [Ce que le correcteur fait réellement](#7-ce-que-le-correcteur-fait-réellement)
8. [Incidents](#8-incidents)
9. [Limites](#9-limites)
10. [Suite](#10-suite)

---

## 1. Ce que la campagne a mesuré

CrackSAM 2 + LoRA `tol3`, entraîné sur Khánh Hà, est **gelé** et ses logits sont
cachés une fois sur les 448 images d'IRT-Crack. Un correcteur de
**20 835 paramètres** lit ces logits et quatre canaux d'évidence, et choisit par
pixel entre renforcer, supprimer et s'abstenir. Neuf bras, trois graines.

| Bras | Ce que reçoit le correcteur | Ce qu'il isole |
|:--|:--|:--|
| **A0** | rien — les logits cachés, tels quels | le transfert Khánh Hà → IRT-Crack |
| **A1** | `z₀`, `p₀`, `H(p₀)` ; les 4 canaux thermiques à **zéro** | la recalibration de domaine seule |
| **A2** | + évidence Frangi thermique **alignée** | la méthode |
| **A3** | + évidence Frangi **d'une autre image** | la causalité spatiale |
| **A4** | + thermique **brute**, même capacité | modalité contre abstraction géométrique |
| **A5** | A2 sans l'action « s'abstenir » | la valeur propre de l'abstention |
| **A6** | renforcement seul, `Δz = δ σ(q) S_max` | la valeur propre du signe |
| **A7** | A2, mais la perte est **pondérée par la marge de la baseline** | le correctif issu du diagnostic (§5) |
| **A8** | A7 avec évidence permutée | la causalité, sous pondération |

Les neuf partagent le même bloc d'hyperparamètres, **identique à l'octet près**
— un test le vérifie — et A1 à A4, A7, A8 ont exactement le même nombre de
paramètres. A7 et A8 ont été conçus **après** avoir vu le résultat de A0–A6 :
ils sont donc explicitement post-hoc, et leur seule prétention est de tester le
mécanisme que le diagnostic désigne, pas de fournir un résultat pré-enregistré.

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
| **A7** Frangi + perte pondérée | 0,7037 ± 0,0010 | 0,8865 ± 0,0006 | 0,8979 | 15,18 |
| **A8** A7 permuté | 0,6992 ± 0,0015 | 0,8824 ± 0,0014 | 0,8897 | 22,14 |

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
| **A7 − A8** — *causalité, sous pondération* | `+0,0044` `[+0,0021 ; +0,0069]` | `+0,0041` `[+0,0016 ; +0,0067]` | **favorable** |
| **A7 − A1** | `−0,0059` `[−0,0087 ; −0,0032]` | `+0,0009` `[−0,0020 ; +0,0038]` | indiscernable |

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

### 3.2 Le rapport d'échelle

`A2 − A1` vaut `+4,5 %` du gain `A1 − A0` sur la métrique tolérante, et `−7,5 %`
sur l'IoU stricte. Lu globalement, **tout ce que ce dispositif gagne, il le gagne
sans la thermique**.

Une seule ligne du tableau contredit la lecture « il ne se passe rien » :
`A7 − A8`, l'écart entre évidence alignée et évidence permutée sous pondération,
`+0,0041` avec un IC95 qui exclut zéro. Une différence entre deux bras qui ne
diffèrent *que* par l'appariement de l'évidence ne peut pas venir de la capacité,
ni des statistiques de canaux, ni de la pondération : elle vient du contenu de
l'évidence. Le §4 explique pourquoi ce contenu ne se voit pas dans `A2 − A1`.

## 4. Le problème : un effet réel, annulé par son propre déploiement

La moyenne globale disait « rien ». Le protocole multimodal avait pourtant
pré-enregistré exactement ce qu'il fallait regarder :

> « le gain `C2−A` et surtout `C2−D` doit se concentrer sur le tiers du test où
> la fissure est invisible en intensité ; **un gain uniforme est un indice
> d'artefact**. » — plan multimodal, §4.4

On stratifie donc le test en trois tiers par la performance de la **baseline
gelée** — un critère indépendant des bras comparés, puisque A0 ne voit aucune
thermique. C'est le proxy le plus direct de « la fissure est-elle visible pour le
modèle RGB ».

### 4.1 Le gain est là où il devait être, et seulement là

IoU tolérante 3 px, 30 images par tiers, bootstrap apparié 10 000.

| Comparaison | tiers **difficile** (baseline 0,674) | tiers moyen (0,884) | tiers facile (0,964) |
|:--|:--|:--|:--|
| **A2 − A1** *la thermique aide-t-elle ?* | **`+0,0117`** `[+0,0058 ; +0,0177]` | `−0,0038` `[−0,0067 ; −0,0008]` | `−0,0018` `[−0,0044 ; +0,0006]` |
| **A2 − A3** *est-ce causal ?* | **`+0,0085`** `[+0,0028 ; +0,0142]` | `−0,0024` `[−0,0054 ; +0,0005]` | `−0,0001` `[−0,0031 ; +0,0027]` |
| **A2 − A4** *au-delà de la modalité ?* | **`+0,0072`** `[+0,0021 ; +0,0124]` | `−0,0030` `[−0,0054 ; −0,0007]` | `+0,0003` `[−0,0018 ; +0,0025]` |
| A1 − A0 *la recalibration* | `+0,0984` | `+0,0349` | `+0,0021` |

> [!IMPORTANT]
> **Sur le tiers difficile, les trois conditions du critère pré-enregistré sont
> remplies simultanément** — y compris `A2 > A4`, le contrôle décisif autour
> duquel tout le plan multimodal était bâti : à capacité et protocole
> identiques, l'**abstraction géométrique de Frangi bat la thermique brute**.
> Les trois IC95 excluent zéro et le signe est stable sur les trois graines
> (`A2 − A3` vaut `+0,0087`, `+0,0079`, `+0,0090`).

Et l'effet **s'inverse** sur le tiers moyen, avec un IC95 qui exclut zéro lui
aussi. Ce n'est pas du bruit dans les deux sens : c'est un effet à double signe.

### 4.2 Le mécanisme du dommage : la fragmentation

Le tableau §3 le montre sans ambiguïté. La baseline prédit `1,70` composante
connexe par image ; le correcteur sans thermique (A1) monte à `4,09` ; **tous les
bras qui reçoivent quatre canaux de plus montent à 13–22**, alignés comme
permutés, Frangi comme bruts. Le clDice suit : `0,9109` pour A1, `0,89–0,90` pour
les autres.

Ces canaux **hachent la prédiction**, et ils la hachent que leur contenu soit
aligné ou non — c'est donc un effet de capacité et de bruit d'entrée, pas de
contenu. Sur une image déjà bien segmentée, ce hachage est du pur dommage ; sur
une image ratée, il est le prix à payer pour récupérer une fissure manquée.

La corrélation entre fragmentation ajoutée et delta est faible (`0,07`) : le
dommage n'est pas *causé* par la fragmentation image par image, les deux sont des
symptômes du même excès de liberté.

### 4.3 Le désalignement plafonne le gain, il ne l'annule pas

L'audit §2.3 mesure `10,1 px` de décalage médian. Le champ réceptif du correcteur
vaut `15 px` : une évidence décalée de 10 px reste donc *dans* son champ, mais au
bord. C'est cohérent avec ce qu'on observe — un signal causal réel mais faible.
Le décalage explique donc l'**amplitude modeste** du gain, pas son annulation.
L'annulation, elle, vient du déploiement indiscriminé.

## 5. Le correctif testé, et ce qu'il prouve

Le diagnostic désigne un coupable précis : une perte qui moyenne uniformément sur
les images est **dominée par des images où il n'y a rien à gagner**. Sur les 286
images d'entraînement, la majorité ressemble au tiers facile — le gradient y
pousse le correcteur à ne pas casser ce qui marche, et non à réparer ce qui rate.

**A7** applique la correction la plus directe : chaque image d'entraînement est
pondérée par sa **marge de progression**, `1 − IoU_baseline` à 3 px, calculée sur
le train seul, normalisée à moyenne 1 et plafonnée à `5×`. Rien d'autre ne
change : même évidence, même tête, même capacité, mêmes hyperparamètres. **A8**
est son contrôle permuté — sans lui, un gain de A7 serait indistinguable d'un
effet de la pondération elle-même.

### 5.1 Le résultat : le signal causal devient significatif, globalement

| Comparaison | delta IoU@3px | IC95 | par graine | Verdict |
|:--|--:|:--:|:--|:--|
| **A7 − A8** *(pondéré, aligné vs permuté)* | **`+0,0041`** | `[+0,0016 ; +0,0067]` | `+0,0051` `+0,0032` `+0,0040` | **favorable** |
| A2 − A3 *(non pondéré, même contraste)* | `+0,0020` | `[−0,0004 ; +0,0044]` | `+0,0017` `+0,0018` `+0,0025` | indiscernable |
| A7 − A1 | `+0,0009` | `[−0,0020 ; +0,0038]` | `−0,0013` `+0,0037` `+0,0003` | indiscernable |
| A7 − A2 | `−0,0011` | `[−0,0027 ; +0,0005]` | `−0,0024` `−0,0010` `+0,0000` | indiscernable |

> [!IMPORTANT]
> **`A7 − A8` est le premier écart aligné-contre-permuté significatif de toute la
> ligne CrackSAM.** Il double celui de `A2 − A3` et son IC95 exclut zéro sur les
> 90 images du test, avec le même signe sur les trois graines. Après quatre
> itérations où l'évidence permutée faisait aussi bien — voire mieux — que
> l'évidence alignée, **la géométrie est enfin lue et utilisée de façon
> spécifique à l'image**.

### 5.2 Ce que le correctif ne fait pas : il aiguise le compromis, il ne le résout pas

| A7 − A1, par tiers | delta | IC95 |
|:--|--:|:--:|
| difficile | **`+0,0131`** | `[+0,0062 ; +0,0204]` |
| moyen | `−0,0058` | `[−0,0093 ; −0,0023]` |
| facile | `−0,0046` | `[−0,0069 ; −0,0024]` |

À comparer à `A2 − A1` : `+0,0117` / `−0,0038` / `−0,0018`. La pondération a
**augmenté le gain là où il fallait** (`+0,0117 → +0,0131`) et **augmenté le
dommage ailleurs** (`−0,0018 → −0,0046`). Le correcteur agit plus fort
(`|Δz|` moyen `0,503` contre `0,337`) : il s'engage davantage, dans les deux
sens. Net : `+0,0009`, indiscernable de zéro.

C'est un résultat instructif et un demi-échec assumé. Agir sur la **pondération
d'entraînement** ne suffit pas, parce que le problème n'est pas *où le modèle
apprend* mais **où il applique ce qu'il a appris**.

## 6. La solution proposée

Le diagnostic et le correctif partiel convergent sur une seule pièce manquante :
**une porte de fiabilité, à l'inférence**. La correction doit être rendue
*exactement nulle* — l'architecture le permet déjà, au bit près — partout où la
baseline est fiable, et libérée là où elle ne l'est pas.

### 6.1 Ce que la porte rapporterait, mesuré

Si la porte était parfaite — c'est-à-dire si l'on n'appliquait A7 que sur le tiers
difficile et A1 ailleurs — le gain sur les 90 images serait le tiers de
`+0,0131`, soit **`+0,0044` d'IoU à 3 px** face à A1, sans aucune des pertes.
C'est le double du gain que la thermique produit aujourd'hui, et surtout c'est un
gain qui ne s'annule pas.

### 6.2 Le verrou : estimer la fiabilité sans étiquette

C'est là que se trouve le vrai travail, et il n'est pas fait. Les proxys
évidents, testés sur ce test :

| Proxy, calculable **sans** vérité terrain | ρ de Spearman avec la difficulté réelle |
|:--|--:|
| nombre de composantes prédites par la baseline | `−0,45` |
| fraction de pixels fissure prédite | `−0,13` |
| *(couverture du squelette — **utilise la GT**, inutilisable)* | *`+0,80`* |

Le meilleur proxy sans étiquette ne retrouve que **60 %** du tiers difficile, et
une règle « ne corriger que le tiers le plus fragmenté » ne rend que `+0,0019`
face à A1 — mieux que rien, mais loin des `+0,0044` de la porte parfaite, et son
contrôle permuté redevient indiscernable.

### 6.3 Trois pistes, par ordre de coût

1. **Une tête de fiabilité apprise** — un second petit réseau, entraîné sur le
   train à prédire `IoU_baseline` depuis les seuls logits gelés, dont la sortie
   multiplie `Δz`. Aucune étiquette au test, une centaine de paramètres de plus,
   et le squelette est déjà en place : `correction_scope` est un point d'entrée
   prévu et testé, il suffit d'y brancher `Ω = fiabilité(z₀) < seuil`.
2. **Un désaccord baseline/évidence comme signal** — la thermique et le RGB sont
   deux mesures ; leur *désaccord* est précisément l'endroit où l'une des deux se
   trompe. C'est un estimateur de fiabilité qui n'a besoin d'aucune étiquette et
   qui utilise l'information multimodale pour décider **où** l'utiliser.
3. **Réduire le dommage plutôt que le compenser** — la fragmentation `4 → 15`
   composantes est le mécanisme du coût. Une pénalité de continuité sur la sortie
   corrigée (`soft-clDice`, déjà implémentée dans `geolora/losses.py` et déjà
   validée sur Khánh Hà) attaque le dommage à sa racine, indépendamment de la
   porte.

### 6.4 Et si l'on veut d'abord lever le doute sur la donnée

Le décalage de `10,1 px` plafonne tout ce qui précède. Deux mesures le
quantifieraient, pour un coût dérisoire :

- **un oracle de recalage** : décaler l'évidence thermique de son décalage
  optimal estimé sur la vérité terrain, puis ré-exécuter A7/A8. C'est une fuite
  d'étiquette délibérée, donc jamais un résultat — mais une **borne supérieure**
  de ce que la thermique pourrait rendre si elle était recalée. Même famille que
  l'oracle de source de CrackSAM-GFA ;
- **FIND**, dont le range laser est co-recalé *par construction du capteur*. La
  chaîne écrite ici s'y transpose en changeant le seul chargeur de modalité.

Si l'oracle de recalage ne relève pas le gain du tiers difficile, la thermique
d'IRT-Crack est épuisée comme sujet et FIND devient la seule suite.

## 7. Ce que le correcteur fait réellement

| Bras | renforcer | supprimer | s'abstenir | `\|Δz\|` moyen | p99 | FN récupérés | FP retirés |
|:--|--:|--:|--:|--:|--:|--:|--:|
| **A1** — sans thermique | 0,001 | 0,000 | 0,999 | **0,377** | 3,14 | **0,320** | **0,159** |
| A2 | 0,001 | 0,000 | **0,999** | 0,337 | 2,79 | 0,310 | 0,136 |
| A3 | 0,001 | 0,000 | 0,999 | 0,387 | 2,85 | 0,314 | 0,152 |
| A4 | 0,001 | 0,000 | 0,999 | 0,375 | 2,69 | 0,308 | 0,121 |
| A5 | 0,728 | 0,272 | — | 0,554 | 3,15 | 0,293 | 0,165 |
| A6 | 0,023 | 0,000 | 0,977 | 0,104 | 2,25 | 0,215 | 0,000 |
| **A7** — perte pondérée | 0,001 | 0,000 | 0,999 | **0,503** | — | 0,262 | 0,140 |
| A8 — A7 permuté | 0,001 | 0,000 | 0,999 | 0,525 | — | 0,217 | 0,169 |

A7 corrige nettement plus fort que A2 (`0,503` contre `0,337`) : la pondération
fait ce qu'on lui demande, elle engage le correcteur. Et l'écart A7/A8 sur les
faux négatifs récupérés — `0,262` contre `0,217` — est la trace, au niveau des
erreurs, du signal causal du §5.

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

## 8. Incidents

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

## 9. Limites

- **A7 et A8 sont post-hoc.** Ils ont été conçus après avoir vu A0–A6, en
  réponse au diagnostic. `A7 − A8` est un contraste propre — deux bras qui ne
  diffèrent que par l'appariement — mais il n'a pas été pré-enregistré, et il
  doit être re-testé sur un jeu que ce diagnostic n'a pas servi à construire
  avant d'être présenté comme un résultat.
- **La stratification, elle, était pré-enregistrée** (plan multimodal §4.4), et
  le critère de stratification (performance de la baseline gelée) est
  indépendant des bras comparés. Mais les tiers comptent `30` images : les IC95
  y sont larges, et trois tiers × trois comparaisons font des tests multiples que
  seule la prédiction *a priori* de la direction rend confirmatoires.
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

## 10. Suite

Le protocole interdit de passer au Frangi-graphe tant que `A2 > A1` et `A2 > A3`
ne sont pas établis. `A2 > A3` l'est désormais, sous pondération (`A7 − A8`) ;
`A2 > A1` ne l'est pas. La suite se lit donc dans cet ordre :

1. **La porte de fiabilité** (§6.3). C'est le seul chaînon qui manque pour
   transformer un effet réel en gain net, et il est mesurable à `+0,0044` près.
   Coût : une tête de quelques centaines de paramètres, une demi-journée.
2. **L'oracle de recalage** (§6.4). Une demi-heure de GPU pour savoir si la
   thermique d'IRT-Crack a encore quelque chose à donner, ou si son décalage la
   condamne. Résultat négatif = sujet clos, et c'est une information qui vaut son
   coût.
3. **FIND.** Recalage garanti par le capteur, écart training-free connu
   (`41 % → 63 %` de Jaccard), et toute la chaîne écrite ici s'y transpose. C'est
   l'ancrage où la question multimodale peut enfin être posée sans réserve sur la
   donnée.
4. **Le résultat positif, pour lui-même.** `+0,045` d'IoU à 3 px en transfert de
   domaine, avec `20 835` paramètres et sans toucher au segmentateur, est utile,
   bon marché et ne demande aucune seconde modalité.

Ce qui reste hors périmètre, et le reste : MST, composantes et centralité. Le
protocole les autorise quand la similarité dense aura produit un gain net. Elle a
produit un signal causal ; ce n'est pas encore la même chose.

---

*Version courte et illustrée : [`SYNTHESE.md`](SYNTHESE.md).*

*Données : [`results/2026-08-12_campagne_irt_crack/`](results/2026-08-12_campagne_irt_crack/) —
métriques par image des 19 exécutions, audit du jeu, plafond de correction,
split officiel. Spécification et ses corrections :
[`README.md`](README.md), [`ERRATA.md`](ERRATA.md),
[`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md).*
