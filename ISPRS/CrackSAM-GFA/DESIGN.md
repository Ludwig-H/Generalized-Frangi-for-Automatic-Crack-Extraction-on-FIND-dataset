# CrackSAM-GFA — conception et pré-enregistrement

> **Date de rédaction :** 8 août 2026
>
> **Statut :** protocole pré-enregistré, écrit **avant** toute mesure sur SAM 2
>
> **Origine :** [`RAPPORT.md` de l'étude anti-ombre du 8 août 2026](../CrackSAM/results/2026-08-08_guidage_geometrique_anti_ombre/RAPPORT.md), §9 et §10

GFA signifie **G**eometric **F**ragment **A**rbitration. Ce document fige la
méthode, les seuils de décision et les contrôles avant de regarder le moindre
résultat. Tout écart ultérieur devra être déclaré explicitement dans
[`RAPPORT.md`](RAPPORT.md).

---

## 1. Ce que l'étude anti-ombre impose

L'étude filtre-seul du 8 août 2026 conclut par un **no-go pour une carte
scalaire autonome et pour tout nouveau `mask_input`**. Cinq contraintes en
découlent, et elles ne sont pas négociables ici.

| # | Contrainte issue du rapport | Conséquence d'implémentation |
|---|---|---|
| C1 | La voie baseline `z0` doit rester exacte | `mask_input=None`, chemin SAM 2 officiel sans prompt, et identité **bit à bit** hors bande acceptée |
| C2 | `mask_input` n'est plus une interface d'entraînement ni de déploiement | Il ne subsiste que comme **contrôle négatif** dans l'oracle d'interface |
| C3 | Les canaux ne doivent **jamais** être multipliés avant l'arbitre | Le tenseur d'évidence garde 11 plans séparés ; aucune moyenne géométrique en amont |
| C4 | L'enveloppe candidate est l'union de fragments courts orientés passant des seuils **absolus** propres à chaque source, et peut être vide | Support ≠ support Frangi ; `∅` est une sortie légale qui renvoie `z0` |
| C5 | Un arbitre local classe chaque bande en `ajouter` / `retirer` / `s'abstenir` | Décision au niveau du fragment, pas du pixel |

La justification causale de C3 est le résultat le plus important du banc
phantom : les quatre cartes `verified_frangi_*` chutent à `≈10 %` de rétention
sur le phantom de traversée, contre `89,5 %` pour OFS seul, parce qu'un « ET »
multiplicatif hérite de l'effondrement de Frangi sous l'ombre.

La justification de C4 est le cas `DJ_Wall_66` : l'historique est dominé par le
linteau sombre et manque la fissure, tandis que la fusion sans Frangi la trouve
(`AP2=0,198`, rappel `0,868`). Un support imposé par Frangi rendrait cette
fissure inatteignable.

## 2. Ce que j'adapte, et pourquoi

Le rapport recommande `verified_local_v2_signed`. Je le suis sur l'ossature et
j'en modifie trois points ; chaque écart est motivé par une observation du
rapport lui-même.

### 2.1 Décision par fragment plutôt que par pixel

**Adaptation.** L'unité de décision est un **fragment court orienté**
(longueur géodésique cible 16–24 px à 448), pas le pixel.

**Motif.** Le rapport demande déjà de « découper le graphe en courts fragments
orientés et ne convertir en points que les fragments vérifiés » (§3.7, leçon
CaPro) et de classer « chaque bande » (§10). Le fragment est aussi la seule
unité à laquelle la sémantique `s'abstenir` est vérifiable : un pixel n'a ni
tangente, ni longueur, ni cohérence d'orientation. Effet secondaire décisif :
il y a `O(10–50)` fragments par image contre `200 704` pixels, ce qui rend
l'arbitre entraînable sur 9 120 images sans sur-paramétrer.

### 2.2 Le veto d'ombre est une *feature d'entrée*, jamais un facteur

**Adaptation.** OFS et OFA entrent dans l'arbitre comme statistiques de
fragment. Aucun produit `Frangi × OFS` n'est calculé nulle part.

**Motif.** C'est C3. Le rapport le dit explicitement : « Le BIC, la phase,
l'énergie et l'échelle restent des features, pas des décisions. »

### 2.3 Deux têtes bornées, et une sélection dure en sortie

**Adaptation.** `Δz = +a_add` sur les corridors `ajouter`, `−a_rem` sur les
corridors `retirer`, avec `a_add, a_rem ∈ [0, A]` et `A = 4,0` nats de logit.
Sortie `torch.where(B, z0 + Δz, z0)`.

**Motif.** Le rapport exige la garantie bit à bit « et non de la supposer à
partir d'une confiance multipliée ». Une sélection dure `torch.where` est
testable ; une pondération douce ne l'est pas. Les projections finales sont
initialisées à zéro, donc le modèle non entraîné vaut exactement `z0`.

`A = 4,0` est choisi avant toute mesure : `σ(z0 ± 4)` déplace une probabilité de
`0,5` à `0,982` / `0,018`, ce qui suffit à faire basculer une bande sans
permettre à une seule correction d'écraser un `z0` déjà très confiant
(`|z0| > 4` reste dominant). Ce nombre n'est pas ajusté après coup.

## 3. Architecture

```text
                    image RGB 448×448
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
   SAM 2 Hiera-L + LoRA r=4 (GELÉ)        évidence géométrique
        │                                  (11 canaux, CPU/GPU)
   encode_images()                              │
        │                                  seuils ABSOLUS par source
   ┌────┴─────┐                                 │
   │          │                          union des supports  ──► peut être ∅
 z0 =      features                             │
 decode(   s4, s8                        squelettisation
 mask_input=None)                               │
   │          │                          fragments courts orientés
   │          │                          + corridor (échelle locale, sans GT)
   │          │                                 │
   │          └────────────┬────────────────────┘
   │                       ▼
   │            statistiques par fragment
   │       (axe, flancs ±1,5·échelle, corridor)
   │                       │
   │                  ARBITRE LOCAL
   │            MLP par fragment + attention
   │            inter-fragments (même image)
   │                       │
   │        ┌──────────────┼──────────────┐
   │    p(ajouter)   p(retirer)    p(s'abstenir)
   │        │              │
   │     a_add ∈[0,A]   a_rem ∈[0,A]        (têtes bornées, init. zéro)
   │        └──────┬───────┘
   │               ▼
   │        Δz sur corridors acceptés,  B = ∪ corridors acceptés
   └───────────────┬───────────────────┘
                   ▼
        z = torch.where(B, z0 + Δz, z0)
        si B = ∅  ►  z ≡ z0  (bit à bit)
```

### 3.1 Les 11 canaux d'évidence

Ce sont exactement ceux listés au §9.2 du rapport. Aucun n'est normalisé par le
maximum spatial de l'image ; le niveau de bruit est estimé par MAD, comme dans
l'étude filtre-seul.

| # | Canal | Rôle | Origine |
|---|---|---|---|
| 1 | `frangi_sim` | proposeur haute précision, sensible à l'ombre | `historical_frangi_similarity_cpu` |
| 2 | `ofs` | symétrie de flux orienté — veto de marche, abstention | `oriented_flux_symmetry` |
| 3 | `ofa` | **anti**symétrie de flux orienté — évidence *positive* de marche | nouveau, dérivé du même anneau |
| 4 | `even_odd` | paire/impair à même σ | `even_odd_derivative_pair` |
| 5 | `dbic` | `BIC(H0) − BIC(H1)`, ligne contre marche | `line_step_bic` |
| 6 | `phase_sym` | symétrie de phase sombre, meilleur rappel squelette | `dark_phase_symmetry` |
| 7 | `abs_energy` | énergie absolue en unités MAD | nouveau |
| 8–9 | `cos2θ`, `sin2θ` | orientation double-angle | Hessienne fusionnée |
| 10 | `scale` | échelle gagnante (rayon) | argmax multi-échelle |
| 11 | `profile` | profil bilatéral vallée/marche | `paired_profile` |

`ofa` est un ajout assumé : le rapport demande de conserver « antisymétrie OFA »
au §9.2 alors que l'étude ne mesurait qu'OFS. Une marche d'ombre doit produire
un `ofa` élevé et un `ofs` faible ; c'est le signal qui permet à l'arbitre de
choisir `retirer` plutôt que simplement `s'abstenir`.

### 3.2 Seuils absolus par source

Chaque source `s` reçoit un seuil `t_s` calibré **une seule fois**, sur le pli
d'entraînement uniquement, comme le quantile `q = 0,995` de la réponse de `s`
sur les images d'entraînement. Ces seuils sont ensuite gelés et écrits dans le
contrat de run. Ils ne sont jamais recalculés par image : c'est précisément le
défaut relatif que le rapport reproche à la chaîne historique (§2.2).

Sources contribuant au support : `frangi_sim`, `phase_sym`, `ofs`, `dbic`.
L'union est prise **après** seuillage individuel, jamais un produit.

## 4. Protocole d'évaluation

### 4.1 Baseline de comparaison

`baseline_r4/best.pt` — SAM 2 Hiera-L + LoRA q/v rang 4, α=4, entraînée
70 époques sur Khánh Hà (9 121 images), seed 3407, 448×448. C'est le checkpoint
gelé archivé dans
`ISPRS/CrackSAM/artifacts/vm_backup_20260714T1435Z_final_checkpoints/`.

**Test de reproduction obligatoire (porte n° 0).** Avant toute chose, `z0`
recalculé sur les 1 695 images de `khanhha_original` doit redonner :

| Métrique | Valeur archivée à reproduire |
|---|---:|
| IoU | `0,623804` |
| Dice | `0,745320` |
| Précision | `0,749322` |
| Rappel | `0,771146` |

Tolérance : `|Δ IoU| ≤ 0,002`. Un écart supérieur invalide toute la chaîne et
doit être résolu avant de continuer.

### 4.2 Partition strictement hors-pli

L'unité statistique est la **scène physique**, jamais le crop perturbé. Les
triplets `original/noisy1/noisy2` d'une même scène restent toujours dans le même
pli. Cinq plis par hachage SHA-256 du nom de scène, comme le pilote existant.
L'arbitre est entraîné sur les plis d'entraînement uniquement et évalué sur des
prédictions `z0` strictement hors pli.

### 4.3 Les deux plafonds à mesurer AVANT d'entraîner

Le rapport l'exige : « Avant de l'entraîner, deux plafonds […] doivent dépasser
des seuils pré-enregistrés. »

#### Oracle de source — la porte décisive

Pour chaque fragment, muni de son corridor **fixé sans GT**, on choisit avec le
GT la meilleure action parmi `{ajouter, retirer, s'abstenir}`, sans jamais
recouper, déplacer, réorienter ni redimensionner le fragment.

> **Seuil pré-enregistré : un gain d'IoU groupé `< +0,01` est un no-go pour
> cette famille de candidats.**

Ce plafond majore ce que *n'importe quel* arbitre peut atteindre avec cette
famille de fragments, ce corridor et cet ensemble d'actions. S'il échoue, la
famille de candidats est réfutée et l'entraînement de l'arbitre est annulé — ce
résultat négatif est alors le livrable.

#### Oracle d'interface — diagnostic uniquement

Face à `None`, avec une géométrie **parfaite** dérivée du GT :

1. `k` points positifs échantillonnés géodésiquement sur le squelette GT ;
2. la même suite de points, plus des points négatifs sur les marches OFA non
   soutenues par OFS ;
3. un corridor parfait injecté comme `mask_input` — **contrôle négatif**.

Budget de points et rasterisation fixés avant lecture des résultats :
`k ∈ {1, 4, 12}`, corridor de rayon 3 px. Son échec élimine une interface ; son
succès ne valide ni le générateur automatique ni l'adapter.

### 4.4 Contrôles causaux obligatoires

Aucun gain ne sera revendiqué sans ces cinq contrôles, tous à couverture
identique (§9.1 du rapport) :

| Contrôle | Ce qu'il réfute |
|---|---|
| `none` | que le gain vienne du seul fait de décoder deux fois |
| `null` | qu'un tenseur nul soit équivalent à l'absence de prompt |
| `permuted` | que l'évidence apporte autre chose qu'une couverture |
| `shifted` | que l'alignement spatial soit indifférent |
| `random_support` | qu'un support quelconque de même taille suffise |

### 4.5 Métriques

Primaire : **IoU** au seuil `0,5`, moyenne par scène physique, avec IC95 par
bootstrap groupé par scène (2 000 rééchantillonnages, graine `20260808`).
Le delta face à `z0` est **apparié**.

Secondaires : Dice, précision, rappel, et — parce que la continuité est l'enjeu
géométrique — nombre de composantes connexes et longueur de squelette couverte.

Conditions : `khanhha_original`, `khanhha_noisy1`, `khanhha_noisy2`. Les jeux
externes `road420` et `facade390` sont des transferts hors domaine et sont
rapportés séparément, jamais agrégés dans la décision principale.

## 5. Critères de succès pré-enregistrés

| Porte | Critère | Si échec |
|---|---|---|
| 0 — reproduction | `|Δ IoU| ≤ 0,002` sur `khanhha_original` | arrêt, chaîne invalide |
| 1 — oracle de source | gain IoU groupé `≥ +0,01` | **no-go**, on ne l'entraîne pas ; on publie le plafond |
| 2 — identité | `z ≡ z0` bit à bit hors `B` et quand `B = ∅` | bug bloquant |
| 3 — gain réel | `Δ IoU` apparié `> 0` avec IC95 excluant `0`, hors pli | pas de revendication de gain |
| 4 — causalité | gain supérieur à `permuted`, `shifted` et `random_support` | gain non attribuable à la géométrie |

Une méthode qui passe 0, 1 et 2 mais échoue en 3 reste un résultat publiable :
elle mesure l'écart entre le plafond oracle et ce qu'un arbitre sans GT atteint.

## 6. Ce que cette étude ne fera pas

- Aucune nouvelle LoRA, aucun GNN : le rapport l'interdit tant que le petit
  adapter n'a pas battu `z0` et ses contrôles (§9.4).
- Aucune suppression d'ombre irréversible en prétraitement : elle peut supprimer
  une fissure qui traverse l'ombre (§3.5).
- Aucune évaluation sur ombres naturelles annotées ni sur Shadow-Crack : ces
  jeux ne sont pas dans le dépôt. La robustesse aux ombres naturelles restera
  donc une **hypothèse**, exactement comme dans l'étude filtre-seul.
- Aucun réglage de seuil sur le jeu de test.

---

## Références internes

- [Étude filtre-seul anti-ombre](../CrackSAM/results/2026-08-08_guidage_geometrique_anti_ombre/RAPPORT.md)
- [Guidage géométrique anti-ombre](../CrackSAM/docs/10_GUIDAGE_GEOMETRIQUE_ANTI_OMBRE_CRACKSAM2.md)
- [Question expérimentale et vocabulaire](../CrackSAM/docs/01_EXPERIMENTAL_QUESTION.md)
- [Raccordement Frangi doublement ancré](../CrackSAM/docs/09_REPONSE_CONCLUSION_FRANGI_SAM2.md)
- [Papier EUVIP — Generalized Frangi](../../EUVIP/EUVIP_2026_Generalized_Frangi_Multimodality_camera-ready.pdf)
