# CrackSAM-MultiModal — guider SAM par une géométrie calculée sur une modalité qu'il ne voit pas

**Cinquième itération de la ligne CrackSAM — à l'état de proposition.** Rien de ce qui suit
n'a été exécuté : ce document est l'inventaire des benchmarks promis dans le mail du
11 août 2026 à Pierre Charbonnier, l'évaluation de faisabilité du projet, et le plan
d'expériences pré-enregistrable.

<div align="center">

| Statut | Question | Jeux d'ancrage | Budget estimé | Origine |
|:--:|:--:|:--:|:--:|:--:|
| **Proposition** | l'évidence Frangi **multimodale** guide-t-elle SAM ? | FIND · IRT-Crack | ~22 runs · <100 € GPU | [Mail du 11/08/2026](../CrackSAM-GeoLoRA/presentations/2026-08-09-cracksam-geolora/Mail_rapport_2026-08-09.pdf) |

</div>

> [!IMPORTANT]
> **Le verdict de faisabilité est GO, sous trois conditions de cadrage** (§4.1) qui coûtent
> ensemble une lecture, un run de ~10 € et une demi-journée — et qui peuvent chacune
> arrêter le projet proprement avant d'avoir dépensé quoi que ce soit.
>
> Le verrou qui a tué les quatre itérations monomodales — l'évidence était une fonction
> déterministe de l'image que SAM voyait déjà — est levé **par construction** sur au moins
> un jeu public : le range laser de FIND est une mesure *active* du creux, information
> absente de l'intensité par la physique, et l'écart training-free du papier EUVIP
> (41 % Jaccard intensité seule → 63 % fusion) prouve qu'elle porte du signal fissure.

---

## Sommaire

1. [La question, et ce qui la rend testable](#1-la-question-et-ce-qui-la-rend-testable)
2. [Recension des benchmarks](#2-recension-des-benchmarks)
3. [Aura-t-on assez de données ?](#3-aura-t-on-assez-de-données-)
4. [Le plan](#4-le-plan)
5. [Objections anticipées et parades](#5-objections-anticipées-et-parades)
6. [La question capteur, pour le Cerema](#6-la-question-capteur-pour-le-cerema)
7. [Références](#7-références)

---

## 1. La question, et ce qui la rend testable

Formulation du mail : *« un SAM → baseline CrackSAM → + léger ré-apprentissage avec
a priori métiers, mais calculés sur une modalité à laquelle n'a pas accès SAM »*.

Ce que les quatre itérations précédentes imposent à cette cinquième :

| Acquis (exécuté) | Conséquence pour ce plan |
|:---|:---|
| Contrôle permuté GeoLoRA : évidence RGB-dérivée ⇒ aucun effet d'alignement (\|Δ\|<0,001) | l'évidence doit être calculée sur une modalité **absente de l'entrée** de SAM — critère n°1 de tout benchmark |
| La matrice causale prouve que SAM **lit** une injection (+0,247 vs permuté) | l'échec monomodal était un échec d'*utilité*, pas de *lecture* : le canal d'injection existe |
| L'adapter additif aveugle fragmentait (3,3→6,0 composantes) et le poly-LR power=6 tuait l'entraînement (best epoch 0) | gating conditionné par l'image, LR constant, validation non augmentée — corrections pré-enregistrées |
| MST, composantes et centralité n'ont **jamais** atteint SAM (`compute_centrality=False` partout) | un bras dédié mesure enfin l'apport propre du graphe |
| La littérature qui réussit en multimodal (MM-SAM, IRFusionFormer…) injecte la **modalité brute**, jamais une évidence géométrique | le contrôle **modalité-brute** est le bras décisif — et cette comparaison n'existe nulle part : publiable dans les deux sens |

La question falsifiable du projet : **l'évidence géométrique Frangi calculée sur la
modalité cachée améliore-t-elle SAM 2 + LoRA au-delà (a) de la baseline mono-entrée et
(b) de la même modalité injectée brute par le même adapter ?**

---

## 2. Recension des benchmarks

### 2.1 Jeux d'ancrage — le plan repose sur eux

| Jeu | Modalités (recalage) | Taille | GT | Accès | Rôle |
|:---|:---|:---|:---|:---|:---|
| **FIND** — Zhou, Canchila & Song, 2022 | intensité laser + **range laser actif** + range filtré + fusion — co-recalés **par construction du capteur** (même balayage) | 2 500 patchs 256² par type | masques denses pixel | [Zenodo 6383044](https://zenodo.org/records/6383044) · CC-BY 4.0 | **jeu principal.** Seul cas où la non-redondance avec l'entrée de SAM est garantie par la physique. Split CrackSegDiff 2000/500 réutilisable ; plafond supervisé connu (91,9 % IoU dense ; 87 % en protocole EUVIP) |
| **IRT-Crack** — Liu, Liu & Wang, IEEE T-ITS 2022 | RGB + thermique FLIR (co-recalage caméra bi-objectif IR-Fusion™) + fusion 50/50 | 448 paires 640×480 · split canonique 358/90 | masques pixel (Photoshop) | [Zenodo 11624965](https://zenodo.org/records/11624965) · CC BY 4.0 | **réplication + voie déployable.** Benchmark RGB-T standard : IRFusionFormer IoU 81,8, ablation publiée RGB 83,3 / IR 64,4 / fusion 85,4 % F1 — le schéma EUVIP en supervisé |

> [!NOTE]
> **Deux corrections de provenance faites ou à faire.** (1) Le lien FIND de `README.md`
> pointait vers `zenodo.org/records/8332767` — le « FIND » du MIT (interprétabilité de
> fonctions, Schwettmann et al. 2023), un homonyme : corrigé vers le DOI canonique
> `10.5281/zenodo.6383044`. (2) Le jeu « IRT-Crack Segmentation » du dépôt, jusqu'ici un
> re-hébergement Google Drive sans provenance, est identifié : c'est le dataset de Liu,
> Liu & Wang (Virginia Tech), citable et redistribuable sous CC BY 4.0.

### 2.2 Compléments conditionnels

| Jeu | Ce qu'il apporte | Condition bloquante |
|:---|:---|:---|
| **Crack900** — XJTLU, [Mendeley kz84t85z66](https://data.mendeley.com/datasets/kz84t85z66/1), CC BY 4.0 | 914 paires RGB-T **maçonnerie** (FLIR E85) — le plus gros corpus RGB-T fissures ; généralisation enrobé→maçonnerie | l'IR natif 384×288 est suréchantillonné ×3,3 vers le RGB 1280×960 : 1 px d'erreur IR = 3,3 px, l'ordre d'une largeur de fissure. **Audit de recalage sur 20-30 paires avant tout engagement**, sinon rétrogradé en jeu de test |
| **CrackDepth** — archive [LIDAR-Mamba](https://github.com/Karl1109/LIDAR-Mamba), 2025 | 655 paires RGB + profondeur light-field 512², baselines multimodales publiées | la profondeur light-field est **passive** (reconstruite du signal optique) : risque de rejouer la redondance Khánh Hà — à traiter *après* FIND, avec le contrôle modalité-brute comme résultat principal |
| **CrackPolar** — même archive | 986 groupes RGB + polarisation (AoP/DoP) : information physiquement absente du RGB | aucun précédent de Hessienne sur canaux polarimétriques — extension de seconde vague |
| **Syncrack étendu** — [générateur public](https://github.com/Sutadasuto/syncrack_generator) (écosystème IFSTTAR) | la fissure y est une **courbe géométrique** : en dériver profondeur et pseudo-thermique donne des paires parfaitement recalées en volume illimité, pour pré-entraîner l'adapter | ne remplace jamais l'évaluation sur données réelles |
| **DeLiVER / MFNet / FMB** | validation du *mécanisme* sur structures fines (poteaux, rails de sécurité) avec recalage parfait (DeLiVER : synthétique CARLA, 7 900 échantillons) | pas de fissures — jamais un résultat du projet |

### 2.3 Écartés, avec motif — pour ne pas les rechercher

| Jeu | Motif d'exclusion |
|:---|:---|
| VT-GraF (interne) | n=5 : rien d'entraînable, aucun statut de redistribution. Vignette qualitative zéro-shot uniquement (paramètres canoniques : 1/3 V + 2/3 IR, Σ=[20,30,40]) |
| CrackForest, OmniCrack30k, lignée Aigle-RN/ESAR/LRIS | **monomodaux** — le critère causal est inapplicable par construction ; la page IRIT d'Aigle-RN est morte (404 vérifié le 11/08/2026) |
| Pothole-600, stereo pothole (Fan et al.) | RGB-D propre mais défauts **surfaciques** (blobs) : le prior linéique (Hessienne, MST, centralité) n'a pas d'objet |
| 3DCrack (Georgia Tech) | range **seul**, sans optique co-recalée : l'évidence redeviendrait fonction de l'entrée. Accès IEEE DataPort sous abonnement, v1.0 réduite à des exemples |
| M2S-RoAD | LiDAR 128 nappes trop clairsemé pour un creux millimétrique ; publication des données non confirmée |
| RSDDS-113 (rail RGB-D) | 113 triplets : inentraînable ; au mieux test de transfert zéro-shot |
| SDNET2021 (IRT+GPR+IE) | l'information est bien absente du RGB (délaminations sous-surface) mais GT **en taches**, modalités non recalées entre elles, pas de RGB apparié |
| Suite Zenodo Liu multi-distress, jeu « silicate spectrum » | annotations en boîtes ou jeu non distribué (dans la constellation Zenodo de Liu, **seul le 11624965 a des masques**) |
| Jeu nocturne IR-visible (CBM 2024) | conceptuellement idéal (RGB nocturne pauvre), **aucun lien public** : écrire aux auteurs (10 min), ne rien budgéter dessus |
| MS-CrackSeg (hyperspectral UAV, 1 031 images) | seul jeu hyperspectral entraînable, mais accès Baidu Pan uniquement — hasardeux depuis la France |
| LTPP/InfoPave | volumes énormes, **aucun GT pixel** |
| CT 3D (VoroCrack3d), CFRP pulsé, GPR (TIGPR), photométrie stéréo industrielle, PolarLITIS/ZJU-RGB-P, NYUDv2, PST900, MUSES | pas de vue optique pour SAM, ou pas de fissures, ou pas de recalage pixel, ou profondeur inexploitable sur structures fines |

### 2.4 Constats de vide — trois niches d'acquisition vacantes

Recherche close en août 2026 : il n'existe **aucun** jeu public de fissures de génie civil
avec (a) polarimétrie, (b) photométrie stéréo, ou (c) UAV visible+IR co-recalé d'ouvrages.
La physique des trois est pourtant favorable (DoP découplé de l'albédo ; discontinuité de
profondeur → signal fort dans les normales ; contraste thermique nocturne). Coût d'entrée
faible (capteur polarimétrique Sony IMX250MZR ≈ 3 k€ ; un appareil + 4 flashs < 2 k€) :
une campagne Cerema y produirait **le premier dataset public du genre** — contribution
possible indépendante du résultat GeoLoRA (§6).

---

## 3. Aura-t-on assez de données ?

**Oui pour le régime visé — adapter + LoRA (~750 k paramètres) sur backbone gelé — non
pour un entraînement lourd, que personne ne propose.**

Le régime 358–2 000 paires est *le régime standard publié* de ce domaine, y compris pour
des entraînements complets :

| Précédent | Effectif d'entraînement | Résultat |
|:---|---:|:---|
| IRFusionFormer (BMVC W 2024) — double branche complète | **358** paires IRT | Dice 90,0 / IoU 81,8 ; fusion > meilleur monomodal de +4-5 pts IoU |
| CrackSegDiff (2024) — modèle de diffusion complet | **2 000** images FIND | IoU dense 91,9 % |
| SPMFNet / MSCrackMamba (2024-25) | ~**731** paires Crack900 | état de l'art RGB-T maçonnerie |
| SAMed (2023) — LoRA q/v sur SAM, 0,1 % des paramètres | ~**2 200** coupes | état de l'art Synapse — le régime paramétrique exact de GeoLoRA |
| Guo et al. (SHM 2025) — SAM+LoRA fissures | **118 à 3 400** images | dépasse les modèles spécialisés |

Trois arguments quantitatifs supplémentaires :

- **La supervision est dense** : FIND train ≈ 2 000 × 65 536 ≈ 131 M de pixels supervisés,
  soit ~175 pixels par paramètre entraînable. Le facteur limitant est la diversité des
  scènes, pas le volume de signal.
- **On part d'un modèle convergé** : la LoRA CrackSAM archivée sert d'initialisation ;
  seul l'adapter (~291 k paramètres) est à apprendre principalement.
- **Le contre-risque documenté n'est pas le manque de données mais le sur-apprentissage
  des rangs élevés** (Segment Any Crack, 2025) : le rang 4 q/v est le bon choix, et c'est
  celui qu'on a déjà.

Stratégies qui compensent réellement un petit N : validation croisée 4-5 folds à deltas
appariés (sur 448 paires agrégées, IC95 ≈ ±0,005 contre ±0,010 sur les 90 test seuls) ;
augmentation **géométrique seulement** (groupe D4, covariante avec la Hessienne) ;
pré-entraînement de l'adapter sur Syncrack étendu ; multi-jeux comme axe de
*généralisation* (entraîner FIND, tester Crack900), pas comme pooling.

Stratégies illusoires, à refuser : monter le rang ; augmentation photométrique de la
thermique (détruit la polarité) ; pooling de jeux au recalage non vérifié ; compter les
images augmentées comme effectif ; ajouter du monomodal.

**Budget** : à ~15-20 min/run sur FIND (2 000 images 256², extrapolé des 41 min/9 121),
l'échelle complète (14 runs FIND + 8 runs IRT) tient en **~55-65 € de GPU Spot**, marge
comprise <100 €.

---

## 4. Le plan

Une seule question, six bras, des critères gelés avant le premier run.

### 4.1 Phase 0 — cadrage : trois conditions qui peuvent arrêter le projet (2-3 j, ~10 €)

> [!WARNING]
> **C'est le chaînon manquant du dossier actuel : la marge supervisée n'est chiffrée
> nulle part.** L'écart training-free 41→63 % ne prédit pas l'écart *supervisé* — si une
> LoRA sur intensité seule atteint déjà 85-90 %, la marge du multimodal est de 2-5 pts,
> pas de 22, et la puissance statistique devient le problème (plancher de détection
> ≈ ±0,004 sur 500 images de test, ≈ ±0,010 sur 90).

1. **Lire Zhou et al. (AutCon 2023)** — le benchmark des auteurs de FIND : 9 DCNN
   entraînés à l'identique sur les 4 types d'image. L'écart supervisé intensity-vs-fused
   y est écrit ; c'est le prior sur l'effet attendu. Lever au passage l'incertitude
   « la GT est tracée sur quelle modalité ? » (fiche Zenodo + papier).
2. **Un run de cadrage** (~10 €) : SAM 2 + LoRA sur intensité seule, split CrackSegDiff —
   mesure la marge restante jusqu'au plafond fused dans notre protocole. Sanity check
   zéro-shot préalable sur 20 images (FIND est à double distance de SA-1B : intensité
   laser grise, 256² — la baseline Khánh Hà ne transfère probablement pas et sera
   ré-entraînée, c'est budgété).
3. **Hygiène de données** : split figé par hashes, **groupé par blocs de numéros
   consécutifs** (les patchs voisins d'une même dalle fuient sinon entre train et test) ;
   décodage JET + polarité vérifiés (le notebook IRT charge aujourd'hui l'IR
   fausses-couleurs en `IMREAD_GRAYSCALE` — piège connu qui corrompt la Hessienne) ;
   audit de recalage sur 20 paires/jeu (seuil d'exclusion pré-enregistré : désalignement
   médian > 3 px) ; protocole d'évaluation double figé (IoU dense primaire — celui des
   baselines publiées — Jaccard squelettisé+dilaté 3 px et Wasserstein en secondaires).

**Go/no-go chiffré** : si la marge mesurée < 2× le plancher de détection au N du test,
changer de jeu ou renoncer — avant toute campagne.

### 4.2 Phase 1 — oracle avant entraînement (1-2 j, ~5 €)

Inférence seule : SAM gelé sur intensité, prompté par des points dérivés de la **GT**
(plafond du canal de guidage) vs par les nœuds/backbone du **graphe Frangi** réel vs non
guidé. Si même les prompts parfaits ne rendent pas +3 pts sur la validation, le canal n'a
pas de marge : arrêt à ~5 € dépensés, note de plafond publiée. Le CSV par image du
training-free (attendu ~63 %) sert ensuite de score de difficulté pour la stratification.

### 4.3 Phase 2 — l'échelle d'ablations FIND (4-5 j, ~35-45 €)

| Bras | Seeds | Ce qu'il isole |
|:---|:--:|:---|
| **A.** `baseline` — SAM+LoRA, intensité seule | 3 | le point de référence de tous les deltas |
| **B.** `earlyfusion` — SAM reçoit directement l'image fused | 1 | « pourquoi pas juste donner la fusion à SAM ? » — l'objection évidente de la réunion |
| **C1.** `frangi_additive` — évidence Frangi sur range, adapter additif init-zéro | 1 | le mécanisme minimal (filet de sécurité) |
| **C2.** `frangi_gated` — même évidence, gating/FiLM conditionné par l'image | 3 | **le bras principal** |
| **D.** `raw_modality` — le range décodé lui-même, injecté brut par le même adapter | 3 | **le contrôle décisif** : C2−D sépare « la géométrie aide » de « la modalité aide » |
| **E.** `permuted` — évidence permutée, re-tirée à chaque époque, symétrique train/éval | 2 | les artefacts de statistiques de canaux |
| **F.** `dense_vs_graph` — C2 sans le canal backbone MST+centralité | 1 | l'apport propre du **graphe**, jamais mesuré en quatre itérations |

Entrées strictement séparées : l'encodeur de SAM ne voit que l'intensité (répliquée 3
canaux) ; le range n'existe que comme évidence pour l'adapter. LR constant partagé,
sélection de checkpoint sur 200 images de validation **non augmentée**, aucun réglage par
bras. S'ajoute la baseline d'équité : un **U-Net early-fusion** (~1-5 M params, mêmes
données) — il gagnera probablement en in-domain, et c'est assumé : la revendication du
foundation model se joue sur la frugalité (courbes à 50/100/500 shots), la robustesse au
bruit (protocole EUVIP) et le transfert (FIND→IRT), pré-enregistrés comme axes co-primaires.

### 4.4 Critère de succès pré-enregistré

« La géométrie aide » exige les **trois** conditions :
1. C2−A > 0, IC95 bootstrap excluant 0 (IoU dense, confirmé en tendance sur le protocole squelette) ;
2. **C2−D > 0, IC95 excluant 0** ;
3. contrôle permuté indiscernable de la baseline (|E−A| < 0,5 pt) — tout « gain » de E ≥ 50 % du gain de C2 gèle les runs.

Issue partielle pré-acceptée : si C2 ≈ D > A, le verdict est « **la modalité aide,
l'abstraction géométrique n'ajoute rien** » — publié tel quel. Prédiction stratifiée
pré-enregistrée : le gain C2−A et surtout C2−D doit se concentrer sur le tiers du test où
la fissure est invisible en intensité (score training-free I-seul le plus bas) ; un gain
uniforme est un indice d'artefact.

### 4.5 Phases 3-5 — analyse, réplication, verdict (4-6 j, ~10-15 €)

Deltas appariés par image + bootstrap 10 000 tirages + variance inter-seeds à part ;
réplication IRT-Crack (8 runs, critère = même **ordre** des bras, IC larges assumés sur
90 images, stratification matin/midi/crépuscule — le contraste thermique de jour est
partiellement corrélé au RGB) ; vignette qualitative VT-GraF ; rédaction dans les deux
issues. Garde-fous : ≤ 25 runs, ≤ 100 €, ≤ 3 semaines mi-temps ; tout bras non prévu
exige un amendement commité du pré-enregistrement.

---

## 5. Objections anticipées et parades

| Objection (gravité) | Parade |
|:---|:---|
| **La marge supervisée n'est pas chiffrée** (bloquante) | Phase 0 entière : lecture Zhou 2023 + run de cadrage + calcul de puissance, avant tout engagement |
| **La GT est tracée sur une modalité inconnue** (sérieuse) — si c'est le range/la fusion, tout gain multimodal est en partie un artefact d'annotation | lecture des sources (1 h) ; les deltas appariés restent valides (même GT partout) ; la stratification par visibilité-en-I dit ce qu'on mesure ; le contrôle D sépare « biais capté par la modalité » de « géométrie » |
| **Le contrôle modalité-brute peut gagner** (sérieuse — issue la plus probable au vu de MM-SAM/IRFusionFormer) | pré-enregistré comme résultat publiable : la comparaison « modalité brute vs évidence géométrique, à adapter identique » n'existe nulle part ; chercher l'avantage de Frangi là où il est plausible — très peu de shots, bruit, transfert |
| **Recalage inter-modalités** (sérieuse) — 2-3 px de désalignement transforment l'évidence en distracteur, et punissent la géométrie fine plus que la carte brute | garanti par construction sur FIND (argument pour en faire le jeu principal) ; audit ECC sur 20 paires ailleurs ; métrique tolérante co-primaire |
| **Fuite spatiale FIND** (sérieuse) — 2 500 patchs issus d'un petit nombre de dalles | split groupé par blocs contigus (primaire) + split CrackSegDiff (comparabilité, réserve écrite) |
| **« Un U-Net early-fusion écrase tout »** (sérieuse) | il est dans le tableau d'office, et la revendication migre vers frugalité/robustesse/transfert *avant* les résultats |
| **SAM à double distance de FIND** (gérable — intensité laser 256² vs SA-1B) | sanity check zéro-shot en Phase 0 ; baseline ré-entraînée (~10 €) ; la baseline Khánh Hà n'est *pas* présentée comme réutilisable |
| **Chaîne d'évidence en dette** (sérieuse — aucun chiffre committé sur IRT, décodage JET incohérent) | étape 0 : corriger, unifier avec `implementation_notes` §4, lancer le training-free batch sur les 448 paires, committer le CSV daté |
| **« CrackSAM » est un nom déjà pris** (gérable — Ge et al. 2023, même créneau) | renommer la méthode avant toute soumission (vérifier l'homonymie du nouveau nom) |

---

## 6. La question capteur, pour le Cerema

La dépendance au capteur coupe le projet en deux pistes d'économies très différentes —
c'est le point à trancher en réunion :

| Piste | Capteur | Coût | Statut scientifique |
|:---|:---|:---|:---|
| **Range laser** (la modalité FIND) | profilomètre type LCMS sur véhicule d'auscultation | ~100-300 k€ — mais c'est le cœur de métier historique IFSTTAR/Cerema (Aigle-RN, LRIS) | la plus forte : non-redondance garantie par la physique. **Question concrète pour Endsum : accès à un véhicule équipé ?** |
| **Thermique** | FLIR portatif classe E85, montable drone | ~8-15 k€ | déployable à court terme ; information non redondante **maximale au crépuscule/nuit** (le contraste de jour est corrélé au visible) — les strates horaires d'IRT-Crack permettent de le quantifier *avant* tout achat |
| **Stéréo/photogrammétrie** | ZED ~0,5 k€ ; drone déjà pratiqué (Palais des Papes) | quasi nul | profondeur *passive* : même réserve de redondance que CrackDepth — ne promettre qu'après un contrôle modalité-brute positif |
| **Niches vacantes** (§2.4) | polarimétrie ~3 k€ ; photométrie stéréo <2 k€ ; UAV bi-caméra | faible | aucun dataset public n'existe : une campagne d'acquisition = première contribution du genre, indépendante du résultat |

Recommandation : le doublé **« preuve causale sur FIND (range actif) + voie déployable
thermique (IRT-Crack, puis Crack900 sous réserve de recalage) »**, stéréo et polarimétrie
en extensions conditionnelles.

---

## 7. Références

**Jeux de données.**
Zhou, Canchila, Song, *FIND — Fused Image dataset for convolutional neural Network-based crack Detection*, Zenodo, 2022, [DOI 10.5281/zenodo.6383044](https://zenodo.org/records/6383044) ·
Liu, Liu, Wang, *Asphalt pavement crack detection based on CNN and infrared thermography*, IEEE T-ITS 23(11), 2022, [Zenodo 11624965](https://zenodo.org/records/11624965), [GitHub](https://github.com/lfangyu09/IR-Crack-detection) ·
Zhang, Huang, Lu, *Crack900*, [Mendeley kz84t85z66](https://data.mendeley.com/datasets/kz84t85z66/1), AutCon 2023 ·
Liu et al., *CrackDepth / CrackPolar / IRTCrack*, archive [LIDAR-Mamba](https://github.com/Karl1109/LIDAR-Mamba), ACM MM 2025 ·
Rill-García, Dokladalova, Dokládal, *Syncrack*, VISAPP 2022, [GitHub](https://github.com/Sutadasuto/syncrack_generator) ·
Zhang et al., *DeLiVER*, CVPR 2023, [GitHub](https://github.com/jamycheung/DELIVER).

**Méthodes.**
Zhou, Canchila, Song, *Deep learning-based crack segmentation for civil infrastructure…*, Automation in Construction, 2023 (le benchmark 9-DCNN par modalité — **à lire en Phase 0**) ·
Jiang et al., *CrackSegDiff*, [arXiv 2410.08100](https://arxiv.org/abs/2410.08100), 2024 ·
Xiao, Chen, *IRFusionFormer*, BMVC 2024 Workshop, [arXiv 2409.20474](https://arxiv.org/abs/2409.20474), [code](https://github.com/sheauhuu/IRFusionFormer) ·
Yuan et al., *SPMFNet*, J. Imaging 11(11), 2025 ·
Zhu, Fang, Fan, *MSCrackMamba*, [arXiv 2412.06211](https://arxiv.org/abs/2412.06211), 2024 ·
Xiao et al., *MM-SAM — Segment Anything with Multiple Modalities*, [arXiv 2408.09085](https://arxiv.org/abs/2408.09085), 2024 (le précédent architectural direct — injecte la modalité **brute**) ·
Zhang, Liu, *SAMed*, [arXiv 2304.13785](https://arxiv.org/abs/2304.13785), 2023 ·
Guo et al., *SAM-based crack segmentation using LoRA fine-tuning*, Structural Health Monitoring, 2025 ·
*Segment Any Crack*, [arXiv 2504.14138](https://arxiv.org/abs/2504.14138), 2025 (les rangs élevés sur-apprennent les petits jeux) ·
Ge et al., *CrackSAM*, [arXiv 2312.04233](https://arxiv.org/abs/2312.04233), 2023 (l'homonyme antérieur) ·
*CMF-Former, depth-aware RGB-D concrete crack segmentation*, Measurement, 2025.

**Interne.**
[RAPPORT GeoLoRA](../CrackSAM-GeoLoRA/RAPPORT.md) (le contrôle permuté monomodal) ·
[Matrice causale du 20/07](../CrackSAM/results/causal_prompt_matrix_2026-07-20/) ·
[Étude anti-ombre du 08/08](../CrackSAM/results/2026-08-08_guidage_geometrique_anti_ombre/RAPPORT.md) ·
[`implementation_notes.md`](../implementation_notes.md) (conventions JET/polarité) ·
[Papier EUVIP](../../EUVIP/EUVIP_2026_Generalized_Frangi_Multimodality_camera-ready.pdf) (les 41/54/63 % et le protocole squelettisé).

---

*Document établi le 11 août 2026 en préparation de la réunion Inria–Cerema du jour.
Recherche : recension systématique (web + dépôt) par 9 agents — familles range/3D,
thermique/IR, modalités alternatives, littérature des méthodes — puis analyse de
faisabilité, conception du plan et contre-expertise contradictoire. Rien de ce document
n'est un résultat exécuté.*
