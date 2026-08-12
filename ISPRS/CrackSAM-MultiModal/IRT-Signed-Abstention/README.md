# CrackSAM-IRT: correction thermique signée avec abstention

<div align="center">

| Statut | Modèle principal | Second modèle | Jeu multimodal | Portée |
|:--:|:--:|:--:|:--:|:--:|
| **Spécification implémentée · campagne GPU non exécutée** | CrackSAM 2 + LoRA `tol3`, entraîné sur Khánh Hà RGB | petit correcteur résiduel thermique (20 835 paramètres) | IRT-Crack, 448 paires | MVP causal avant Frangi-graphe |

</div>

> [!IMPORTANT]
> **État au 12 août 2026.** Le code de cette spécification est écrit. `121 tests`
> passent sur CPU, et la chaîne a été validée de bout en bout — **SAM 2 réel
> compris**, sur des images de substitution. **Aucun chiffre expérimental
> n'existe pour autant** : la campagne A0–A6 demande une VM G4, le jeu IRT-Crack
> et une baseline gelée (`tol3` est perdu ; `baseline_r4` est disponible).
>
> * ce que l'implémentation a dû corriger dans ce document : [`ERRATA.md`](ERRATA.md) — trois points étaient **bloquants**, dont un qui aurait rendu la campagne inconcluante sans l'empêcher de tourner ;
> * ce qui a été écrit, lancé, et ce qui ne l'a pas été : [`IMPLEMENTATION_REPORT.md`](IMPLEMENTATION_REPORT.md).
>
> Les sections ci-dessous sont la spécification d'origine, amendée en place là où
> elle était inexécutable ; chaque amendement renvoie à l'errata correspondant.

> [!IMPORTANT]
> Le but n'est **pas** d'entraîner un second SAM, de réentraîner la LoRA Khánh Hà ni de
> fournir la thermique comme prompt à SAM.
>
> Le but est de conserver le segmentateur RGB existant comme une baseline gelée et
> d'apprendre, sur IRT-Crack, un petit modèle qui choisit localement entre trois actions :
>
> 1. **renforcer** le logit de fissure ;
> 2. **supprimer** un faux positif probable ;
> 3. **s'abstenir** et restituer exactement la prédiction RGB.
>
> La première version doit utiliser uniquement une carte de similarité Frangi calculée
> sur la modalité thermique. Le graphe complet, les MST, les centralités et les GNN sont
> explicitement hors périmètre tant que ce signal minimal n'a pas été établi.

---

## Sommaire

1. [Question scientifique](#1-question-scientifique)
2. [Contraintes héritées du dépôt](#2-contraintes-héritées-du-dépôt)
3. [Architecture proposée](#3-architecture-proposée)
4. [Évidence thermique](#4-évidence-thermique)
5. [Second modèle](#5-second-modèle)
6. [Fonction de coût](#6-fonction-de-coût)
7. [Données et protocole](#7-données-et-protocole)
8. [Ablations obligatoires](#8-ablations-obligatoires)
9. [Métriques et décision](#9-métriques-et-décision)
10. [Structure de code demandée](#10-structure-de-code-demandée)
11. [Interfaces attendues](#11-interfaces-attendues)
12. [Commandes attendues](#12-commandes-attendues)
13. [Tests obligatoires](#13-tests-obligatoires)
14. [Critères d'acceptation](#14-critères-dacceptation)
15. [Ordre d'implémentation](#15-ordre-dimplémentation)
16. [Périmètre interdit](#16-périmètre-interdit)
17. [Consigne prête à transmettre à Claude Code](#17-consigne-prête-à-transmettre-à-claude-code)
18. [Tableau de résultats à compléter](#18-tableau-de-résultats-à-compléter)
19. [Références internes](#19-références-internes)

---

## 1. Question scientifique

On dispose de deux sources d'information :

- une image visible $I_R$, traitée par un CrackSAM déjà entraîné sur Khánh Hà ;
- une image thermique co-recalée $I_T$, disponible uniquement sur IRT-Crack.

Le modèle RGB produit un logit de fissure :

$$
z_0 = M_{\mathrm{RGB}}(I_R),
\qquad
p_0 = \sigma(z_0).
$$

La thermique produit une évidence géométrique :

$$
E_T = \Phi_{\mathrm{Frangi}}(I_T).
$$

La question falsifiable est :

> **Une évidence Frangi thermique alignée permet-elle de corriger les erreurs d'un
> CrackSAM RGB gelé au-delà d'une simple recalibration au domaine IRT-Crack ?**

La sortie finale doit prendre la forme :

$$
z_{\mathrm{final}}
=
z_0
+
\Delta z_\phi(z_0,E_T),
$$

avec les propriétés suivantes :

$$
E_T=\varnothing
\quad\Longrightarrow\quad
\Delta z_\phi=0,
$$

et, à l'initialisation,

$$
\Delta z_\phi \equiv 0.
$$

Le système non entraîné et le système privé de thermique doivent donc être **strictement
identiques** à la baseline CrackSAM.

---

## 2. Contraintes héritées du dépôt

### 2.1 Ce qui existe déjà et doit être réutilisé

Le dépôt contient déjà :

- `ISPRS/CrackSAM/cracksam2/model.py`
  - construction de SAM 2 ;
  - LoRA q/v ;
  - `CrackSAM2.forward(...)` ;
  - sortie explicite `output["logits"]`.
- `ISPRS/CrackSAM/cracksam2/residual.py`
  - baseline gelée ;
  - correction résiduelle au niveau des logits ;
  - projection initialisée à zéro ;
  - repli exact via `select_residual_logits(...)`.
- `ISPRS/CrackSAM/cracksam2/frangi.py`
  - extraction GPU du Frangi généralisé ;
  - carte de similarité dans $[0,1]$ ;
  - diagnostics `tau_mask` et `comp_mask` ;
  - cache sans pickle.
- `ISPRS/implementation_notes.md`
  - décodage correct des thermiques encodées en JET ;
  - avertissement contre la conversion naïve en niveaux de gris ;
  - conventions de polarité.

> [!CAUTION]
> Ne pas dupliquer ces implémentations dans un nouveau mini-projet. Le nouveau code doit
> importer les composants maintenus, ou ajouter une extension localisée lorsque
> l'interface actuelle ne suffit pas.

### 2.2 Résultat négatif à ne pas reproduire

GeoLoRA a montré que des cartes Frangi dérivées du même RGB que SAM ne produisaient aucun
effet causal mesurable : l'évidence alignée et l'évidence permutée donnaient pratiquement
les mêmes résultats.

IRT-Crack est intéressant parce que la thermique est une mesure absente de l'entrée RGB.
Le contrôle permuté reste néanmoins obligatoire. Un réseau peut parfaitement apprendre
une recalibration de domaine et prétendre ensuite avoir découvert la physique. Les réseaux
ont ce genre de sens de l'humour.

### 2.3 Contrainte de simplicité

La première version doit :

- geler intégralement CrackSAM ;
- pré-calculer ses logits sur IRT-Crack ;
- entraîner moins de **100 000 paramètres** ;
- ne pas utiliser les features internes de SAM ;
- ne pas construire de graphe ;
- ne pas employer de LoRA supplémentaire ;
- pouvoir être entraînée à partir de caches sans charger SAM à chaque époque.

---

## 3. Architecture proposée

```text
RGB I_R
  │
  ▼
CrackSAM Khánh Hà gelé
  │
  ├── logit z0
  ├── probabilité p0 = sigmoid(z0)
  └── entropie H(p0)
                         ┐
Thermique I_T            │
  │                      │
  ▼                      │
Décodage physique        │
  │                      │
  ▼                      │
Frangi double polarité   │
  │                      │
  └── évidence E_T ──────┤
                         ▼
             correcteur signé à abstention
                         │
                         ├── π+ : renforcer
                         ├── π- : supprimer
                         └── π0 : s'abstenir
                         │
                         ▼
              Δz = δmax (π+ - π-)
                         │
                         ▼
                  zfinal = z0 + Δz
```

Le modèle principal ne voit jamais la thermique. La thermique n'intervient qu'après
production des logits RGB.

---

## 4. Évidence thermique

### 4.1 Décodage

Le pipeline doit accepter :

```yaml
thermal_encoding: auto  # auto | grayscale | jet
```

Règles :

1. une thermique réellement monochrome est normalisée dans $[0,1]$ ;
2. une thermique encodée en JET est inversée par recherche du plus proche élément de la
   palette, conformément à `ISPRS/implementation_notes.md` ;
3. `cv2.cvtColor(..., COLOR_BGR2GRAY)` est interdit sur une image JET ;
4. le type de décodage, l'erreur moyenne à la palette et les percentiles doivent être
   écrits dans le manifeste de cache ;
5. une galerie de contrôle de 16 exemples doit être produite avant tout entraînement.

La normalisation par défaut est robuste :

$$
\widetilde I_T
=
\operatorname{clip}
\left(
\frac{I_T-Q_{0.01}(I_T)}
{Q_{0.99}(I_T)-Q_{0.01}(I_T)+\varepsilon},
0,1
\right).
$$

Les percentiles sont calculés image par image. La valeur brute décodée doit également
pouvoir être conservée dans le cache pour les contrôles.

### 4.2 Double polarité

Une fissure peut apparaître plus chaude ou plus froide selon les conditions
d'acquisition. La version principale ne doit donc pas imposer une polarité unique.

Calculer :

$$
S_{\mathrm{dark}}
=
\Phi_{\mathrm{Frangi}}(\widetilde I_T),
$$

$$
S_{\mathrm{bright}}
=
\Phi_{\mathrm{Frangi}}(1-\widetilde I_T),
$$

puis :

$$
S_{\max}
=
\max(S_{\mathrm{dark}},S_{\mathrm{bright}}).
$$

Le tenseur thermique minimal est :

```text
E_T[0] = similarity_dark
E_T[1] = similarity_bright
E_T[2] = similarity_max
E_T[3] = support_union
```

où `support_union` est l'union des supports de nœuds des deux polarités.

> [!CAUTION]
> **Amendement — [errata §1](ERRATA.md#1-tau_mask-est-un-plan-de-zéros-sans-mst--bloquant).**
> Ce support ne peut **pas** être lu dans `diagnostics["tau_mask"]` : avec
> `compute_centrality=False`, l'extracteur y renvoie un plan de zéros
> (`ISPRS/src/graph_extraction.py:287`), le seuillage des nœuds n'étant exécuté
> que sur la branche MST. Il est reconstruit par
> `thermal_residual.thermal_frangi.support_from_similarity`, dont l'égalité
> bit-à-bit avec le `tau_mask` réel est vérifiée par test.

Toutes les cartes sont `float32`, enregistrées à la résolution originale, avec valeurs
finies. `support_union` est binaire.

### 4.3 API d'extraction attendue

L'implémentation doit réutiliser l'extracteur maintenu :

```python
from thermal_residual import _repo   # insère ISPRS/CrackSAM dans sys.path
from cracksam2.frangi import extract_frangi_graph_gpu
```

> [!CAUTION]
> **Amendement — [errata §2](ERRATA.md#2-limport-de-la-spécification-est-syntaxiquement-impossible--bloquant).**
> `from ISPRS.CrackSAM.cracksam2.frangi import …` ne peut pas fonctionner :
> `CrackSAM` et `CrackSAM-MultiModal` contiennent des tirets, interdits dans un
> identifiant Python. `thermal_residual/_repo.py` est le seul module qui touche à
> `sys.path`, et il le fait comme les tests de `ISPRS/CrackSAM`.

Le MVP appelle deux fois l'extracteur avec :

```python
compute_centrality=False
return_raster_features=False
```

La carte utilisée est la deuxième valeur renvoyée, `similarity_img`. Le MST et la
centralité ne doivent pas être calculés dans ce MVP.

Signature proposée :

```python
def generate_dual_polarity_thermal_evidence(
    thermal_image: np.ndarray,
    *,
    encoding: str = "auto",
    scales: tuple[float, ...] = (1.0, 3.0, 5.0, 9.0, 15.0),
    R: int = 3,
    tau: float = 0.18,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Return decoded thermal image and four registered Frangi channels."""
```

Valeurs de retour minimales :

```python
{
    "thermal_decoded": float32[H, W],
    "similarity_dark": float32[H, W],
    "similarity_bright": float32[H, W],
    "similarity_max": float32[H, W],
    "support_union": float32[H, W],
}
```

---

## 5. Second modèle

### 5.1 Entrées

Le correcteur reçoit :

$$
X=
\operatorname{concat}
\left[
\bar z_0,
p_0,
H(p_0),
S_{\mathrm{dark}},
S_{\mathrm{bright}},
S_{\max},
S_{\mathrm{support}}
\right],
$$

avec :

$$
\bar z_0
=
\frac{\operatorname{clip}(z_0,-10,10)}{10},
$$

et :

$$
H(p_0)
=
-
p_0\log(p_0+\varepsilon)
-
(1-p_0)\log(1-p_0+\varepsilon).
$$

Le nombre d'entrées du modèle principal est donc **7 canaux**.

Les logits de la baseline sont toujours détachés :

```python
baseline_logits = baseline_logits.detach()
```

### 5.2 Architecture minimale

```text
Conv 3x3, 7 -> 32, dilation 1
GroupNorm
GELU

Conv 3x3, 32 -> 32, dilation 2
GroupNorm
GELU

Conv 3x3, 32 -> 32, dilation 4
GroupNorm
GELU

Conv 1x1, 32 -> 3
Softmax sur les 3 actions
```

Les trois canaux de sortie sont ordonnés ainsi :

```python
ACTION_REINFORCE = 0
ACTION_SUPPRESS = 1
ACTION_ABSTAIN = 2
```

et donnent :

$$
(\pi^+,\pi^-,\pi^0)
=
\operatorname{softmax}(q).
$$

### 5.3 Correction signée

La correction différentiable est :

$$
\Delta z
=
m\,
\delta_{\max}
\left(
\pi^+-\pi^-
\right),
$$

où :

- $m\in\{0,1\}$ indique la présence de la modalité thermique ;
- $\delta_{\max}>0$ borne l'amplitude de correction ;
- valeur initiale recommandée : `delta_max = 4.0`.

> [!CAUTION]
> **Amendement — [errata §3](ERRATA.md#3-delta_max--4-borne-la-fenêtre-corrigeable--bloquant).**
> Cette borne est aussi une **borne de faisabilité**, et elle contredit le §5.1
> du même document : la décision étant `z₀ + Δz > 0` avec `|Δz| ≤ δ_max`, un
> pixel n'est corrigeable que si `|z₀| < δ_max`. À `δ_max = 4`, tout faux négatif
> confiant de la baseline est hors d'atteinte, quelle que soit l'évidence
> thermique — et une campagne plate ne saurait alors pas distinguer « la
> thermique n'aide pas » de « la borne est trop petite ».
>
> `scripts/08_correction_ceiling.py` mesure ce plafond **avant** tout
> entraînement : quantiles de `|z₀|`, fraction des erreurs hors portée, et IoU
> d'un oracle borné par `δ_max`. La valeur retenue doit être inscrite dans les
> sept configurations avant le premier run.

La sortie est :

$$
z_{\mathrm{final}}=z_0+\Delta z.
$$

La probabilité d'activité est :

$$
\pi_{\mathrm{active}}=1-\pi^0.
$$

La carte d'action interprétable est :

$$
a(x)=
\operatorname*{arg\,max}
\{\pi^+(x),\pi^-(x),\pi^0(x)\}.
$$

> [!NOTE]
> La segmentation finale utilise la correction **douce**. La carte `argmax` sert aux
> diagnostics et aux visualisations, pas à introduire un seuillage non différentiable
> dans la première version.

### 5.4 Initialisation identité avec gradients non nuls

La dernière convolution doit être initialisée comme suit :

```python
nn.init.zeros_(action_head.weight)
action_head.bias.data[:] = torch.tensor([-2.0, -2.0, 2.0])
```

Ainsi :

$$
\pi^+=\pi^-,
$$

donc :

$$
\Delta z=0
$$

exactement à l'initialisation, tandis que les dérivées de
$\pi^+-\pi^-$ par rapport aux logits d'action restent non nulles.

Cette initialisation évite le mauvais couplage consistant à mettre simultanément à zéro
une porte scalaire et une projection résiduelle, ce qui peut empêcher tout signal
d'apprentissage d'atteindre la branche auxiliaire.

### 5.5 Abstention exacte en l'absence de thermique

L'argument `modality_present` doit être obligatoire dans le wrapper :

```python
modality_present: torch.BoolTensor  # shape (B,)
```

Il est diffusé en `(B,1,1,1)` et multiplie la correction.

La propriété suivante doit être vérifiée par `torch.equal` :

```python
output_without_thermal["logits"] == baseline_logits
```

Aucune approximation numérique n'est acceptable pour ce repli.

### 5.6 Portée spatiale

Le protocole causal principal utilise :

```yaml
correction_scope: global
```

Cela donne la même liberté au contrôle `raw_thermal` et à la variante Frangi. L'effet
d'une restriction spatiale ne doit pas être confondu avec l'effet de la représentation.

Une ablation de sécurité peut ajouter :

```yaml
correction_scope: evidence_union
support_dilation: 3
baseline_scope_threshold: 0.35
```

avec :

$$
\Omega
=
\operatorname{Dilate}(S_{\mathrm{support}},r)
\cup
\{p_0>\tau_{\mathrm{RGB}}\},
$$

et :

$$
\Delta z
\leftarrow
\Omega\,\Delta z.
$$

Cette variante autorise :

- l'ajout près d'une évidence thermique ;
- la suppression sur un premier plan prédit par CrackSAM ;
- l'abstention exacte ailleurs.

Elle est secondaire, car le contrôle `raw_thermal` doit recevoir une portée strictement
comparable.

### 5.7 Interface proposée

```python
class ThermalSignedAbstentionAdapter(nn.Module):
    def __init__(
        self,
        evidence_channels: int = 4,
        hidden_channels: int = 32,
        delta_max: float = 4.0,
        correction_scope: str = "global",
        support_dilation: int = 3,
        baseline_scope_threshold: float = 0.35,
    ) -> None:
        ...

    def forward(
        self,
        baseline_logits: torch.Tensor,
        thermal_evidence: torch.Tensor,
        modality_present: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            logits
            baseline_logits
            residual_logits
            action_logits
            action_probabilities
            reinforce_probability
            suppress_probability
            abstain_probability
            active_probability
            hard_action
            correction_scope
        """
```

Pseudo-code normatif :

```python
z0 = baseline_logits.detach()
p0 = torch.sigmoid(z0)
entropy = binary_entropy(p0)

x = torch.cat(
    [
        z0.clamp(-10.0, 10.0) / 10.0,
        p0,
        entropy,
        thermal_evidence,
    ],
    dim=1,
)

hidden = self.encoder(x)
action_logits = self.action_head(hidden)
action_probabilities = torch.softmax(action_logits, dim=1)

pi_plus = action_probabilities[:, 0:1]
pi_minus = action_probabilities[:, 1:2]
pi_zero = action_probabilities[:, 2:3]

present = modality_present[:, None, None, None].to(z0.dtype)
delta = present * self.delta_max * (pi_plus - pi_minus)

if self.correction_scope == "evidence_union":
    delta = delta * build_scope(...)

logits = z0 + delta
```

---

## 6. Fonction de coût

La loss principale est :

$$
\mathcal L
=
\mathcal L_{\mathrm{seg}}
+
\lambda_{\Delta}\mathcal L_{\Delta}
+
\lambda_{\mathrm{active}}\mathcal L_{\mathrm{active}}
+
\lambda_{\mathrm{conflict}}\mathcal L_{\mathrm{conflict}}.
$$

### 6.1 Segmentation

Réutiliser les pertes déjà validées dans la lignée CrackSAM :

$$
\mathcal L_{\mathrm{seg}}
=
\lambda_{\mathrm{BCE}}\mathcal L_{\mathrm{BCE}}
+
\lambda_{\mathrm{Dice}}\mathcal L_{\mathrm{Dice}}
+
\lambda_{\mathrm{tol}}\mathcal L_{\mathrm{tol3}}.
$$

Configuration initiale :

```yaml
loss:
  bce_weight: 0.5
  dice_weight: 0.5
  tolerant_weight: 0.25
  tolerant_radius: 3
```

Le test strict reste la métrique principale. La loss tolérante ne doit pas modifier la
définition du masque de vérité terrain.

### 6.2 Régularisation de la correction

$$
\mathcal L_{\Delta}
=
\frac{1}{HW}
\sum_x |\Delta z(x)|.
$$

Elle impose au second modèle de ne modifier CrackSAM que lorsque la segmentation y gagne.

### 6.3 Régularisation de l'activité

$$
\mathcal L_{\mathrm{active}}
=
\frac{1}{HW}
\sum_x (1-\pi^0(x)).
$$

Ce terme encourage l'abstention. Son poids doit rester faible pour ne pas rendre la
solution triviale.

### 6.4 Conflit renforcer/supprimer

$$
\mathcal L_{\mathrm{conflict}}
=
\frac{1}{HW}
\sum_x \pi^+(x)\pi^-(x).
$$

Ce terme évite une fausse abstention obtenue par annulation simultanée de deux grandes
probabilités opposées.

Configuration initiale :

```yaml
regularization:
  residual_l1_weight: 1.0e-3
  active_weight: 1.0e-4
  conflict_weight: 1.0e-3
```

Ces valeurs sont des valeurs de départ. Toute modification doit être décidée sur la
validation, jamais sur les 90 images de test.

---

## 7. Données et protocole

### 7.1 Jeu de données

Jeu principal :

- **IRT-Crack** ;
- 448 paires RGB-thermique, `640×480` ;
- masque pixel — distribué en `.jpg`, donc avec artefacts de compression ;
- split annoncé par le benchmark : 358 entraînement, 90 test.

Source :

- <https://zenodo.org/records/11624965> — une archive ZIP de 618 Mo, quatre
  dossiers d'images, **sans liste de split** ;
- <https://github.com/lfangyu09/IR-Crack-detection> — `LICENSE` et `README.md`
  seulement ;
- les listes `00_List/{train_val.txt, test.txt}` sont sur le Google Drive du
  benchmark [IRFusionFormer](https://github.com/sheauhuu/IRFusionFormer/blob/main/Prepare.md).

> [!CAUTION]
> **Amendement — [errata §3](ERRATA.md#3-le-split-officiel-35890-nest-pas-distribué-avec-le-jeu--sérieux).**
> Le split 358/90 n'est **pas** distribué avec le jeu. `--official-split` accepte
> le dossier `00_List` quand on l'a ; sinon un split *dérivé* de même effectif est
> construit — déterministe, stratifié, invariant à l'ordre d'énumération — et
> marqué `origin: "derived"`. Dans ce cas les chiffres publiés ne sont comparables
> qu'en ordre de grandeur, et le rapport doit le dire.

### 7.2 Manifeste obligatoire

Aucun script d'entraînement ne doit déduire silencieusement les appariements à partir
d'ordres de fichiers.

Créer un `manifest.csv` contenant au minimum :

```text
sample_id
rgb_path
thermal_path
mask_path
official_split
time_stratum
height
width
rgb_sha256
thermal_sha256
mask_sha256
```

`time_stratum` vaut `morning`, `noon`, `evening` ou `unknown` selon les métadonnées
réellement disponibles. Ne pas inventer la strate à partir de la luminosité.

Le script doit échouer si :

- deux modalités ne partagent pas le même identifiant ;
- un fichier manque ;
- deux chemins pointent vers le même fichier avec des identifiants différents ;
- un masque est vide sans que ce cas soit explicitement autorisé ;
- les dimensions ne correspondent pas après la procédure de recalage définie ;
- un échantillon apparaît dans plusieurs splits.

### 7.3 Split de travail

Les 90 images du test officiel ne doivent être ouvertes par aucun script de sélection
d'hyperparamètres.

Dans les 358 images d'entraînement :

```text
train interne : 80 %
validation     : 20 %
```

Le split interne est :

- déterministe ;
- stratifié si possible par `time_stratum` ;
- équilibré approximativement par quartiles de fraction de pixels fissure ;
- enregistré dans un fichier versionné ;
- partagé par toutes les variantes.

Une validation croisée à cinq plis peut être ajoutée après le MVP. Elle ne doit pas
retarder l'implémentation initiale.

### 7.4 Augmentations

MVP autorisé :

- flip horizontal ;
- flip vertical ;
- aucune augmentation photométrique de la thermique ;
- aucune rotation arbitraire ;
- mêmes transformations pour logits, évidence et masque.

Les augmentations sont appliquées aux caches de manière covariante.

### 7.5 Caches

> [!CAUTION]
> **Amendement — [errata §5](ERRATA.md#5-la-résolution-de-travail-nétait-pas-spécifiée--sérieux).**
> La résolution de travail n'était pas fixée, alors qu'IRT-Crack est en `640×480`
> et que CrackSAM 2 a été entraîné en `448×448`. Chaîne retenue : RGB
> redimensionné en `448²` bicubique pour l'entrée de SAM (sa distribution
> d'entraînement), logits ré-échantillonnés vers `480×640` pour le cache, puis
> évidence, masque et métriques au natif. **La vérité terrain n'est jamais
> rééchantillonnée** — seuls des logits lisses le sont.

#### Cache CrackSAM

Pour chaque image RGB :

```text
sample_id
baseline_logits : float32[1,H,W]
checkpoint_sha256
source_rgb_sha256
model_config
```

#### Cache thermique

Pour chaque thermique :

```text
sample_id
thermal_decoded  : float32[H,W]
similarity_dark  : float32[H,W]
similarity_bright: float32[H,W]
similarity_max   : float32[H,W]
support_union    : float32[H,W]
extractor_config
source_thermal_sha256
extractor_sha256
```

Utiliser `.npz` ou `.npy` avec `allow_pickle=False`.

Chaque dossier de cache doit contenir un manifeste JSON avec :

- version de schéma ;
- commit Git ;
- paramètres complets ;
- SHA-256 des sources ;
- date UTC ;
- nombre d'échantillons ;
- statistiques min/max/moyenne ;
- liste des erreurs ou exclusions.

### 7.6 Entraînement

Configuration initiale :

```yaml
training:
  optimizer: adamw
  learning_rate: 3.0e-4
  weight_decay: 1.0e-4
  batch_size: 8
  max_epochs: 100
  early_stopping_patience: 15
  gradient_clip_norm: 1.0
  amp: true
  seeds: [13, 37, 73]
```

Le checkpoint est choisi sur l'**IoU tolérante à 3 px** de la validation non
augmentée.

> [!NOTE]
> **Amendement — [errata §8](ERRATA.md#8-sélection-de-checkpoint--iou-stricte--tolérante-3 px--changé-sur-demande).**
> La spécification disait « IoU stricte ». La campagne étant cadrée sur « une
> baseline `tol3` avec une tolérance de 3 pixels », la sélection se fait sur
> `iou_buffered_tol3`, gelée à l'identique dans les sept configurations.
> L'IoU stricte reste rapportée partout, et `selection_metric` reste un champ de
> configuration tracé dans `training.json`.

Le seuil de segmentation final est fixé à `0.5`. Une éventuelle calibration de seuil doit
constituer un bras séparé et être appliquée de manière identique à toutes les variantes.

### 7.7 Mode sans thermique

Deux usages doivent être distingués :

1. **production sans thermique** :
   `modality_present=False`, donc correction strictement nulle ;
2. **contrôle RGB-only de capacité égale** :
   `modality_present=True`, mais les quatre canaux thermiques sont remplacés par zéro.
   Le correcteur peut alors apprendre une recalibration basée uniquement sur
   $z_0,p_0,H(p_0)$.

Cette distinction est indispensable pour isoler l'apport de la thermique.

---

## 8. Ablations obligatoires

| ID | Variante | Entrée du correcteur | Ce qu'elle mesure |
|:--:|:--|:--|:--|
| **A0** | `baseline` | aucune correction | CrackSAM Khánh Hà transféré tel quel |
| **A1** | `rgb_recalibration` | logits, probabilité, entropie ; thermique à zéro | adaptation au domaine IRT-Crack |
| **A2** | `frangi_signed_abstention` | Frangi thermique aligné | méthode principale |
| **A3** | `frangi_permuted` | Frangi d'une autre image | causalité spatiale et sémantique |
| **A4** | `raw_thermal` | thermique décodée, capacité identique | modalité brute contre abstraction Frangi |
| **A5** | `frangi_no_abstention` | renforcer/supprimer seulement | valeur propre de l'action d'abstention |
| **A6** | `positive_only` | renforcement uniquement | comparaison à une modulation de type SERD |

### 8.1 Contrôle permuté

À l'entraînement :

- la permutation est re-tirée à chaque époque ;
- elle ne mélange jamais train, validation et test ;
- si `time_stratum` est fiable, la permutation reste dans la même strate.

À l'évaluation :

- la permutation est fixe ;
- elle est déterminée par la graine du run ;
- le mapping `sample_id -> thermal_sample_id` est enregistré.

### 8.2 Contrôle thermique brut

Pour garder quatre canaux d'évidence :

```text
raw[0] = T
raw[1] = 1 - T
raw[2] = 2 * abs(T - 0.5)
raw[3] = 1
```

Le même encodeur, le même nombre de paramètres, la même loss et le même protocole sont
utilisés.

### 8.3 Contrôle sans abstention

Remplacer la tête trois actions par deux actions :

$$
(\pi^+,\pi^-)=\operatorname{softmax}(q^+,q^-),
$$

$$
\Delta z=\delta_{\max}(\pi^+-\pi^-).
$$

L'encodeur et le budget paramétrique doivent rester aussi proches que possible.

### 8.4 Contrôle positif uniquement

Utiliser :

$$
\Delta z
=
\delta_{\max}
\sigma(q)
S_{\max}.
$$

Cette variante ne peut que renforcer une réponse, jamais supprimer ni s'abstenir
explicitement. Elle mesure l'intérêt réel de la correction signée.

> [!CAUTION]
> **Amendement — [errata §4](ERRATA.md#4-lidentité-bit-à-bit-est-impossible-pour-a6--sérieux-et-cest-un-théorème).**
> Ce bras ne peut **pas** satisfaire l'identité bit-à-bit du §14, et aucune
> initialisation ne le permettrait : une correction non négative, nulle à
> l'initialisation, y aurait un minimum global, donc un gradient nul. A6 part de
> `bias = −8`, soit `|Δz| ≤ 1,3·10⁻³`. Le critère d'identité exacte porte sur les
> têtes signées, où il est vérifié avant **et après** entraînement.

---

## 9. Métriques et décision

### 9.1 Métriques de segmentation

Rapporter :

- IoU stricte, métrique principale ;
- Dice/F1 ;
- précision ;
- rappel ;
- IoU tolérante à 1, 2, 3, 5 et 8 pixels ;
- couverture du squelette ;
- nombre de composantes connexes ;
- clDice si l'implémentation maintenue est disponible.

### 9.2 Métriques propres au correcteur

Rapporter :

- fraction de pixels `reinforce` ;
- fraction de pixels `suppress` ;
- fraction de pixels `abstain` ;
- moyenne de $|\Delta z|$ ;
- quantiles de $|\Delta z|$ ;
- fraction d'images pour lesquelles le candidat bat la baseline ;
- delta IoU moyen et médian par image ;
- delta IoU par strate horaire ;
- delta sur les faux négatifs de la baseline ;
- delta sur les faux positifs de la baseline.

Produire au minimum les cartes qualitatives suivantes :

```text
RGB
thermique décodée
similarity_dark
similarity_bright
baseline probability
reinforce probability
suppress probability
abstain probability
residual logits
prediction finale
vérité terrain
```

### 9.3 Statistique

Les comparaisons sont appariées par image.

Utiliser :

- bootstrap apparié ;
- 10 000 réplications ;
- intervalle à 95 % ;
- variance entre graines rapportée séparément.

### 9.4 Critère de succès

La revendication « la Frangi thermique aide » exige simultanément :

$$
\operatorname{IoU}(A2)>\operatorname{IoU}(A1),
$$

avec IC95 du delta excluant zéro, et :

$$
\operatorname{IoU}(A2)>\operatorname{IoU}(A3),
$$

avec un écart cohérent sur les graines.

La revendication « l'abstraction Frangi aide au-delà de la modalité » exige :

$$
\operatorname{IoU}(A2)>\operatorname{IoU}(A4).
$$

Résultats négatifs pré-acceptés :

- $A2\simeq A1$ : la thermique Frangi n'apporte pas de signal complémentaire ;
- $A2\simeq A3>A1$ : gain non causal ou artefact de statistiques de canaux ;
- $A4>A2>A1$ : la thermique aide, mais Frangi perd de l'information ;
- $A2>A4>A1$ : résultat favorable à l'abstraction géométrique ;
- $A5>A2$ : l'abstention est trop fortement régularisée ou mal calibrée.

> [!IMPORTANT]
> Aucun passage au Frangi-graphe, au GNN ou à une LoRA multimodale ne doit être engagé
> avant que $A2>A1$ et $A2>A3$ soient établis.

---

## 10. Structure de code demandée

Sous-dossier autonome, sans modification des résultats historiques. Arborescence
**réellement écrite** — les ajouts par rapport à la demande initiale sont
annotés :

```text
ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/
├── README.md                   cette spécification, amendée
├── ERRATA.md                   +  ce que l'implémentation a dû corriger
├── IMPLEMENTATION_REPORT.md    +  fichiers, commandes, tests, non-exécuté
├── pyproject.toml
├── conftest.py                 +  amorçage sys.path pour pytest
├── configs/
│   ├── irt_baseline.yaml            + (A0, pour que A0 passe par la même chaîne)
│   ├── irt_rgb_recalibration.yaml
│   ├── irt_signed_abstention_v1.yaml
│   ├── irt_frangi_permuted.yaml
│   ├── irt_raw_thermal.yaml
│   ├── irt_no_abstention.yaml
│   ├── irt_positive_only.yaml
│   └── ablation_matrix.yaml
├── thermal_residual/
│   ├── __init__.py
│   ├── _repo.py                +  seul module qui touche à sys.path
│   ├── constants.py
│   ├── provenance.py
│   ├── manifest.py
│   ├── splits.py               +  split déterministe et stratifié
│   ├── thermal_decode.py
│   ├── thermal_frangi.py
│   ├── cache.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── stats.py                +  bootstrap apparié
│   ├── ceiling.py              +  oracle borné par delta_max
│   ├── config.py               +  chargement et validation des YAML
│   ├── data.py
│   ├── training.py
│   └── evaluation.py
├── scripts/
│   ├── 00_build_manifest.py    (manifeste + split figé)
│   ├── 01_audit_dataset.py
│   ├── 02_cache_cracksam_logits.py
│   ├── 03_cache_thermal_frangi.py
│   ├── 04_train.py
│   ├── 05_evaluate.py
│   ├── 06_run_ablations.py
│   ├── 07_report.py            +  deltas appariés et tableaux
│   └── 08_correction_ceiling.py +  porte de plafond, avant tout entraînement
├── workflows/
│   └── run_irt_vm.sh           +  chaîne VM reprenable par jalons
├── tests/                      11 fichiers, 121 tests, CPU
└── results/.gitkeep
```

### 10.1 Imports autorisés depuis le dépôt

Le nouveau code peut importer directement :

```python
from ISPRS.CrackSAM.cracksam2.model import CrackSAM2, build_cracksam2
from ISPRS.CrackSAM.cracksam2.frangi import extract_frangi_graph_gpu
```

Il peut réutiliser les losses et métriques existantes après vérification de leur API.

### 10.2 Modifications du code existant

Par défaut : aucune.

Une modification dans `ISPRS/CrackSAM/cracksam2/` n'est autorisée que si :

- elle factorise une fonction réellement commune ;
- elle conserve toutes les interfaces historiques ;
- elle est couverte par les tests existants ;
- elle ne modifie aucun résultat archivé.

---

## 11. Interfaces attendues

### 11.1 Manifeste

```python
@dataclass(frozen=True)
class IRTSample:
    sample_id: str
    rgb_path: Path
    thermal_path: Path
    mask_path: Path
    official_split: str
    time_stratum: str
    height: int
    width: int
    rgb_sha256: str
    thermal_sha256: str
    mask_sha256: str
```

### 11.2 Cache baseline

```python
def cache_baseline_logits(
    manifest_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    *,
    device: str,
    batch_size: int,
) -> Path:
    """Cache frozen CrackSAM logits and return the cache manifest path."""
```

### 11.3 Cache thermique

```python
def cache_thermal_evidence(
    manifest_path: Path,
    output_dir: Path,
    *,
    encoding: str,
    scales: tuple[float, ...],
    R: int,
    tau: float,
    device: str,
) -> Path:
    """Cache dual-polarity thermal Frangi evidence."""
```

### 11.4 Dataset de caches

```python
class IRTResidualDataset(torch.utils.data.Dataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return {
            "sample_id": ...,
            "baseline_logits": ...,   # (1,H,W)
            "thermal_evidence": ...,  # (4,H,W)
            "mask": ...,              # (1,H,W)
            "modality_present": ...,  # bool
        }
```

### 11.5 Checkpoint

Le checkpoint du correcteur doit contenir :

```text
format_version
model_state_dict
model_config
training_config
baseline_checkpoint_sha256
baseline_cache_manifest_sha256
thermal_cache_manifest_sha256
dataset_manifest_sha256
git_commit
seed
best_epoch
best_validation_metrics
```

Ne pas sérialiser le modèle SAM dans ce checkpoint.

---

## 12. Commandes attendues

Les commandes exactes peuvent être ajustées, mais l'ergonomie suivante doit exister.

### 12.1 Construire le manifeste

```bash
python scripts/00_build_manifest.py \
  --dataset-root /data/IRT-Crack \
  --official-split /data/IRT-Crack/split.json \
  --output data/manifest.csv
```

### 12.2 Auditer le décodage et le recalage

```bash
python scripts/01_audit_dataset.py \
  --manifest data/manifest.csv \
  --thermal-encoding auto \
  --output results/dataset_audit
```

Sorties minimales :

```text
results/dataset_audit/report.json
results/dataset_audit/gallery.png
results/dataset_audit/pairing_errors.csv
```

### 12.3 Cacher les logits CrackSAM

```bash
python scripts/02_cache_cracksam_logits.py \
  --manifest data/manifest.csv \
  --sam2-checkpoint /models/sam2_hiera_large.pt \
  --lora-checkpoint /models/cracksam_khanhha_best.pt \
  --output cache/baseline \
  --device cuda
```

### 12.4 Cacher les cartes Frangi thermiques

```bash
python scripts/03_cache_thermal_frangi.py \
  --manifest data/manifest.csv \
  --config configs/irt_signed_abstention_v1.yaml \
  --output cache/thermal_frangi \
  --device cuda
```

### 12.5 Entraîner

```bash
python scripts/04_train.py \
  --config configs/irt_signed_abstention_v1.yaml \
  --manifest data/manifest.csv \
  --baseline-cache cache/baseline/manifest.json \
  --thermal-cache cache/thermal_frangi/manifest.json \
  --output results/frangi_signed_abstention/seed_13 \
  --seed 13
```

### 12.6 Évaluer

```bash
python scripts/05_evaluate.py \
  --run results/frangi_signed_abstention/seed_13 \
  --split test \
  --bootstrap 10000
```

### 12.7 Lancer la matrice d'ablations

```bash
python scripts/06_run_ablations.py \
  --protocol configs/ablation_matrix.yaml \
  --seeds 13 37 73 \
  --output results/ablation_matrix
```

### 12.8 Tests

```bash
pytest -q
```

---

## 13. Tests obligatoires

### 13.1 Identité à l'initialisation

```python
model = ThermalSignedAbstentionAdapter(...)
model.eval()

out = model(
    baseline_logits=z0,
    thermal_evidence=evidence,
    modality_present=torch.ones(batch, dtype=torch.bool),
)

assert torch.equal(out["logits"], z0)
assert torch.count_nonzero(out["residual_logits"]) == 0
```

### 13.2 Gradients à l'initialisation

Après une backward sur une loss non triviale :

```python
assert model.action_head.weight.grad is not None
assert torch.count_nonzero(model.action_head.weight.grad) > 0
```

### 13.3 Repli sans thermique

```python
out = model(
    baseline_logits=z0,
    thermal_evidence=evidence,
    modality_present=torch.zeros(batch, dtype=torch.bool),
)

assert torch.equal(out["logits"], z0)
```

Ce test doit rester vrai après entraînement.

### 13.4 Sens de la correction

Construire des poids synthétiques donnant :

- $\pi^+ \approx 1$ : `residual_logits > 0` ;
- $\pi^- \approx 1$ : `residual_logits < 0` ;
- $\pi^0 \approx 1$ : `residual_logits ≈ 0`.

### 13.5 Bornage

```python
assert residual.abs().max() <= delta_max + 1e-6
```

### 13.6 Décodage JET

Tester :

- récupération monotone des 256 indices de palette ;
- erreur faible sur une image JET synthétique ;
- différence explicite avec la conversion OpenCV en gris ;
- stabilité sur une image monochrome.

### 13.7 Appariement et fuite

Tester :

- détection des fichiers manquants ;
- détection des doublons ;
- disjonction stricte train/validation/test ;
- invariance du split à l'ordre d'énumération des fichiers.

### 13.8 Contrôle permuté

Tester :

- absence de point fixe si le nombre d'échantillons le permet ;
- aucune permutation entre splits ;
- déterminisme à graine fixe ;
- permutation différente à une autre époque d'entraînement.

### 13.9 Provenance des caches

Un cache doit être refusé si :

- le hash de l'image source a changé ;
- le hash du checkpoint CrackSAM diffère ;
- la configuration Frangi diffère ;
- le nombre ou l'ordre des échantillons est incohérent.

### 13.10 Smoke test d'apprentissage

Sur quatre échantillons synthétiques ou réels :

- la loss doit diminuer ;
- le modèle doit pouvoir sur-apprendre le mini-lot ;
- la baseline ne doit recevoir aucun gradient ;
- le nombre de paramètres entraînables doit être inférieur à 100 000.

---

## 14. Critères d'acceptation

État au 12 août 2026. Coché = vérifié par un test ou par une exécution réelle.

- [x] le manifeste IRT-Crack est reproductible et validé — `test_manifest.py` ;
- [x] le décodage JET est testé et illustré — `test_thermal_decode.py`, `01_audit_dataset.py` produit la planche ;
- [x] les logits CrackSAM peuvent être cachés sans charger le second modèle — `02_cache_cracksam_logits.py` ;
- [x] les cartes Frangi double polarité sont cachées et visualisables ;
- [x] le correcteur possède exactement trois actions ;
- [x] la sortie initiale est bit-à-bit égale à la baseline — têtes signées ; A6 en est **théoriquement incapable**, [errata §4](ERRATA.md) ;
- [x] le mode sans thermique est bit-à-bit égal à la baseline après entraînement ;
- [x] les gradients du correcteur sont non nuls à l'initialisation — sur la tête ; l'encodeur démarre au pas 1, [errata §6](ERRATA.md) ;
- [x] aucun paramètre de CrackSAM n'est entraînable — il n'est même pas chargé à l'entraînement ;
- [x] le correcteur contient moins de 100 000 paramètres — **20 835** ;
- [x] les variantes A0 à A4 sont exécutables par configuration ;
- [x] les variantes A5 et A6 sont couvertes par la même interface d'évaluation ;
- [x] le contrôle permuté est enregistré et reproductible — `test_permutation_control.py` ;
- [x] la chaîne produit des deltas appariés et leurs IC95 — `07_report.py`, validé à blanc ;
- [x] `pytest -q` passe sur CPU — **116 tests**, sans SAM 2 ni GPU ni jeu réel ;
- [x] un `IMPLEMENTATION_REPORT.md` décrit les fichiers, les commandes, les tests et les écarts ;
- [ ] **le rapport contient les deltas appariés mesurés sur IRT-Crack** — bloqué : demande une VM G4, le jeu et le checkpoint `tol3`.

---

## 15. Ordre d'implémentation

### Étape 1 : données

1. construire le manifeste ;
2. valider les paires ;
3. figer le split ;
4. écrire les tests de fuite.

### Étape 2 : thermique

1. coder le décodage `grayscale|jet|auto` ;
2. produire la galerie ;
3. coder l'extraction Frangi double polarité ;
4. écrire le cache et sa provenance.

### Étape 3 : baseline

1. charger CrackSAM Khánh Hà ;
2. cacher les logits ;
3. vérifier les dimensions et hashes ;
4. interdire tout gradient.

### Étape 4 : modèle

1. coder la tête à trois actions ;
2. coder l'initialisation identité ;
3. coder le repli `modality_present=False` ;
4. écrire les tests analytiques avant l'entraînement.

### Étape 5 : entraînement

1. coder la loss ;
2. sur-apprendre quatre exemples ;
3. lancer un fold train/validation ;
4. produire les diagnostics d'actions.

### Étape 6 : causalité

1. baseline A0 ;
2. recalibration A1 ;
3. Frangi aligné A2 ;
4. Frangi permuté A3 ;
5. thermique brute A4.

### Étape 7 : mécanisme

Uniquement après la matrice A0-A4 :

1. sans abstention A5 ;
2. positif uniquement A6 ;
3. portée `evidence_union` ;
4. éventuellement cartes d'échelle et d'orientation.

### Étape 8 : suite conditionnelle

Le Frangi-graphe ne devient pertinent que si la carte de similarité alignée produit déjà
un signal causal. La progression autorisée est alors :

```text
similarité dense
    ↓
fragments indépendants
    ↓
adjacences réelles contre adjacences permutées
    ↓
message passing léger
```

Ne pas sauter directement à la dernière ligne.

---

## 16. Périmètre interdit

Pour cette première implémentation, ne pas :

- réentraîner SAM 2 ;
- modifier la LoRA Khánh Hà ;
- entraîner un second SAM sur la thermique ;
- utiliser la thermique comme `mask_input` de SAM ;
- fusionner RGB et thermique avant l'encodeur ;
- générer un pseudo-thermique depuis le RGB ;
- calculer un MST ou une centralité ;
- ajouter un GNN ;
- choisir des hyperparamètres sur le test ;
- changer le split selon les variantes ;
- utiliser une capacité différente pour Frangi et thermique brute ;
- normaliser une image JET par une conversion standard en gris ;
- présenter un gain face à A0 comme preuve multimodale sans comparaison à A1 et A3 ;
- modifier ou écraser les résultats historiques du dépôt.

---

## 17. Consigne prête à transmettre à Claude Code

> [!TIP]
> Le bloc ci-dessous peut être transmis tel quel à Claude Code avec ce README comme
> fichier de référence.

```text
Implémente la spécification de ce README dans
ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/.

Avant de coder, lis attentivement :
- ISPRS/CrackSAM/cracksam2/model.py
- ISPRS/CrackSAM/cracksam2/residual.py
- ISPRS/CrackSAM/cracksam2/frangi.py
- ISPRS/CrackSAM/cracksam2/graph_types.py
- ISPRS/implementation_notes.md
- ISPRS/CrackSAM-GeoLoRA/RAPPORT.md
- ISPRS/CrackSAM-MultiModal/README.md

Objectif :
- conserver CrackSAM Khánh Hà entièrement gelé ;
- cacher ses logits sur les RGB d'IRT-Crack ;
- calculer une similarité Frangi double polarité sur la thermique ;
- entraîner un petit correcteur résiduel produisant trois probabilités par pixel :
  renforcer, supprimer, s'abstenir ;
- utiliser Δz = delta_max * (pi_plus - pi_minus) ;
- garantir une identité exacte avec la baseline à l'initialisation et lorsque
  modality_present=False ;
- implémenter les ablations A0 à A6 et les contrôles permutés ;
- produire des tests complets et un IMPLEMENTATION_REPORT.md.

Contraintes :
- ne modifie pas les interfaces historiques sans nécessité ;
- ne réentraîne pas SAM ;
- moins de 100 000 paramètres entraînables ;
- aucune donnée de test utilisée pour choisir les hyperparamètres ;
- caches sans pickle et accompagnés de SHA-256 ;
- tests CPU pour toute la logique indépendante de SAM/Frangi GPU ;
- chaque script doit accepter --help et échouer clairement en cas de donnée incohérente.

Travaille par étapes :
1. manifeste et tests ;
2. décodage thermique et galerie ;
3. cache Frangi ;
4. cache CrackSAM ;
5. modèle et tests d'identité/gradient ;
6. entraînement smoke test ;
7. ablations ;
8. rapport.

Ne lance pas une campagne complète tant que les tests d'identité, de gradient, de
permutation et de provenance ne passent pas.

À la fin :
- exécute pytest -q ;
- donne la liste exacte des fichiers créés ou modifiés ;
- donne les commandes de reproduction ;
- signale honnêtement ce qui n'a pas pu être exécuté ;
- écris IMPLEMENTATION_REPORT.md.
```

---

## 18. Tableau de résultats à compléter

### 18.1 Segmentation

| Variante | Seed | IoU | Dice | Précision | Rappel | IoU tol. 3 | clDice | Composantes |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| A0 baseline | 13 |  |  |  |  |  |  |  |
| A1 RGB recalibration | 13 |  |  |  |  |  |  |  |
| A2 Frangi signé + abstention | 13 |  |  |  |  |  |  |  |
| A3 Frangi permuté | 13 |  |  |  |  |  |  |  |
| A4 thermique brute | 13 |  |  |  |  |  |  |  |
| A5 sans abstention | 13 |  |  |  |  |  |  |  |
| A6 positif uniquement | 13 |  |  |  |  |  |  |  |

### 18.2 Actions

| Variante | Renforcer | Supprimer | Abstention | Moy. $|\Delta z|$ | Images améliorées | Delta IoU médian |
|:--|--:|--:|--:|--:|--:|--:|
| A2 |  |  |  |  |  |  |
| A3 |  |  |  |  |  |  |
| A4 |  |  |  |  |  |  |

### 18.3 Deltas appariés

| Comparaison | Delta IoU moyen | IC95 | Delta Dice moyen | IC95 | Verdict |
|:--|--:|:--:|--:|:--:|:--|
| A2 - A1 |  |  |  |  |  |
| A2 - A3 |  |  |  |  |  |
| A2 - A4 |  |  |  |  |  |
| A2 - A5 |  |  |  |  |  |
| A2 - A6 |  |  |  |  |  |

---

## 19. Références internes

- [`ISPRS/CrackSAM/cracksam2/model.py`](../../CrackSAM/cracksam2/model.py)
- [`ISPRS/CrackSAM/cracksam2/residual.py`](../../CrackSAM/cracksam2/residual.py)
- [`ISPRS/CrackSAM/cracksam2/frangi.py`](../../CrackSAM/cracksam2/frangi.py)
- [`ISPRS/CrackSAM/cracksam2/graph_types.py`](../../CrackSAM/cracksam2/graph_types.py)
- [`ISPRS/implementation_notes.md`](../../implementation_notes.md)
- [`ISPRS/CrackSAM-GeoLoRA/RAPPORT.md`](../../CrackSAM-GeoLoRA/RAPPORT.md)
- [`ISPRS/CrackSAM-MultiModal/README.md`](../README.md)
- [IRT-Crack sur Zenodo](https://zenodo.org/records/11624965)
- [Dépôt de référence IRT-Crack](https://github.com/lfangyu09/IR-Crack-detection)

---

*Spécification rédigée le 12 août 2026. Le document définit un protocole et une
architecture à implémenter. Il ne constitue pas un résultat expérimental.*
