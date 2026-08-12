# Rapport d'implémentation — CrackSAM-IRT, correction thermique signée

*12 août 2026. Ce document dit ce qui a été écrit, ce qui a été **exécuté**, et
ce qui ne l'a pas été. Il ne contient aucun résultat sur IRT-Crack, parce
qu'aucun n'existe.*

<div align="center">

| Code | Tests | Chaîne validée à blanc | Campagne sur IRT-Crack |
|:--:|:--:|:--:|:--:|
| 48 fichiers · ≈ 7 200 lignes | **116, tous verts, CPU** | **oui**, bout en bout | **non exécutée** |

</div>

> [!WARNING]
> **Rien de ce dépôt ne permet aujourd'hui de dire si la thermique aide.**
> La chaîne complète a tourné sur un jeu **synthétique** — c'est une validation
> de plomberie, pas un résultat. Trois ressources manquent, toutes listées au
> §5 : le jeu IRT-Crack, le checkpoint `tol3`, une VM G4.

---

## 1. Ce qui a été écrit

### 1.1 Le paquet

| Module | Rôle | Lignes |
|:--|:--|--:|
| `_repo.py` | seul point de contact avec `sys.path` ; commit Git, marque `+dirty` | 97 |
| `constants.py` | tout ce qui est gelé : actions, canaux, versions de schéma, bras | 173 |
| `provenance.py` | SHA-256, écriture atomique, `.npz` sans pickle, manifestes | 173 |
| `manifest.py` | découverte, appariement, six contrôles bloquants, CSV | 413 |
| `splits.py` | split déterministe, stratifié, invariant à l'ordre des fichiers | 279 |
| `thermal_decode.py` | `auto` / `grayscale` / `jet`, normalisation robuste, audit | 243 |
| `thermal_frangi.py` | double polarité, reconstruction du support sans MST | 237 |
| `cache.py` | deux caches reprenables, refus sur provenance | 289 |
| `model.py` | correcteur à trois actions, **20 835** paramètres | 331 |
| `losses.py` | segmentation + trois régularisations, sur les pertes maintenues | 123 |
| `metrics.py` | IoU tolérante, topologie, clDice, statistiques d'actions | 235 |
| `stats.py` | bootstrap apparié, variance inter-graines, plancher de détection | 165 |
| `config.py` | chargement et validation stricte des YAML | 134 |
| `data.py` | dataset sur caches, contrôle permuté, augmentation covariante | 274 |
| `training.py` | boucle reprenable, sélection sur validation, checkpoints tracés | 410 |
| `evaluation.py` | métriques par image, diagnostics, deltas d'erreurs | 238 |

Huit CLI (`scripts/00` à `07`), sept configurations de bras plus la matrice, une
orchestration VM par jalons (`workflows/run_irt_vm.sh`), dix fichiers de tests.

### 1.2 Ce qui est réutilisé plutôt que réécrit

- `cracksam2.frangi.extract_frangi_graph_gpu` — l'extracteur Frangi, appelé deux
  fois par image, `compute_centrality=False` ;
- `cracksam2.model.build_cracksam2` / `load_adapter_state_dict` — la baseline ;
- `cracksam2.losses.binary_dice_loss` ;
- `geolora.losses.tolerant_loss` — la perte dont le barreau `tol3` a battu la
  baseline en IoU stricte sur Khánh Hà.

Une seule chose est dupliquée : l'IoU tolérante de
`CrackSAM-GeoLoRA/scripts/05_tolerant_iou.py`, parce qu'un script numéroté n'est
pas importable. La duplication est convertie en équivalence **vérifiée** :
`test_metrics.py` charge le script d'origine par chemin et compare les deux
implémentations sur 24 combinaisons (6 masques × 4 tolérances), à `10⁻¹²` près.

**Aucun fichier existant du dépôt n'a été modifié.**

## 2. Ce qui a été exécuté

### 2.1 Les tests

```console
$ python -m pytest "ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/tests" -q
116 passed in 7.57s
```

Sans GPU, sans SAM 2, sans le jeu réel. Les fixtures écrivent un faux IRT-Crack
complet — thermique réellement encodée en JET, masques `.jpg` face à des images
`.png` — sur lequel le **vrai** extracteur Frangi tourne en CPU.

Les résultats qui valent d'être cités :

| Ce qui est établi | Où | Valeur mesurée |
|:--|:--|:--|
| identité bit-à-bit à l'initialisation | `test_identity_fallback` | `torch.equal` vrai, `nnz(Δz) = 0` |
| repli exact sans thermique, **après** entraînement | idem | `torch.equal` vrai |
| gradient de la tête non nul au pas 0 | idem | 96 / 96 coefficients |
| gradient de l'encodeur : nul au pas 0, vivant au pas 1 | idem | 0 puis 2 016 |
| budget paramétrique | `test_signed_abstention` | **20 835** < 100 000 |
| capacité identique A1 = A2 = A3 = A4 | `test_configs` | écart 0 |
| support reconstruit = `tau_mask` réel | `test_thermal_frangi` | `array_equal` vrai, 3 tirages |
| similarité identique avec/sans centralité | idem | `array_equal` vrai |
| polarité séparée sombre / claire | idem | rapport > 3 dans les deux sens |
| permutation sans point fixe, jamais entre splits | `test_permutation_control` | vérifié par split |
| gris naïf non monotone sur une palette JET | `test_thermal_decode` | vert médian > rouge maximal |
| IoU tolérante ≡ implémentation GeoLoRA | `test_metrics` | 24 cas, `10⁻¹²` |

### 2.2 La chaîne complète, à blanc

Sur un jeu **synthétique** de 60 paires `64×80` (fissure chaude en JET, moitié
droite effacée des logits « baseline » fabriqués), la chaîne a tourné de bout en
bout :

```console
$ python scripts/00_build_manifest.py --dataset-root … --test-size 20
60 échantillons appariés, résolutions ['64x80']
split « derived » : {'train': 32, 'validation': 8, 'test': 20}

$ python scripts/01_audit_dataset.py --manifest … --output …
60 images auditées, 0 erreur(s) · décodages retenus : ['jet']
erreur de palette moyenne : 0.0023 (alerte au-delà de 0.06)
écart au gris naïf : 0.2890
désalignement médian : 0.00 px → accepté (seuil 3.0 px)

$ python scripts/03_cache_thermal_frangi.py --device cpu
60 entrées, 0 erreur(s) · décodages appliqués : {'jet': 60}

$ python scripts/06_run_ablations.py --seeds 13 --device cpu --max-epochs 6
A0 … A6 : 7 exécutions

$ python scripts/07_report.py --bootstrap 2000
[tableaux + verdict pré-enregistré]
```

Cette exécution prouve que les huit scripts s'enchaînent, que les jalons de
reprise fonctionnent, que les caches se valident et que `07_report.py` produit
les deltas appariés, les IC95 et le verdict. **Elle ne prouve rien d'autre** :
les logits baseline étaient fabriqués, six époques ne convergent pas, et le jeu
n'a rien de la difficulté d'une chaussée réelle. Les chiffres de cette
répétition ne sont pas reportés ici pour qu'ils ne puissent pas être confondus
avec un résultat.

`scripts/02_cache_cracksam_logits.py` est le **seul** script qui n'a jamais été
exécuté : il demande SAM 2, absent de cette machine. Il est écrit contre l'API
lue dans `cracksam2/model.py` (`build_cracksam2`, `load_adapter_state_dict`, clé
`adapter` du checkpoint, sortie `output["logits"]`), mais cette API n'a pas été
appelée pour de vrai.

## 3. Écarts à la spécification

Neuf points, dont deux bloquants, détaillés avec leur vérification dans
[`ERRATA.md`](ERRATA.md). En résumé :

1. `tau_mask` est vide sans MST → support reconstruit, égalité prouvée ;
2. l'import `ISPRS.CrackSAM.…` est impossible (tirets) → `_repo.py` ;
3. le split 358/90 n'est pas distribué → split dérivé, marqué comme tel ;
4. l'identité bit-à-bit est **impossible** pour A6 → borne `1,3·10⁻³`, démontrée ;
5. la résolution de travail n'était pas fixée → `448²` pour SAM, natif partout ailleurs ;
6. l'encodeur démarre au pas 1, pas au pas 0 → mesuré, testé ;
7. terme d'activité constant sans abstention → neutralisé sur A5 et A6 ;
8. sélection de checkpoint stricte → tolérante 3 px, sur consigne de campagne ;
9. plusieurs affirmations de la spécification vérifiées **exactes**, dont le
   contrat à cinq valeurs de l'extracteur et l'initialisation `(−2, −2, +2)`.

Ajouts non demandés, tous justifiés : `configs/irt_baseline.yaml` (pour que A0
passe par la même chaîne d'évaluation que les autres, sans ré-inférence),
`splits.py`, `stats.py`, `config.py`, `07_report.py`, `workflows/run_irt_vm.sh`.

## 4. Reproduire

```bash
# tests — aucune dépendance externe
python -m pytest "ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/tests" -q

# chaîne complète sur VM G4, reprenable par jalons
export IRT_DATA_ROOT="$HOME/irt-crack"          # ZIP Zenodo 11624965 décompressé
export IRT_RUN_ROOT="$HOME/irt-run"
export SAM2_CHECKPOINT="$HOME/checkpoints/sam2_hiera_large.pt"
export CRACKSAM_LORA_CHECKPOINT="$HOME/checkpoints/tol3_best.pt"
export IRT_OFFICIAL_SPLIT="$HOME/irt-crack/00_List"   # facultatif
bash ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/workflows/run_irt_vm.sh
```

Chaque étape écrit un jalon dans `${IRT_RUN_ROOT}/state` ; une préemption Spot ne
coûte au pire qu'une étape. Le script ne démarre ni n'arrête aucune VM.

## 5. Ce qui manque pour avoir un résultat

| Ressource | État | Comment l'obtenir |
|:--|:--|:--|
| **Jeu IRT-Crack** | absent de la machine | ZIP de 618 Mo, [Zenodo 11624965](https://zenodo.org/records/11624965), CC BY 4.0. Le disque local n'a que 571 Mo libres : téléchargement **sur la VM** |
| **Split officiel 358/90** | non distribué | dossier `00_List` du Google Drive IRFusionFormer ; à défaut, split dérivé signalé comme tel |
| **Checkpoint `tol3`** | non localisé | produit par la campagne GeoLoRA des 8–9 août ; à chercher dans `~/cracksam2-artifacts` sur le disque de `cracksam-frangigraph-g4-spot-ew8c` |
| **Poids SAM 2 Hiera-L** | non localisé | même disque, variable `SAM2_CHECKPOINT` |
| **VM G4** | `TERMINATED` | `cracksam-frangigraph-g4-spot-ew8c`, `europe-west8-c`, disque 200 Go. Démarrage = action explicite, à confirmer |

**Coût estimé de la campagne.** L'entraînement ne charge pas SAM : une époque est
un petit réseau convolutif sur 286 images `480×640`, soit quelques secondes.
Sept bras × trois graines ≈ 20 entraînements, plus deux caches (448 images
chacun). L'ordre de grandeur est **une à deux heures de VM**, soit quelques
euros — très en dessous de l'enveloppe de 100 € du plan multimodal.

## 6. Ce que cette implémentation ne fera pas dire

- Elle **n'établit pas** que la thermique aide : aucun bras n'a tourné sur des
  données réelles.
- Un gain de A2 contre A0 ne serait pas un résultat multimodal : le critère
  pré-enregistré exige `A2 > A1` **et** `A2 > A3`, et la question de
  l'abstraction géométrique se joue sur `A2 − A4`.
- Le MST, les composantes et la centralité ne sont toujours pas mesurés — c'est
  délibéré, le protocole les interdit tant que la similarité dense n'a pas
  produit de signal causal.
- La robustesse aux ombres, la frugalité et le transfert restent des hypothèses.
