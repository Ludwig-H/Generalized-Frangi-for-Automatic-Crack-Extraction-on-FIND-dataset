# CrackSAM 2 — baseline, diagnostic Frangi et piste Frangi-graphe

Ce dossier rassemble une expérience de segmentation automatique de fissures
fondée sur **SAM 2 Hiera Large**. Il poursuit trois objectifs distincts :

1. conserver une baseline SAM 2 + LoRA reproductible ;
2. documenter honnêtement l'expérience historique où une similarité
   Frangi-graphe était injectée comme masque dense ;
3. préparer une nouvelle méthode où Frangi fournit des candidats structuraux
   que les features de SAM vérifient avant une correction résiduelle sûre.

> [!IMPORTANT]
> La baseline locale est entraînée sur SAM 2. Elle reprend l'idée PEFT de
> CrackSAM, mais ce n'est pas une reproduction architecturale exacte de
> CrackSAM SAM 1 : le backbone, le décodeur, la loss et le périmètre des
> paramètres entraînables diffèrent.

## Résultat acquis

| Variante locale | Backbone | Guidage | IoU macro, 6 conditions |
|---|---|---|---:|
| `baseline_sam2_lora` | SAM 2 Hiera-L | aucun | **0,5675** |
| `frangi_dense_prompt_sam2_lora` | SAM 2 Hiera-L | pseudo-logits denses toujours actifs | 0,5563 |

Le delta apparié pondéré de la variante Frangi historique vaut `−0,00985`, avec
un IC95 `[-0,01198 ; -0,00779]`. Ce résultat invalide le pipeline complet
« similarité → logit → `mask_input` obligatoire + LoRA entraînées séparément ».
Il n'invalide pas la similarité Frangi comme feature auxiliaire, ni le graphe
topologique complet, qui n'a pas été utilisé dans cette expérience.

- [Comparaison et limites causales](docs/02_BASELINE_COMPARISON.md)
- [Rapport chiffré exhaustif](results/frangi_milestone_report/RAPPORT_FRANGI_MILESTONES.md)
- [Diagnostic SafeFrangi historique](results/frangi_safe_recommendation/RAPPORT_RECOMMANDATION_SAFE_FRANGI.md)

## Piste principale

La piste retenue est **FrangiGraph-Residual** :

- une voie baseline sans prompt produit `z0` ;
- la similarité et le graphe Frangi deviennent une évidence auxiliaire
  réfutable, jamais une probabilité de segmentation ;
- un petit adaptateur vérifie cette évidence avec les features multi-échelles
  de SAM et prédit une correction signée `Δz` ;
- une porte d'abstention permet de rendre exactement `z0` lorsque le candidat
  géométrique est incertain.

Le premier MVP est raster et résiduel. Le vérificateur nœuds/arêtes/composantes
n'est ajouté qu'après preuve que le signal Frangi est utile à poids fixes et que
ses gains sont prédictibles sur validation.

- [Architecture FrangiGraph-Residual](docs/03_FRANGI_GRAPH_RESIDUAL.md)
- [Feuille de route complète](docs/04_IMPLEMENTATION_ROADMAP.md)

## Parcours de lecture

1. [Question expérimentale et vocabulaire](docs/01_EXPERIMENTAL_QUESTION.md)
2. [Baseline et comparaison avec CrackSAM](docs/02_BASELINE_COMPARISON.md)
3. [Piste Frangi-graphe principale](docs/03_FRANGI_GRAPH_RESIDUAL.md)
4. [Roadmap d'implémentation et matrice d'ablations](docs/04_IMPLEMENTATION_ROADMAP.md)
5. [Exécution sur une VM G4](docs/05_GCP_EXECUTION.md)
6. [Références qui motivent l'architecture](docs/06_DESIGN_REFERENCES.md)

## Organisation

```text
CrackSAM/
├── cracksam2/       cœur Python actif : données, modèle, Frangi, losses, métriques
├── workflows/       pré-calcul, entraînement, évaluation et suivi
├── reporting/       génération de rapports et sondes diagnostiques
├── protocol/        listes de splits figées du protocole CrackSAM
├── docs/            explications, architecture et feuille de route
├── results/         résultats publiés et manifestes historiques
├── reference/       papier, code SAM 1, notebook et présentation
└── tests/           tests du pipeline actif
```

Les cinq CLI maintenues restent à la racine afin de préserver les imports et les
contrats expérimentaux :

- `prepare_cracksam2_data.py` ;
- `precompute_frangi_prompts.py` ;
- `train_sam2.py` ;
- `evaluate_sam2.py` ;
- `compute_exact_wasserstein.py`.

## Reproduire l'expérience historique

Après installation de `requirements-sam2.txt` et préparation de la VM :

```bash
export CRACKSAM2_DATA_ROOT="$HOME/cracksam2-data"
export CRACKSAM2_ARTIFACT_ROOT="$HOME/cracksam2-artifacts"
export CRACKSAM2_PROMPT_ROOT="$HOME/cracksam2-prompts"

bash ISPRS/CrackSAM/workflows/run_full_cracksam2_experiment.sh
```

Ce workflow reproduit l'expérience 2026-07 historique, pas la future méthode
FrangiGraph-Residual. Les documents qui prescrivaient directement le
pseudo-masque sont conservés sous `docs/archive/` uniquement pour provenance.

## Provenance et stabilité

- Les chemins absolus présents dans certains JSON sont des traces de la VM
  d'origine et ne doivent pas être réécrits.
- Les checkpoints historiques lient leurs résultats au code de leur commit ;
  la réorganisation actuelle ne prétend pas permettre une reprise bit-à-bit.
- `data/`, `prompt_cache/` et `artifacts/` restent locaux et ignorés par Git.
