# Feuille de route complète — FrangiGraph-Residual

## Objectif livré

Produire une comparaison reproductible entre :

1. la baseline historique SAM 2 + LoRA ;
2. une baseline CrackSAM 2 plus fidèle et sa variante SAM 2.1 ;
3. FrangiGraph-Residual, utilisant la similarité comme évidence auxiliaire ;
4. son extension structurée nœuds/arêtes/composantes.

La contribution n'est acceptée que si elle améliore la moyenne et réduit le
risque de pertes catastrophiques, sans dépendre d'un test déjà consulté.

## Contraintes de départ

- conserver les résultats 2026-07 et leurs manifestes sans les régénérer ;
- ne pas reconstruire un cache v2 depuis les pseudo-logits v1 ;
- ne pas changer simultanément backbone, interface Frangi et loss ;
- ne pas appeler « graphe complet » une variante utilisant seulement
  `node_sim_max` ;
- encoder l'image une seule fois par variante lorsque cela est possible ;
- utiliser la VM G4 uniquement pour les étapes réellement GPU.

## Phase 0 — Geler le protocole

### Travail

- publier les SHA-256 des listes de splits ;
- grouper les crops issus d'une même image physique ;
- définir quatre familles : Khanhha, Road420, Facade390, Concrete3k ;
- réserver un split interne pour les gates ;
- choisir un holdout final dédupliqué ;
- pré-enregistrer métriques, marges et seuils.

### Livrables

- `protocol/cracksam_paper/manifest.json` ;
- `protocol/next_experiment.yaml` ;
- table `sample_id → source_group → split` ;
- fiche de décision signée par le commit.

### Gate de sortie

Aucune image physique ne traverse deux splits et le test final n'a pas servi à
choisir l'architecture.

## Phase 1 — Ablation causale des checkpoints existants

### Matrice obligatoire

| ID | Poids | Prompt | But |
|---|---|---|---|
| C0 | baseline époque 20 | `None` | référence |
| C1 | baseline époque 20 | Frangi historique | effet direct sur poids baseline |
| C2 | Frangi époque 20 | `None` | effet des poids appris sans prompt |
| C3 | Frangi époque 20 | Frangi historique | expérience historique appariée |
| C4 | baseline époque 20 | tenseur nul | `no_mask` versus masque nul |
| C5 | baseline époque 20 | Frangi permuté | dépendance au bon contenu |
| C6 | baseline époque 20 | Frangi décalé | dépendance à l'alignement |

Répéter C0–C3 avec les deux checkpoints best. Sauvegarder les logits avant
seuillage et calculer une courbe seuil-IoU sur validation seulement.

### Implémentation proposée

- ajouter à `evaluate_sam2.py` un mode qui commute le prompt sans optimiseur ;
- créer `workflows/run_prompt_causal_matrix.sh` ;
- écrire les sorties sous `artifacts/causal_prompt_matrix/` ;
- tester qu'aucun paramètre ne change et qu'aucun optimiseur n'est construit.

### Décision

Si C1 n'apporte aucun signal conditionnel et si l'alignement Frangi n'est pas
corrélé au delta, arrêter définitivement la voie `mask_input`. Dans tous les
cas, ne pas réutiliser le pseudo-logit comme interface finale.

## Phase 2 — Baselines contrôlées

### B0 : référence historique

Restaurer le checkpoint depuis le manifeste 2026-07 et vérifier son SHA. Ne pas
le réentraîner pour modifier a posteriori son résultat.

### B1 : port CrackSAM 2 plus fidèle

- Hiera gelé hors LoRA q/v ;
- prompt encoder entraînable ;
- mask decoder entraînable ;
- mêmes données, seed et budget que les variantes comparées ;
- nombre et noms des paramètres entraînables publiés.

### B2 : SAM 2.1

Rejouer B1 avec `sam2.1_hiera_large.pt` et
`configs/sam2.1/sam2.1_hiera_l.yaml`, sans modifier le reste du protocole.

### Ablations PEFT secondaires

- decoder-only ;
- LoRA Hiera + decoder ;
- normalisations + decoder ;
- LoRA Hiera + prompt encoder + decoder.

### Gate de sortie

Choisir la baseline de développement uniquement sur validation. Conserver B0
dans toutes les tables, même si B1 ou B2 devient la nouvelle référence.

## Phase 3 — API Frangi structurée et cache v2

### API cible

Ajouter une API explicite sans changer silencieusement la fonction historique :

```python
graph = extract_frangi_graph(..., return_graph=True)
```

Objet logique :

```text
FrangiGraphSample
├── raster: float32[C,H,W]
├── nodes: float32[N,F_node]
├── node_xy: int32[N,2]
├── edges: int32[E,2]
├── edge_features: float32[E,F_edge]
├── component_id: int32[N]
└── metadata: paramètres, dimensions, SHA et versions
```

### Features raster minimales

1. similarité `node_sim_max` ;
2. support ;
3. magnitude Hessienne absolue ;
4. échelle gagnante ;
5. `sin(2θ)` ;
6. `cos(2θ)` ;
7. distance au squelette.

### Features structurées

- valeurs propres brutes et normalisées ;
- rapport d'élongation, réponse et magnitude absolue ;
- direction, échelle, degré, endpoint et jonction ;
- longueur et composantes de similarité de chaque arête ;
- vecteur spatial et compatibilité avec la tangente ;
- courbure, MST, centralité et composante ;
- profils vallée/marche à plusieurs rayons ;
- stabilité de position et d'orientation entre échelles.

### Format de cache

- un `.npz` sans pickle par échantillon ;
- un manifeste JSON global publié atomiquement une fois le split complet ;
- `schema_version=2` ;
- SHA-256 image, masque et liste ;
- tous les paramètres Frangi et le SHA du code d'extraction ;
- statistiques de taille, densité, NaN/Inf et durée.

### Tests

- déterminisme CPU/GPU à tolérance définie ;
- sortie historique inchangée avec `return_graph=False` ;
- indices d'arêtes et composantes valides ;
- aucune valeur non finie ;
- lecture refusée sur divergence de SHA ou paramètres ;
- reprise idempotente après interruption Spot.

### Gate de sortie

Le cache d'un mini-split est relu intégralement et ses cartes sont visualisables
sans charger SAM.

## Phase 4 — MVP résiduel raster

### Modules proposés

```text
cracksam2/
├── graph_types.py
├── graph_cache.py
├── graph_raster_adapter.py
├── residual_fusion.py
└── gating.py
```

Nouveaux points d'entrée :

```text
precompute_frangi_graph_cache.py
train_frangi_graph_residual.py
evaluate_frangi_graph_residual.py
```

### Architecture

- baseline gelée et en mode `eval` ;
- une seule passe Hiera ;
- projection convolutionnelle des cartes vers les features haute résolution ;
- tête `Δz` dont la dernière couche est initialisée à zéro ;
- candidat `z1 = z0 + Δz` ;
- aucune gate apprise pendant ce premier entraînement.

### Loss initiale

\[
L = L_{seg,image}(z_1,y)
  + \lambda_{topo} L_{clDice}(z_1,y)
  + \lambda_{safe}\max(0,L_i(z_1,y)-L_i(z_0,y)+m).
\]

- BCE/Dice calculées par image avant moyenne ;
- clDice comme auxiliaire et métrique ;
- Skeleton Recall seulement en ablation ;
- dégradation pénalisée image par image.

### Contrôles

- Frangi nul, permuté et décalé ;
- dropout complet de la branche ;
- similarité seule versus canaux absolus ;
- features finales seules versus pyramide haute résolution.

### Gate de sortie

Poursuivre seulement si le candidat présente sur validation :

- un oracle baseline/candidat d'au moins `+0,01` macro ;
- une amélioration moyenne non négative sans gate ;
- une relation prédictible entre qualité Frangi et gain ;
- aucune violation du test de neutralité à l'initialisation.

## Phase 5 — Ombres et fiabilité géométrique

### Génération appariée

- ombres multiplicatives polygonales ;
- pénombres avec plusieurs largeurs ;
- intensité et dominante colorée variables ;
- cas traversant une fissure et cas éloigné ;
- recalcul Frangi après transformation RGB ;
- même seed et même masque GT avant/après.

### Supervision

- nouvelles arêtes sur la frontière synthétique comme hard negatives ;
- conservation du squelette GT à travers l'ombre ;
- cohérence des logits hors zone affectée ;
- densité de signal dans une bande autour de la frontière ;
- rappel du squelette dans cette bande.

### Ablations

- vallée/marche ;
- dérive multi-échelle ;
- compatibilité tangent/arête ;
- magnitude absolue ;
- chroma comme ablation séparée seulement.

### Gate de sortie

Une feature anti-ombre doit réduire les faux positifs de frontière sans réduire
le rappel des fissures traversant une ombre au-delà de la marge pré-enregistrée.

## Phase 6 — Vérificateur nœuds/arêtes/composantes

### Modèle

Un petit Graph Transformer ou GAT reçoit le graphe réellement généré par Frangi
et les features SAM échantillonnées. Il prédit :

- `p_node` : nœud correspondant à une fissure ;
- `p_edge` : connexion valide ;
- `p_component` : composante plausible ;
- éventuellement largeur et confiance locale.

Le modèle voit les candidats Frangi bruités de l'inférence. Il ne doit pas être
entraîné uniquement sur des sommets ou arêtes dérivés du GT.

Ajouter des arêtes factices, composantes supprimées, graphes permutés,
décalages, perturbations d'orientation et frontières d'ombre synthétiques.

### Comparaisons

| ID | Entrée |
|---|---|
| G0 | similarité raster seule |
| G1 | raster multicanal absolu |
| G2 | nœuds/arêtes sans features SAM |
| G3 | nœuds/arêtes + features SAM |
| G4 | G3 + classification composante |
| G5 | G4 + features anti-ombre |

### Gate de sortie

Le graphe doit améliorer G1 sur validation et sur au moins deux familles
indépendantes. Sinon, conserver le modèle raster plus simple.

## Phase 7 — Gate d'abstention

### Ordre

1. gate globale ;
2. calibration out-of-fold ;
3. gate par composante ;
4. gate spatiale seulement si nécessaire.

Entrées possibles : statistiques du graphe, entropie baseline, stabilité sous
augmentation, confiance du vérificateur et désaccord `z0/z1`.

À l'inférence :

```text
si q < seuil_abstention : sortie = z0 exactement
sinon                    : sortie = z0 + g * Δz
```

### Métriques

- courbe risque-couverture ;
- Brier score et ECE ;
- taux d'abstention par dataset ;
- gains conditionnels et faux accords ;
- taux de pertes `< −0,05` et `< −0,10`.

### Gate de sortie

La gate ne doit être ni toujours ouverte ni toujours fermée. Elle doit améliorer
le P05 des deltas sans masquer une dégradation systématique d'un dataset.

## Phase 8 — Entraînement final

### Curriculum

1. baseline choisie gelée ;
2. adaptateur raster sans gate ;
3. features de fiabilité ;
4. vérificateur de graphe si G3/G4 passent ;
5. gate sur prédictions out-of-fold ;
6. fine-tuning conjoint court à faible LR seulement si nécessaire.

### Graines

- une seed de développement pour les ablations ;
- trois seeds complètes pour les finalistes ;
- seeds du data loader, des augmentations et des ombres consignées séparément.

### Contrat de checkpoint

Chaque checkpoint lie :

- checkpoint SAM et SHA ;
- checkpoint baseline et SHA ;
- cache v2 et manifeste ;
- paramètres entraînables ;
- loss, optimiseur et scheduler ;
- seed, commit et état du worktree ;
- version du schéma de résultats.

## Phase 9 — Évaluation confirmatoire

### Critères pré-enregistrés

- gain macro quatre familles `≥ +0,005` ;
- borne basse de l'IC95 clusterisé `> 0` ;
- aucune famille sous `−0,005` ;
- réduction d'au moins 50 % des pertes `< −0,05` par rapport au candidat sans
  sécurité ;
- P05 amélioré ;
- clDice/clIoU non dégradé ;
- gate calibrée et non dégénérée ;
- mémoire et temps documentés.

### Ordre d'ouverture

1. validation de développement ;
2. tests historiques, marqués exploratoires ;
3. gel du code, des poids, du seuil et du manifeste ;
4. holdout final dédupliqué ;
5. rapport généré une seule fois.

## Pilote parallèle SAM 3

SAM 3 n'est pas la piste principale. Un pilote court peut néanmoins comparer :

- masque natif SAM 3 avec texte « surface crack » ;
- réponse sémantique interne de type SERD ;
- SERD avec Sobel/top-hat ;
- SERD avec similarité Frangi ;
- Frangi comme score de composante plutôt que masque final.

Le pilote ne justifie une migration que si Frangi améliore la réponse sémantique
sur validation sous un protocole fixé. SAM 3.1 n'est pas prioritaire pour cette
tâche statique.

## Ordonnancement sur G4

| Session | Travail GPU | Durée cible | Sortie durable |
|---|---|---:|---|
| G4-0 | smoke test causal C0–C6 | 1 h | logits + rapport |
| G4-1 | baseline fidèle et profil mémoire | 2 h | contrat validé |
| G4-2 | cache v2 mini-split | 1 h | manifeste test |
| G4-3 | cache v2 complet | 4–8 h reprenables | cache complet |
| G4-4 | MVP raster seed dev | 4–8 h reprenables | checkpoints |
| G4-5 | ablations raster/ombres | sessions bornées | table validation |
| G4-6 | vérificateur graphe | sessions bornées | G0–G5 |
| G4-7 | trois seeds finalistes | une session par seed | modèles finaux |

Chaque session commence par le preflight, écrit hors du dépôt les données et
checkpoints lourds, synchronise les résultats avant arrêt, puis exige l'état GCE
`TERMINATED`. Voir [l'exécution GCP](05_GCP_EXECUTION.md).

## Définition de « terminé »

Le projet est terminé lorsque :

- la baseline et chaque variante sont restaurables par SHA ;
- la neutralité et le fallback exact ont des tests automatiques ;
- la valeur marginale du graphe est isolée de celle des cartes raster ;
- les ombres sont évaluées sur un protocole causal ;
- le résultat final respecte les critères statistiques pré-enregistrés ;
- le coût G4, la latence et la mémoire sont rapportés ;
- une expérience négative reste publiable avec ses causes identifiées.
