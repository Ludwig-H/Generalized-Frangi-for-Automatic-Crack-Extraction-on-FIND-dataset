# Workflows reproductibles

Ces scripts orchestrent l'expérience historique SAM 2 et le pilote
FrangiGraph-Residual. Ils résolvent leurs chemins depuis la racine
`ISPRS/CrackSAM`, indépendamment du répertoire courant.

Le point d'entrée principal est :

```bash
bash ISPRS/CrackSAM/workflows/run_full_cracksam2_experiment.sh
```

## Pilote FrangiGraph + porte logistique

`run_frangigraph_logistic_gate_pilot.sh` enchaîne le cache à sept cartes, cinq
correcteurs résiduels hors-fold, les prédictions OOF, l'ajustement de la
régression logistique sur les folds 0–3 et le choix du seuil sur le seul fold 4.
Le mode `FULL` refuse un worktree sale ; le mode `SMOKE` est explicitement non
scientifique. Les deux modes sont reprenables après une préemption Spot.

```bash
export CRACKSAM2_DATA_ROOT=/home/codespace/cracksam2-data
export SAM2_CHECKPOINT=/chemin/absolu/sam2_hiera_large.pt
export BASELINE_CHECKPOINT=/chemin/absolu/baseline_r4_best.pt
export FRANGIGRAPH_RUN_ROOT=/chemin/durable/frangigraph_run

bash ISPRS/CrackSAM/workflows/run_frangigraph_logistic_gate_pilot.sh --mode SMOKE
```

Pour reprendre un cache complet déjà construit et vérifié :

```bash
export FRANGIGRAPH_GRAPH_CACHE=/chemin/durable/cache_khanhha_train_original
bash ISPRS/CrackSAM/workflows/run_frangigraph_logistic_gate_pilot.sh --mode FULL
```

Le label primaire du mode `FULL` exige un gain strict `ΔIoU > 0,005`. Le mode
`SMOKE` utilise zéro uniquement pour exercer la chaîne technique. Aucun jeu
historique n'entre dans l'apprentissage des coefficients ou du seuil.

La condition par défaut est `correct` : les sept cartes Frangi correspondantes
alimentent le correcteur pendant l'entraînement et l'évaluation OOF. Pour le
contrôle causal à capacité égale, utilisez un nouveau répertoire de run et :

```bash
export FRANGIGRAPH_RASTER_CONDITION=no_evidence
bash ISPRS/CrackSAM/workflows/run_frangigraph_logistic_gate_pilot.sh --mode FULL
```

Ce contrôle conserve exactement la même architecture et remplace les cartes
par l'encodage canonique d'absence de preuve. Seules les valeurs littérales
`correct` et `no_evidence` sont acceptées. La variable absente, vide ou définie
à `correct` produit la même valeur aval et les mêmes octets dans le contrat
immuable ; la condition est enregistrée dans `workflow_contract.json`, ce qui
interdit de la changer lors d'une reprise.

## Matrice causale du prompt historique

La première phase de la nouvelle feuille de route isole l'effet du prompt sans
réentraînement :

```bash
bash ISPRS/CrackSAM/workflows/run_prompt_causal_matrix.sh
```

Le script charge chaque checkpoint une seule fois et compare explicitement
`none`, `frangi`, `zero_logit`, `permuted` et `shifted`. La condition
`zero_logit` n'est pas une absence de masque : elle teste le passage d'un
tenseur nul dans l'encodeur de masque, alors que `none` utilise le chemin SAM
officiel sans masque.
