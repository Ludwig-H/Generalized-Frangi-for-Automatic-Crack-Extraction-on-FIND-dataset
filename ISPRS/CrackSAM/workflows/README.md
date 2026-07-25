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
correcteurs résiduels sélectifs hors-fold, les prédictions OOF, l'ajustement de
la régression logistique sur les folds 0–3 et le choix du seuil sur le seul
fold 4. Le mode `FULL` refuse un worktree sale ; le mode `SMOKE` est
explicitement non scientifique. Les deux modes sont reprenables après une
préemption Spot.

Le contrat actif est `schema_version=3`. Un ancien
`FRANGIGRAPH_RUN_ROOT` créé sous le schéma 2 n'est **pas reprenable** avec ce
workflow : conservez-le pour provenance et choisissez une nouvelle racine pour
le schéma 3. Une reprise n'est possible qu'avec un contrat schéma 3 strictement
identique.

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

L'architecture par défaut est `verified_local_v1` : elle vérifie localement les
sept cartes avec les features SAM et des profils vallée/marche calculés sur la
luminance. Elle apprend un score d'alignement avec une annotation dilatée, pas
une probabilité calibrée ni l'utilité marginale par rapport à la baseline. Le
seuil local `0,5` est un défaut opérationnel à calibrer ultérieurement sur des
groupes tenus à l'écart. Les rayons de profil et la dilation sont exprimés en
cellules de la grille de fusion Hiera ; la tolérance de cible est exprimée en
pixels du masque de sortie avant projection.

La condition d'entraînement est `correct`. Le support Frangi est un masque
structurel ; il serait donc inutile d'entraîner cette architecture sous
`no_evidence`, et le workflow le refuse.

Le test de nécessité d'entrée consiste à évaluer le **même checkpoint** avec
l'encodage absent. Le workflow principal n'exécute pas automatiquement cette
seconde passe :

```bash
python ISPRS/CrackSAM/evaluate_frangi_graph_residual.py \
  ... \
  --raster-condition no_evidence \
  --allow-input-ablation-raster-override
```

Avec le même forward, ce test doit restituer `z0` bit à bit. Il vérifie la
dépendance du résidu à l'entrée Frangi et le fallback, mais ne constitue pas un
contraste causal complet à capacité égale. Une reproduction du protocole
historique `legacy_raster_v1` sous `no_evidence` reste disponible en fixant
explicitement
`FRANGIGRAPH_ADAPTER_MODE=legacy_raster_v1`,
`FRANGIGRAPH_EVIDENCE_LOSS_WEIGHT=0` et une nouvelle racine de run. Tous les
paramètres du sélecteur sont enregistrés dans `workflow_contract.json`, ce qui
interdit de les changer lors d'une reprise. Cette reproduction legacy n'est pas
un contrôle apparié du nouveau vérificateur.

Les contrôles réellement informatifs restent à exécuter : réentraînements à
architecture et budget identiques avec Frangi décalé, permuté ou aléatoire à
couverture comparable, puis ablation des profils photométriques. Le pilote
n'intègre actuellement ni augmentation d'ombre ni évaluation de benchmark
externe ; sa robustesse aux ombres reste une hypothèse.

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
