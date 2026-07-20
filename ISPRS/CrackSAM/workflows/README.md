# Workflows reproductibles

Ces scripts orchestrent l'expérience historique SAM 2 : pré-calcul des prompts,
entraînement, évaluation et génération des jalons. Ils résolvent leurs chemins
depuis la racine `ISPRS/CrackSAM`, indépendamment du répertoire courant.

Le point d'entrée principal est :

```bash
bash ISPRS/CrackSAM/workflows/run_full_cracksam2_experiment.sh
```

Ils ne mettent pas encore en œuvre FrangiGraph-Residual. Les nouveaux workflows
seront ajoutés par phase, avec un nom distinct et un manifeste de protocole.

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
