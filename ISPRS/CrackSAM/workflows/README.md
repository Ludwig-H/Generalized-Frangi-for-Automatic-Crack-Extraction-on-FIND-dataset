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
