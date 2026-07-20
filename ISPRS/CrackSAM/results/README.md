# Résultats et provenance

Ce dossier contient des résultats légers, rapports et manifestes déjà produits.
Il ne sert pas de répertoire d'entraînement.

- `frangi_milestone_report/` : comparaison quantitative baseline/Frangi ;
- `frangi_safe_recommendation/` : diagnostic et protocole SafeFrangi historique ;
- `frangi_chrominance_cpu_probe/` : sonde CPU exploratoire ;
- fichiers datés à la racine : contrat des checkpoints et faisabilité.

Les chemins absolus contenus dans les JSON sont des traces de provenance. Ils
ne doivent pas être réécrits après déplacement. Les futurs runs lourds vont sous
un `artifact root` externe ; seuls les résultats validés et leur manifeste sont
rapatriés ici.

## Analyse bootstrap du pilote FrangiGraph

Après les cinq évaluations OOF et le gel de la porte logistique, lancer
`analyze_frangigraph_pilot_bootstrap.py` sur les artefacts du run :

```bash
python ISPRS/CrackSAM/analyze_frangigraph_pilot_bootstrap.py \
  --gate-json "$RUN_ROOT/gate/logistic_gate.json" \
  --oof-manifest "$RUN_ROOT/gate_data/oof_manifest.json" \
  --fold-dir "0=$RUN_ROOT/oof_evaluations/fold_0" \
  --fold-dir "1=$RUN_ROOT/oof_evaluations/fold_1" \
  --fold-dir "2=$RUN_ROOT/oof_evaluations/fold_2" \
  --fold-dir "3=$RUN_ROOT/oof_evaluations/fold_3" \
  --fold-dir "4=$RUN_ROOT/oof_evaluations/fold_4" \
  --group-assignments \
    ISPRS/CrackSAM/protocol/frangigraph_v1/train_group_folds.csv \
  --gated-csv "$RUN_ROOT/gate/oof_analysis/per_image_gated.csv" \
  --output "$RUN_ROOT/bootstrap_analysis"
```

Par défaut, l'analyse utilise 20 000 réplications, la graine 3407 et un
bootstrap percentile à 95 %, clusterisé par `dataset::source_group`. Le seuil
est relu dans le JSON gelé et n'est jamais recalibré. `summary.json` conserve
la provenance, l'audit des jointures, les strates et les limites
d'interprétation ; `estimands.csv` fournit la table compacte des estimations et
intervalles. Les résultats de porte sur les folds 0--3 sont apparents, ceux du
fold 4 sont descriptifs de la calibration, et l'agrégat des cinq folds reste
apparent. Seul le résidu est évalué OOF sur les cinq folds.
