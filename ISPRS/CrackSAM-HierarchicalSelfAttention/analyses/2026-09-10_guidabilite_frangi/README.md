# Prédire le bénéfice de Frangi à partir des features de SAM 2

[**Rapport illustré**](RAPPORT.md) · [Protocole statistique et références](methods.md) · [Catégories image par image](tables/categories.csv)

[**Complément : comparer 14 représentations avec UMAP**](COMPARAISON_FEATURES.md),
avec grille visuelle complète, sondes groupées, trois graines et comparaison 3D.

L’analyse reprend **SAM 2 + LoRA baseline, époque 20**, contre **Frangi-similarité,
époque 25**, choisis par validation : 8 895 observations, 2 122 groupes physiques.
La neutralité est définie par **|ΔIoU| ≤ 0,01**, avec sensibilité à 0,005 et 0,02.
Les catégories comparent deux checkpoints ; elles n’isolent pas l’effet causal
du prompt Frangi et ne constituent pas un test du futur biais hiérarchique.

Les features viennent exclusivement de la **baseline avant prompt**. UMAP est
non supervisée ; les sondes linéaires évaluent la séparabilité dans les features
originales, avec des folds par scène. Les images, poids, shards et features
restent dans `cache/` ou sur le stockage de calcul, hors Git.

## Reproduire

Depuis ce sous-dossier, préparer les catégories et les illustrations archivées :

```bash
python prepare_cases.py
```

Extraire les features avec **l’environnement historique SAM 2**, ses dépendances
épinglées dans [requirements-sam2.txt](../../../CrackSAM/requirements-sam2.txt) et
les deux poids identifiés par SHA-256. Les chemins ci-dessous correspondent au
stockage historique. Aucun entraînement n’est lancé ; les 8 895 prédictions
baseline sont vérifiées contre leurs IoU archivées.

```bash
/home/codespace/.venv-cracksam2/bin/python extract_features.py \
  --cases tables/categories.csv --reference-csv tables/categories.csv \
  --data-root /home/codespace/cracksam2-data \
  --foundation /home/codespace/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/ISPRS/CrackSAM/checkpoints/sam2_hiera_large.pt \
  --adapter /home/codespace/cracksam2-artifacts/baseline_r4/best.pt \
  --output-dir cache/extraction-b1 \
  --batch-size 1 --workers 4 --include-pre-global \
  --verify-count 8895 --verify-tolerance 0.005
```

La taille de batch **1** reproduit l’évaluation historique. L’extraction reprend
les shards déjà terminés uniquement si son contrat est identique : poids,
images, code, précision et taille des batches.

Pour l’analyse CPU, utiliser un environnement séparé de celui de SAM 2 :

```bash
python -m venv /tmp/frangi-guidabilite-analysis
/tmp/frangi-guidabilite-analysis/bin/python -m pip install -r requirements-analysis.txt
/tmp/frangi-guidabilite-analysis/bin/python analyze_features.py \
  --cases tables/categories.csv --features cache/extraction-b1/features.npz \
  --output-dir results --permutations 199 --bootstrap 1000 \
  --seed 42 --umap-extra-seeds 7 123
```

Les versions effectivement utilisées sont conservées dans
[requirements-gcp.txt](results/requirements-gcp.txt). Le calcul G4 du 10 septembre
est terminé et [son arrêt a été vérifié](results/gcp_execution.json).

Le fichier `results/umap_3d.html` est autonome. Télécharger ce fichier pour
explorer les images en 3D ; les PNG restent lisibles directement sur GitHub.

Enfin, publier les métadonnées légères et construire le rapport :

```bash
cp cache/extraction-b1/metadata.json results/extraction_metadata.json
python build_report.py --extraction-metadata results/extraction_metadata.json
```

Le générateur refuse les résultats incomplets ou incohérents : SHA des entrées,
8 895 contrôles IoU, statistiques, coordonnées UMAP 2D/3D et figures sont requis.
Il lit les résultats enregistrés sans relancer les calculs ni choisir une sonde.

## Fichiers

- `tables/` : catégories, sensibilité, statistiques descriptives et provenance des cas.
- `figures/cases/` : panneaux historiques inchangés et distributions des catégories.
- `results/` : projections, prédictions hors fold, performances, permutations et canaux.
- `cache/extraction-b1/` : artefacts lourds nécessaires à une reprise, non versionnés.

Les contrôles logiciels ciblés se lancent avec :

```bash
python -m pytest test_analyze_features.py tests/test_extract_features.py -q
```
