# Protocole de développement FrangiGraph-Residual

Les anciennes listes mélangent parfois des découpes provenant de la même image
physique entre entraînement, validation et test. Elles restent nécessaires pour
reproduire la comparaison publiée, mais elles ne suffisent pas à mesurer
honnêtement la porte de confiance.

Pour le nouveau modèle, la liste d'entraînement est donc divisée en cinq parts.
Toutes les découpes d'une même image source restent dans la même part. La part
4 est une calibration externe : elle ne participe à aucun des quatre modèles
qui produisent les prédictions OOF des parts 0 à 3. Pour une part `f` parmi 0 à
3, le résidu est entraîné sur les trois autres parts de 0 à 3, avec `f` et 4
exclues. Un cinquième résidu est ensuite entraîné sur 0 à 3 et prédit la part 4.

Le nombre d'époques est fixé avant ces cinq entraînements. Les mesures de la
part laissée de côté sont descriptives : elles ne choisissent ni l'époque ni le
checkpoint. Chaque prédiction hors entraînement provient de la dernière époque
fixée ; sinon le choix du « meilleur » checkpoint regarderait indirectement les
images censées rester inconnues.

- prédictions OOF des parts 0 à 3 : apprentissage des coefficients de la
  régression logistique ;
- prédictions OOF de la part 4, jamais utilisées en amont : choix du seuil qui
  ouvre ou ferme la porte ;
- jeux Road420, Facade390, Concrete3k et test Khanhha : jamais utilisés pour
  apprendre le résidu ni la porte ; ils restent des évaluations historiques ;
- un nouveau jeu indépendant sera nécessaire pour une confirmation définitive.

Ce cycle reste exploratoire : la baseline historique fournie a elle-même été
entraînée sur l'intégralité de `train.txt`. Les résidus et la porte respectent
la séparation ci-dessus, mais le système complet n'est pas OOF tant que la
baseline n'est pas elle aussi réentraînée par fold. Aucune conclusion
confirmatoire de fiabilité ne doit donc reposer sur ces seules cinq parts.

Les affectations sont déterministes dans `train_group_folds.csv`. Le manifeste
publie les empreintes des listes, les recouvrements historiques et les tailles
de chaque part. Pour les reconstruire :

```bash
python ISPRS/CrackSAM/protocol/build_next_protocol.py
```

Un protocole minuscule, explicitement non scientifique, peut être construit
pour valider la chaîne réelle sur GPU avant toute campagne coûteuse :

```bash
python ISPRS/CrackSAM/protocol/build_smoke_protocol.py \
  --output /chemin/vers/frangigraph-smoke
```
