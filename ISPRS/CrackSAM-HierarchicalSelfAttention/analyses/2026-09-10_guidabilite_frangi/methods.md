# Mesurer si H renseigne le bénéfice de Frangi

**Cible.** Pour chaque image, ΔIoU compare les deux checkpoints historiques.
Les catégories utilisent ±1 point d’IoU, avec sensibilité à ±0,5 et ±2 points.
Elles décrivent ces modèles ; elles ne mesurent pas l’effet causal d’une
activation de Frangi sur des poids identiques.

**Représentation.** La sonde principale utilise la moyenne spatiale des canaux H
de la baseline SAM 2 + LoRA, avant le prompt. La moyenne de l’entrée de la
dernière attention globale, si extraite, est plus proche de l’emplacement de la
future porte. Moyennes par grille 2×2 et par échelle : comparaisons secondaires.
L’ajout des écarts-types n’est plus une opération linéaire dans H.

## Projection et séparabilité sont deux questions différentes

UMAP est ajustée **sans catégories**, en 2D et 3D : canaux standardisés,
distance euclidienne, 30 voisins, `min_dist=0.1`, graine 42. Les couleurs sont
ajoutées ensuite. La même carte montre catégories, domaines et ΔIoU. PCA fournit
une comparaison linéaire. La silhouette est mesurée dans les features
standardisées, la *trustworthiness* évalue la fidélité locale d’UMAP, sur au plus
2 000 images. Des amas visuels ne prouvent pas une séparabilité linéaire :
UMAP peut créer des coupures artificielles ([documentation UMAP](https://umap-learn.readthedocs.io/en/latest/clustering.html)).

## Test principal : prédire des images laissées de côté

Régression logistique L2, `C=1`, classes équilibrées, cinq folds. Standardisation
et sélection de canaux sont ajustées dans chaque entraînement. Les recadrages
d’une scène et ses versions bruitées restent ensemble, selon le parser
historique du dépôt ([validation groupée](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)).

Critères : balanced accuracy (hasard : 1/3 avec trois classes présentes),
macro-F1, rappels et AUROC par classe, matrice de confusion. Les intervalles à
95 % rééchantillonnent 1 000 fois les scènes au sein des domaines, en conservant
les prédictions hors fold : ils n’incluent pas la variabilité du réentraînement
de SAM. Une politique secondaire choisit le modèle guidé seulement si la sonde
prédit « amélioration » ; son ΔIoU moyen est comparé à l’usage systématique de
Frangi et à un oracle descriptif.

Témoins : classe majoritaire, domaine seul, domaine et famille source. Une
évaluation laisse chaque domaine entièrement de côté ; les trois conditions
Khanhha constituent un seul domaine. Les résultats sont aussi détaillés sur les
scènes absentes des listes historiques d’entraînement et de validation. Cette
vérification n’efface pas les chevauchements du modèle historique.

## Permutations et canaux

Le test secondaire retient une image déterministe par scène, version propre
préférée, sans consulter sa catégorie. Il permute 199 fois les catégories **au
sein de chaque couple domaine et famille source**, réajuste la sonde et compare sa balanced accuracy observée
à cette distribution : `p = (1 + nombre de scores permutés ≥ observé) / 200`.
Les folds sont fixes et indépendants des catégories. Ce test cherche un signal
au-delà du domaine et de la famille source sur ce sous-échantillon ; il ne teste pas tous les recadrages
comme s’ils étaient indépendants. Le nombre de strates et d’images dont la
catégorie peut effectivement être permutée est enregistré ; une strate à
catégorie unique ne fournit aucune permutation informative. L’argument `groups` de
[`permutation_test_score`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html)
permuterait à l’intérieur des scènes, raison d’utiliser une boucle explicite.

Les sondes à 1, 5, 10 et 25 canaux sélectionnent leur classement ANOVA uniquement
sur leur entraînement. Le tableau global présente coefficients standardisés,
corrélation de Spearman avec ΔIoU et fréquence de sélection entre folds ; ces
rangs sont **descriptifs**, sans p-valeurs univariées. Des canaux corrélés peuvent
se partager l’information. Un succès prépare une porte de confiance, mais ne
prouve pas son transfert au futur guidage hiérarchique.

## Reproduction

```bash
python analyze_features.py --cases tables/categories.csv --features /chemin/features.npz --output-dir results --permutations 199 --bootstrap 1000 --seed 42
```

Archive NPZ : `ids` et `mean` obligatoires ; `std`, `grid2`, `multiscale_mean` et
`pre_global_mean` optionnels. Aucun masque réel ou résultat du modèle guidé
n’entre dans ces features. Les empreintes des entrées et les folds sont publiés
avec les résultats. Dépendances : NumPy, pandas, SciPy, scikit-learn, matplotlib,
umap-learn ; Plotly ajoute la vue 3D HTML autonome. Les figures ne servent jamais
à choisir les paramètres du classifieur.
