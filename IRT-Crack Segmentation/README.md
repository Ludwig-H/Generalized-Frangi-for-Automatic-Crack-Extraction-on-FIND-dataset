# Évaluation de Frangi-Graph sur le Benchmark IRT-Crack Segmentation

Ce dossier contient l'implémentation et les expériences permettant d'évaluer l'approche **Generalized Frangi Graph** (introduite dans notre article EUSIPCO *Multi-Modal, Training-Free Crack Extraction via Generalized Frangi Graph*) sur la base de données publique **IRT-Crack Segmentation** (images optiques et infrarouges).

L'objectif est de démontrer la capacité de notre approche non supervisée à extraire les réseaux de fissures en fusionnant les modalités (Visible + Infrarouge) au niveau des matrices Hessiennes, le tout accéléré sur GPU.

## 1. Caractéristiques du Benchmark IRT-Crack

La base de données d'origine (téléchargée automatiquement via Google Drive dans notre pipeline) présente des défis architecturaux spécifiques que notre Dataloader PyTorch gère rigoureusement :

*   **Topologie en 4 sous-espaces :** `01-Visible Image` (RGB), `02-Infrared Image` (Fausses couleurs), `03-Fusion Image` (Non utilisé, nous fusionnons nous-mêmes au niveau Hessien), et `04-Ground Truth` (Masques binaires).
*   **Indexation Creuse (Sparse Indexing) :** La numérotation des fichiers s'étend jusqu'à `2484` mais ne contient que 448 images valides. Toute boucle séquentielle (`for i in range(2485)`) échouerait. Notre code cartographie dynamiquement l'espace présent sur le disque via `pathlib`.
*   **Asymétrie des extensions :** Les entrées sont au format `.png` (sans perte), mais les cibles (Ground Truth) sont étrangement fournies au format `.jpg` (avec perte).
*   **Artéfacts de compression :** À cause du format JPEG du Ground Truth, des artefacts de compression de type Gibbs corrompent les bords des masques. Le Dataloader applique donc un **seuillage mathématique strict** (`(img > 127).astype(np.float32)`) avant toute évaluation de métriques (Jaccard, Tversky) pour ne pas fausser les scores.

## 2. Implémentation PyTorch 100% GPU (A100)

L'algorithme a été repensé pour exploiter pleinement les architectures GPU modernes (comme le NVIDIA A100 sur Google Colab). Le code se trouve dans le notebook `Frangi_IRT_Crack_GPU.ipynb`.

### A. Fusion Multimodale au Niveau Hessien
Contrairement aux approches par apprentissage profond qui fusionnent les "features" dans les couches latentes, nous appliquons une **fusion d'opérateurs physiques** :
1. Les dérivées partielles spatiales ($I_{xx}, I_{xy}, I_{yy}$) sont calculées de manière très optimisée en utilisant `torch.nn.functional.conv2d` avec des noyaux Gaussiens pré-calculés.
2. La Hessienne de l'image *Visible* et celle de l'image *Infrarouge* sont calculées séparément.
3. Chaque Hessienne est normalisée par sa norme spectrale maximale.
4. Les matrices sont sommées linéairement (par défaut **50% Visible / 50% Infrarouge**). *Note : les images pré-fusionnées du dossier `03-Fusion Image` sont volontairement ignorées.*
5. Les valeurs propres et vecteurs propres sont extraits de cette Hessienne fusionnée pour annuler le bruit décorrélé entre les capteurs.

### B. Graphe de Similarité Frangi
Pour dépasser le simple filtrage pixel par pixel, nous construisons un graphe géométrique. Le calcul de la similarité (qui combine l'élongation spatiale, le contraste d'intensité et l'alignement angulaire topologique) implique de comparer chaque pixel candidat à ses voisins de rayon $R$.
Pour éviter l'explosion mémoire $\mathcal{O}(N^2)$ (erreur de type `OutOfMemoryError`), l'architecture du graphe a été optimisée et rendue **creuse (Sparse)** :
1. Une recherche des plus proches voisins (K-NN) est effectuée ultra-rapidement via `scipy.spatial.cKDTree` pour isoler uniquement les paires valides dans le rayon $R$.
2. Le calcul intensif des tenseurs de similarité par PyTorch n'est effectué **que sur ces arêtes locales**.
3. Les résultats alimentent une matrice creuse (`scipy.sparse.coo_matrix`), minimisant drastiquement l'empreinte VRAM et RAM.

### C. Extraction Topologique (Squelettisation)
Une fois la matrice d'affinité spatiale construite sur le GPU :
1. Elle est basculée sur le CPU.
2. Un Arbre Couvrant de Poids Minimum (MST - *Minimum Spanning Tree*) est calculé via SciPy.
3. La Centralité d'Intermédiarité Pondérée (*Weighted Betweenness Centrality*) est accumulée pour élaguer l'arbre et ne conserver que le "backbone" (squelette majeur) continu de la fissure.

## 3. Utilisation

1. Ouvrez **Google Colab**.
2. Importez le notebook `Frangi_IRT_Crack_GPU.ipynb` généré dans ce dossier.
3. Assurez-vous d'activer l'accélérateur matériel **GPU** (idéalement A100 ou T4) dans `Exécution > Modifier le type d'exécution`.
4. Exécutez toutes les cellules. Le notebook s'occupera :
    * De télécharger le dataset de manière autonome. (Par défaut, il télécharge une **archive ZIP** pour plus de rapidité, mais une option de secours par dossier est sélectionnable via une case à cocher).
    * D'instancier le Dataloader robuste.
    * D'extraire les Fissures sur un échantillon et d'afficher de riches visualisations multi-axes (Modalités isolées, réponse Frangi, carte de Centralité, Superposition au GT).
    * De calculer les métriques de segmentation (IoU/Jaccard, Tversky) sur un batch de validation.

## 4. Pistes d'expérimentation futures
* Ajuster le dictionnaire `weights = {'visible': 0.5, 'infrared': 0.5}` pour mesurer l'impact isolée de chaque capteur (ex: `1.0` / `0.0`).
* Évaluer l'algorithme sur le dataset entier avec la distance de Wasserstein (nécessite une fonction d'évaluation spatiale supplémentaire sur les squelettes).