# Hiérarchie Frangi et modèles gelés : pistes comparées

Recherche vérifiée le **5 septembre 2026**. Contrainte : conserver les poids de
SAM 2, y compris son adaptation aux fissures lorsqu'elle existe. Les mécanismes
ci-dessous ont des précédents ; leur combinaison avec notre hiérarchie reste
une perspective.

## 1. HSA : partager les coefficients selon l'arbre

[Amizadeh et al., NeurIPS 2025](https://arxiv.org/html/2509.15448v1#S4.SS3)
remplacent certaines attentions de RoBERTa sans réentraînement. L'arbre fourni
détermine quels coefficients sont partagés entre sous-groupes ; les
caractéristiques du modèle déterminent leurs valeurs. Aucun paramètre ajouté.

**Rapport à Frangi :** fournir notre arbre de fusions. C'est la référence la plus
directe pour le poster. Les expériences publiées portent sur un compromis
coût–précision, avec des pertes de précision ; elles ne démontrent pas une
amélioration de segmentation. Le théorème suppose notamment des requêtes/clés
normalisées : sa garantie ne se transfère pas automatiquement à Hiera.

## 2. Regrouper les tokens : une autre intervention interne

Trois travaux apportent des éléments complémentaires :

- **[ATC, Haurum et al., ECCV 2024](https://arxiv.org/html/2409.11923v1)** :
  classification hiérarchique ascendante des tokens, sans paramètres nouveaux.
  Évaluations avec et sans fine-tuning, dont segmentation avec ViT-Adapter.
- **[CubistMerge, Gong et Lis, prépublication, v2 de 2026](https://arxiv.org/html/2509.21764v2)** :
  fusion spatiale préservant une grille 2D, effectivement évaluée sur SAM 2
  sans entraînement supplémentaire (§4–4.1). Ses groupes restent contraints
  par les lignes, colonnes et fenêtres.
- **[StructSAM, Nguyen et al., 2026](https://arxiv.org/html/2603.07307v2)** :
  fusion → attention → restitution de la grille dense, avec protection des
  contours, sur SAM/MedSAM/EfficientSAM gelés. SAM 2 est seulement un comparateur.
  Version courte au workshop AdaptFM@ICML ; ne pas citer comme ICML principal.

**Notre piste :** les groupes Frangi autoriseraient ou interdiraient certaines
fusions. Garder séparés les tokens ambigus, les bifurcations et les détails fins.
Plusieurs coupes emboîtées organiseraient progressivement les regroupements.
Restituer la grille ne récupère pas l'information perdue par une fusion.
Ces articles visent surtout l'accélération ; un bénéfice sur les fissures reste
une question distincte. ATC signale aussi les difficultés du single linkage
lorsque les fusions deviennent agressives.

## 3. CASS : transférer des relations vers l'attention gelée

[Kim et al., CVPR 2025](https://arxiv.org/html/2411.17150v3#S3.SS2)
transfèrent la structure spectrale d'un graphe DINO vers l'attention finale de
CLIP, sans entraînement. **Rapport à Frangi :** transmettre des relations du
graphe à un modèle gelé a donc un précédent visuel. Remplacer ces relations par
notre hiérarchie demanderait une définition nouvelle ; CASS ne réalise pas
déjà un guidage hiérarchique de SAM 2.

## Choix pour la suite

Conserver **HSA comme référence du poster**. Explorer le **regroupement guidé**
comme alternative pratique : un bloc global, fusions prudentes, puis comparaison
à nombre de tokens égal avec un regroupement visuel et un arbre permuté.
Pour revendiquer l'apport de la hiérarchie, comparer aussi plusieurs niveaux à
une partition unique. L'arbre provient des seuils du MST avant élagage, pas de
la profondeur d'un parcours arbitrairement enraciné.

Si un petit apprentissage séparé est acceptable,
[GraphAdapter, Li et al., NeurIPS 2023](https://arxiv.org/abs/2309.13625)
offre un autre précédent : entraîner des modules de graphe sur les
représentations d'un modèle gelé. Notre transposition utiliserait les
caractéristiques SAM 2 sauvegardées et un petit module parcourant l'arbre,
sans rétropropagation dans SAM 2. Elle ne serait pas entièrement sans apprentissage.
