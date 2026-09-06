# Archives des pistes examinées

Synthèse arrêtée au **6 septembre 2026**. La [proposition actuelle](README.md) conserve SAM préentraîné gelé et apprend LoRA avec un biais de proximité ultramétrique. Les variantes ci-dessous ne constituent pas des recommandations concurrentes.

## Ce que les essais ont appris

Dans [GeoLoRA](../CrackSAM-GeoLoRA/tables/generated/), les onze cartes de Hessienne, gradient, orientation et profils corrigent les caractéristiques de SAM, avec LoRA coentraîné. Sur 1695 images, l’IoU stricte moyenne vaut **0,62763** avec la perte tolérante seule, **0,62698** avec les cartes et **0,62655** avec les cartes permutées à l’entraînement. Elles ne transmettent pas une hiérarchie de groupes.

Le [code versionné](../CrackSAM-GeoLoRA/scripts/02_train.py) présente une réserve d’alignement des cartes avec les augmentations ; sa correspondance exacte aux entraînements archivés reste à établir. Ces scores ne prouvent donc pas que SAM représente exactement la Hessienne. Ils ne donnent aucun motif convaincant de reprendre l’ajout de cartes locales.

La [perte clDice](../CrackSAM-GeoLoRA/tables/generated/eval_cldice.json) améliore la couverture du squelette mais réduit l’IoU : **0,60659**, contre **0,62414** pour sa baseline. L’[arbitre GFA sur fragments plats](../CrackSAM-GFA/RAPPORT.md) obtient **0,61714**, contre **0,62375** pour sa référence. Les [prompts denses et correcteurs raster](../CrackSAM/docs/08_AUDIT_CRACKSAM2_FRANGIGRAPH_LORA.md) n’établissent pas davantage un gain propre à Frangi. Ces protocoles restent distincts ; aucun ne teste le biais hiérarchique retenu.

## Les variantes écartées du premier essai

| Piste | Idée conservée et raison de l’écarter ici |
|---|---|
| Biais avec LoRA également gelée | Transmettre les relations sans nouvel apprentissage ; l’adaptation n’apprend toutefois pas à utiliser ce signal. [CASS, CVPR 2025](https://arxiv.org/abs/2411.17150), transfère des relations DINO vers CLIP, sans hiérarchie Frangi ni SAM. |
| Attention hiérarchique contrainte | [HSA, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/0480adaf62a918405a5e3b1031e0c056-Paper-Conference.pdf), partage des coefficients entre sous-arbres. Cela peut effacer des différences utiles aux extrémités et jonctions ; ses transferts sur RoBERTa ne valident pas SAM. |
| Fusion de tokens | ATC, ToMe et StructSAM inspirent des regroupements pour réduire le calcul ; une coupe ou des mises à jour communes ne testent pas toute notre hiérarchie. |
| Moyennes dans LoRA | [Conv-LoRA, ICLR 2024](https://arxiv.org/abs/2401.17868), insère des opérations spatiales apprises entre projections de faible rang. Des moyennes Frangi seraient une transposition économique, mais mélangeraient les voisins avant leur comparaison visuelle. |
| Lissage des prédictions | Risque d’imposer les faux raccords du graphe. Des paires contrastives validées par les annotations seraient plus prudentes, sans avantage propre à Frangi établi. |
| Lecteur extérieur appris | Un petit réseau lit les groupes au-dessus de caractéristiques gelées. Il ajoute une tête et déplace l’apprentissage hors des attentions de SAM. |

## Le programme des polyèdres LiDAR

L’idée plus large reste de lire **les éléments géométriques, leurs incidences et leurs regroupements**. [Superpoint Transformer, ICCV 2023](https://arxiv.org/abs/2306.08045), fournit un précédent de segmentation 3D avec une hiérarchie externe ; [Cell Attention Networks, IJCNN 2023](https://arxiv.org/abs/2209.08179), distingue les voisinages entre cellules ; [Tree-structured Attention, ICLR 2020](https://arxiv.org/abs/2002.08046), lit un arbre fourni dans le domaine du texte.

La variante multirésolution conserve moyennes des groupes et différences enfant–parent, puis apprend leur importance. [Saito et al., 2024](https://link.springer.com/article/10.1007/s43670-023-00076-4), motive des représentations à plusieurs échelles sur les simplexes. Des gains tous identiques font disparaître l’effet des regroupements intermédiaires : un contrôle essentiel.

Pour K ≥ 2, les supports peuvent se recouvrir. Un dendrogramme de pixels ne remplace ni ces incidences ni les connecteurs apparaissant sans fusion. La piste SAM actuelle constitue un premier essai K = 1 ; elle ne valide pas ce programme LiDAR.

## Retrouver les analyses détaillées

Les anciens documents et dessins sont conservés dans l’[état Git du 6 septembre 2026, `7c91bee`](https://github.com/Ludwig-H/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/tree/7c91beeeec7cc393bd2e9dba68c2e6fe47789422/ISPRS/CrackSAM-HierarchicalSelfAttention). Ils détaillent bibliographie, comparaisons, contrats géométriques et limites. Cette archive en est l’unique point d’entrée dans le dossier courant.
