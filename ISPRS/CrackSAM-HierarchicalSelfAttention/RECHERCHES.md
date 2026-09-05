# Des descripteurs locaux aux relations hiérarchiques

Lecture du chapitre 12, d’EUVIP et des 119 pages du PDF de soutenance fourni ; recherches ciblées vérifiées le **5 septembre 2026**.

## Ce que montre la diapositive « ne change rien »

La page PDF **64**, numérotée **55**, résume les essais détaillés page PDF **109**, numérotée **98**. Les [résultats GeoLoRA enregistrés](../CrackSAM-GeoLoRA/tables/generated/) donnent, sur 1695 images de test :

| Variante | IoU stricte |
|---|---:|
| SAM 2 adapté aux fissures | 0,62414 |
| Même modèle, perte tolérante | 0,62763 |
| Perte tolérante + descripteurs | 0,62698 |
| Descripteurs permutés à l’entraînement | 0,62655 |

L’ajout des cartes perd **0,00065** face à la même perte sans cartes. L’écart entre les deux entraînements géométriques est **0,00043**. Le résultat utile est donc : **aucun bénéfice convaincant de ces descripteurs dans ces essais**. Une graine, cinq époques et l’absence d’intervalle de confiance empêchent de transformer cette proximité en preuve générale d’équivalence.

Les [onze cartes](../CrackSAM-GeoLoRA/geolora/evidence.py) résument plusieurs échelles de courbure, orientation, flux et profils. Elles sont ajoutées aux caractéristiques **après l’encodeur**, avant le décodeur, par un [adaptateur](../CrackSAM-GeoLoRA/geolora/adapter.py). Elles ne transmettent ni groupes emboîtés ni niveaux de fusion. Plusieurs résolutions de cartes ne constituent pas une hiérarchie de groupes.

### Une réserve sur l’interprétation causale

Dans le [code d’entraînement versionné](../CrackSAM-GeoLoRA/scripts/02_train.py), `EvidenceDataset.__getitem__` charge les cartes originales après les augmentations spatiales de l’image/masque ; elles ne suivent pas ces transformations. La validation active aussi ces augmentations. Le [script d’évaluation](../CrackSAM-GeoLoRA/scripts/03_evaluate.py) utilise des cartes alignées pour tous les checkpoints : le contrôle permute à l’entraînement, pas à l’inférence sur un modèle unique.

Ces propriétés, également présentes dans la [version du 9 août](https://github.com/Ludwig-H/Generalized-Frangi-for-Automatic-Crack-Extraction-on-FIND-dataset/blob/14a619bccc1fe50079e411169693b7730c667b41/ISPRS/CrackSAM-GeoLoRA/scripts/02_train.py), limitent l’attribution à une « redondance déjà apprise ». Il reste à établir la correspondance exacte avec le code déployé pour les checkpoints archivés. Les scores sont conservés ; aucun réentraînement ni correctif GeoLoRA n’est effectué ici.

## Ce que la littérature apporte

| Source primaire | Résultat pertinent et limite |
|---|---|
| [SAM, §7.2](https://arxiv.org/html/2304.02643v1#S7.SS2) | Retrouve des contours sans entraînement spécifique à cette tâche. Cela ne prouve pas qu’il représente exactement une Hessienne. |
| [SAUGE, introduction et méthode](https://arxiv.org/html/2412.12892v2) | Exploite les caractéristiques d’un SAM gelé pour produire des contours à plusieurs granularités. Appuie l’idée de représentations visuelles riches, sans tester la redondance de Frangi. |
| [HQ-SAM, tableau 3](https://papers.nips.cc/paper/2023/file/5f828e38160f31935cfe9f67503ad17c-Paper-Conference.pdf) | Mieux exploiter des caractéristiques déjà calculées améliore les frontières. Une information disponible peut donc rester mal utilisée. Étude sur SAM 1. |
| [Graphormer, §3.1.2, équation 6](https://proceedings.neurips.cc/paper_files/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf) | Transmet des relations de graphe par un biais ajouté aux scores d’attention. Précédent d’interface, sans validation sur SAM ou les fissures. |
| [HSA, §3–4](https://arxiv.org/html/2509.15448v1) | Introduit effectivement un a priori hiérarchique, avec des contraintes de coefficients partagés. Ses essais de transfert ne démontrent ni réussite ni impossibilité pour SAM 2. |

**Interprétation :** les résultats locaux sont compatibles avec une redondance des descripteurs, mais ne l’établissent pas. La piste suivante consiste à expliciter les **relations entre structures**, tout en testant si SAM en tire réellement parti. Aucun article consulté ne valide déjà cette intégration Frangi–SAM 2.

## Ancrage de la proposition

- [EUVIP, version finale](../../EUVIP/EUVIP_2026_Generalized_Frangi_Multimodality_camera-ready.pdf), §III et conclusion : fusion des Hessiennes, graphe, arbre couvrant et centralité ; complémentarité envisagée avec les méthodes apprises.
- [Manuscrit](../../Manuscrit_de_these_LouisHauseux.pdf), chapitre 12, pages imprimées **147–163** (PDF **173–189**), notamment §12.4 : annotation assistée et intégration aux modèles de fondation. Les formulations de centralité diffèrent de la version EUVIP ; la proposition repose sur les poids du graphe, pas sur leur assimilation.
- [Soutenance fournie](https://github.com/Ludwig-H/Manuscrit-de-th-se/blob/main/Soutenance/soutenance/Soutenance_These_2026-09-08_LouisHauseux.pdf), pages PDF **64–65, 109, 111–114, 119** : résultats négatifs et ancienne perspective d’attention. Pages PDF distinguées des numéros affichés, en raison des animations.
- [Gower–Ross, 1969](https://academic.oup.com/jrsssc/article/18/1/54/6882518) : l’arbre couvrant conserve la hiérarchie Single-Linkage. La forme du MST dessiné n’est donc pas un verdict sur les regroupements qu’il encode.
- SAM 2 officiel, révision `2b90b9f` : [configuration Hiera-L](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2/sam2_hiera_l.yaml), [attention](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/modeling/backbones/hieradet.py#L51), [prédicteur sans gradients](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/sam2_image_predictor.py#L79). Le biais proposé nécessite une modification de ce module.
