# Hiérarchie Frangi et SAM 2 gelé : quelles pistes retenir ?

**Périmètre historique : LoRA également gelée.** La [décision actuelle](DECISION_SAM_LORA.md) retient le biais ultramétrique **pendant l’apprentissage de LoRA**. Les comparaisons ci-dessous documentent le régime antérieur.

Cette comparaison privilégie une intervention **dans les attentions de SAM 2 gelé**. Pour le critère de cohérence avec la voie polyèdres du LiDAR 3D, la [note consacrée au lecteur géométrique](VOIE_POLYEDRES.md) approfondit et privilégie la voie B.

**Verdict : commencer par guider une attention de SAM 2 avec les relations multi-échelles du graphe Frangi (A).** Cette intervention répond directement à « guider SAM 2 », conserve ses tokens et ses poids et permet un retour au modèle initial. Un petit lecteur hiérarchique entraîné sur ses caractéristiques gelées (B) constituerait ensuite un programme plus proche de l’« alphabet–grammaire » de la thèse. HSA et la compression des clés/valeurs restent des comparateurs.

Il s’agit de propositions : **aucune n’est implémentée ou mesurée ici**. Les poids de SAM 2, y compris une adaptation préalable aux fissures, resteraient gelés. La lecture s’appuie sur EUVIP et la [soutenance figée au commit `8ddbae760c5a337c4e033c45e5f60c16ca58cc67`](https://github.com/Ludwig-H/Manuscrit-de-th-se/blob/8ddbae760c5a337c4e033c45e5f60c16ca58cc67/Soutenance/soutenance/Soutenance_These_2026-09-08_LouisHauseux.pdf). Les pages ci-dessous désignent les pages PDF, puis les numéros imprimés des diapositives.

## Ce que la hiérarchie doit transmettre

La soutenance distingue **construire tous les regroupements** et **choisir une segmentation** : composantes aux différents seuils, single linkage et MST (p. 18–23, diapos 15–16), puis dendrogramme et distance de fusion (p. 110, diapo 99). L’intérêt recherché est de conserver plusieurs niveaux et leurs événements, sans décider immédiatement quelle échelle représente une fissure.

Pour EUVIP, repartir des [dissimilarités publiées](../../EUVIP/LaTeX/main.tex), issues des compatibilités de courbure, de forme et d’orientation : `d_ij = min(1, ρ_ij(1 − S⁰_ij))`. Fixer les candidats, les arêtes et leurs coûts, **avant sélection de la plus grande composante et élagage par centralité**, puis construire la filtration. Ce choix exploite une étape antérieure au squelette final du papier ; il ne prétend pas que celui-ci conserve déjà toute cette information.

Ne pas confondre **hauteur de fusion**, **centralité dans le MST** et **échelle gaussienne σ du Hessien**. Un arbre spatial enraciné au sommet le plus central n’est pas le dendrogramme des fusions. Les ex æquo ne doivent pas créer artificiellement des fusions binaires ordonnées.

En **K2**, les feuilles sont des arêtes. L’exemple de la soutenance regroupe `AB, BC, AC` et `DE, EF, DF`, avec `CD` séparément ; les supports partagent C et D (p. 27–31, diapos 20–24). Imposer un parent unique à chaque pixel effacerait ce recouvrement. Les objets hiérarchisés et leur projection vers l’image sont distincts. Enfin, les résultats de densité HGP dans leur cadre géométrique ne deviennent pas automatiquement des théorèmes sur les coûts Frangi.

## A — Ajouter une relation multi-échelle à l’attention

### Une relation définie par tous les seuils

Dans une composante du graphe fixé, la première fusion de deux candidats vérifie :

$$
u_{ij}=\max_{e\in\operatorname{chemin}_{\mathrm{MST}}(i,j)} d_e.
$$

Pour la hiérarchie K1 des composantes, définir :

$$
\kappa_{ij}=\int_0^1 \mathbf{1}[i,j\text{ dans la même composante au seuil }t]\,dt
=1-u_{ij}.
$$

Deux candidats réunis tôt partagent davantage de seuils. Poser `κ_ii = 1` et `κ_ij = 0` entre composantes déconnectées, sans inventer de connexion physique. Le MST préserve ces regroupements ; κ ne conserve pas toutes ses incidences géométriques ni les événements K≥2.

Le **noyau κ complet est positif semi-défini** : chaque matrice de coappartenance à une partition l’est, ainsi que leur moyenne. C’est le principe des [noyaux de partitions de Davies et Ghahramani, 2014](https://mlg.eng.cam.ac.uk/pub/pdf/DavGha14a.pdf), appliqué ici aux seuils de notre filtration. Les coûts bornés sont empiriques, pas des probabilités. L’intégrale uniforme exploite leurs hauteurs numériques mais reste un choix à valider. Une transformation croissante des coûts peut conserver les fusions tout en changeant κ.

### Passer des candidats aux tokens

Construire une matrice `P` à poids non négatifs reliant chaque token aux candidats qu’il couvre. Normaliser chaque ligne non vide à somme un ; laisser les lignes vides nulles. Puis calculer :

$$
\kappa_{\mathrm{tok}}=P\kappa P^\top.
$$

Un token peut recevoir plusieurs contributions géométriques. Cette projection reste un **résumé** : un patch traversé par deux branches peut les mélanger. Mesurer couverture, mélanges de groupes et dépendance à la résolution. La normalisation ne distingue pas, à elle seule, un patch presque vide d’un patch fortement couvert.

Pour K2, une incidence pondérée entre arêtes et pixels respecterait mieux les recouvrements qu’une affectation exclusive. Elle ne préserverait pas automatiquement orientations, événements et géométrie complète. Le premier essai doit annoncer clairement **K1**, sans revendiquer déjà la grammaire K2 des diapos 104–107.

### Une modification bornée et réversible

À partir de `κ_tok`, construire `B` en neutralisant sa diagonale et les interactions non couvertes. **B n’est plus nécessairement positif semi-défini** ; cette propriété n’est pas requise pour modifier des logits d’attention :

$$
A'=\operatorname{softmax}_{\text{lignes}}(L+\alpha B),
\qquad L=QK^\top/\sqrt{d_h}.
$$

Partager B et un unique `α` entre les têtes, tout en conservant leurs scores visuels distincts. Choisir `0 ≤ α ≤ α_max` dans un intervalle annoncé, sur validation, avec `α = 0` comme référence obligatoire. Aucun gradient : cela reste un réglage expérimental.

Commencer dans un seul bloc global, par exemple l’indice 43 de [Hiera-L](https://github.com/facebookresearch/sam2/blob/2b90b9f5ceec907a1c18123530e92e794ad901a4/sam2/configs/sam2/sam2_hiera_l.yaml), sans supprimer tokens, projections, résidus, MLP ou informations de position. Conserver prompts et décodeur identiques. **Hors graphe, le logit ajouté est nul, mais les probabilités peuvent changer par renormalisation** lorsqu’une requête reçoit d’autres corrections. Une matrice B partagée de 4096² éléments FP16 représente environ 32 Mio ; temps et mémoire supplémentaires dépendent aussi du backend d’attention et restent à mesurer.

### Le précédent utile, et sa limite

[CASS, CVPR 2025, §3.2](https://arxiv.org/html/2411.17150v3#S3.SS2), transfère une structure relationnelle issue de DINO vers CLIP sans entraînement : graphe construit avec les clés, correspondance spectrale entre têtes, puis transfert de composantes spectrales. Il établit un précédent de **transfert de relations vers une attention visuelle gelée**, sans traiter une hiérarchie visuelle externe ni SAM 2. Nous ne reprenons ni son `KKᵀ` comme définition Frangi, ni les suppressions de résidu/FFN propres à sa configuration CLIP. Le biais fondé sur κ est notre hypothèse à évaluer.

## Trois objections à A, et les expériences décisives

1. **Une liaison fausse devient transitive.** Une arête trop favorable peut rapprocher deux structures entières. Les grandes composantes peuvent aussi attirer beaucoup de masse d’attention. Mesurer faux raccords, ruptures et détails fins, en complément de l’IoU.
2. **La projection peut détruire la distinction recherchée.** Des branches distinctes peuvent occuper le même patch. Rapporter les ambiguïtés et comparer sur un support de candidats identique ; davantage de pixels couverts ne prouverait pas l’utilité de l’arbre.
3. **Le modèle n’a pas appris à interpréter ce signal.** En monomodal, Frangi réorganise une observation déjà accessible à SAM ; une observation supplémentaire peut venir du multimodal. Ni complémentarité ni bon placement du biais ne sont acquis. Si `α = 0` gagne, conserver ce résultat.

Contrôler chaque explication : bonus uniforme sur le **masque plat de candidats** ; **adjacence locale** ; **partition unique** ; arbre construit sur les caractéristiques visuelles ou la proximité spatiale ; puis hiérarchie Frangi complète. Comparer les hauteurs originales à des hauteurs canoniques conservant seulement la topologie, ainsi que des associations feuilles–arbre permutées. Pour les niveaux permutés, préserver la monotonie parent–enfant : une permutation naïve ne définit plus une filtration valide.

Ces tests prolongent les témoins de la p. 118, diapo 107 : graphe latéral, arbre aléatoire, niveaux permutés, topologie seule et trajectoires complètes. Conserver mêmes images, prompts, budget de calcul et budget de recherche de `α`, puis évaluer sur un test séparé. Compter construction du graphe, projection, mémoire et temps total ; aucune accélération n’est acquise.

## B — Apprendre une petite grammaire sur des représentations gelées

[Superpoint Transformer, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Robert_Efficient_3D_Semantic_Segmentation_with_Superpoint_Transformer_ICCV_2023_paper.pdf), utilise une véritable hiérarchie géométrique externe et des voisinages à plusieurs niveaux. Son module est **entraîné** pour la segmentation 3D. Ses partitions initiales limitent les frontières restituables ; sa compacité ne garantit pas celle d’une transposition à nos images.

Notre adaptation mettrait en cache les caractéristiques de SAM 2, sans graphe de gradients. Un petit lecteur apprendrait les échanges entre éléments Frangi, groupes et ensembles, avec orientations, niveaux et événements ; une voie fine conserverait les caractéristiques détaillées pour produire le masque. Aucun gradient ne traverserait SAM. Cette piste concrétise davantage l’« alphabet–grammaire » des diapos 58–59 et 104–107. [GraphAdapter, NeurIPS 2023](https://arxiv.org/abs/2309.13625), est un précédent plus général : des convolutions apprises sur des graphes de connaissances entre classes, sans cette hiérarchie géométrique.

**Trois objections :** il faut des annotations et entraîner un nouveau module ; une sortie indépendante constitue un lecteur sur SAM, pas un guidage de ses attentions ; des groupes erronés et une projection grossière peuvent encore limiter les détails. Comparer le même lecteur avec caractéristiques seules, graphe local et véritable hiérarchie. Un gain dû au module supplémentaire ne validerait pas la grammaire.

## HSA et compression : des comparateurs

[HSA, NeurIPS 2025, §§3.2 et 5.3](https://proceedings.neurips.cc/paper_files/paper/2025/file/0480adaf62a918405a5e3b1031e0c056-Paper-Conference.pdf), partage des coefficients entre sous-arbres frères. Il conserve les feuilles mais peut supprimer leurs différences d’interaction ; il ne renforce pas automatiquement les bonnes connexions. Les remplacements sans entraînement concernent RoBERTa : au tableau 3, l’exactitude passe de 95,58 à 94,94 % sur IMDB, mais de 92,67 à 50,72 % sur QNLI. La normalisation Q/K et le traitement de la diagonale de sa preuve ne correspondent pas directement à SAM 2. Les hauteurs Frangi ne sont pas automatiquement utilisées. C’est un comparateur de contrainte hiérarchique, avec ces écarts explicités.

[ATC, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07441.pdf), construit des regroupements hiérarchiques mais exploite une coupe ; [CubistMerge, prépublication, v2 de 2026](https://arxiv.org/pdf/2509.21764v2), évalue effectivement des fusions sur SAM 2 gelé. Les regroupements de ce dernier suivent toutefois une grille, peu adaptée aux branches irrégulières de Frangi. Ces précédents visent d’abord le coût.

[StructSAM, v2 de 2026](https://arxiv.org/pdf/2603.07307v2), propose fusion → attention → restitution, avec un résidu dense préservé. Les différences entrantes subsistent, mais les membres d’un groupe partagent leur mise à jour d’attention. Ses évaluations portent sur SAM, MedSAM et EfficientSAM ; SAM 2 est un comparateur. Il protège les contours sans fournir de hiérarchie métier. Sa version courte relève du workshop AdaptFM@ICML, pas de la conférence principale. Pour les fissures, **appartenir à une même branche ne signifie pas posséder des caractéristiques interchangeables** : extrémités, jonctions et changements de matériau peuvent demander des échanges distincts.

Une variante prudente conserverait toutes les requêtes et le résidu dense, mais résumerait les clés/valeurs par groupes disjoints, avec la correction de masse `log(taille)` également utilisée par [ToMe, ICLR 2023](https://arxiv.org/abs/2210.09461). Elle serait exacte pour des clés identiques dans chaque groupe, approximative sinon. Une coupe ne testerait pas toute la hiérarchie ; mélanger parents et enfants compterait certains éléments plusieurs fois. Pour garder plusieurs niveaux, il faudrait des coupes emboîtées ou une couverture de l’arbre adaptée à chaque requête. A conserve aussi les clés/valeurs individuelles et évite cette approximation.

**Priorité : A pour tester le guidage ; B pour développer la grammaire.** Les contrôles devront montrer ce qu’apportent relations et niveaux au-delà du seul détecteur Frangi.
