# Apprendre à lire la hiérarchie de polyèdres

Cette note concerne le programme LiDAR 3D. Pour la solution simple **SAM gelé + LoRA + Frangi-graphe**, lire la [décision recentrée](DECISION_SAM_LORA.md).

**Choix pour prolonger la soutenance LiDAR 3D : un petit lecteur appris sur les éléments géométriques, leurs recollements et leurs fusions, alimenté par un encodeur gelé.** Le biais dans l’attention de SAM reste un essai relationnel plus restreint. Il ne remplace pas ce programme.

**Mécanisme à essayer en premier :** la [lecture des différences entre enfants et parents](LECTURE_MULTIECHELLE.md), conditionnée par la géométrie et les niveaux. Elle conserve explicitement les détails ; l’attention décrite ci-dessous reste une variante plus expressive à comparer.

Cette note du **5 septembre 2026** approfondit les chapitres 8–9 du [manuscrit](../../Manuscrit_de_these_LouisHauseux.pdf) et les diapos 59, 104–107 de la [soutenance vérifiée](https://github.com/Ludwig-H/Manuscrit-de-th-se/blob/8ddbae760c5a337c4e033c45e5f60c16ca58cc67/Soutenance/soutenance/Soutenance_These_2026-09-08_LouisHauseux.pdf). L’architecture décrite est une proposition ; aucun modèle n’est entraîné ou évalué ici.

## 1. Ce que signifie l’alphabet géométrique

L’alphabet n’est pas un catalogue universel de polyèdres identiques dans toutes les scènes. La règle de construction est fixée ; les objets dépendent du nuage. Leurs descripteurs pourraient transférer. La figure de la diapo 59 représente un objet géométrique **à chaque nœud**, pas seulement aux feuilles. La tokenisation exacte reste ouverte dans les [notes orales](https://github.com/Ludwig-H/Manuscrit-de-th-se/blob/8ddbae760c5a337c4e033c45e5f60c16ca58cc67/Soutenance/soutenance/README.md#L173).

Pour un premier lecteur, je propose de conserver un état pour chaque atome et chaque nœud de fusion. Les atomes restent identifiés ; les nœuds internes portent leur composition et leur contexte. On évite ainsi de choisir une seule coupe ou de donner tous les groupes au réseau comme une liste sans relations.

| Ordre de la thèse | Atome τ | Connecteur σ | Un contact qui ne suffit pas au recollement |
|---|---|---|---|
| K = 1 | Point | Arête | Proximité sans arête admise |
| K = 2 | Arête / segment | Triangle | Sommet partagé |
| K = 3 | Facette triangulaire | 4-uplet / simplexe abstrait de dimension 3 | Arête partagée seule |

**L’ordre K, la dimension ambiante et le niveau de fusion sont trois choses différentes.** K = 3 fournit des triangles, mais ne garantit pas un maillage de surface régulier ou une enveloppe orientée. Des attributs de surface demandent une convention de réalisation et un traitement des dégénérescences.

![Éléments, regroupements et lecteur](figures/lecteur_polyedres.png)

## 2. Deux structures doivent rester visibles

**Les incidences disent de quoi un objet est fait.** En K = 2, les groupes `{AB, BC, AC}`, `{CD}`, `{DE, EF, DF}` peuvent rester distincts tout en partageant C et D. Le voisinage par point partagé peut fournir du contexte ; il ne doit pas devenir silencieusement la règle de fusion.

**La hiérarchie dit quand les objets se regroupent.** Conserver naissance, enfants, parent, niveau et connecteurs témoins. Une profondeur entière ne remplace pas un rayon ou une densité. Les atomes K ≥ 2 peuvent naître après le rayon zéro ; les événements simultanés ne doivent pas recevoir un ordre binaire artificiel.

Un arbre seul ne conserve pas toutes les incidences ; un graphe de facettes à un seul niveau ne conserve pas la trajectoire. Le théorème du K-MST porte sur les supports des polyèdres non triviaux : il ne livre pas automatiquement toutes les faces et tous les attributs nécessaires au lecteur.

Un connecteur peut aussi apparaître dans un groupe déjà connecté, sans nouvelle fusion. Lire toute l’évolution du complexe demande ses naissances internes, ou des attributs d’incidence selon l’échelle. L’arbre avec les seuls témoins des fusions décrit une information plus restreinte.

## 3. Le lecteur minimal proposé

### Des observations gelées vers les éléments

Calculer une fois les caractéristiques `f_x` et leur correspondance aux points. Pour les images EUVIP, geler la [baseline locale SAM 2 + LoRA](../CrackSAM/README.md), adaptation distincte du [CrackSAM publié sur SAM 1](https://arxiv.org/abs/2312.04233). Pour un nuage, employer un encodeur 3D adapté aux canaux disponibles : SAM 2 ne prend pas nativement un LiDAR en entrée.

[Sonata, CVPR 2025](https://arxiv.org/html/2503.16429v1), fournit un précédent concret : encodeur 3D gelé avec tête linéaire ou décodeur appris. Son [code officiel](https://github.com/facebookresearch/sonata) restitue des caractéristiques aux points via les indices de sous-échantillonnage. Il faut vérifier checkpoint, canaux, domaine et correspondances ; cette restitution ne recrée pas des détails déjà perdus.

Le §9.1 du manuscrit fournit une interface particulièrement utile, avec les notations originales :

$$
S_\tau=\sum_{\sigma\supset\tau}\psi(\rho(\sigma)),\qquad
T_x=\sum_{\tau\ni x}S_\tau,\qquad
w_{x\tau}=\mathbf 1[x\in\tau]\frac{S_\tau}{T_x}.
$$

Ici, ρ(σ) est le rayon de naissance du connecteur et ψ une fonction de pondération déclarée. Poser les poids à zéro si `T_x = 0`. **Sτ est un score issu des connecteurs, pas une aire.** Sur les points couverts, les poids incidents somment à un. Ils doivent être calculés avec les connecteurs et la fonction ψ déclarés, avant de perdre cette information dans le MST.

Une initialisation possible est la moyenne pondérée des `f_x` incidents à τ, normalisée par `m_τ = Σ_x w_xτ`, complétée par sa géométrie et sa naissance. Garder aussi les caractéristiques originales : une moyenne reste une compression. Les atomes sans masse positive demandent un traitement déclaré, pas une division par zéro.

### Variante avec attention : lire les compositions et restituer le contexte

Un seul petit module peut réaliser trois opérations :

1. **Décrire** chaque élément et chaque groupe par ses caractéristiques et les attributs géométriques effectivement disponibles.
2. **Monter** dans la forêt : chaque parent lit ses enfants par attention, avec le niveau et les témoins de l’événement. Partager les poids entre événements, plutôt que créer une couche apprise par profondeur. L’agrégation doit être indépendante de l’ordre arbitraire des enfants.
3. **Redescendre** : rendre le contexte des parents aux enfants, en conservant leur état local. Les relations latérales éventuelles ont un type distinct des liens parent–enfant.

Ce parcours utilise tous les événements exportés. Il résume néanmoins leurs informations dans des vecteurs de taille finie : ce n’est pas un encodage injectif garanti. Deux couches locales sur les seules arêtes parent–enfant ne suffiraient pas à lire un arbre profond ; un parcours explicite évite cette confusion.

Les grands groupes peuvent contenir plusieurs classes. On ne leur impose donc pas automatiquement une étiquette unique. Pour le premier essai, superviser la sortie fine suffit ; les pertes auxiliaires sur les groupes seraient des variantes à isoler.

### Revenir vers les points sans décider trop tôt

Après la lecture, projeter les contextes atomiques `h_τ` par `c_x = Σ_τ w_xτ h_τ`, puis prédire à partir de `f_x` **et** `c_x`. Le chemin local traite aussi les points non couverts. Pour EUVIP, une correction résiduelle du masque gelé constitue une variante possible, avec correction initiale nulle.

Conserver les appartenances pondérées pendant le calcul. Le vote majoritaire du §9.1 garantit une partition pour une sélection donnée ; le répéter à toutes les coupes ne garantit pas des partitions de points emboîtées. Exemple : x vote `(A,B,C)=(.40,.35,.25)`, y vote `(.60,.20,.20)`. Tous deux choisissent A ; lorsque B et C fusionnent, x choisit BC et y reste A. Une affectation exclusive préalable changerait donc la structure étudiée.

**Seuls les projections d’entrée, le lecteur et la sortie sont appris.** Les caractéristiques de l’encodeur restent en cache, sans rétropropagation dans celui-ci. Cette architecture utilise un modèle de fondation ; elle n’est pas une modification de ses attentions internes.

## 4. Les références utiles et leurs limites

**Superpoint Transformer — Robert, Raguet et Landrieu, ICCV 2023.** C’est le meilleur point de départ en segmentation 3D : géométrie des groupes, voisinages et passages entre niveaux. L’étude utilise deux niveaux de partition ; en supprimer un dégrade les résultats, en ajouter davantage n’y apporte pas de gain. Elle prédit une classe par superpoint fin. Notre extension doit lire les événements, préserver les recouvrements et rendre une sortie fine. [Méthode et ablations](https://arxiv.org/html/2306.08045v2).

**Cell Attention Networks — Giusti et al., IJCNN 2023.** Les attentions séparent arêtes partageant un sommet et arêtes appartenant à une même face. C’est le précédent le plus éclairant pour le recollement K = 2. Reprendre cette distinction, avec nos triangles admis ; écarter son pooling qui supprime des arêtes. CAN ne fournit pas notre filtration. [Article](https://arxiv.org/abs/2209.08179).

**Tree-structured Attention with Hierarchical Accumulation — Nguyen et al., ICLR 2020.** Le Transformer traite feuilles, nœuds internes et branches d’un arbre fourni. C’est un précédent de lecture de compositions. Ses expériences portent sur le texte et ses positions exploitent un ordre des mots : elles ne conviennent pas directement à des enfants géométriques non ordonnés. Son attention reste quadratique ; il ne démontre pas un lecteur LiDAR économique. [Article](https://arxiv.org/abs/2002.08046).

**HSA — Amizadeh et al., NeurIPS 2025.** Le régime pertinent ici est la petite tête apprise au-dessus d’embeddings préentraînés, évaluée avec environ 1,2 M de paramètres pour le texte. Il se distingue du remplacement sans entraînement d’attentions RoBERTa. HSA redevient donc un comparateur sérieux pour le lecteur ; son partage de coefficients et l’encodage des niveaux restent des choix à tester. [§4.1 et annexe I de l’arXiv](https://arxiv.org/html/2509.15448v1#S4.SS1).

**PolyhedronNet** suppose des objets déjà décrits par faces polygonales orientées ; il apporte des idées de géométrie locale, pas notre arbre de fusion. **Cellular Transformer** permet des échanges entre types de cellules ; sa hiérarchie de rangs n’est pas celle de nos regroupements. Leur mécanisme peut inspirer une extension, sans nécessiter leur architecture complète. [PolyhedronNet, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d551343f85fcf5e1a230fd393406306e-Abstract-Conference.html), [Cellular Transformer, prépublication 2024](https://arxiv.org/abs/2405.14094v2).

## 5. Ce qu’il faut exporter et tester en premier

Le contrat minimum contient : identifiants des atomes et sommets ; incidences et scores de vote ; naissances ; parents/enfants ; niveaux et témoins des multifusions ; correspondance des caractéristiques ; attributs géométriques avec leur validité. Conserver les incidences nécessaires, plutôt que reconstruire arbitrairement une surface à partir du seul arbre.

**Commencer en K = 2**, dont le manuscrit documente l’implémentation, pour vérifier le lecteur et les recouvrements. Passer à K = 3 après qualification de la construction géométrique et de ses attributs. Le [statut E-HGP consulté](https://github.com/Ludwig-H/E-HGP/blob/764c80b995268cebfc7a38c9d4bb4fb31605d7aa/README.md) distingue source à recouvrements et projection ponctuelle laminaire ; il ne suffit pas à annoncer une chaîne LiDAR complète prête à apprendre.

Comparer le même lecteur, sur **les mêmes atomes et caractéristiques**, avec : géométrie seule ; incidences seules ; arbre seul ; incidences et arbre ; hauteurs canoniques contre vraies hauteurs ; deux coupes contre tous les événements. Même budget de paramètres et de réglages, voie fine identique, test séparé par scène. Ajouter un lecteur des mêmes nœuds sans parcours explicite, puis les arbres témoins des slides avec attributs recalculés. Changer ces arbres change aussi les objets internes : ce contrôle ne sépare pas à lui seul alphabet et grammaire. Voir les [contrôles du lecteur minimal](LECTURE_MULTIECHELLE.md#ce-qui-décidera-de-la-suite).

Mesurer également stabilité sous raréfaction, occultation et changement de portée. Des retours surfaciques dans R³ ne donnent pas automatiquement une densité physique de surface à partir de `r⁻³`. La stabilité de la sémantique ne découle pas du théorème géométrique fini. Les objets filiformes et leurs naissances tardives restent un test prioritaire.

Enfin, compter construction, cache, incidences et profondeur réelle : peu de paramètres ne signifie pas peu de calcul. Pour EUVIP, le graphe publié fournit un cas K = 1 ; passer à des arêtes comme atomes serait une nouvelle construction. Le lien 2D–3D est d’abord **un principe de lecteur commun**, pas un transfert démontré des mêmes poids ni une preuve sur les surfaces LiDAR.
