# Audit de CrackSAM 2 et proposition FrangiGraph-Verified Pyramid LoRA

> Date : 29 juillet 2026
>
> Statut : audit architectural et expérimental, puis recommandation de recherche
>
> Périmètre : implémentations, protocoles et artefacts versionnés dans
> `ISPRS/CrackSAM`, ainsi que l'extracteur de graphe dans `ISPRS/src`

## Résumé exécutif

Les expériences disponibles ne permettent pas de conclure que la géométrie du
graphe Frangi est inutile. Elles montrent en revanche que son **interface avec
SAM 2 était inadaptée**.

La tentative historique ne conditionne pas réellement SAM 2 par un graphe. Elle
réduit une affinité locale à une carte `node_sim_max`, l'interprète comme une
probabilité, la convertit en pseudo-logits, puis l'injecte dans l'interface
`mask_input`. Cette interface représente un masque antérieur à raffiner, et non
une évidence géométrique incertaine. La majorité des pixels est ainsi présentée
comme du fond quasi certain. L'expérience échoue nettement :

- baseline SAM 2-LoRA : IoU macro `0,5675` ;
- pseudo-masque Frangi : IoU macro `0,5563` ;
- différence appariée pondérée : `−0,00985`, IC95
  `[−0,01198 ; −0,00779]`.

La matrice causale montre néanmoins que l'alignement spatial contient de
l'information : le bon prompt est très supérieur aux versions permutées ou
décalées. Elle démontre que `mask_input` est une interface nuisible pour ce
signal. La valeur complémentaire du contenu Frangi via une autre interface
reste, elle, à établir.

Le pilote résiduel raster est plus sûr, mais son gain apparent n'est pas une
preuve d'utilisation de Frangi. Recalculé par image physique, `cracktree200`
contribue seul `+0,005218` à un gain global de `+0,005029`. Sans cette famille,
le résidu vaut `−0,000208` et le gain de la porte devient pratiquement nul. Le
correcteur reçoit aussi les logits et les features SAM ; il peut donc effectuer
une correction de domaine sans exploiter le contenu Frangi.

La meilleure piste est un **FrangiGraph-Verified Pyramid LoRA** :

1. construire une baseline SAM 2.1-LoRA forte et la geler ;
2. conserver un véritable objet graphe — nœuds, arêtes et composantes ;
3. vérifier ses candidats avec les features Hiera échantillonnées aux nœuds ;
4. projeter le graphe vérifié en pyramide aux résolutions `s4`, `s8` et `s16` ;
5. fusionner cette pyramide par des adapters résiduels initialisés à zéro ;
6. réserver une LoRA de decoder à la voie conditionnée ;
7. produire une voie baseline exacte `z0` et une voie candidate `zG`, avec
   retour structurellement garanti à `z0`.

La LoRA doit assurer l'adaptation au domaine fissure. Le canal qui transporte la
géométrie propre à chaque image doit être un adapter conditionnel explicite, pas
une LoRA statique recevant indirectement un pseudo-masque.

## 1. Périmètre et méthode d'audit

L'audit couvre :

- la baseline et la variante Frangi dense ;
- la matrice causale des prompts ;
- le pilote `FrangiGraph-Residual` à cinq folds ;
- la porte logistique ;
- le prototype `verified_local_v1` ;
- l'extraction Frangi et les caches raster ;
- les splits et unités statistiques ;
- les points d'injection LoRA dans SAM 2.

Les résultats publiés ont été relus dans :

- [le rapport des jalons Frangi](../results/frangi_milestone_report/RAPPORT_FRANGI_MILESTONES.md) ;
- [la matrice causale](../results/causal_prompt_matrix_2026-07-20/RAPPORT_MATRICE_CAUSALE.md) ;
- [le diagnostic SafeFrangi](../results/frangi_safe_recommendation/RAPPORT_RECOMMANDATION_SAFE_FRANGI.md) ;
- [la documentation du sélecteur local](07_SELECTIVE_FRANGI_EVIDENCE.md) ;
- [les estimands du pilote](../artifacts/frangigraph_logistic_pilot_e3_seed3407/bootstrap_group_20000/estimands.csv).

La décomposition supplémentaire par famille a été recalculée depuis
[`per_image_gated.csv`](../artifacts/frangigraph_logistic_pilot_e3_seed3407/gate/oof_analysis/per_image_gated.csv),
en suivant une réduction `crop → source_group → famille`, cohérente avec
l'estimand group-balanced du pilote.

## 2. Ce que les expériences ont réellement testé

### 2.1 Pseudo-masque Frangi historique

La variante historique :

1. calcule des réponses Hessiennes multi-échelles ;
2. construit des relations locales entre pixels candidats ;
3. conserve une fraction supérieure des arêtes ;
4. réduit chaque nœud au maximum de similarité d'une arête incidente ;
5. rasterise ce scalaire ;
6. le convertit en logit ;
7. le transmet au prompt encoder standard de SAM 2.

Lorsque `compute_centrality=False`, l'extracteur retourne immédiatement
`node_sim_max`. Le MST, la centralité et les composantes ne participent pas à
l'essai historique :

- [calcul de `node_sim_max`](../../src/graph_extraction.py#L258) ;
- [retour anticipé sans centralité](../../src/graph_extraction.py#L263).

Il serait donc incorrect de décrire cette expérience comme un test de la
topologie complète du graphe Frangi.

### 2.2 Cache raster récent

Le cache actuel appelle l'extracteur avec `compute_centrality=True`, mais empile
seulement sept cartes :

1. similarité ;
2. support ;
3. magnitude Hessienne ;
4. échelle gagnante ;
5. `sin(2θ)` ;
6. `cos(2θ)` ;
7. distance au squelette rasterisé.

La centralité calculée n'est pas transmise. Les identifiants de nœuds, les
arêtes, les composantes, les endpoints, les jonctions et la persistance
multi-échelle ne sont pas sérialisés
([construction des sept canaux](../cracksam2/frangi.py#L238)).

Le modèle résiduel est donc un modèle **raster-conditionné**, pas un modèle de
graphe.

### 2.3 Correcteur résiduel

Le pilote `legacy_raster_v1` combine :

- les sept cartes Frangi ;
- les logits baseline détachés ;
- les features Hiera haute résolution détachées ;
- un CNN résiduel de 60 929 paramètres.

Cette architecture ne force pas la sortie à dépendre du contenu Frangi. Elle
peut apprendre une recalibration de `z0`, une correction de domaine ou un
post-traitement de features SAM
([branches d'entrée du résidu](../cracksam2/residual.py#L191)).

Le prototype `verified_local_v1` ajoute des profils photométriques et une
enveloppe locale révocable. Ses invariants de neutralité sont utiles, mais :

- il opère encore sur des cellules raster ;
- son score est non calibré ;
- la cible apprend la proximité au GT dilaté, pas l'utilité marginale ;
- la correction est impossible hors du support Frangi ;
- aucun résultat GPU n'est disponible.

Ce prototype doit rester un contrôle raster solide, pas être présenté comme la
solution finale.

## 3. Diagnostic des échecs

### 3.1 Erreur de sémantique : affinité versus probabilité

`node_sim_max` mesure la meilleure compatibilité locale entre un nœud et une
arête retenue. Il ne s'agit ni :

- d'une probabilité de fissure ;
- d'une probabilité calibrée de masque ;
- d'une estimation d'utilité par rapport à SAM ;
- d'une preuve d'appartenance à une composante plausible.

Le convertir par `logit(P)` crée donc une sémantique artificielle. Les pixels
faibles ou absents deviennent des valeurs proches de `−11,5`, c'est-à-dire une
affirmation de fond extrêmement forte pour le prompt encoder.

### 3.2 Normalisation relative et absence d'abstention

Chaque Hessienne modale est divisée par son maximum spatial propre
([normalisation](../../src/graph_extraction.py#L37)). Une image faiblement
structurée reçoit donc elle aussi un maximum proche de 1.

Le seuil candidat vaut ensuite 1 % du maximum de l'image et le nombre d'arêtes
retenues est relatif, avec au moins une arête forcée
([sélection relative](../../src/graph_extraction.py#L243)). Les conséquences
sont :

- la densité dépend peu de la confiance absolue ;
- une image médiocre produit presque toujours une pseudo-structure ;
- le signal ne possède pas de véritable état « aucune évidence » ;
- la comparaison des scores entre images est mal définie.

Dans le pilote, la densité Frangi moyenne vaut environ `0,1507` avec un
écart-type d'environ `0,0101`, ce qui confirme qu'elle reflète largement la
politique de sélection relative.

### 3.3 Mauvaise interface SAM 2

Le code encode d'abord l'image par Hiera. Frangi n'est visible qu'au moment du
prompt encoder et du mask decoder
([chemin d'encodage/décodage](../cracksam2/model.py#L240)).

Le `mask_input` standard signifie « estimation de masque à raffiner ». Il est
convolué et réduit à la résolution de l'embedding principal. Cela provoque trois
pertes critiques pour les fissures :

- disparition des lignes subcellulaires ;
- perte de la tangente, de l'échelle et de la largeur ;
- perte des relations explicites entre nœuds et composantes.

`masks=None`, un tenseur nul et un pseudo-masque faible empruntent par ailleurs
des chemins différents. Le no-mask embedding appris n'est pas équivalent à un
tableau de valeurs numériques nulles.

### 3.4 Adaptation LoRA trop restrictive

La fonction locale :

- gèle tout SAM 2 ;
- ajoute des LoRA Q/V dans tous les blocs Hiera ;
- ajoute des LoRA Q/V dans les attentions du mask decoder.

L'upsampling, les hyperréseaux de masque, les convolutions haute résolution et
les autres projections restent gelés
([injection LoRA](../cracksam2/model.py#L126)).

Cette baseline n'est pas une reproduction fidèle de CrackSAM 1. Le modèle
publié entraînait un périmètre différent et utilisait un autre backbone, un
autre decoder et une autre formulation de sortie
([comparaison détaillée](02_BASELINE_COMPARISON.md#pourquoi-le-port-nest-pas-fidèle)).

Une partie du signal attribué à Frangi peut ainsi correspondre à la correction
d'une baseline insuffisamment adaptée, notamment sur certains domaines.

### 3.5 Objectifs d'apprentissage non causaux

L'essai dense emploie BCE et Dice mais aucune supervision explicite de :

- continuité d'arête ;
- existence d'arête ;
- cohérence d'orientation ;
- appartenance à une composante ;
- utilité marginale par rapport à `z0`.

Le pseudo-masque est toujours présent pendant l'apprentissage. Aucun dropout
complet, graphe corrompu ou exemple nul n'apprend au réseau à l'ignorer.

Le sélecteur local récent améliore la sûreté, mais sa cible indique uniquement
si un candidat est proche du GT dilaté
([cible d'évidence](../cracksam2/losses.py#L153)). Une région déjà parfaitement
traitée par SAM reste positive. Ce n'est pas la question pertinente pour un
correcteur sélectif.

### 3.6 Entraînements séparés et coadaptation

La baseline et la variante dense ont été entraînées séparément. Leur différence
finale mélange :

- l'effet instantané du prompt ;
- l'effet des poids LoRA coadaptés ;
- la stochasticité de l'optimisation ;
- la sélection de checkpoints.

La matrice causale ultérieure sépare partiellement ces effets et montre que la
LoRA entraînée avec Frangi apprend surtout à survivre à l'interface :

- effet du prompt sur les poids Frangi : seulement `+0,0029` ;
- effet des poids Frangi sans prompt : `−0,0146` ;
- système conjoint : `−0,0122` sous la baseline.

## 4. Lecture quantitative des résultats

### 4.1 Expérience dense

| Comparaison | ΔIoU macro |
|---|---:|
| Prompt Frangi sur poids baseline versus `None` | `−0,0979` |
| Logits nuls sur poids baseline versus `None` | `−0,1641` |
| Prompt permuté versus `None` | `−0,3452` |
| Prompt décalé versus `None` | `−0,3679` |
| Bon alignement versus prompt permuté | `+0,2473` |
| Bon alignement versus prompt décalé | `+0,2700` |
| Meilleur système Frangi versus baseline | `−0,0122` |

Le bon alignement est très supérieur aux contrôles faux : la sortie dépend donc
fortement de la correspondance spatiale. Cependant, le prompt correct reste
nettement inférieur au chemin sans prompt. L'expérience démontre que cet
encodage est nuisible ; elle ne démontre pas encore que le contenu Frangi
apporterait une utilité complémentaire via une autre interface.

Le meilleur checkpoint Frangi est légèrement supérieur sur la validation
historique, puis inférieur sur les six conditions de test. Les checkpoints
tardifs se dégradent encore. Ce profil est compatible avec une coadaptation ou
un raccourci dépendant du domaine.

### 4.2 Pilote résiduel

Le pilote contient 9 121 crops regroupés en 1 727 images physiques.

| Estimand | Group-balanced | Moyenne par crop |
|---|---:|---:|
| Résidu toujours ouvert | `+0,005029` | `+0,001318` |
| Oracle baseline/candidat | `+0,007869` | `+0,002983` |
| Système avec porte | `+0,002568` | `+0,000484` |

La différence entre les deux agrégations montre l'importance du regroupement
des crops issus de la même image.

#### Décomposition group-first par famille

| Famille | Groupes physiques | Δ résidu | Δ système gated |
|---|---:|---:|---:|
| CFD | 95 | `−0,00937` | `0` |
| forest | 95 | `−0,00895` | `0` |
| noncrack | 104 | `−0,00274` | `0` |
| DeepCrack | 141 | `−0,00152` | `+0,00044` |
| Volker | 90 | `−0,00013` | `0` |
| CRACK500 | 441 | `−0,00004` | `0` |
| Eugen | 5 | `+0,00062` | `0` |
| Rissbilder | 182 | `+0,00132` | `≈0` |
| GAPS384 | 264 | `+0,00276` | `−0,00023` |
| Sylvie | 148 | `+0,00657` | `0` |
| cracktree200 | 162 | `+0,05563` | `+0,02737` |

`cracktree200` contribue `+0,005218` au gain global `+0,005029`, soit plus de
100 % du gain net. Sans cette famille :

- résidu toujours ouvert : `−0,000208` ;
- système gated : environ `−2,6 × 10⁻⁹`.

La baseline `cracktree200` a un IoU moyen groupé d'environ `0,040` et une
fraction de foreground prédite d'environ `0,00075`. Le correcteur peut donc
apprendre un régime de récupération spécifique à cette famille.

Les associations entre qualité Frangi et gain sont faibles :

| Variable | Pearson avec ΔIoU | Spearman avec ΔIoU |
|---|---:|---:|
| Similarité Frangi moyenne sur support | `−0,023` | `−0,010` |
| Densité Frangi | `+0,044` | `+0,072` |

La porte atteint une AUC globale d'environ `0,755`, mais elle utilise aussi des
variables génériques de la baseline et du désaccord. Son coefficient standardisé
pour la similarité Frangi est légèrement négatif. Elle semble surtout
reconnaître un régime de domaine, pas une géométrie utile de manière générale.

Enfin, l'oracle `+0,00787` reste sous le critère exploratoire `+0,01` fixé dans
la [feuille de route existante](04_IMPLEMENTATION_ROADMAP.md#condition-pour-passer-à-la-suite-3).

## 5. Limites du protocole et de l'inférence causale

### 5.1 Recouvrement des images physiques

Le manifeste documente des recouvrements historiques entre les listes Khanhha :

| Paires de listes | Groupes physiques communs |
|---|---:|
| train / test | 730 |
| train / validation | 325 |
| validation / test | 248 |

Il s'agit souvent de crops différents d'une même image source, non
nécessairement de fichiers identiques
([manifeste du protocole](../protocol/frangigraph_v1/manifest.json#L7)).

Les comparaisons restent utiles pour reproduire le protocole historique, mais la
partie Khanhha ne constitue pas un test indépendant au niveau physique.

### 5.2 Baseline non cross-fittée

Les résidus du pilote sont produits hors fold, mais la baseline a été entraînée
sur l'intégralité de `train.txt`. Le système complet n'est donc pas OOF
([limite documentée](../protocol/frangigraph_v1/README.md#L29)).

### 5.3 Porte sans test indépendant

Les coefficients de la porte sont ajustés sur les folds 0–3. Le fold 4 sert à
choisir son seuil. Les performances agrégées sur ces mêmes folds sont
exploratoires :

- folds 0–3 : résultats apparents sur les données ayant ajusté les coefficients ;
- fold 4 : résultat descriptif sur les données ayant choisi le seuil ;
- aucun outer holdout ne teste la combinaison complète.

### 5.4 Incertitudes trop optimistes

Les IC historiques de l'expérience dense et de la matrice causale
rééchantillonnent les crops plutôt que les images physiques. Les signes des
effets négatifs très grands sont robustes, mais la largeur des intervalles est
trop optimiste pour de petits effets.

### 5.5 Étendue limitée du pilote

Le pilote résiduel utilise :

- trois époques ;
- une seule seed ;
- aucune augmentation géométrique ;
- aucun contrôle réentraîné à capacité égale ;
- aucun benchmark externe ;
- aucun test d'ombres appariées.

Il doit être traité comme une sonde de faisabilité, non comme une validation
d'architecture.

## 6. Architecture recommandée

### 6.1 Principe général

L'architecture proposée sépare trois fonctions :

1. **sémantique de fissure** : SAM 2.1 adapté par LoRA ;
2. **proposition géométrique** : vrai graphe Frangi ;
3. **arbitrage** : vérificateur de graphe utilisant les features SAM.

```text
                                      ┌── decoder baseline gelé ───► z0
Image ──► SAM 2.1 Hiera-LoRA/FPN ──► F4,F8,E16
  │                                   │
  └──► graphe Frangi V,E ─► GNN vérificateur
                             │
                             └──► pyramide G4,G8,G16
                                      │
                           fusers résiduels zéro-init
                                      │
                           decoder + LoRA graphe ─────► zG

Sortie : z = where(M(G, F, z0), zG, z0)
```

Une seule passe Hiera suffit. Les deux voies réutilisent les mêmes features et
ne doublent que le décodage, beaucoup moins coûteux que l'encodeur.

`F4` et `F8` désignent les deux features haute résolution exposées au mask
decoder local. `E16` désigne l'embedding principal après FPN ; le niveau Hiera
plus profond contribue à cet embedding par le FPN, mais n'est pas fourni comme
entrée nodale indépendante dans le MVP.

### 6.2 Baseline `B1`

Conserver la baseline historique `B0` pour la reproductibilité, puis établir une
baseline actuelle `B1` :

- checkpoint SAM 2.1 Hiera-L ;
- LoRA Q/V rang 4 dans Hiera comme point initial ;
- entraînement réel du petit mask decoder ;
- apprentissage au minimum de `conv_s0`, `conv_s1`, de l'upsampling et des
  hyperréseaux ;
- sortie binaire prompt-free ;
- loss foreground mieux adaptée aux structures rares ;
- sélection de checkpoint sur groupes physiques disjoints.

Le dépôt officiel fournit les checkpoints SAM 2.1 et le code de fine-tuning
([SAM 2 officiel](https://github.com/facebookresearch/sam2)).

Après validation, `B1` est gelée. Sa LoRA devient la **LoRA domaine**. Elle
produit la sortie sûre `z0`.

### 6.3 FrangiGraph V3

Le nouvel extracteur doit publier un objet graphe explicite.

#### Attributs de nœud

- coordonnées normalisées ;
- réponse Hessienne absolue avec normalisation `σ²` ;
- valeurs propres ou descripteurs équivalents ;
- tangente `sin(2θ), cos(2θ)` ;
- échelle et largeur estimée ;
- persistance entre échelles ;
- polarité sombre/claire ;
- profils transverses par rayon, sans réduire tous les rayons à un maximum ;
- degré, endpoint, jonction et identifiant de composante.

#### Attributs d'arête

- longueur et direction ;
- accord entre la direction de l'arête et les tangentes des extrémités ;
- termes séparés `shape`, `intensity` et `alignment` ;
- courbure locale ;
- évolution d'échelle ;
- statistiques photométriques le long et au travers de l'arête ;
- indicateur de gap ou de pont hypothétique.

#### Attributs de composante

- longueur totale ;
- nombre d'endpoints et de jonctions ;
- distribution des courbures ;
- cohérence d'orientation et d'échelle ;
- persistance multi-échelle ;
- confiance absolue ;
- proximité d'autres composantes.

La sélection doit autoriser un graphe réellement vide. Une normalisation robuste
apprise sur le train remplace les maxima par image. Le MST peut servir de
descripteur, mais ne doit jamais imposer la topologie avant vérification
sémantique.

### 6.4 Vérificateur de graphe

Pour chaque nœud, échantillonner par interpolation bilinéaire :

- `F4` pour la localisation et les bords fins ;
- `F8` pour la largeur et le contexte local ;
- `E16` pour le contexte sémantique ;
- les logits ou probabilités baseline autour du nœud ;
- les profils photométriques transverses.

Un GNN edge-aware ou un petit Graph Transformer de deux à trois couches prédit :

- `p_valid_node` ;
- `p_valid_edge` ;
- `p_valid_component` ;
- `p_add_utility` ;
- `p_harm_risk`.

Cette séparation entre géométrie dense et prédiction topologique est cohérente
avec [SAM-Road](https://arxiv.org/abs/2403.16051), qui utilise les embeddings SAM
pour estimer l'existence des arêtes d'un réseau.

### 6.5 Rasterisation vérifiée et fusion pyramidale

Après vérification seulement, les arêtes pondérées sont projetées en champs
multi-échelles :

- confiance de squelette ;
- `sin(2θ)` et `cos(2θ)` ;
- échelle ou largeur ;
- distance au squelette normalisée par la largeur locale ;
- heatmaps endpoints et jonctions ;
- embedding de composante diffusé autour de ses arêtes.

Pour `l ∈ {4, 8, 16}` :

\[
\widetilde H_l
=
H_l
+
\gamma_l\,
g_l(H_l,G_l)
\odot
A_l(H_l,G_l),
\qquad
\gamma_l(0)=0.
\]

Les dernières projections sont initialisées à zéro. Un masque explicite doit
forcer la branche adapter à un tenseur exactement nul lorsque le graphe est
absent ou rejeté. Cet invariant se teste au niveau des features, mais le
fallback final ne doit pas dépendre d'une seconde exécution numériquement
identique du decoder.

L'adaptation multi-niveaux est soutenue par le précédent
[SAM2-Adapter](https://arxiv.org/abs/2408.04579). Elle est particulièrement
importante ici : les fissures fines dépendent des features haute résolution
`s4/s8`, que le pseudo-masque historique ne pouvait pas modifier directement.

### 6.6 Rôle de LoRA

Deux ensembles de paramètres doivent être séparés :

#### LoRA domaine

- incluse dans `B1` ;
- apprend l'apparence générique des fissures ;
- active dans `z0` et `zG` ;
- gelée pendant l'apprentissage du graphe.

#### LoRA graphe

- ajoutée au mask decoder de la voie candidate ;
- initialisée à zéro ;
- désactivée dans la voie baseline ;
- entraînée avec le GNN et les fusers ;
- Q/V rang 4 comme MVP, puis ablation Q/K/V/O seulement si nécessaire.

Une LoRA standard est un changement de poids fixe. Elle ne peut être le canal
géométrique que si le graphe entre explicitement dans les activations. Dans le
MVP, ce rôle appartient au GNN et aux fusers.

Une extension ultérieure pourrait spatialement moduler la branche bas rang :

\[
\Delta h_l(p)
=
m_l(p,G)\,
B_lA_lh_l(p).
\]

Cette « Spatial Graph-LoRA » guiderait littéralement l'amplitude de la LoRA par
la géométrie. Elle ne doit être testée qu'après démonstration de la valeur du
graphe, car elle complique la stabilité et l'attribution causale.

### 6.7 Sortie sûre à deux voies

La baseline calcule et conserve le tenseur `z0`. La voie conditionnée produit
`zG`. Une porte locale dure par composante ou par bande d'arête construit `M`.
À l'inférence, le fallback sélectionne directement le tenseur `z0` déjà calculé,
par exemple avec `torch.where`, au lieu de redécoder une seconde « baseline » :

\[
z(x)
=
\begin{cases}
z_G(x), & M(x)=1,\\
z_0(x), & M(x)=0.
\end{cases}
\]

Les invariants requis sont :

- graphe vide : le tenseur retourné est directement `z0` ;
- composante rejetée : les pixels concernés sont sélectionnés directement dans
  `z0` ;
- correction signée autorisée seulement dans une bande calibrée ;
- aucune dégradation silencieuse par un biais du correcteur.

La voie candidate doit être déterministe à l'inférence : mode évaluation,
dropout désactivé et kernels déterministes lorsque l'environnement le permet.

Le recours à la mémoire SAM 2 n'est pas recommandé pour le MVP. Sur une image
statique, une pseudo-mémoire issue de la même image risquerait surtout de
recopier les faux positifs Frangi à faible résolution.

## 7. Supervision recommandée

### 7.1 Baseline entièrement OOF

Les cibles d'utilité doivent être construites avec des prédictions baseline
issues d'un modèle n'ayant jamais vu le groupe physique correspondant.

Pour chaque outer fold :

1. entraîner `B1_f` sur les autres groupes ;
2. prédire `z0_f` sur le fold tenu à l'écart ;
3. construire les cibles graphe depuis `z0_f` et le GT ;
4. entraîner le vérificateur uniquement sur les folds autorisés ;
5. calibrer les seuils sur un fold distinct ;
6. évaluer le système complet sur un outer holdout.

### 7.2 Deux notions séparées

Le vérificateur doit posséder au moins deux objectifs :

#### Validité sémantique

« Cette arête correspond-elle plausiblement à une fissure ? »

Une cible continue peut combiner :

- proportion de l'arête dans le GT dilaté ;
- distance au squelette GT ;
- accord de tangente avec le squelette ;
- cohérence de largeur.

#### Utilité marginale

« Accepter cette arête améliore-t-il la baseline ? »

Les labels doivent distinguer :

- faux négatif baseline couvert par une vraie arête ;
- région déjà correcte, à ignorer ;
- fausse arête qui ajouterait des faux positifs ;
- composante qui améliore localement IoU ou clDice ;
- composante à risque sur masque vide.

### 7.3 Fonction objectif

Une formulation initiale raisonnable est :

\[
\mathcal L
=
\mathcal L_{\text{focal/tversky}}
+
\lambda_d\mathcal L_{\text{dice-fg}}
+
\lambda_c\mathcal L_{\text{clDice}}
+
\lambda_e\mathcal L_{\text{edge}}
+
\lambda_o\mathcal L_{\text{orientation}}
+
\lambda_s\mathcal L_{\text{safety}}.
\]

`clDice` combine précision et rappel topologiques pour les structures tubulaires
([article clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html)).
Son poids doit rester modéré : une contrainte topologique trop forte peut
sur-connecter des discontinuités réellement présentes.

La pénalité de sûreté compare la loss candidate à celle de `z0` par image :

\[
\mathcal L_{\text{safety}}
=
\max\left(
0,
\mathcal L(z_G,y)-\mathcal L(z_0,y)+m
\right).
\]

Avec `m=0`, cette pénalité vise la non-dégradation. Avec `m>0`, elle impose
explicitement que la candidate améliore la baseline d'au moins cette marge ;
ce choix plus strict doit être annoncé et validé séparément.

### 7.4 Corruptions nécessaires

Chaque entraînement doit inclure :

- graphe entièrement nul ;
- dropout de composantes ;
- suppression et ajout d'arêtes ;
- permutation d'arêtes ;
- décalage spatial ;
- jitter des nœuds ;
- graphe aléatoire à couverture identique ;
- suppression des attributs Frangi tout en gardant la capacité du modèle.

Un objectif de ranking peut demander que le graphe correct obtienne une
meilleure loss que sa version corrompue. Cela force l'usage de la bonne
géométrie, au lieu de laisser le réseau exploiter uniquement la parcimonie ou
les features SAM.

### 7.5 Ombres

Les ombres sont une hypothèse plausible de faux positif, mais le dépôt ne
démontre pas qu'elles expliquent la majorité des pertes. Le protocole doit aussi
considérer joints, marquages, textures, bords d'objets et changements
d'illumination.

Pour chaque augmentation d'ombre :

- modifier l'image ;
- recalculer intégralement Frangi ;
- garder la version propre et ombrée dans le même groupe ;
- mesurer les faux positifs dans une bande autour de la frontière ;
- mesurer le rappel du squelette lorsque l'ombre traverse la fissure.

## 8. Ablations décisives

### 8.1 Contrôle à capacité égale

Entraîner avec les mêmes folds, seeds, paramètres et budget :

1. `SAM-only` : `z0` et features Hiera, sans Frangi ;
2. raster Frangi correct ;
3. raster décalé ;
4. raster permuté ;
5. raster aléatoire à même couverture ;
6. vrai graphe correct ;
7. vrai graphe sans attributs Frangi.

Ce test doit précéder toute affirmation sur la valeur du contenu Frangi.

### 8.2 Valeur propre de la topologie

Comparer :

- vrai graphe ;
- mêmes nœuds et mêmes attributs, arêtes recâblées ;
- nœuds indépendants sans message passing ;
- graphe de voisinage purement spatial ;
- raster produit depuis le vrai graphe.

Un GNN n'est justifié que si les vraies arêtes battent les contrôles qui
préservent les nœuds et la capacité.

### 8.3 Lieu d'injection

Comparer :

- `s16` seulement ;
- `s8+s16` ;
- `s4+s8+s16` ;
- tokens de graphe dans le decoder ;
- fusion pyramidale plus tokens ;
- `mask_input` historique comme contrôle négatif.

### 8.4 LoRA

Comparer après gel de l'architecture :

- aucune LoRA graphe ;
- Q/V rang 4 ;
- Q/K/V/O rang 4 ;
- Spatial Graph-LoRA ;
- entraînement du petit decoder candidat.

Le rang ne doit pas devenir le premier axe de recherche. L'interface et la
représentation sont actuellement les facteurs dominants.

### 8.5 Généralisation par famille

Les résultats doivent inclure :

- leave-one-family-out ;
- entraînement sans `cracktree200`, test sur `cracktree200` ;
- score global avec et sans `cracktree200` ;
- moyenne macro par famille ;
- au moins trois seeds pour les finalistes.

## 9. Métriques et critères de décision

### 9.1 Unité statistique

- unité primaire : image physique ;
- toutes les découpes et perturbations dans le même cluster ;
- bootstrap par groupe physique ;
- reporting secondaire par crop uniquement à titre descriptif.

### 9.2 Métriques

- IoU et Dice ;
- clDice ou clIoU ;
- précision et rappel du squelette ;
- erreur d'endpoints et de jonctions ;
- continuité par composante ;
- P05 du delta par image ;
- taux de pertes `< −0,05` ;
- taux de gains `> +0,005` ;
- couverture de la porte ;
- calibration et courbe risque-couverture ;
- temps, mémoire et nombre de paramètres.

### 9.3 Conditions de passage

Poursuivre vers une LoRA spatialement conditionnée seulement si :

1. le vrai graphe bat le contrôle `SAM-only` ;
2. il bat le raster à capacité comparable ;
3. il bat le graphe recâblé ;
4. l'IC95 groupé du gain est au-dessus de zéro ;
5. le gain reste positif sans `cracktree200` ;
6. la majorité des familles ne se dégrade pas ;
7. le graphe nul restitue exactement `z0` ;
8. le taux de pertes sévères n'augmente pas ;
9. un holdout dédupliqué confirme le résultat.

Si le vrai graphe ne bat pas les contrôles recâblés ou sans Frangi, il faudra
abandonner Frangi comme entrée apprise et le conserver seulement comme outil
diagnostique ou générateur de cas difficiles.

## 10. Feuille de route recommandée

### Phase A — preuve causale à faible coût

1. Évaluer les checkpoints résiduels existants sous Frangi correct,
   `no_evidence`, décalé et permuté.
2. Réentraîner un contrôle `SAM-only` à capacité égale.
3. Publier la décomposition avec et sans `cracktree200`.
4. Ne pas interpréter le passage du même checkpoint sous `no_evidence` comme une
   preuve causale complète.

### Phase B — baseline `B1`

1. Migrer à SAM 2.1.
2. Adapter le decoder haute résolution.
3. Construire des outer folds par groupe physique.
4. Produire les logits baseline OOF.
5. Geler `B1`.

### Phase C — FrangiGraph V3

1. Sérialiser nœuds, arêtes, composantes et attributs.
2. Ajouter une vraie abstention absolue.
3. Valider les transformations géométriques et les orientations.
4. Tester la stabilité multi-échelle.
5. Versionner le contrat et les empreintes du cache.

### Phase D — MVP graphe vérifié

1. GNN léger ;
2. échantillonnage `F4/F8/E16` ;
3. rasterisation vérifiée ;
4. fusers zéro-init ;
5. voie `z0/zG` ;
6. LoRA graphe Q/V dans le decoder ;
7. objectifs validité et utilité séparés.

### Phase E — confirmation

1. contrôles réentraînés à capacité égale ;
2. arêtes recâblées ;
3. leave-one-family-out ;
4. trois seeds ;
5. propre/ombre apparié ;
6. benchmark externe et holdout dédupliqué ;
7. gel de l'architecture avant test final.

## 11. Recommandation finale

La priorité ne doit pas être d'ajouter davantage de LoRA autour du
pseudo-masque historique. Cette voie a déjà été invalidée.

La meilleure architecture à poursuivre est :

> **une baseline SAM 2.1-LoRA prompt-free forte et gelée, complétée par un vrai
> graphe Frangi vérifié sémantiquement avec les features Hiera, fusionné par des
> adapters résiduels aux résolutions `s4/s8/s16`, puis décodé par une voie LoRA
> géométrique révocable.**

Cette architecture répond directement aux quatre défaillances observées :

- elle conserve la topologie au lieu de la réduire à `node_sim_max` ;
- elle ne confond plus affinité et probabilité de masque ;
- elle injecte la géométrie aux résolutions adaptées aux fissures fines ;
- elle garantit une sortie baseline exacte lorsque Frangi est inutile ou
  trompeur.

Si une seule nouvelle campagne doit être financée, elle doit comparer, dans un
outer split où `cracktree200` est tenu à l'écart :

1. `SAM-only` à capacité égale ;
2. raster Frangi ;
3. vrai graphe ;
4. graphe recâblé.

C'est le test minimal capable de déterminer si la géométrie Frangi-graphe
apporte réellement une information complémentaire à SAM 2.

## Références externes

1. Ravi et al., [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024.
2. Meta AI, [dépôt officiel SAM 2 et checkpoints SAM 2.1](https://github.com/facebookresearch/sam2).
3. Chen et al., [SAM2-Adapter](https://arxiv.org/abs/2408.04579), 2024.
4. Hetang et al., [SAM-Road](https://arxiv.org/abs/2403.16051), 2024.
5. Shit et al., [clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html), CVPR 2021.
