# Perspective : apprendre ce qui compte à chaque regroupement

**Premier lecteur recommandé : conserver les différences entre enfants et parents, puis apprendre leur importance selon la géométrie et le niveau de fusion.** L’encodeur reste gelé. Cette variante précise le [lecteur de polyèdres](VOIE_POLYEDRES.md) ; l’attention entre enfants demeure un comparateur plus expressif.

## Pourquoi cette piste

Une moyenne décrit ce que les éléments partagent ; leurs écarts décrivent ce qui les distingue. Garder les deux permet de parcourir les regroupements sans effacer systématiquement les détails. L’hypothèse devient précise : **la géométrie des compositions aide à choisir les échelles utiles à la prédiction**.

Une hiérarchie déterministe n’ajoute pas une observation indépendante des données qui la construisent. Elle organise des relations que le modèle pourrait autrement devoir apprendre avec davantage d’exemples ou de calcul. La justification porte sur cette organisation et sa généralisation.

## Une référence particulièrement proche des polyèdres

[Saito, Schonsheck et Shvarts, *Multiscale transforms for signals on simplicial complexes*, 2024](https://link.springer.com/article/10.1007/s43670-023-00076-4), construit des représentations multirésolution de signaux sur **arêtes, faces et simplexes**, à partir d’une hiérarchie de leurs identités. Leurs §§4.2–5 autorisent d’autres bipartitions et des poids. Leur dimension κ correspond à **K−1** dans la thèse.

Nous conserverions notre filtration, plutôt que leur partitionnement spectral. Leur construction binaire est remplacée ci-dessous par des différences enfant–parent adaptées aux multifusions. C’est notre proposition, pas une architecture SAM démontrée par cet article.

[Gavish, Nadler et Coifman, ICML 2010](https://www.weizmann.ac.il/math/Nadler/sites/math.Nadler/files/publications/wavelets_trees_p18.pdf), fournit le précédent des ondelettes sur arbres, y compris à plusieurs enfants. [Haar Graph Pooling, ICML 2020](https://proceedings.mlr.press/v119/wang20m.html), fournit un précédent neural, mais son pooling supprime les détails : nous les gardons.

## Le mécanisme minimal

Partir des caractéristiques atomiques `z_τ` et de masses **positives, fixes**, par exemple les masses de vote `m_τ` du §9.1 lorsqu’elles sont positives. Pour un nœud v, sommer les masses de ses atomes descendants et calculer leur moyenne μᵥ. Les appartenances portent sur les identités d’atomes ; leurs supports ponctuels peuvent se recouvrir.

Pour chaque enfant v de parent p, conserver `δ_v = μ_v − μ_p`. Pour une feuille τ de racine R :

$$
z_\tau=\mu_R+\sum_{v\in(R\rightsquigarrow\tau)\setminus\{R\}}\delta_v.
$$

La somme se simplifie exactement. Il n’est donc pas nécessaire de perdre les détails pour décrire les groupes. Cette représentation est redondante ; nous ne prétendons pas construire ici une base orthonormale.

Un petit réseau partagé apprend ensuite un gain γₚ, commun aux enfants d’un événement p :

$$
\widetilde z_\tau=\mu_R+
\sum_{v\in(R\rightsquigarrow\tau)\setminus\{R\}}
\gamma_{p(v)}\,\delta_v.
$$

Le gain dépend des attributs disponibles du polyèdre et de ses recollements, du niveau numérique déclaré et du désaccord entre enfants. Une première version scalaire suffit ; `γ_p = 2 sigmoid(g_θ(...))`, avec dernière couche initialisée à zéro, démarre à **γₚ = 1**, donc à l’identité atomique. Le nombre de couches apprises ne dépend pas de la profondeur de l’arbre.

Le calcul est indépendant de l’ordre des enfants si les entrées de `g_θ` le sont. Le gain commun conserve la moyenne de chaque racine. Il reste limité : il module des écarts existants, sans apprendre toutes les interactions entre enfants.

Les caractéristiques gelées et les moyennes fixes peuvent être pré-calculées. Deux parcours calculent moyennes et sommes descendantes en `O(nombre de liens × canaux)`, hors géométrie et réseau des gains. Éviter de matérialiser toutes les relations ancêtre–feuille ; la profondeur peut encore limiter la parallélisation.

## Le contrôle qui révèle un faux usage de la hiérarchie

Si tous les gains valent une même constante γ :

$$
\widetilde z_\tau=\mu_R+\gamma(z_\tau-\mu_R).
$$

**Les regroupements intermédiaires disparaissent du calcul.** Transmettre tout l’arbre ne suffit donc pas. Il faut comparer gains constants et gains variables selon les événements.

Exemple arithmétique, avec caractéristiques `(a,b,c,d)=(0,1; 0,3; 0,7; 0,9)` et masses unitaires :

| Groupements | Tous les gains à 1 | Gain de la racine à 1, ceux des deux groupes à 0 |
|---|---|---|
| `(a,b)` et `(c,d)` | `(0,1; 0,3; 0,7; 0,9)` | `(0,2; 0,2; 0,8; 0,8)` |
| `(a,c)` et `(b,d)` | `(0,1; 0,3; 0,7; 0,9)` | `(0,4; 0,6; 0,4; 0,6)` |

Zéro est ici un gain imposé pour l’illustration, à la limite du paramétrage sigmoid. Ces valeurs illustrent un mécanisme ; elles ne sont pas des prédictions de segmentation.

## Conserver la géométrie et la sortie fine

Le retour par W n’est pas inversible, même si le parcours atomique l’est à γ=1. Pour EUVIP, une première correction peut partir de `Δc_x = Σ_τ w_xτ(ẑ_τ − z_τ)` et s’ajouter aux logits de la baseline par une projection linéaire sans biais, initialisée avec des poids non nuls. À γ=1, cette correction est exactement nulle ; mettre aussi la projection à zéro bloquerait le démarrage des deux apprentissages. La voie fine traite aussi les points hors graphe. Le choix des masses nulles reste explicite.

Avec `m_τ = Σ_x w_xτ` et cette projection commune, la somme des corrections de logits est nulle : ce premier lecteur redistribue l’évidence mais ne corrige pas son biais moyen global. Cette limite ne s’applique pas à un décodeur arbitraire et doit être distinguée d’un échec de la hiérarchie.

**Le dendrogramme ne décrit pas toutes les naissances de connecteurs.** Une arête peut fermer un cycle sans fusionner de composantes. De même, un connecteur peut enrichir un polyèdre déjà formé. Pour lire cette géométrie, exporter leurs naissances ou des statistiques d’incidence selon l’échelle ; les seuls témoins des fusions ne suffisent pas.

Conserver masse de vote, nombre d’atomes et étendue comme grandeurs distinctes. Une moyenne softmax peut confondre deux et quatre enfants identiques ([Zhang et Xie, IJCAI 2020](https://www.ijcai.org/proceedings/2020/194)). Cela justifie des attributs explicites, sans supposer que la densité brute LiDAR soit un indice sémantique stable.

## Ce qui décidera de la suite

Comparer successivement : correction locale sans arbre ; gain constant ; gains par événement ; composition avec portes ; attention hiérarchique. [Child-Sum Tree-LSTM, ACL 2015](https://aclanthology.org/P15-1150/), est le précédent pour une composition apprise d’enfants non ordonnés. [Superpoint Transformer, ICCV 2023](https://arxiv.org/abs/2306.08045), reste la référence pour l’attention géométrique en segmentation 3D.

Séparer trois questions : mêmes nœuds lus avec ou sans organisation relationnelle explicite ; mêmes relations avec hauteurs réelles ou neutralisées ; autres regroupements avec leurs attributs recalculés. Ce dernier contrôle change aussi les objets internes. Donner toutes les appartenances emboîtées permet déjà de reconstruire l’arbre : retirer seulement ses arêtes ne supprime pas toute information hiérarchique.

Commencer en K=2 ; isoler frontières, objets fins et raréfaction. L’identité, le cas constant, la conservation d’énergie pondérée de la décomposition **avant les gains** et l’invariance à l’ordre des enfants ont été vérifiés sur exemples arithmétiques avec masses inégales et multifusions. **L’intérêt sémantique reste la question expérimentale.**
