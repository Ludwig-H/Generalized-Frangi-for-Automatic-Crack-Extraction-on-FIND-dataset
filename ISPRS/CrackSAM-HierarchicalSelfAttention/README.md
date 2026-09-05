# Guider SAM 2 avec les relations de la hiérarchie Frangi

**Perspective : conserver SAM 2 gelé et tester une organisation explicite de ses échanges.** La [comparaison des pistes](PISTES_SANS_REENTRAINEMENT.md) explique le choix au regard du papier EUVIP, de la soutenance et des articles vérifiés le 5 septembre 2026.

## Le choix pour EUVIP

Conserver chaque token et ses caractéristiques. Utiliser la hiérarchie pour indiquer **à quel niveau deux régions se réunissent**, puis moduler leur attention. Cette expérience ne suppose pas que Frangi segmente mieux que SAM. En monomodal, la hiérarchie organise des observations déjà disponibles ; une modalité supplémentaire peut aussi apporter une observation nouvelle.

![Principe du guidage](figures/guidage_hierarchique.png)

## Du graphe à une relation entre régions

Reprendre les dissimilarités `d_ij` du papier. Sur un graphe fixé, avant sélection de la plus grande composante et élagage par centralité, faire varier le seuil des arêtes. Les composantes fusionnent : on conserve ces événements et leurs niveaux. Le MST de chaque composante suffit à les retrouver.

Dans une même composante, `u_ij` est le plus grand coût sur le chemin du MST entre deux sommets. La relation `κ_ij = 1 − u_ij` mesure la fraction des seuils de `[0,1]` auxquels ils appartiennent au même groupe ; elle vaut zéro entre composantes séparées. Elle exploite toute la filtration. La profondeur d’un parcours et les échelles gaussiennes ne définissent pas cette hiérarchie.

## Un premier essai sans apprentissage

Reporter ces relations sur les tokens, puis ajouter un biais borné au score visuel :

$$
A' = \operatorname{softmax}(QK^\top/\sqrt{d_h} + \alpha B).
$$

`B` est la relation projetée, avec diagonale neutralisée. Les poids de SAM et de son adaptation aux fissures restent gelés ; `α` est choisi sur validation, avec `α = 0` comme référence. Un bloc global de Hiera-L, par exemple le bloc 43, permet une intervention initiale sans modifier la grille. La projection des candidats vers les tokens et les faux raccordements du graphe sont les principaux risques.

[CASS, CVPR 2025](https://arxiv.org/html/2411.17150v3) fournit un précédent de transfert de relations vers une attention visuelle gelée. **La hiérarchie Frangi dans SAM 2 est notre proposition**, pas un résultat de cet article. Comparer impérativement au même support de candidats, à une partition unique, au graphe local et à des arbres témoins.

## La suite vers la soutenance

Un petit lecteur hiérarchique inspiré de [Superpoint Transformer, ICCV 2023](https://arxiv.org/abs/2306.08045), entraîné sur les caractéristiques SAM sauvegardées, prolongerait davantage l’idée « alphabet–grammaire ». Il demanderait un apprentissage nouveau, sans rétropropagation dans SAM.

Les [formulations pour la soutenance](SOUTENANCE.md) résument cette distinction. Les [résultats négatifs](RECHERCHES.md) restent documentés. Le montage est à implémenter ; aucun gain n’a été mesuré. Le schéma vectoriel se régénère avec `python figures/make_figure.py` depuis ce dossier.
