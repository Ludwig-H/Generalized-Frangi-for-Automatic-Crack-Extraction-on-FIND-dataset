# Perspective : donner une organisation aux représentations de SAM

## Une diapositive pour EUVIP

**Titre : « Perspective — guider SAM gelé par une hiérarchie de relations »**

![Principe du guidage](figures/guidage_hierarchique.png)

À afficher :

- SAM fournit les représentations visuelles.
- Le graphe Frangi fournit les groupes et leurs niveaux de fusion.
- Ces relations modulent l’attention ; chaque token est conservé.
- Premier test : aucun apprentissage, un seul bloc modifié.

À dire, environ 40 secondes :

> Notre graphe ne doit pas produire un meilleur masque que SAM pour être utile. Nous proposons d’exploiter la succession de ses regroupements : quels fragments se réunissent tôt, lesquels seulement plus tard. Cette relation pourrait orienter les échanges entre les représentations de SAM, sans modifier ses poids ni fusionner ses tokens. CASS montre qu’un transfert de relations vers une attention visuelle gelée est possible ; l’emploi de notre hiérarchie dans SAM reste à tester.

Référence technique : [Kim et al., CASS, CVPR 2025](https://arxiv.org/html/2411.17150v3). L’article transfère un graphe DINO vers CLIP ; il ne valide pas déjà notre combinaison.

## Le lien avec le programme hiérarchique de la thèse

La diapositive **59** présente l’alphabet et la grammaire ; les **104–107** précisent les unités, leurs recollements, leurs niveaux et les contrôles. Le premier essai EUVIP traduit seulement une partie de ce programme : la filtration des connexions du graphe ordinaire.

Pour aller plus loin : **SAM gelé → caractéristiques sauvegardées → petit lecteur de la hiérarchie → prédiction fine**. [Superpoint Transformer, Robert et al., ICCV 2023](https://arxiv.org/abs/2306.08045) est ici une référence plus directe : une hiérarchie géométrique organise un Transformer. Ce lecteur serait appris séparément ; SAM servirait d’extracteur gelé.

## Deux réponses au jury

**« Est-ce vraiment la hiérarchie qui aide ? »** Comparer au même ensemble de candidats sans niveaux, au graphe local, à une seule partition et à des arbres témoins. SAM seul ne suffit pas comme contrôle.

**« Cela réalise-t-il déjà le programme des K-polyèdres ? »** Le premier montage porte sur le graphe ordinaire. À K ≥ 2, la hiérarchie regroupe des arêtes ou facettes dont les supports peuvent se recouvrir ; une affectation exclusive des pixels ne la représente pas correctement.

Les mécanismes et les limites de HSA, des fusions de tokens et de ces deux voies sont détaillés dans la [comparaison](PISTES_SANS_REENTRAINEMENT.md).
