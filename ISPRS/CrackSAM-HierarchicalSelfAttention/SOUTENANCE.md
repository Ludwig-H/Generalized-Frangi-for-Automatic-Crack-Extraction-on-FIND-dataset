# Une perspective en deux diapositives

## 1 — SAM voit déjà les détails ; donnons-lui leur organisation

**À afficher**

- Nos essais : ajouter des cartes de Frangi, d’orientation et de dérivées n’apporte pas de gain convaincant.
- Une autre information explicite : quels morceaux appartiennent aux mêmes groupes, du détail à l’ensemble.
- La hiérarchie du graphe peut guider les échanges entre les représentations visuelles de SAM 2.

**À dire — environ 40 secondes**

> Nos premiers essais suggèrent que rajouter des descripteurs visuels n’est pas la bonne voie. SAM dispose déjà de représentations très riches. Ce que nous proposons maintenant, c’est de lui donner une organisation : quels morceaux se regroupent, et à quel niveau. L’arbre couvrant du graphe Frangi permet de retrouver cette succession de regroupements. Nous passons ainsi d’une description locale de l’image à une structure qui peut organiser les échanges dans le modèle.

Cette diapositive prolonge la page PDF **64** (numéro affiché **55**). Pour rester défendable, remplacer « SAM a déjà tout vu » par **« Dans nos essais, les descripteurs ajoutés n’apportent pas de gain convaincant »**. Réserver les limites du protocole aux questions du jury ; elles sont documentées dans [RECHERCHES.md](RECHERCHES.md).

## 2 — La hiérarchie module l’attention

**À afficher**

![Schéma de la perspective](figures/guidage_hierarchique.png)

- Deux régions se retrouvent dans un groupe commun : son niveau fournit un biais à l’attention.
- Premier essai : un seul bloc modifié, trois coefficients appris, initialisés à zéro.
- Question décisive : la hiérarchie Frangi aide-t-elle davantage qu’une seule partition ou un arbre témoin ?

**À dire — environ 45 secondes**

> Pour chaque paire de régions, on regarde à quel niveau elles se réunissent dans l’arbre. On utilise cette relation pour moduler le score d’attention que SAM calcule déjà. Chaque paire conserve son propre score visuel ; le modèle apprend à y ajouter la préférence hiérarchique. Le premier essai peut être très limité, avec trois coefficients dans un seul bloc. Pour être convaincant, il faudra montrer que les vrais regroupements apportent davantage qu’une partition unique ou qu’un arbre dont on a mélangé les feuilles. Aujourd’hui, c’est une perspective, pas un gain acquis.

Cette diapositive peut remplacer la page PDF **65** (numéro **56**). Le [SVG](figures/guidage_hierarchique.svg) est réutilisable dans la présentation.

## Trois réponses courtes pour le jury

**« SAM n’a-t-il pas déjà une hiérarchie ? »** Hiera organise plusieurs résolutions. Ici, les groupes dépendent de l’image et des relations du graphe Frangi. Leur intérêt supplémentaire reste à mesurer.

**« Frangi doit-il mieux segmenter que SAM ? »** Le pari porte sur l’utilité de certaines relations entre régions. Il n’exige pas que le masque Frangi soit meilleur. Des relations erronées peuvent cependant nuire.

**« Pourquoi ne pas remplacer l’attention par HSA ? »** HSA est une possibilité plus contraignante. Le biais additif permet un premier test ciblé et retrouve le modèle initial lorsque ses coefficients sont nuls. Cela ne réfute pas HSA.

## Alléger les diapositives de secours

La page PDF **111** peut conserver l’explication de SAM, en déplaçant l’entrée hiérarchique vers l’attention de l’encodeur. Remplacer **112–114 et 119** par le montage et les contrôles du [README](README.md). Les anciennes mesures sur arbres synthétiques ne justifient ni une impossibilité générale, ni un plafond de segmentation. Un essai utilisant la vérité terrain resterait un diagnostic d’interface, pas une borne universelle.
