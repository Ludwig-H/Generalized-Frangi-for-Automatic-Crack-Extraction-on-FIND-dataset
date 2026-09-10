# Notes pour la discussion

## Deux idées distinctes

**Graphormer** apprend un biais indexé par la distance de plus court chemin. Nous reprenons son ajout aux scores, avant softmax. **Notre proposition** utilise la proximité Frangi-graphe dans une attention globale de SAM 2, avec LoRA sur Q et V. La moyenne des proximités entre candidats vers les jetons n’est généralement plus une ultramétrique ; les slides de soutenance en donnent une lecture simplifiée.

## Un poids appris par image

Prendre les caractéristiques visuelles juste avant le bloc guidé. Pour chaque image, les moyenner sur les jetons puis appliquer une couche linéaire et une sigmoïde :

$$\bar h(I)=\frac{1}{N}\sum_{i=1}^{N}h_i(I),\qquad\beta(I)=\mathrm{sigmoid}(w^\top\bar h(I)+b).$$

$$A_H(I)=\mathrm{softmax}(Q_{\mathrm{LoRA}}K^\top/\sqrt{d_h}+\beta(I)B_H(I)).$$

La sortie est partagée entre têtes. Seuls LoRA, `w` et `b` apprennent ; la hiérarchie reste calculée. Le module ajoute `d + 1` paramètres pour des caractéristiques de dimension `d`. Aucun détecteur d’ombres n’est nécessaire : on teste si la supervision de segmentation suffit pour apprendre quand diminuer le guidage.

Pour cette variante, initialiser `w = 0` et `b = log(0.01 / 0.99)` : guidage faible au départ. Une sigmoïde n’atteint pas exactement zéro ; le témoin sans biais impose donc séparément `β = 0`. Le coefficient constant de la slide de soutenance reste, lui, initialisé à zéro comme proposé initialement ; comparer aussi une initialisation commune à `0.01` pour isoler l’effet du conditionnement.

## Décider si cela aide

Comparer, à LoRA et protocole identiques : sans biais, biais hiérarchique à coefficient constant, puis coefficient dépendant de l’image. Conserver le témoin de proximité spatiale pour distinguer l’effet de la hiérarchie de celui d’un simple rapprochement.

Rapporter IoU, ruptures et faux raccords, globalement et sur les images ombragées ou granuleuses. Fixer les choix sur validation, puis tester sur des scènes tenues à l’écart. Un coefficient appris presque nul partout indiquerait que le modèle délaisse le guidage dans ce protocole ; vérifier aussi l’optimisation avant de conclure à son inutilité.

**Limite du premier choix :** un poids global ne distingue pas une ombre localisée d’une zone fiable. Si les résultats le justifient, étudier ensuite des poids locaux `g_i(I) g_j(I)` comme dans la note du dossier parent. Même alors, la confiance aux extrémités ne garantit pas la fiabilité du chemin qui les relie. Aucun de ces gains n’est établi à ce stade.
