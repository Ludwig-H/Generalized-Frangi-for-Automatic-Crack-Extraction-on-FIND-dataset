# FrangiGraph-SelectiveResidual — vérifier avant de corriger

## Décision et statut

Frangi ne doit plus être interprété comme une vérité dense, ni même comme une
preuve uniformément fiable à l'échelle d'une image. Le prototype l'emploie
comme un **générateur de candidats conçu pour privilégier le rappel**. Un
vérificateur local estime ensuite quelle partie de ce signal est proche d'une
annotation de fissure avant d'autoriser une correction de SAM 2.

Cette architecture est un **prototype expérimental** : elle teste l'hypothèse
qu'une sélection locale peut mieux résister aux réponses Frangi provoquées par
les ombres. Une même image peut contenir une fissure et une frontière d'ombre
trompeuse, ce qu'une décision globale ne peut pas séparer spatialement. Aucun
résultat ne démontre encore que le prototype résout ce cas.

## Méthode implémentée : `verified_local_v1`

La baseline gelée produit les logits `z0` et les features Hiera haute
résolution. Le cache existant fournit les sept cartes Frangi. Trois descripteurs
photométriques de **luminance** sont calculés en ligne, sans modifier le cache
ni utiliser le masque vrai à l'inférence. Il ne s'agit pas de trois profils RGB
indépendants : les canaux RGB sont d'abord convertis en une luminance scalaire.

À la normale Hessienne reconstruite par `sin(2θ), cos(2θ)`, on échantillonne la
luminance de part et d'autre du candidat, à plusieurs rayons. En notant

\[
d_- = Y(x-rn)-Y(x), \qquad d_+ = Y(x+rn)-Y(x),
\]

le modèle reçoit notamment :

\[
v = \frac{2\min(\max(d_-,0),\max(d_+,0))}
{|d_-|+|d_+|+\epsilon}
\]

pour la symétrie d'une vallée sombre, et

\[
s = \frac{|d_- - d_+|}{|d_-|+|d_+|+\epsilon}
\]

pour la marche unilatérale typique d'une limite d'éclairage. Ces valeurs sont
des features, jamais un veto déterministe : une vraie fissure traversée par une
ombre peut elle-même être asymétrique.

Pour chacun des trois descripteurs, l'implémentation prend son maximum sur les
rayons configurés. Ces maxima sont indépendants : vallée, marche et contraste
peuvent donc provenir de rayons différents, sans rayon gagnant commun ni preuve
de cohérence multi-échelle. Un quatrième canal mesure la norme du double-angle
après agrégation des orientations sur chaque cellule ; une annulation de
directions incompatibles devient ainsi une faible cohérence plutôt qu'une
orientation arbitraire, et atténue les trois profils orientés dans la même
proportion.

Un petit CNN prédit alors

\[
a_\theta(x)=\sigma\!\left(h_\theta(R_G,F_{SAM},z_0,Y)\right).
\]

`aθ` est un **score d'alignement local non calibré** avec le voisinage de
l'annotation. La BCE équilibrée change le poids effectif des classes : ce score
n'est ni une probabilité de fissure, ni une probabilité de preuve utile, ni une
mesure de l'utilité marginale par rapport à `z0`.

Le score est forcé à zéro hors du support Frangi. Pendant l'entraînement, le
masque reste doux pour transmettre les gradients. En évaluation, il devient
dur au seuil enregistré dans le checkpoint. La valeur `0,5` est le défaut
opérationnel actuel ; elle devra être calibrée sur des groupes tenus à l'écart.
Une dilation étend ensuite la zone autorisée autour des cellules acceptées,
sans garantir qu'elle épouse la largeur réelle de la fissure, puis la structure
du modèle impose :

\[
z_1(x)=z_0(x)+B(a_\theta)(x)\,\widehat{\Delta z}(x).
\]

Les rayons par défaut `1,5` et `3` et la dilation par défaut `2` sont exprimés
en **cellules de la grille de fusion Hiera haute résolution**, pas en pixels du
masque de sortie. La tolérance de cible `3` est appliquée en pixels du masque de
sortie avant projection par max-pooling sur la grille Hiera.

Deux propriétés sont ainsi garanties par construction et testées bit à bit :

- hors de la bande acceptée, `z1 == z0` exactement ;
- avec l'encodage canonique `no_evidence`, `z1 == z0` partout, même si la tête
  résiduelle a appris un biais non nul.

La porte globale existante reste un second coupe-circuit, après cette sélection
locale.

## Supervision sans fuite

Le masque vrai ne rentre jamais dans `forward` et n'est jamais ajouté au cache.
Il sert uniquement à construire une cible de proximité pendant l'entraînement :

- support Frangi dans le masque vrai dilaté de trois pixels : positif ;
- autre support Frangi : négatif, quelle que soit la cause de la réponse ;
- absence de support : loss exactement nulle et finie.

Cette cible signifie « proche de l'annotation dilatée », pas « utile en plus de
la baseline ». Une réponse déjà correctement traitée par `z0` reste positive,
et une frontière d'ombre qui traverse l'annotation peut être positive au point
d'intersection. La BCE équilibre la masse des classes positives et négatives
**par image** ; elle n'empêche pas un long contour de dominer les autres
composantes à l'intérieur d'une même classe. La loss de segmentation, clDice et
la pénalité de dégradation par image sont conservées.

## Compatibilité et périmètre exact

- `legacy_raster_v1` reste chargeable pour reproduire les checkpoints du
  pilote déjà publié ;
- `verified_local_v1` est la valeur par défaut des nouveaux entraînements ;
- le cache Frangi v2, ses sept canaux, ses manifestes et ses SHA restent
  inchangés ;
- l'architecture, les rayons, le rayon de dilation et le seuil sont enregistrés
  dans le checkpoint et le contrat immuable du workflow ;
- entraîner `verified_local_v1` avec `no_evidence` est refusé : le support nul
  rendrait volontairement tous les gradients de segmentation nuls. Évaluer
  **le même checkpoint entraîné avec le vrai Frangi** sous `no_evidence` teste
  la nécessité de cette entrée et le fallback exact. Ce n'est pas un contraste
  causal complet à capacité égale.

Ce MVP sélectionne des zones raster, pas encore des composantes du vrai graphe.
Le cache actuel ne publie ni identifiants de nœuds/arêtes, ni composantes, ni
persistance multi-échelle. Il serait donc abusif de présenter cette version
comme un GNN de graphe complet.

## Résultats disponibles et statut

Il n'existe pas encore de résultat GPU pour `verified_local_v1`. Le pilote
exploratoire de trois époques de l'adaptateur raster non sélectif donne, sur
9 121 crops regroupés en 1 727 images physiques :

| Estimand groupé | Gain IoU | IC95 bootstrap |
|---|---:|---:|
| résidu toujours ouvert | `+0,00503` | `[+0,00434 ; +0,00568]` |
| oracle image baseline/candidat | `+0,00787` | `[+0,00730 ; +0,00843]` |
| système avec porte globale | `+0,00257` | `[+0,00194 ; +0,00320]` |

Ce résultat reste apparent : la baseline n'était pas cross-fittée, la porte
globale a été ajustée sur les folds 0–3 et son seuil choisi sur le seul fold 4,
et aucun contrôle réentraîné apparié n'a encore isolé la valeur propre du
contenu Frangi. Aucun benchmark externe n'a été évalué pour ce pilote et aucune
expérience avec augmentation d'ombre n'a encore été exécutée. Frangi dense dans
`mask_input` reste, lui, nettement inférieur à la baseline. Ces observations
motivent la nouvelle méthode mais ne permettent pas d'en annoncer le gain ni
la robustesse aux ombres.

La nouvelle implémentation valide déjà les invariants logiciels sur CPU :

- vallée sombre versus marche nette, pénombre colorée et ridge diagonal sur
  motifs jouets ;
- chemin intégré extracteur Frangi → orientation en cache → profil local ;
- atténuation explicite des profils lorsque les orientations s'annulent dans
  une cellule ;
- neutralité exacte à l'initialisation ;
- rejet local dur et absence d'évidence ;
- correction exactement nulle hors bande ;
- labels tolérants, équilibrage de classe et support vide ;
- chargement distinct des checkpoints historiques et sélectifs.

## Expérience suivante, avant toute conclusion

1. Exécuter le smoke test cinq folds avec `verified_local_v1`.
2. Entraîner cinq producteurs OOF sur `correct`, à budget et seed figés.
3. Évaluer chaque checkpoint deux fois : vrai Frangi puis `no_evidence`. Ce
   contraste mesure la nécessité de l'entrée pour le checkpoint entraîné, pas
   un effet causal complet.
4. Réentraîner des contrôles à architecture et budget identiques avec Frangi
   décalé, permuté ou aléatoire à couverture comparable. Ajouter un contrôle
   avec profils photométriques neutralisés et comparer à couverture locale
   égale. Ces contrôles doivent séparer contenu Frangi, parcimonie et capacité
   du correcteur.
5. Construire des paires propre/ombre déterministes — ce jeu n'existe pas
   encore. Recalculer Frangi sur l'image ombrée ; ne jamais réutiliser le raster
   propre.
6. Garder toutes les variantes d'une image physique dans le même fold et
   bootstrapper par image physique, pas par crop ou perturbation.
7. Seulement après gel de l'architecture : outer cross-fitting de la baseline,
   du résidu et des seuils, puis évaluation indépendante.

L'évaluateur enregistre le score local, le support accepté, la couverture de
l'enveloppe, l'amplitude résiduelle dedans/dehors et le recouvrement descriptif
avec l'annotation dans un artefact séparé des features de la porte globale. Les
métriques anti-ombre prioritaires restent le faux positif dans une bande autour
de la frontière d'ombre, le rappel du squelette lorsqu'elle traverse la fissure,
la chute propre→ombre et la courbe risque-couverture du sélecteur local.

Après gel du protocole, le benchmark primaire externe à évaluer est
[Shadow-Crack de Fan et al. (2023)](https://www.ieee-jas.net/article/doi/10.1109/JAS.2023.123447).
Ses auteurs motivent ce jeu par le fait que les ombres de chaussée peuvent avoir
une intensité proche de celle des fissures. Il s'agit d'une validation future,
pas d'un résultat acquis par ce dépôt.

## Critères de décision proposés

- gain IoU macro d'au moins `+0,005`, borne basse de l'IC95 groupé positive ;
- contraste apparié `correct − no_evidence` rapporté comme test de nécessité
  d'entrée, sans l'interpréter seul comme preuve causale ;
- au moins 20 % de faux signal en moins sur les bandes d'ombre par rapport au
  résidu toujours ouvert ;
- perte de rappel squelette sous ombre au plus égale à cinq points ;
- aucune famille sous une marge de `−0,005` ;
- au moins deux fois moins de groupes avec `ΔIoU < −0,05` ;
- sélecteur non dégénéré : au moins 5 % de zones ouvertes et 5 % fermées.

Ces marges sont des critères d'ingénierie proposés à pré-enregistrer avant les
données, et non des seuils déjà étayés. Un gain qui disparaît face à une
sélection aléatoire à couverture identique signifierait que seule la parcimonie
aide. Une baisse forte du rappel aux
traversées d'ombre signifierait que le sélecteur supprime les cas difficiles au
lieu d'identifier les ombres.

## Références de conception

- [Frangi et al., *Multiscale Vessel Enhancement Filtering*, MICCAI
  1998](https://research.manchester.ac.uk/en/publications/multiscale-vessel-enhancement-filtering/) :
  la réponse Hessienne décrit une structure, pas sa sémantique.
- [Geifman et El-Yaniv, *SelectiveNet*, ICML
  2019](https://proceedings.mlr.press/v97/geifman19a.html) : l'abstention doit
  être évaluée par le compromis risque-couverture.
- [Fan et al., *Pavement Cracks Coupled With Shadows*, IEEE/CAA JAS,
  2023](https://www.ieee-jas.net/article/doi/10.1109/JAS.2023.123447) :
  benchmark Shadow-Crack à évaluer, conçu pour les fissures de chaussée
  couplées aux ombres.
