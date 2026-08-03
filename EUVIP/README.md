# EUVIP 2026 — versions finales

Ce dossier conserve l'historique de la camera-ready préparée à partir de
`EUVIP_2026_Generalized_Frangi_Multimodality.zip`, des revues CMT et des
remarques successives de Josiane et Pierre Charbonnier. Les documents d'origine
de `Reviews/` n'ont pas été modifiés.

## Livrables

- `EUVIP_2026_Generalized_Frangi_Multimodality_camera-ready.pdf` est destiné à
  EUVIP et à IEEE Xplore ; son wrapper est `LaTeX/camera-ready.tex`.
- `EUVIP_2026_Generalized_Frangi_Multimodality_author-version.pdf` est destiné
  à HAL ; son wrapper est `LaTeX/author-version.tex`.
- `LaTeX/main.tex` contient le corps commun aux deux variantes afin d'éviter
  toute divergence scientifique.
- `modifications_pierre_charbonnier.txt` résume succinctement les changements
  de cette révision.
- `EUVIP_2026_Generalized_Frangi_Multimodality_differences.pdf` reste une
  comparaison historique de travail et ne doit pas être déposée.
- `code/` et `Frangi-EUVIP.zip` contiennent l'artefact logiciel autonome,
  également publié dans le dépôt GitHub `Ayana-Inria/Frangi-EUVIP`.

Les deux versions finales ont le même contenu et respectent la règle officielle
EUVIP : six pages au total, références comprises, avec une page 6 contenant
uniquement des références. Il ne faut donc pas déplacer les acknowledgments
après la bibliographie s'ils apparaissent sur la sixième page. Aucune page
supplémentaire payante n'est annoncée par EUVIP 2026.

### Différences administratives entre les variantes

- **Camera-ready** : aucun numéro de page, aucun en-tête et aucun pied de page
  hormis, au bas de la première page, la mention
  `979-8-3195-3697-6/26/$31.00 ©2026 IEEE`.
- **Author-version** : aucun copyright EUVIP/IEEE et pagination arabe visible
  de 1 à 6.
- Le paquet `flushend` est chargé dans la source commune pour équilibrer les
  colonnes de la dernière page dans les deux variantes.

Sources officielles : [Paper Submission EUVIP 2026](https://euvip2026.github.io/information/paper-submission/),
[Paper Kit & Guidelines](https://euvip2026.github.io/information/paper-kit-guidelines/)
et [instructions camera-ready](https://euvip2026.github.io/information/camera-ready/).

## Corrections de Pierre Charbonnier — 3 août 2026

- L'abstract ne nomme plus CrackSegDiff et parle d'une méthode de deep learning
  état de l'art ; `are still rare` est devenu `remain rare`.
- Les formulations `from the eigenvalues` et `In learned systems` ont été
  remplacées respectivement par une formulation fondée sur l'eigensystème et
  par `In machine learning`.
- Le schéma de la Fig. 1 a été élargi et coloré selon les sous-parties de la
  Section III. Des flèches parallèles noires et grises matérialisent les
  modalités et les composantes multiples ; `minimum spanning trees` est au
  pluriel.
- La similarité de base, la dissimilarité
  \(d_{ij}=\operatorname{clip}_{[0,1]}[\rho_{ij}(1-S_{ij}^{(0)})]\) et le rôle
  pénalisant de \(\rho_{ij}\) sont désormais expliqués. La similarité sert au
  seuillage et à la centralité ; la dissimilarité fournit le coût des arbres
  couvrants minimum.
- Les seuils d'arêtes \(\tau_E\) et de nœuds \(\tau_V\) sont introduits
  séparément avant de préciser le choix expérimental
  \(\tau=\tau_E=\tau_V\).
- La Fig. 2 précise que les cartes affichent, en chaque pixel, la réponse de
  Frangi maximale sur les arêtes ou les échelles pertinentes. La courbure utile
  est écrite \(\max(\lambda_2,0)\).
- Les benchmarks évoqués en III-C renvoient à leur section de présentation avec
  `\emph{cf.}`. Les formulations demandées emploient désormais `false alarms`,
  `subsequent tree` et `consists of four steps`.
- Le Tableau I commence par \(S_{\mathrm{int}}\), présenté comme le support de
  forte courbure qui porte l'essentiel de l'information de Frangi. Les termes
  de forme et d'alignement sont ensuite décrits comme des raffinements qui
  découragent les réponses indésirables sur ce support.
- Le choix des composantes connexes est explicité : FIND contient une seule
  structure d'intérêt, donc la plus grande composante est gardée ; sans cet a
  priori, les composantes sont retenues au-dessus d'une fraction de la taille
  de l'image adaptée aux structures recherchées.
- La section de disponibilité ne donne plus d'adresse de stockage séparée : les
  données non publiques sont disponibles dans le dépôt `Frangi-EUVIP` fourni.

## Historique utile des révisions précédentes

- Le manuscrit rétablit les cinq auteurs, les affiliations Inria/Cerema,
  l'auteur correspondant et les acknowledgments Bpifrance, DS4H et 3IA.
- La passe éditoriale de Josiane a harmonisé les majuscules après deux-points et
  ajouté l'explication du parcours en largeur puis de l'accumulation inversée
  des masses de sous-arbres pour la centralité.
- Les réponses aux reviewers ont ajouté une vue d'ensemble de la méthode,
  clarifié l'interprétation de Frangi et précisé la nouveauté : couplage des
  Hessiennes multimodales normalisées, du graphe par paires, des arbres
  couvrants et de la centralité.
- Le lien avec l'étude GRETSI, les expériences FIND propres et bruitées, les
  deux patchs des Vaches Noires et le cas du Palais des Papes sont distingués.
  Les essais hors domaine sont présentés comme des transferts initiaux, non
  comme une validation générale.
- Les paramètres ont été recoupés avec les notebooks : FIND et Palais des Papes
  utilisent \(s_s=2\), \(s_i=0{,}25\), \(s_a=0{,}125\),
  \(\Sigma=\{1,3,5,7\}\), \(R=3\), \(\tau=0{,}25\) ; Vaches Noires utilise
  \(s_s=0{,}5\), \(\Sigma=\{1,3,5,7,9\}\) et \(\tau=0{,}30\), les autres
  valeurs étant identiques.
- La conclusion présente les comparaisons SAM/CrackSAM comme des travaux futurs
  et ne revendique aucun résultat supplémentaire.
- Les entrées BibTeX, plusieurs formulations trop affirmatives, les légendes et
  la disposition des courbes de bruit ont été corrigées sans changer les
  résultats numériques.

## Compilation et contrôles

Compiler chaque variante depuis `EUVIP/LaTeX/` :

```text
pdflatex camera-ready.tex
bibtex camera-ready
pdflatex camera-ready.tex
pdflatex camera-ready.tex

pdflatex author-version.tex
bibtex author-version
pdflatex author-version.tex
pdflatex author-version.tex
```

Contrôles attendus avant livraison :

- deux PDF A4 de six pages et page 6 réservée aux références ;
- colonnes finales équilibrées ;
- copyright visible uniquement dans la version `camera-ready` ;
- pagination 1–6 visible uniquement dans la version `author-version` ;
- aucune référence ou citation non résolue, aucun `Overfull \hbox` et toutes les
  polices incorporées ;
- vérification visuelle des flèches, couleurs, équations, tableaux et légendes.

Le PDF camera-ready doit ensuite être validé dans IEEE PDF eXpress avec
l'identifiant de conférence **72486X**, puis déposé dans CMT sans nouvelle
modification. La camera-ready et l'IEEE electronic Copyright Form sont attendus
le **5 août 2026 à 23 h 59 AoE**. Le papier doit être couvert par une inscription
Full et présenté sur place.
