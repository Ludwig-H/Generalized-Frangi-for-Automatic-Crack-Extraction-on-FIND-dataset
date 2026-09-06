# Poster EUVIP 2026

[**PDF A0 portrait — prêt à imprimer**](Poster_EUVIP_2026_Hauseux_A0.pdf)
· [Source LaTeX](poster.tex)

[![Aperçu du poster](apercu.png)](Poster_EUVIP_2026_Hauseux_A0.pdf)

Poster en anglais, au format **841 × 1189 mm**, adapté du template Inria / 3IA /
DS4H fourni dans `../Template_beamer_Inria_Poster_NEO_team.zip`. Il conserve
la palette, les bandeaux et la typographie EB Garamond. Tous les blocs utilisent
le bleu nuit, y compris la perspective. Les logos Inria, Ayana,
3IA et DS4H viennent du template ; les auteurs et affiliations sont ceux du
papier final. Imprimer à **100 %**, sans ajustement à la page.
Le bloc bibliographique réunit dix références ciblées sur les fissures,
les données, SAM et le guidage de l'attention, dont la revue de Zhang et al. (2025),
avec un QR code de 6 cm.

## Contenu et sources

La méthode et les résultats suivent [le papier final](../LaTeX/main.tex)
et sa [camera-ready](../EUVIP_2026_Generalized_Frangi_Multimodality_camera-ready.pdf).
Les illustrations montrent la fusion hessienne, les trois contraintes de la
similarité, la réduction du graphe, FIND propre/bruité et les cas géologiques.
Le Palais des Papes (France) illustre la fusion intensité–profondeur.
Les images expérimentales sont copiées sans modification depuis `../LaTeX/`.
La courbe de bruit est celle du papier, avec des axes et une légende lisibles ;
les bandes représentent l'écart-type. Aucun résultat nouveau n'est ajouté.

Les schémas sont insérés directement en TikZ, sans fond blanc ni PDF intermédiaire.
Deux dessins sont adaptés de la soutenance de Louis Hauseux :

- [frangi-hessian.tex](figures/frangi-hessian.tex) reprend
  [`frangi_hessienne.tex`](https://github.com/Ludwig-H/Manuscrit-de-th-se/blob/8ddbae760c5a337c4e033c45e5f60c16ca58cc67/Soutenance/soutenance/figs/frangi_hessienne.tex).
- [frangi-terms.tex](figures/frangi-terms.tex) reprend
  [`frangi_termes.tex`](https://github.com/Ludwig-H/Manuscrit-de-th-se/blob/8ddbae760c5a337c4e033c45e5f60c16ca58cc67/Soutenance/soutenance/figs/frangi_termes.tex).

Les légendes sont traduites en anglais, avec les notations du papier. Un bandeau
commun regroupe intensité/contraste et forme comme critères du Frangi classique
adaptés aux paires ; un second identifie l’alignement comme ajout du graphe. Les
connexions défavorables sont affaiblies, conformément aux pénalités souples.
Le petit schéma de réduction du graphe dans `poster.tex` est illustratif.

## Perspective : SAM gelé + LoRA + hiérarchie Frangi

Le [TikZ](figures/foundation-hierarchy.tex) reprend la [piste retenue dans ISPRS](../../ISPRS/CrackSAM-HierarchicalSelfAttention/README.md) : deux fragments éloignés peuvent fusionner tôt dans l’arbre Frangi. Construire cette hiérarchie avant sélection de composante et élagage. Sa proximité donne un biais à une attention globale de SAM 2, qui conserve ses scores visuels. **Les poids préentraînés restent gelés ; LoRA et un seul coefficient β apprennent ensemble.**

Les hauteurs et les relations du dessin sont illustratives. Le titre « Perspective » distingue cette proposition des résultats EUVIP. L’objectif est de réduire les ruptures sans multiplier les faux raccords.

- **Graphormer [8]** : ajout d’un biais relationnel avant softmax ; nous remplaçons la relation de plus court chemin par une proximité issue des hauteurs de fusion.
- **LoRA [9]** : adaptation par des matrices de faible rang ; ici sur les projections Q/V.
- **MALIS [10]** : lien entre affinités, chemins et connexité après seuillage ; nous ne reprenons ni sa perte ni l’apprentissage des arêtes.

CrackSAM [4] adapte le SAM original ; la proposition utilise SAM 2 [6]. Les [références complètes et leurs limites](../../ISPRS/CrackSAM-HierarchicalSelfAttention/REFERENCES.md) précisent cette transposition. Aucun de ces articles ne démontre le gain de la combinaison proposée.

## Consignes EUVIP vérifiées

Consultation le **5 septembre 2026** du [site EUVIP](https://euvip2026.github.io/),
du [programme](https://euvip2026.github.io/program/) et des
[instructions auteurs](https://euvip2026.github.io/information/paper-kit-guidelines/).
Aucune consigne publique spécifique à la taille ou à l'orientation des posters
n'a été trouvée. Le format A0 portrait suit donc la demande et le template
fourni. La consigne A4 du « Paper Kit » concerne les articles, pas les posters.

## Modifier et compiler

Dépendances : XeLaTeX, Beamer/beamerposter, TikZ, `qrcode` et EB Garamond.
Sur Debian/Ubuntu :

```bash
sudo apt-get install texlive-xetex texlive-latex-extra texlive-fonts-recommended fonts-ebgaramond
cd EUVIP/poster
make
```

`make` compile le poster et ses fragments TikZ en deux passes. Les fichiers
temporaires restent dans `build/`, ignoré par Git ; `make clean` les supprime.
Le PDF final et son aperçu PNG sont conservés comme livrables du poster.
Contrôler après modification : une seule page A0, polices incorporées,
aucun débordement, légendes visibles et QR code lisible.

Livraison contrôlée : une page A0, toutes les polices incorporées, aucun
avertissement ni débordement LaTeX, QR code décodé vers le dépôt attendu,
revue visuelle de la page et des détails. Les images expérimentales sont
identiques octet par octet aux sources du papier.
Le QR code imprimé et son lien cliquable dans le PDF pointent tous deux vers
[`Ayana-Inria/Frangi-EUVIP`](https://github.com/Ayana-Inria/Frangi-EUVIP).
