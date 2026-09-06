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
Le bloc bibliographique réunit huit références ciblées sur les fissures,
les données, SAM et le guidage de l'attention, dont la revue de Zhang et al. (2025),
avec un QR code de 6 cm.

## Contenu et sources

Le contenu scientifique suit exclusivement [le papier final](../LaTeX/main.tex)
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
[foundation-hierarchy.tex](figures/foundation-hierarchy.tex) illustre uniquement
une perspective : construire des groupes emboîtés depuis le graphe Frangi pour
guider l'attention d'un modèle de fondation. Le titre « Perspective » indique ce statut.
Les trois filaments A, B et C correspondent aux trois groupes de l'arbre et de
la matrice. La référence retenue est
[Amizadeh et al., *Hierarchical Self-Attention: Generalizing Neural Attention Mechanics
to Multi-Scale Problems*, NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/0480adaf62a918405a5e3b1031e0c056-Abstract-Conference.html),
déjà citée dans le dossier ISPRS. HSA partage les coefficients d'attention entre
sous-arbres frères ; leurs valeurs dépendent des caractéristiques du modèle.
La matrice illustre cette structure, sans représenter des résultats mesurés.
Le §4.3 démontre un remplacement partiel des attentions de RoBERTa sans nouvel
entraînement, avec un compromis coût–précision. Il ne démontre aucun gain sur SAM 2.
**Tous les poids de SAM 2 resteraient gelés** : la perspective consiste à fournir
une hiérarchie Frangi à HSA, sans LoRA ni module appris. Cela demande une
adaptation du calcul d'attention et une correspondance arbre–tokens, fond compris.
CrackSAM [4] adapte le SAM original ; SAM 2 est documenté par Ravi et al. [6].
Les [archives des pistes examinées](../../ISPRS/CrackSAM-HierarchicalSelfAttention/ARCHIVES.md)
conservent la comparaison avec le regroupement de tokens et le transfert de relations.

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
