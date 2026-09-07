# Poster EUVIP 2026

[**PDF A0 portrait**](Poster_EUVIP_2026_Hauseux_A0.pdf) · [Source LaTeX](poster.tex) · [Bibliographie BibTeX](references.bib)

[![Aperçu](apercu.png)](Poster_EUVIP_2026_Hauseux_A0.pdf)

Poster en anglais, **841 × 1189 mm**, selon le template Inria / 3IA / DS4H. Imprimer à **100 %**. Les blocs utilisent le bleu nuit ; le QR code de 6 cm renvoie à [Ayana-Inria/Frangi-EUVIP](https://github.com/Ayana-Inria/Frangi-EUVIP).

Inria et Cerema sont centrés dans le bandeau, entre 3IA et DS4H ; le logo Ayana est retiré. Le [logo Cerema blanc](logos/cerema-white.svg) est une adaptation monochrome locale du [SVG du site Cerema](https://www.cerema.fr/themes/custom/uas_base/images/LogoCerema_horizontal.svg), consulté le 7 septembre 2026. Les contours et le fond transparent sont conservés ; le PDF vectoriel est converti depuis ce SVG avec PyMuPDF. Il ne s’agit pas d’une variante blanche publiée par Cerema.

## Mise en page et contenu

Les deux colonnes ont une hauteur commune, calculée à partir des blocs. L’espace restant est réparti entre eux ; l’espacement minimal se règle avec `\posterblockgap` dans [le thème](beamerthemegemini.sty). FIND propre et bruité occupent deux blocs distincts.

La méthode, les notations et les résultats suivent [le papier final](../LaTeX/main.tex). Les images expérimentales proviennent de `../LaTeX/`. Le Palais des Papes (France) illustre la multimodalité. Les TikZ distinguent les critères Frangi classiques de l’alignement apporté par le graphe.

La perspective reprend [SAM gelé + LoRA + biais hiérarchique](../../ISPRS/CrackSAM-HierarchicalSelfAttention/README.md). Graphormer inspire l’insertion du biais ; MALIS éclaire la connexité ; LoRA fournit l’adaptation. Les hauteurs du dessin sont illustratives.

## Bibliographie automatique

Modifier les métadonnées dans [references.bib](references.bib), puis citer les clés avec `\cite{...}` dans `poster.tex`. BibTeX génère la numérotation et les dix entrées, sans liste manuelle.

Le style [poster.bst](poster.bst) affiche le premier auteur et « et al. », conserve les titres et les lie au DOI ou à la source. Les auteurs complets restent dans le fichier `.bib`.

## Compiler

Dépendances : XeLaTeX, BibTeX, Beamer/beamerposter, TikZ, `natbib`, `multicol`, `qrcode` et EB Garamond.

```bash
cd EUVIP/poster
make
```

`make` enchaîne XeLaTeX → BibTeX → XeLaTeX × 2. Les auxiliaires restent dans `build/`, ignoré par Git. Contrôler le PDF : une page A0, blocs alignés, références résolues, polices incorporées et QR lisible. L’aperçu PNG est actualisé avec le PDF lors de la livraison.
