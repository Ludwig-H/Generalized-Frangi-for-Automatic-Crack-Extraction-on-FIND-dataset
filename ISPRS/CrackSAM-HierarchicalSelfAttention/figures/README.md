# Trois figures TikZ pour la soutenance

Les figures sont vectorielles, sans fond de page imposé. Couleurs et styles restent locaux. Les PNG transparents servent aux aperçus GitHub ; le [PDF rassemble les trois figures](figures.pdf).

| Source | Aperçu | Message |
|---|---|---|
| [01_proximite.tikz](01_proximite.tikz) | [PNG](01_proximite.png) | Des fragments éloignés peuvent fusionner tôt dans la hiérarchie. |
| [02_biais.tikz](02_biais.tikz) | [PNG](02_biais.png) | La hauteur de fusion devient un bonus relationnel. |
| [03_sam_lora.tikz](03_sam_lora.tikz) | [PNG](03_sam_lora.png) | Une attention guidée, avec LoRA et un coefficient appris. |

Les figures 1 et 2 utilisent exactement quatre candidats : A–B fusionnent à 0,2 ; C–D à 0,4 ; les deux groupes à 0,9. Ce sont des valeurs illustratives. La matrice suppose un candidat par token et neutralise la diagonale ; le cas général demande une projection vers les tokens.

## Réutilisation

Ajouter `\usepackage{tikz}` au préambule Beamer, puis :

```latex
\resizebox{\linewidth}{!}{\input{figures/03_sam_lora.tikz}}
```

Pour compiler les trois figures, installer TikZ/PGF et `standalone`, puis exécuter depuis ce dossier :

```bash
pdflatex -halt-on-error figures.tex
```

## Idées reprises

Le dendrogramme représente les fusions du graphe ; Graphormer inspire l'ajout d'un biais relationnel à l'attention. Leur combinaison avec l'ultramétrique Frangi et LoRA est la perspective proposée. Voir les [références et attributions précises](../REFERENCES.md).
