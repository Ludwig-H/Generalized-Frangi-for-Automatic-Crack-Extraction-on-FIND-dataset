# Cerema — 10 septembre 2026

[**Présentation : 4 slides**](Cerema_2026-09-10_Hauseux_SAM_Hierarchie.pdf) · [Source LaTeX](main.tex) · [Article Graphormer](articles/Graphormer_NeurIPS_2021.pdf)

[![Aperçu des quatre slides](apercu.png)](Cerema_2026-09-10_Hauseux_SAM_Hierarchie.pdf)

## Contenu et provenance

1. **Guider l’attention** : slide 56 de la soutenance.
2. **SAM gelé + LoRA + hiérarchie** : slide 90 de la soutenance, dans les compléments.
3. **Graphormer** : le principe repris est le biais avant softmax ; la hiérarchie Frangi dans SAM reste notre proposition.
4. **Poids dépendant de l’image** : hypothèse d’un petit module prédisant le poids du guidage.

Les deux premières slides reprennent les textes et TikZ de la [soutenance du 8 septembre](https://github.com/Ludwig-H/Manuscrit-de-th-se/blob/3593fbf25434c4715b37537eae1287bf49239fa7/Soutenance/soutenance/main.tex), au commit `3593fbf`. Ce sont les pages PDF **65 et 101**, conservées dans [l’extrait original](sources/Soutenance_extraits_56_90.pdf). Le thème Inria vient de la même source ; seules la date et la numérotation du pied de page changent. Les images d’ombres proviennent de [nos essais du 9 août](../../../CrackSAM-GeoLoRA/presentations/2026-08-09-cracksam-geolora/README.md).

Graphormer : Ying et al., NeurIPS 2021, §3.1.2, équation (6), page 4. Le [PDF officiel complet](https://proceedings.neurips.cc/paper_files/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf) est fourni dans `articles/`, sans modification. Il utilise une distance de plus court chemin, pas notre hiérarchie ni SAM. Les références sont générées par **BibTeX**, avec les libellés et pieds de slide habituels ; aucune citation dans les titres.

## Variante à étudier

La hiérarchie dépend déjà de l’image. Nous proposons d’en adapter **le poids**, prédit à partir de la moyenne des caractéristiques SAM avant l’attention guidée. Une seule couche et une sigmoïde suffisent ; elle apprend avec LoRA par la perte de segmentation, sans annotation d’ombre. Les poids préentraînés restent gelés.

Il s’agit d’un poids utile à la tâche, **pas d’une confiance calibrée**. Une ombre peut contenir une fissure ; un coefficient global peut affaiblir aussi des relations utiles. Le gain reste à mesurer. [Notes pour le premier test](NOTES.md).

## Compiler

```bash
cd ISPRS/CrackSAM-HierarchicalSelfAttention/presentations/2026-09-10-cerema
make
```

Dépendances : LuaLaTeX, BibTeX, Babel français et Beamer/TikZ (`texlive-luatex`, `texlive-lang-french`, `texlive-latex-extra` sous Debian). Le thème utilise ici Latin Modern, comme le PDF de soutenance, faute de fontes Inria installées ; son avertissement à ce sujet est attendu. Les auxiliaires restent dans `build/`. L’aperçu accompagne le PDF compilé.
