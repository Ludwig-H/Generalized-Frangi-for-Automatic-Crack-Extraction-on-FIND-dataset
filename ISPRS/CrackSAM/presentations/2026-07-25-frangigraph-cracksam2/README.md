# Frangi-similarité × CrackSAM 2 — présentation du 25 juillet 2026

Présentation Beamer fondée sur le template Inria et sur la réunion
Inria–Cerema du 10 juillet 2026.

La narration révisée :

- résume SAM, SAM 2, SAM 3 et la baseline CrackSAM 2 en deux diapositives ;
- définit précisément le score pair-à-pair Frangi-similarité et sa
  transformation en pseudo-logits ;
- montre des cartes issues du rapport dans plusieurs régimes visuels ;
- compare dix cas qualitatifs : gains, pertes, cas stable et échec commun ;
- relie les images au bilan apparié et à la matrice causale ;
- conclut sur une intégration locale, sélective et révocable.

Compilation reproductible :

```sh
make
```

`tools/make_figures.py` reconstruit tous les crops depuis les rapports et
artefacts versionnés du dépôt. Le PDF final est `main.pdf` (20 diapositives).
