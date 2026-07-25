# Frangi-graphe × CrackSAM 2 — présentation du 25 juillet 2026

Présentation Beamer fondée sur le template Inria et sur la réunion
Inria–Cerema du 10 juillet 2026.

Le PDF distingue explicitement :

- les résultats publiés de CrackSAM sur SAM 1 ;
- la baseline locale portée sur SAM 2 ;
- l'expérience négative du prompt Frangi dense ;
- le pilote exploratoire de correction résiduelle ;
- le prototype local sélectif, implémenté mais pas encore évalué sur GPU ;
- les propositions issues de la littérature jusqu'à SAM 3 et CVPR 2026.

Compilation reproductible :

```sh
make
```

`tools/make_figures.py` reconstruit les crops depuis les artefacts versionnés
du dépôt. Le PDF final est `main.pdf`.
