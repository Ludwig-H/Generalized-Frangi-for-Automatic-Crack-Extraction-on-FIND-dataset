# EUVIP 2026 — mise à jour camera-ready

Mise à jour réalisée le 28 juillet 2026 à partir de l’archive
`EUVIP_2026_Generalized_Frangi_Multimodality.zip`, des revues CMT et des deux
courriels fournis. Les PDF d’origine dans `Reviews/` n’ont pas été modifiés.

## Fichiers livrés

- `EUVIP_2026_Generalized_Frangi_Multimodality.pdf` : version propre à déposer.
- `EUVIP_2026_Generalized_Frangi_Multimodality_differences.pdf` : comparaison
  avec le manuscrit de l’archive. Les ajouts et modifications sont signalés en
  rouge, sans soulignement des formules mathématiques, et les suppressions sont
  barrées en rouge.
- `LaTeX/` : source camera-ready et figures. Le fichier principal est
  `LaTeX/main.tex`. Le fichier `LaTeX/latexdiff-red-preamble.tex` fixe le
  balisage rouge de la version comparative.

La version « différences » est un document de travail. Elle fait 8 pages parce
qu’elle conserve les passages supprimés et montre les déplacements de figures ;
elle ne doit pas être envoyée comme camera-ready.

## Réponses apportées aux revues

### Reviewer 1 — présentation de Frangi et vue d’ensemble

- Ajout d’un schéma global au début de la méthodologie. Il montre la chaîne
  complète : modalités scalaires, Hessiennes multi-échelles, normalisation et
  fusion, graphe de similarité, composantes, arbre couvrant, centralité et
  squelette final.
- Ajout d’une explication courte de l’interprétation du rapport de Frangi
  \(R_B\) : valeur proche de zéro pour une structure allongée, proche de un pour
  une structure de type « blob », rôle du signe de la courbure et de
  l’orientation locale.
- La grande figure FIND est désormais présentée comme une vue détaillée des
  cartes intermédiaires, et non comme le schéma général de la méthode.

### Reviewer 2 et méta-review — nouveauté

- La contribution est formulée plus précisément. Les Hessiennes, l’arbre
  couvrant minimum et la centralité ne sont pas revendiqués comme nouveaux
  séparément ; l’apport est leur couplage dans un graphe de Frangi par paires,
  avec fusion multimodale au niveau des Hessiennes.
- La relation avec l’étude préliminaire GRETSI est explicitée, ainsi que ce qui
  est ajouté ici : fusion multimodale normalisée, expériences FIND propres et
  bruitées, et second cas géologique.

### Reviewer 5 — paramètres, données et lisibilité

- Les rôles de \(s_s\), \(s_i\), \(s_a\), \(\Sigma\), \(R\) et du seuil
  d’élagage sont expliqués. Le texte distingue les paramètres de sensibilité
  choisis empiriquement des paramètres liés à la largeur attendue des fissures,
  aux lacunes à franchir et au compromis parcimonie–rappel.
- Le facteur de rétention \(\tau\) est défini de la même façon pour les arêtes
  et les nœuds : \(0{,}25\) pour FIND et le Palais des Papes, et
  \(0{,}30\) pour Vaches Noires.
- `Test1` et `Test2` sont définis comme deux patchs UAV visibles annotés de
  \(512\times512\) pixels, acquis aux Vaches Noires à Villers-sur-Mer. Le texte
  précise qu’ils sont distincts de l’image de \(3760\times2058\) utilisée pour
  adapter le U-Net.
- Le tableau des composants suit maintenant l’ordre du texte : forme,
  intensité/courbure, alignement, distance, centralité.
- Le facteur de distance a reçu une notation propre,
  \(\rho_{ij}=\lVert x_i-x_j\rVert_2\), réutilisée dans la définition de
  \(d_{ij}\) et dans le tableau récapitulatif. La définition distingue
  désormais la similarité locale multi-échelle de la similarité du graphe,
  qui inclut la pénalisation spatiale employée dans le code.
- Les conclusions hors domaine ont été tempérées. Les deux patchs géologiques
  annotés et le cas qualitatif du Palais des Papes sont décrits comme des tests
  de transfert initiaux, pas comme une validation générale sur plusieurs
  domaines.

### Modèles de fondation

- La conclusion nomme explicitement les modèles de fondation comme des
  baselines incontournables et fait de la comparaison directe avec SAM et son
  adaptation CrackSAM une priorité. Aucun résultat nouveau n’est revendiqué :
  il s’agit d’une perspective clairement identifiée.

## Autres modifications du papier

- Rétablissement des cinq auteurs, avec les noms de famille en petites
  capitales, des affiliations Inria/Cerema et de l’auteur correspondant.
- Remplacement de l’institution anonymisée par Cerema.
- Rétablissement des remerciements Bpifrance pour les auteurs Inria, ainsi que
  des financements DS4H et 3IA du premier auteur.
- Mise à jour de la section de disponibilité du code avec le dépôt public.
- Suppression de la citation `chiu2026automated` placée à tort après
  CrackSegDiff.
- Correction des types BibTeX de FIND et de la revue Zhang afin de supprimer
  les avertissements bibliographiques, ainsi que de l’ordre des auteurs de la
  revue Zhang d’après la version primaire.
- Reformulation de plusieurs passages trop affirmatifs sur la généralisation,
  le bruit intermodal et les changements de domaine.
- Réécriture légère de l’abstract, des contributions, de la partie sur les
  données géologiques et de la conclusion. Le contenu scientifique et les
  résultats numériques ont été conservés.
- Alignement des équations de distance et de centralité sur l’implémentation
  du notebook : similarité corrigée par la distance, poids d’un nœud défini
  par sa meilleure similarité incidente, puis masse de sous-arbre.
- Simplification de quelques légendes, suppression de dépendances LaTeX
  inutilisées et correction des tableaux qui dépassaient des colonnes.
- Réorganisation des trois courbes de bruit sur une ligne à deux colonnes afin
  de dégager de la place sans changer les marges, la police ou le gabarit IEEE.
- Ajout d’un saut de page avant la bibliographie pour que la page 6 ne contienne
  que les références, conformément à l’appel à communications EUVIP.

## Vérification des paramètres

Les valeurs ont été recoupées avec le notebook et ses sorties, et pas seulement
avec le texte ancien du manuscrit :

- FIND et Palais des Papes : \(s_s=2\), \(s_i=0{,}25\),
  \(s_a=0{,}125\), \(\Sigma=\{1,3,5,7\}\), \(R=3\) et
  \(\tau=0{,}25\).
- Le même facteur \(\tau\) fixe la fraction des arêtes puis des nœuds
  conservés ; aucune expérience décrite dans le papier n’emploie une
  rétention de 40 %.
- Vaches Noires : \(s_s=0{,}5\), \(s_i=0{,}25\),
  \(s_a=0{,}125\), \(\Sigma=\{1,3,5,7,9\}\), \(R=3\) et
  \(\tau=0{,}30\), valeurs recoupées avec l’étude préliminaire GRETSI.

Le script GPU exact qui a produit les deux figures Vaches Noires n’est pas
présent dans ce dépôt. Les valeurs ci-dessus sont documentées dans les sources
du manuscrit, de la thèse et de l’article GRETSI. Le papier ne prétend pas
qu’elles ont été choisies sans consultation de la vérité terrain. Une
confirmation des auteurs reste souhaitable avant le dépôt si une traçabilité
exécutable complète est exigée.

## Compilation et contrôles

La version propre a été compilée avec :

```text
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

La version comparative a été régénérée avec `latexdiff` entre le `main.tex`
contenu dans l’archive d’origine et `LaTeX/main.tex`. Le préambule
`latexdiff-red-preamble.tex` impose le rouge pour les ajouts comme pour les
suppressions, sans souligner les mathématiques ; les équations modifiées sont
traitées comme des blocs avec l’option `--math-markup=1`. Pour garder un
document compilable, le contenu des tableaux modifiés est lui aussi traité
comme un bloc rouge. Le bloc auteurs est affiché dans sa version finale sans
balisage ; la comparaison porte sur le contenu scientifique du manuscrit. Les
corrections du fichier BibTeX sont consignées dans ce README, mais ne sont pas
balisées dans la bibliographie générée par `latexdiff`.

Contrôles effectués :

- PDF A4 de 6 pages ;
- page 6 réservée aux références ;
- aucune citation ou référence croisée non résolue ;
- aucun dépassement de boîte (`Overfull \hbox`) ;
- polices de texte incorporées en Type 1 ;
- vérification visuelle des six pages et de la version annotée.

Il reste un avertissement non bloquant de `caption` lié à l’emploi historique
de `subcaption` avec `IEEEtran`. Il n’a pas d’effet visible sur le document.
Le remplacer aurait demandé une reprise inutile de toutes les sous-figures.
La version comparative émet en plus des avertissements `Underfull \hbox` dus
aux passages conservés et barrés ; elle ne présente aucun débordement.

## Points administratifs à ne pas oublier

- Date limite camera-ready : **5 août 2026, 23 h 59 AoE**.
- Le courriel d’acceptation annonce que les instructions finales de dépôt
  suivront séparément ; vérifier CMT avant l’envoi.
- Chaque article accepté doit être couvert par l’inscription requise par la
  conférence.
- Josiane a demandé un accusé de réception et propose une relecture jeudi ou
  vendredi. Cette réponse par courriel reste à faire hors du dépôt.

Les nouveaux essais SAM 2/SAM 3 et Khanhha n’ont pas été ajoutés. Le courriel
indique qu’ils ne sont pas encore concluants et qu’ils doivent être discutés le
11 août, après la date limite. Les intégrer maintenant aurait ajouté des
résultats fragiles sans répondre à une demande des reviewers.
