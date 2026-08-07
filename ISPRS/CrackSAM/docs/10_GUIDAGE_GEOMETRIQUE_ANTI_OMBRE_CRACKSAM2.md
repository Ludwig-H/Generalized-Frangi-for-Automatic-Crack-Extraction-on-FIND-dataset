# Guidage géométrique anti-ombre pour CrackSAM 2

> **Date :** 6 août 2026  
> **Statut :** synthèse critique et recommandation expérimentale  
> **Emplacement recommandé :** `ISPRS/CrackSAM/docs/10_GUIDAGE_GEOMETRIQUE_ANTI_OMBRE_CRACKSAM2.md`  
> **Résultat nouveau sur FIND :** aucun. Les seuls tests nouveaux mentionnés ci-dessous sont des contrôles synthétiques exploratoires.

Documents associés :

- [audit de CrackSAM 2 et du guidage Frangi](08_AUDIT_CRACKSAM2_FRANGIGRAPH_LORA.md) ;
- [raccordement Frangi doublement ancré](09_REPONSE_CONCLUSION_FRANGI_SAM2.md) ;
- [sélection locale Frangi actuelle](07_SELECTIVE_FRANGI_EVIDENCE.md) ;
- [rapport des jalons Frangi](../results/frangi_milestone_report/RAPPORT_FRANGI_MILESTONES.md) ;
- [matrice causale du prompt](../results/causal_prompt_matrix_2026-07-20/RAPPORT_MATRICE_CAUSALE.md).

## Résumé exécutif

L'expérience historique n'invalide pas l'idée d'utiliser une géométrie de type
Frangi avec SAM 2. Elle invalide surtout l'interface choisie :

1. `node_sim_max` est une affinité locale, pas une probabilité de fissure ;
2. cette affinité a été convertie en pseudo-logits ;
3. les pseudo-logits ont été injectés dans `mask_input`, qui représente pour
   SAM 2 un masque antérieur à raffiner ;
4. une réponse Frangi erronée sur une ombre devient ainsi une hypothèse de
   segmentation trop forte.

La baseline SAM 2-LoRA atteint une IoU macro de `0,5675`, contre `0,5563` pour
la variante dense Frangi. Sur les poids baseline gelés, le prompt Frangi correct
retire `0,0979` d'IoU macro. Il reste toutefois très supérieur à un prompt
permuté ou décalé : l'alignement géométrique contient donc de l'information,
mais son encodage est nuisible.

La piste prioritaire est désormais :

> **garder SAM 2-LoRA comme voie de référence, utiliser Frangi seulement pour
> proposer des structures, vérifier localement ces structures par une mesure
> ligne-versus-marche calculée à échelle commune, puis autoriser une correction
> résiduelle signée et exactement révocable.**

La vérification locale doit combiner :

- une réponse **paire** de ligne sombre ;
- une réponse **impaire** de bord ou de marche d'illumination ;
- un petit modèle explicite comparant « ombre seule » à « ombre plus fissure » ;
- l'énergie absolue, l'échelle et la cohérence d'orientation ;
- les logits et les features gelés de SAM 2.

Une seconde piste, plus conservatrice, consiste à utiliser le graphe uniquement
pour raccorder deux fragments SAM déjà fiables. Elle borne fortement les
dégradations possibles, mais ne retrouve pas une fissure entièrement manquée et
ne supprime pas les faux positifs déjà produits par SAM.

Il ne faut pas commencer par un GNN, des points positifs automatiques ou une
fusion profonde dans tout Hiera. Une frontière d'ombre peut elle-même former un
graphe long, continu et très cohérent ; propager un mauvais signal ne le rend
pas plus vrai, seulement plus coûteux.

## 1. Ce que les expériences du dépôt établissent

### 1.1 Le prompt dense est la mauvaise interface

La matrice causale donne les résultats principaux suivants :

| Comparaison | Delta d'IoU macro |
|---|---:|
| prompt Frangi sur les poids baseline contre `None` | `-0,0979` |
| tenseur de logits nuls contre `None` | `-0,1641` |
| prompt correct contre prompt permuté | `+0,2473` |
| prompt correct contre prompt décalé | `+0,2700` |
| meilleur système Frangi historique contre baseline | `-0,0122` |

`None` n'est pas équivalent à un masque numérique nul dans SAM 2. Le bon
alignement est lu par le modèle, mais l'entrée de masque dense lui donne la
mauvaise signification.

### 1.2 L'essai historique n'utilise pas réellement le graphe complet

Lorsque `compute_centrality=False`, l'extracteur retourne la carte
`node_sim_max` avant le calcul du MST, des composantes et de la centralité. Le
cache récent contient sept cartes raster, mais pas :

- les identifiants de nœuds ;
- les arêtes ;
- les composantes ;
- les extrémités et jonctions ;
- la persistance multi-échelle.

La variante historique est donc un modèle **raster-conditionné**, pas un test
de la valeur propre de la topologie Frangi.

### 1.3 Le petit résidu actuel ne prouve pas l'utilité de Frangi

Le pilote résiduel peut apprendre à partir des logits et des features SAM, même
sans exploiter le contenu géométrique. Son gain groupé apparent est en outre
fortement dominé par la famille `cracktree200`. Il doit rester un contrôle
architectural, pas une preuve que Frangi apporte une information complémentaire.

## 2. Pourquoi une ombre large trompe quand même Frangi

La largeur globale de la zone sombre n'est pas le bon critère. Une ombre peut
couvrir une grande partie de l'image, tandis que sa frontière reste localement :

- fine ;
- contrastée ;
- allongée ;
- orientée ;
- cohérente sur une longue distance.

Une Hessienne locale ne voit pas « une grande région d'ombre ». Elle voit une
transition étroite. Une frontière nette peut donc produire une réponse
curvilinéaire forte.

Le code actuel accentue ce risque :

1. chaque Hessienne modale et chaque échelle sont normalisées par leur maximum
   spatial propre ;
2. les candidats sont définis relativement au maximum de l'image ;
3. une fraction relative des arêtes est conservée ;
4. au moins une arête est forcée lorsque des candidats existent.

Une image sans fissure nette peut ainsi produire une pseudo-structure forte
relativement à elle-même. Il faut réintroduire une **confiance absolue** et un
véritable état « aucune évidence ».

Le sélecteur `verified_local_v1` distingue déjà une vallée sombre d'une marche,
mais il prend indépendamment le maximum de la vallée, de la marche et du
contraste sur plusieurs rayons. Ces maxima peuvent provenir de rayons
différents. Le vecteur final peut donc décrire une combinaison qui n'existe à
aucune échelle physique unique.

## 3. Piste principale : `verified_local_v2_signed`

### 3.1 Séparer proposition géométrique et décision sémantique

Le pipeline recommandé est :

```text
Image ──► SAM 2-LoRA gelé ──► logits de référence z0 et features Hiera
  │
  └──► Frangi ──► support, orientation et échelle de candidats
                          │
                          ▼
             vérification paire/impaire + ligne/marche
                          │
                 ┌────────┴────────┐
                 │                 │
          correction positive  correction négative
                 │                 │
                 └────────┬────────┘
                          ▼
             résidu local, signé et révocable
```

Frangi propose avec un rappel élevé. Il ne décide jamais seul qu'un pixel est
une fissure.

### 3.2 Première version : dérivées orientées paire et impaire

Le MVP ne nécessite pas immédiatement un banc complet de filtres log-Gabor.
L'orientation normale `n` et l'échelle `sigma` sont déjà fournies par Frangi.
Sur la luminance `Y`, on peut calculer à la même position, à la même orientation
et à la même échelle :

\[
E_\sigma(x)
=
\sigma^2\,\partial_{nn}\bigl(G_\sigma * Y\bigr)(x),
\]

\[
O_\sigma(x)
=
\sigma\,\partial_n\bigl(G_\sigma * Y\bigr)(x).
\]

- `E_sigma` est une réponse paire : elle favorise une vallée ou une ligne
  centrée ;
- `O_sigma` est une réponse impaire : elle favorise une marche ou un bord
  unilatéral.

Après avoir fixé la convention de polarité pour une fissure sombre, on définit
par exemple :

\[
q_\sigma(x)
=
\frac{
\left[
E_{\sigma,\mathrm{dark}}(x)
-\kappa |O_\sigma(x)|
-T_\sigma
\right]_+
}{
|E_\sigma(x)|+|O_\sigma(x)|+\varepsilon
}.
\]

`q_sigma` n'est pas une probabilité de fissure. C'est une mesure de compatibilité
locale avec une ligne sombre plutôt qu'avec une marche.

Les règles importantes sont :

- comparer les réponses paire et impaire à la **même échelle** ;
- garder l'énergie absolue
  \(\sqrt{E_\sigma^2+O_\sigma^2}\), sans normalisation uniquement relative ;
- demander une persistance sur plusieurs échelles voisines ;
- conserver l'état « aucune évidence » lorsque l'énergie absolue est trop
  faible.

Une version ultérieure pourra remplacer les dérivées gaussiennes par une vraie
paire en quadrature log-Gabor ou monogène. La littérature sur la symétrie de
phase motive cette direction, mais le MVP gaussien est plus simple à tester et
réutilise directement les orientations existantes.

### 3.3 Modèle explicite « ombre seule » contre « ombre plus fissure »

La parité locale doit être complétée par un test de profil. On travaille sur la
log-luminance

\[
u(t)=\log(Y(t)+\varepsilon),
\]

car une variation multiplicative d'éclairage devient approximativement
additive.

Sous l'hypothèse d'une ombre ou d'une variation d'éclairage seule :

\[
H_0:\qquad
u(t)
=
a_0+a_1t+a_2t^2
+
b\,\Phi\!\left(\frac{t-t_s}{s}\right),
\]

où `Phi` est une marche lissée.

Sous l'hypothèse d'une ombre et d'une fissure :

\[
H_1:\qquad
u(t)
=
a_0+a_1t+a_2t^2
+
b\,\Phi\!\left(\frac{t-t_s}{s}\right)
-
c\,\psi\!\left(\frac{t-t_c}{w}\right),
\qquad c\geq 0,
\]

où `psi` est un profil de tranchée sombre, gaussien dans le premier prototype.

Le score

\[
\Delta_{\mathrm{ligne}}
=
\operatorname{BIC}(H_0)-\operatorname{BIC}(H_1)
\]

mesure si l'ajout d'une fissure étroite explique réellement mieux le profil.

Cette formulation est préférable à un veto « profil asymétrique = ombre » :
une vraie fissure traversant une ombre peut être asymétrique. `H_1` autorise
simultanément une marche d'éclairage et une tranchée sombre.

Le score reste une feature douce. Il ne doit pas être utilisé seul comme
classificateur déterministe.

### 3.4 Features minimales du vérificateur

Pour chaque cellule candidate, le vérificateur devrait recevoir :

1. la similarité, le support et la magnitude Hessienne absolue ;
2. l'échelle gagnante et l'orientation ;
3. `E_sigma`, `O_sigma`, leur énergie et le score `q_sigma` ;
4. la persistance de `q_sigma` sur les échelles voisines ;
5. `Delta_ligne` et les paramètres ajustés de largeur et de profondeur ;
6. la cohérence d'orientation dans le voisinage ;
7. les logits baseline `z0` ;
8. les features Hiera gelées ;
9. facultativement, une feature de réflectance Retinex ou la profondeur lorsque
   cette modalité est fiable.

La chrominance peut être ajoutée comme indice secondaire, jamais comme règle
principale. Une ombre colorée, un matériau hétérogène ou une fissure humide
rendent les règles chromatiques fragiles.

### 3.5 Correction résiduelle signée

La sortie recommandée est :

\[
z(x)
=
z_0(x)
+
B^+(x)\,\Delta^+(x)
-
B^-(x)\,\Delta^-(x),
\]

avec :

- `B+` : bande où une fissure manquée peut être ajoutée ;
- `B-` : bande où un faux positif de type ombre peut être supprimé ;
- `Delta+` et `Delta-` : amplitudes non négatives apprises ;
- `z0` : logits de la baseline gelée.

Les garanties doivent être structurelles :

- sans évidence acceptée, `z == z0` bit à bit ;
- hors des bandes acceptées, `z == z0` bit à bit ;
- les projections finales sont initialisées à zéro ;
- une porte globale peut encore restaurer exactement la baseline.

Cette voie est préférable à un nouveau `mask_input`, car elle ne transforme
jamais l'évidence géométrique en masque antérieur obligatoire.

### 3.6 Superviser l'utilité marginale, pas la proximité au masque vrai

La cible actuelle indique essentiellement si un candidat Frangi est proche de
l'annotation dilatée. Elle ne dit pas si le candidat améliore SAM.

Le nouveau vérificateur doit apprendre trois états :

| État local | Baseline | Vérité terrain | Action |
|---|---|---|---|
| faux négatif | fond ou faible confiance | fissure | correction positive |
| faux positif | fissure ou forte confiance | fond | correction négative |
| baseline correcte | décision correcte | décision correcte | abstention |

La baseline doit être gelée et ses prédictions d'entraînement doivent être
obtenues hors fold, afin que la cible d'utilité ne soit pas construite à partir
de prédictions artificiellement optimistes.

La loss finale combine :

- la loss de segmentation du candidat ;
- une loss auxiliaire à trois états ;
- une pénalité de couverture pour éviter une porte toujours ouverte ;
- une pénalité explicite sur les images où le candidat dégrade la baseline ;
- éventuellement `clDice` pour la continuité du squelette.

## 4. Piste secondaire : raccordement Frangi doublement ancré

Le raccordement doublement ancré reste la variante la plus sûre à court terme.

SAM 2 fournit des fragments de fissure fiables. Le graphe ne peut ajouter un
chemin que si celui-ci relie deux extrémités distinctes et compatibles :

- les deux fragments ont une confiance élevée ;
- les extrémités se font face ;
- leurs orientations et largeurs sont compatibles ;
- le chemin reste dans la zone SAM indécise ;
- il ne traverse jamais le fond certain ;
- il ne touche pas une troisième composante ;
- il ne crée ni nouvelle composante isolée ni nouvelle jonction.

Le coût de chemin peut devenir :

\[
\begin{aligned}
C(P)
={}&
\lambda_L\,\operatorname{longueur}(P)
+
\lambda_\kappa\int_P |\kappa(s)|\,ds \\
&+
\lambda_O\int_P
\frac{|O_{\sigma(s)}(s)|}
{|E_{\sigma(s)}(s)|+|O_{\sigma(s)}(s)|+\varepsilon}\,ds \\
&-
\lambda_Q\int_P q_{\sigma(s)}(s)\,ds
-
\lambda_B\int_P \Delta_{\mathrm{ligne}}(s)\,ds \\
&+
\lambda_{\mathrm{SAM}}\int_P c_{\mathrm{fond}}(s)\,ds .
\end{aligned}
\]

Le contrôle indispensable est le segment droit entre les mêmes ancrages. Le
graphe n'est utile que si son chemin bat cette interpolation géométrique
triviale.

Cette piste possède une limitation assumée :

- elle peut réparer de petites coupures ;
- elle ne peut pas retrouver une fissure entièrement manquée ;
- elle ne supprime pas les faux positifs déjà présents dans la baseline.

Elle est donc complémentaire du résidu signé, pas équivalente.

## 5. Données et protocole anti-ombre

### 5.1 Paires propre/ombrée

Créer des paires partageant exactement la même annotation :

- ombres dures et pénombres ;
- bandes larges, contours courbes et ombres partielles ;
- ombres neutres et colorées ;
- fissure entièrement dans l'ombre ;
- fissure traversant la frontière d'ombre ;
- image sans fissure avec seulement une ombre.

Frangi et toutes les features géométriques doivent être recalculés après chaque
augmentation. Réutiliser le graphe de l'image propre supprimerait précisément
les faux candidats que l'expérience doit mesurer.

Toutes les variantes d'une même image physique restent dans le même groupe et
dans le même fold.

### 5.2 Benchmark externe

Après gel du protocole, le benchmark externe prioritaire est
**Shadow-Crack**, conçu spécifiquement pour les fissures de chaussée couplées à
des ombres.

Une branche de réflectance inspirée de Retinex peut être testée comme feature
auxiliaire, notamment pour les scènes peu éclairées. Elle ne doit pas remplacer
l'image originale : une décomposition imparfaite peut effacer les fissures
fines en même temps que l'illumination.

## 6. Expérience minimale et ablations

### 6.1 Étape zéro : plafond oracle

Avant tout entraînement GPU supplémentaire, mesurer sur un sous-ensemble
d'analyse de l'entraînement :

1. la fraction des faux négatifs SAM couverte par un candidat Frangi ;
2. la fraction des faux positifs d'ombre couverte par un candidat Frangi ;
3. le gain oracle si seules les corrections géométriques utiles étaient
   acceptées ;
4. le gain oracle du raccordement doublement ancré.

Si ce plafond est négligeable, aucun sélecteur limité aux mêmes candidats ne
pourra produire un gain utile.

### 6.2 Matrice principale

Comparer le même SAM 2-LoRA gelé :

| Variante | Question |
|---|---|
| baseline seule | référence |
| `verified_local_v1` | valeur du prototype actuel |
| profils vallée/marche au même rayon | effet de la cohérence d'échelle |
| paire/impaire gaussienne | valeur du test ligne contre bord |
| modèle explicite `H0/H1` | valeur du modèle d'illumination |
| fusion paire/impaire + `H0/H1` | complémentarité des deux preuves |
| résidu signé | ajout et suppression locales |
| raccordement doublement ancré | réparation conservatrice des coupures |

### 6.3 Contrôles causaux

Chaque modèle doit être comparé à :

- absence totale d'évidence ;
- géométrie décalée ;
- géométrie permutée entre images ;
- support aléatoire à couverture identique ;
- suppression du canal impair ;
- suppression de l'énergie absolue ;
- maxima indépendants contre échelle commune ;
- vraies arêtes contre arêtes mélangées, si le graphe explicite est utilisé.

Un gain qui persiste avec une géométrie permutée indique une correction de
domaine ou un effet de capacité, pas une exploitation de la bonne structure.

## 7. Métriques et critères de décision

Les métriques principales sont :

- IoU et Dice macro par famille ;
- bootstrap groupé par image physique ;
- faux positifs dans une bande autour des frontières d'ombre ;
- rappel du squelette lorsqu'une ombre traverse une fissure ;
- chute propre vers ombré pour chaque paire ;
- courbe risque-couverture du sélecteur ;
- nombre de groupes avec `Delta IoU < -0,05` ;
- continuité et nombre de composantes du squelette.

Critères d'ingénierie à pré-enregistrer avant le test final :

- gain macro d'au moins `+0,005` avec borne basse de l'IC95 positive ;
- au moins `20 %` de faux signal en moins dans les bandes d'ombre ;
- perte de rappel du squelette sous ombre inférieure ou égale à cinq points ;
- aucune famille sous une marge de `-0,005` ;
- au moins deux fois moins de groupes fortement dégradés ;
- porte non dégénérée, avec des zones ouvertes et fermées en proportion
  mesurable.

Ces seuils sont des critères de décision proposés, pas des résultats acquis.

## 8. Contrôle synthétique exploratoire

Un contrôle 1D a comparé des profils de fissures et d'ombres sous changements de
forme, de largeur, de bruit et de modèle d'éclairage. Le classifieur utilisé
était identique pour toutes les familles de features.

| Descripteur | AUC difficile | AUC hors famille | FPR pour 90 % de rappel, hors famille |
|---|---:|---:|---:|
| maxima indépendants actuels | `0,916` | `0,877` | `0,635` |
| profils conservés au même rayon | `0,941` | `0,911` | `0,339` |
| parité paire/impaire multi-échelle | `0,962` | `0,925` | `0,239` |
| modèle explicite ligne/marche | `0,932` | `0,909` | `0,249` |
| fusion des nouvelles features | `0,966` | `0,929` | `0,243` |

Ce contrôle suggère deux priorités :

1. ne plus agréger indépendamment les rayons ;
2. tester une information impaire de bord en plus de la Hessienne paire.

Il ne permet pas de prévoir le gain sur FIND. Les profils, les distributions et
les ombres synthétiques ont été choisis manuellement.

Le précédent contrôle 2D donnant des réponses Frangi numériques presque égales
pour fissures et ombres n'est **pas retenu comme preuve quantitative** : le
paramètre de contraste de l'implémentation utilisée y était adapté par image, ce
qui empêchait une comparaison absolue propre entre scènes. Le risque d'ombre
reste démontré par le mécanisme de normalisation relative du code du dépôt et
doit être mesuré directement sur les images réelles.

## 9. Mise en œuvre recommandée

### MVP raster

Ajouter :

- `cracksam2/phase_evidence.py` :
  dérivées paire/impaire à échelle commune et ajustement `H0/H1` ;
- un mode `verified_local_v2_signed` dans `cracksam2/residual.py` ;
- une cible d'utilité marginale à trois états dans `cracksam2/losses.py` ;
- des métriques spécifiques aux bandes d'ombre dans l'évaluateur.

Tests unitaires indispensables :

- invariant au signe de l'orientation non orientée ;
- réponse paire forte sur une vallée sombre ;
- réponse impaire forte sur une marche ;
- fissure traversant une ombre acceptée par `H1` ;
- absence d'évidence donnant exactement la baseline ;
- correction exactement nulle hors bande ;
- branche positive et branche négative testées séparément ;
- cohérence stricte des échelles et des coordonnées.

### Graphe explicite

Seulement après un résultat positif du MVP raster :

- sérialiser les nœuds, arêtes, composantes, extrémités et jonctions ;
- calculer les features paire/impaire aux nœuds ;
- comparer les vrais chemins aux arêtes mélangées ;
- implémenter le raccordement doublement ancré ;
- envisager un GNN uniquement si les vraies arêtes apportent un gain
  reproductible au-delà d'un classifieur indépendant par nœud.

## 10. Recommandation finale

Ordre de priorité :

1. **corriger la preuve photométrique** : même échelle, paire/impaire, énergie
   absolue et modèle ombre plus fissure ;
2. **corriger la supervision** : utilité marginale positive, négative ou nulle
   par rapport à la baseline gelée ;
3. **injecter par résidu signé révocable**, jamais par `mask_input` ;
4. tester en parallèle le **raccordement doublement ancré** comme variante
   conservatrice ;
5. ne développer un GNN ou une fusion pyramidale profonde qu'après preuve que
   la topologie réelle apporte quelque chose face aux contrôles.

La proposition la plus pertinente est donc :

> **Frangi comme générateur de candidats, vérification locale paire/impaire et
> ligne/marche, puis correction résiduelle signée avec retour exact à SAM 2.**

## Références

1. A. F. Frangi, W. J. Niessen, K. L. Vincken et M. A. Viergever,
   *Multiscale Vessel Enhancement Filtering*, MICCAI, 1998.
2. P. Kovesi, *Symmetry and Asymmetry from Local Phase*, Australian Joint
   Conference on Artificial Intelligence, 1997, p. 185-190.
3. P. Kovesi, *Image Features from Phase Congruency*, Videre, vol. 1, no 3,
   p. 1-26, 1999.
4. C. Steger, *An Unbiased Detector of Curvilinear Structures*, IEEE TPAMI,
   vol. 20, no 2, p. 113-125, 1998. DOI: 10.1109/34.659930.
5. L. Fan et al., *Pavement Cracks Coupled With Shadows: A New Shadow-Crack
   Dataset and a Shadow-Removal-Oriented Crack Detection Approach*, IEEE/CAA
   Journal of Automatica Sinica, vol. 10, no 7, p. 1593-1607, 2023.
   DOI: 10.1109/JAS.2023.123447.
6. T. Chen et al., *SAM2-Adapter: Evaluating and Adapting Segment Anything 2 in
   Downstream Tasks*, arXiv:2408.04579, 2024.
7. Z. Yao et al., *CrackNex: a Few-shot Low-light Crack Segmentation Model
   Based on Retinex Theory for UAV Inspections*, ICRA, 2024,
   arXiv:2403.03063.
