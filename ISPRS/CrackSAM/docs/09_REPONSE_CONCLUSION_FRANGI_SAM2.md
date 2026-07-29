# Raccordement Frangi doublement ancré après SAM 2-LoRA

> Date : 29 juillet 2026
>
> Statut : proposition révisée, sans nouveau résultat expérimental
>
> Cette version remplace la proposition antérieure « GNN vers points SAM ».
>
> Sources : [conclusion du papier soumis au European Workshop on Visual
> Information Processing](../../../EUVIP/LaTeX/main.tex#L619) et
> [audit de CrackSAM 2](08_AUDIT_CRACKSAM2_FRANGIGRAPH_LORA.md)

Dans ce rapport, **SAM 2** désigne *Segment Anything Model 2*, **LoRA** désigne
l'adaptation par matrices de faible rang et **GNN** désigne un réseau neuronal
de graphes. Ce sont les trois seuls sigles employés.

## Décision

La bonne séparation des rôles est :

> **SAM 2 décide si une fissure est présente ; le graphe de Frangi peut
> seulement raccorder deux fragments que SAM 2 reconnaît déjà avec une forte
> confiance.**

Frangi n'entre ni dans le transformer, ni dans l'entrée de masque dense, ni
comme point positif. Il intervient après la segmentation, dans un raccordement
géométrique contraint.

Cette solution est volontairement plus limitée que la proposition précédente.
Elle ne retrouve pas une fissure entièrement manquée par SAM 2. En contrepartie,
elle interdit à Frangi :

- de créer une composante isolée ;
- d'agir dans une région considérée comme du fond certain ;
- d'étendre globalement un masque à partir d'un faux candidat ;
- de modifier la sortie lorsqu'il ne relie pas deux fragments certains.

Il faut la présenter comme une **réparation locale conservatrice de petites
coupures**, et non comme une solution générale aux ombres.

## Pourquoi abandonner les points et le GNN au premier essai

Le résultat historique invalide déjà le masque Frangi dense :

- SAM 2-LoRA sans Frangi : intersection sur union moyenne `0,5675` ;
- Frangi injecté comme masque dense : `0,5563` ;
- différence appariée : `−0,00985`.

Un point positif est moins dense, mais sa signification reste forte : il dit à
SAM 2 qu'un objet est présent à cet endroit. Une seule erreur de sélection sur
une ombre peut donc commander une fausse segmentation.

Un GNN n'apporte pas de protection structurelle. Une frontière d'ombre peut
être longue, continue et bien orientée ; elle peut former un meilleur graphe
qu'une fissure fragmentée. La propagation entre voisins risque alors de
renforcer le faux signal.

Enfin, réduire un graphe à quelques points détruit précisément les arêtes, les
chemins et la continuité qui constituent l'apport géométrique de la méthode
Frangi.

## Architecture

```text
Image ──► SAM 2 adapté par LoRA ──► score de fissure par pixel
                                      │
                         ┌────────────┼────────────┐
                         │            │            │
                  fissure certaine  indécis   fond certain
                         │            │            │
                  extrémités          │        zone interdite
                         │            │
Image ──► graphe de Frangi ──► chemins candidats dans la zone indécise
                                      │
                           pénalité vallée / marche d'ombre
                                      │
                        soutien de la profondeur si disponible
                                      │
                       chemin court entre deux extrémités compatibles
                                      │
                                      ▼
                    ajout local au masque de référence, sinon aucun changement
```

Le transformer n'est exécuté qu'une fois et ne voit jamais Frangi.

## Méthode

### 1. Construire un modèle de référence

SAM 2 est adapté aux fissures par LoRA, sans guidage Frangi. Cette adaptation
doit apprendre l'apparence des fissures et des principaux faux positifs,
notamment les ombres. Une fois le modèle choisi, ses paramètres sont gelés.

LoRA apprend une adaptation générale du domaine. Elle ne doit pas transporter
le graphe : ses poids sont fixes alors que la géométrie Frangi change pour
chaque image.

### 2. Définir le masque de référence et trois zones de confiance

Trois seuils ordonnés sont nécessaires :

\[
t_{\mathrm{fond}} < t_{\mathrm{masque}} < t_{\mathrm{certain}}.
\]

Le seuil central produit le masque SAM de référence. Les deux autres
définissent :

1. **fissure certaine** : score supérieur à \(t_{\mathrm{certain}}\) ;
2. **zone indécise** : score compris entre \(t_{\mathrm{fond}}\) et
   \(t_{\mathrm{certain}}\) ;
3. **fond certain** : score inférieur à \(t_{\mathrm{fond}}\).

Le masque de référence n'est jamais remplacé par la seule zone très certaine.

Les seuils ne doivent pas être choisis arbitrairement. Ils sont réglés
sur des images de validation physiquement distinctes :

- \(t_{\mathrm{masque}}\) conserve le seuil de décision du modèle de référence ;
- \(t_{\mathrm{certain}}\) privilégie une forte précision des ancrages ;
- \(t_{\mathrm{fond}}\) conserve le rappel nécessaire pour ne pas interdire
  tous les vrais raccords.

Le score brut de SAM 2 n'est pas nécessairement une probabilité fiable sur un
nouveau domaine. Le réglage doit donc vérifier la précision réellement observée
au-dessus du seuil haut.

Chaque composante du masque de référence est réduite à sa ligne centrale. Une
extrémité devient un **segment d'ancrage**, et non un simple pixel, seulement
si :

- la composante possède une longueur minimale ;
- une courte portion derrière l'extrémité contient assez de pixels de fissure
  certaine ;
- sa direction locale est stable ;
- sa largeur peut être mesurée sur la portion déjà reconnue.

### 3. Chercher uniquement des raccords doublement ancrés

Une paire d'extrémités devient candidate si :

- les deux segments d'ancrage appartiennent à des fragments distincts ;
- leur distance reste sous une limite exprimée par rapport à leur largeur ;
- les extrémités se font face et leurs directions annoncent le même raccord ;
- leurs largeurs ne sont pas fortement incompatibles ;
- un chemin existe dans le graphe de Frangi ;
- ce chemin reste dans la zone indécise et ne traverse jamais le fond certain ;
- l'intérieur du chemin reste hors du masque de référence, sauf au contact des
  deux ancrages ;
- le premier prototype ne crée ni intersection ni nouvelle jonction.

Cette dernière exclusion est nécessaire parce que la zone indécise chevauche
une partie du masque de référence. Sans elle, un chemin pourrait toucher une
troisième composante.

Le chemin de moindre coût est recherché dans le graphe. Son coût augmente avec :

- une faible similarité Frangi ;
- un désaccord entre l'orientation de l'arête et celle du chemin ;
- une forte courbure ;
- la longueur du raccord ;
- une apparence de marche d'ombre ;
- un score SAM proche du fond certain.

Un chemin ne possédant qu'un seul ancrage est rejeté. Le graphe Frangi peut être
non vide sans que le système soit obligé d'agir.

Pour limiter la recherche opportuniste, chaque extrémité ne conserve que son
meilleur chemin. Celui-ci n'est accepté que s'il est nettement meilleur que le
deuxième meilleur chemin et si sa longueur reste sous un multiple fixé de la
largeur des deux segments SAM. Une extrémité ne peut être utilisée qu'une fois
dans la première version.

La magnitude Hessienne absolue doit également dépasser un seuil appris sur les
données d'entraînement. La normalisation relative par image ne suffit pas :
elle peut rendre artificiellement forte la meilleure structure d'une image
sans fissure. Ce seuil n'est comparable entre images que si la luminosité et les
échelles Frangi suivent une normalisation fixe définie sur l'apprentissage.

### 4. Distinguer vallée sombre et marche d'ombre

Pour chaque point du chemin et pour plusieurs distances liées à l'échelle
Frangi, on mesure la luminosité au centre, notée \(Y_0\), et de chaque côté de
la ligne, notées \(Y_-\) et \(Y_+\).

La symétrie d'une vallée sombre est :

\[
v_r =
\frac{
2\min\left([Y_- - Y_0]_+,[Y_+ - Y_0]_+\right)
}{
|Y_- - Y_0| + |Y_+ - Y_0| + \varepsilon
}.
\]

La force d'une marche d'éclairage est :

\[
m_r =
\frac{
|Y_- - Y_+|
}{
|Y_- - Y_0| + |Y_+ - Y_0| + \varepsilon
}.
\]

Une fissure sombre tend à avoir \(v_r > m_r\). Une frontière d'ombre tend à
avoir \(m_r > v_r\).

Les deux valeurs doivent être calculées à la **même distance**, puis leur
risque conjoint \(m_r(1-v_r)\) est agrégé par la médiane des distances. Prendre
des maxima indépendants, comme dans le prototype actuel, pourrait combiner une
vallée observée à une distance avec une marche observée à une autre.

La notation \([a]_+\) signifie ici \(\max(a,0)\).

Ce terme reste une pénalité douce, pas un veto. Une vraie fissure traversant une
ombre peut devenir asymétrique ; un rejet absolu diminuerait son rappel.

### 5. Utiliser la profondeur comme indice complémentaire lorsqu'elle existe

Dans les données multimodales du papier, la profondeur peut fournir une
vérification moins directement affectée par une ombre visible. Une frontière
d'ombre présente dans l'intensité ne devrait pas former le même chemin dans la
profondeur.

La profondeur reste d'abord une variante expérimentale, pas une condition de la
méthode principale. Lorsqu'elle est utilisée, elle ne peut confirmer un raccord
que si elle est fiable, correctement alignée avec l'intensité et qu'elle
présente un soutien proche d'orientation compatible. Sinon, cet indice est
déclaré indisponible.

Cette règle doit être comparée à une variante intensité seule, car la profondeur
peut elle-même manquer les fissures très fines. Les deux ancrages SAM ne doivent
pas être qualifiés de preuves indépendantes : une même ombre peut produire deux
faux fragments SAM. Le soutien de profondeur ne sera conservé que s'il améliore
le compromis entre précision des raccords et couverture.

### 6. Ajouter seulement un pont local

Un chemin accepté produit un squelette de raccordement. Le premier essai ajoute
seulement une ligne d'un pixel, contenue dans la zone où SAM 2 reste indécis.

L'échelle gagnante de Frangi n'est pas considérée comme une mesure fiable de
largeur. Une extension ultérieure pourra épaissir le raccord, au plus jusqu'à
la plus petite largeur observée sur les deux segments SAM.

Le masque final est la sortie de référence augmentée de ce pont local. Aucun
pixel extérieur au couloir accepté n'est modifié.

La première version ajoute seulement des pixels. Elle ne tente pas de supprimer
les faux positifs déjà produits par SAM 2 : cette responsabilité appartient à
l'apprentissage de LoRA.

## Garanties par construction

Les propriétés suivantes doivent être testées automatiquement :

1. moins de deux segments d'ancrage : sortie identique à la référence ;
2. aucun chemin Frangi admissible : sortie identique à la référence ;
3. chemin traversant le fond certain : rejet ;
4. chemin avec un seul ancrage : rejet ;
5. chemin trop long ou mal orienté : rejet ;
6. hors du couloir retenu : sortie identique à la référence ;
7. aucune composante entièrement nouvelle ne peut apparaître ;
8. un pont accepté fusionne exactement deux fragments sans en toucher un
   troisième.

Ces propriétés ne garantissent pas une amélioration moyenne. Elles bornent la
manière dont Frangi peut dégrader la segmentation.

## Apprentissage et réglage

Le premier essai ne nécessite ni GNN ni seconde LoRA.

### Étape zéro : vérifier que le problème est réellement raccordable

Avant toute implémentation complète, utiliser les annotations d'une partie
d'apprentissage réservée à l'analyse, jamais celles du test final, pour
mesurer :

- la longueur de fissure manquée qui se trouve entre deux bons segments SAM ;
- la fraction de ces coupures qui possède un chemin Frangi admissible ;
- le meilleur gain possible si seuls les raccords corrects étaient acceptés.

Si même ce meilleur sélecteur connaissant l'annotation ne permet pas un gain
utile, la piste doit être arrêtée. Aucun sélecteur limité aux mêmes candidats et
aux mêmes contraintes ne pourra dépasser ce plafond.

### Place de LoRA

La première expérience de raccordement réutilise exactement la LoRA de
référence déjà gelée. Elle ne doit pas modifier à la fois la segmentation SAM
et le raccordement, car leurs effets deviendraient impossibles à séparer.

Dans une campagne ultérieure, le modèle de référence adapté par LoRA pourra être
renforcé avec des paires propres et ombrées partageant la même annotation. Les
ombres synthétiques devront varier en largeur, orientation, opacité et douceur
de frontière. Le raccordement sera alors réévalué avec ce nouveau modèle gelé.

### Recalcul obligatoire de Frangi

Frangi est recalculé après l'ajout de chaque ombre. Réutiliser le graphe de
l'image propre supprimerait précisément les faux candidats que l'expérience
doit mesurer.

### Réglage du raccordement

Les trois seuils SAM, la longueur maximale et les poids du coût de chemin sont
réglés sur les seules images d'apprentissage et de validation. Toutes les
versions d'une même image physique restent dans le même groupe.

Une version initiale peut employer quelques poids scalaires et une recherche de
chemin classique. Un GNN ne sera envisagé que si les vraies arêtes battent des
arêtes mélangées à couverture identique.

## Expérience décisive

Comparer exactement le même SAM 2-LoRA gelé :

| Variante | Question |
|---|---|
| SAM 2-LoRA seul | référence |
| union directe avec Frangi | contrôle volontairement non sûr |
| segment droit entre les mêmes ancrages | Frangi fait-il mieux qu'un raccord géométrique trivial ? |
| raccordement doublement ancré sans terme d'ombre | valeur des contraintes topologiques |
| raccordement doublement ancré avec vallée contre marche | solution proposée |
| même méthode avec graphe décalé | la bonne position est-elle nécessaire ? |
| même méthode avec arêtes mélangées, après conservation du vrai graphe | les vraies connexions sont-elles utiles ? |

Les mesures principales sont :

- intersection sur union et coefficient de Dice ;
- indice de Tversky et distance de Wasserstein pour rester comparable au
  papier ;
- rappel et précision du squelette ;
- précision des seuls pixels ajoutés ;
- longueur de fissure manquée récupérée ;
- longueur fausse ajoutée ;
- nombre de ruptures réparées et de connexions incorrectes créées ;
- faux positifs dans une bande autour des frontières d'ombre ;
- rappel lorsqu'une ombre traverse une fissure ;
- proportion d'images laissées exactement inchangées ;
- fréquence et amplitude des pertes sévères ;
- résultat global avec et sans la famille `cracktree200`.

La méthode n'est retenue que si :

1. l'étape zéro montre un plafond de gain suffisant ;
2. le vrai graphe bat le segment droit et les graphes décalés ou mélangés à
   longueur totale de raccordement acceptée identique ;
3. la borne basse de l'incertitude du gain, calculée par image physique, reste
   positive ;
4. le gain moyen reste positif lorsque `cracktree200` est retiré ;
5. les faux positifs près des ombres n'augmentent pas ;
6. le rappel aux traversées d'ombre ne diminue pas fortement ;
7. les garanties logicielles d'identité exacte sont toutes satisfaites ;
8. une évaluation sur des images physiques jamais utilisées confirme le gain.

La mesure prioritaire du raccordement est la précision des pixels qu'il ajoute,
pas seulement la variation moyenne du masque complet.

## Mise en œuvre minimale dans le dépôt

Le prototype peut réutiliser les éléments existants :

- [`evaluate_sam2.py`](../evaluate_sam2.py) peut déjà enregistrer les scores
  bruts de SAM 2-LoRA avec l'option `--save-logits` ;
- [`frangi.py`](../cracksam2/frangi.py) produit la similarité, l'échelle,
  l'orientation et la distance au squelette Frangi ;
- [`graph_types.py`](../cracksam2/graph_types.py) décrit leur format et
  [`graph_cache.py`](../cracksam2/graph_cache.py) les enregistre ;
- [`evidence_selection.py`](../cracksam2/evidence_selection.py) calcule déjà les
  profils vallée et marche. Le raccordement doit ajouter son propre calcul à
  distances appariées sans modifier l'agrégation utilisée par le prototype
  existant ;
- [`graph_extraction.py`](../../src/graph_extraction.py) construit déjà les
  nœuds, les arêtes et l'arbre couvrant de poids minimal.

Ajouter `frangi_bridge.py` dans `cracksam2` suffit pour le cœur de la méthode :

1. construire les trois zones SAM ;
2. extraire les extrémités des fragments certains ;
3. reconstruire un graphe à huit voisins depuis le squelette Frangi
   précalculé ;
4. calculer les coûts d'ombre et d'orientation ;
5. rechercher et filtrer les chemins ;
6. produire le masque final et les diagnostics.

Un petit évaluateur séparé, `evaluate_frangi_bridge.py`, peut relire les scores
déjà enregistrés. Tous les réglages sont alors comparés sans réexécuter SAM 2
et sans modifier `model.py`.

Le premier test peut utiliser les pixels où la distance au squelette vaut zéro,
à condition que le fichier ait été produit avec le calcul de centralité et la
configuration d'arbre couvrant `K=1`. Il faut toutefois le présenter comme une
approximation : le fichier actuel contient la distance à un arbre rasterisé,
pas les identifiants des nœuds et des arêtes. Deux lignes qui se croisent
peuvent ainsi devenir artificiellement connectées. La conservation explicite
du vrai graphe ne devient nécessaire que si ce test passe.

Les tests minimaux vérifient :

- aucun graphe ou un seul ancrage : identité exacte ;
- deux ancrages et une coupure courte : raccord ;
- structure Frangi isolée : rejet ;
- fond certain, détour, mauvaise orientation ou marche d'ombre : rejet ;
- chaque pixel ajouté appartient à un chemin reliant deux composantes
  antérieures.

La confirmation multimodale devra en plus conserver séparément le soutien
Frangi de l'intensité et celui de la profondeur. Une fusion préalable des deux
modalités ne permettrait plus de savoir si le chemin est réellement confirmé
par la profondeur.

## Limites assumées

- Une fissure entièrement manquée par SAM 2 ne sera pas récupérée.
- Une fausse fissure déjà prédite avec forte confiance par SAM 2 ne sera pas
  supprimée.
- Deux faux fragments SAM proches d'une ombre peuvent encore fournir deux
  ancrages erronés.
- Une ombre étroite ou entourée de texture peut ressembler à une vallée sombre.
- Une vraie fissure traversant une ombre peut recevoir une pénalité trop forte.
- Sans profondeur, le rejet des ombres repose sur des indices moins
  indépendants.
- Deux fissures distinctes mais proches peuvent être raccordées à tort.
- Le squelette rasterisé peut créer une connexion artificielle à un croisement.
- Des seuils mal réglés peuvent rendre le système inactif ou trop permissif.

Ces limites doivent apparaître dans les résultats, pas être masquées par une
seule moyenne.

## Recommandation finale

La réponse la plus crédible à la conclusion du papier n'est pas d'insérer
Frangi plus profondément dans SAM 2. Elle consiste à intégrer les deux méthodes
selon leurs rôles naturels :

> **LoRA adapte SAM 2 à la sémantique des fissures ; le graphe de Frangi répare
> uniquement leur continuité, sous double ancrage SAM et pénalité d'ombre.**

Cette proposition est plus simple que le GNN vers points, conserve réellement
les chemins du graphe et laisse le transformer intact. Si elle n'améliore pas
la continuité face aux graphes décalés et mélangés, il faudra conclure que
Frangi n'apporte pas de valeur complémentaire exploitable dans ce cadre.

Si la version déterministe se révèle trop prudente malgré un plafond de gain
suffisant, un GNN pourra seulement classer les chemins déjà doublement ancrés.
Il ne devra ni produire des points pour SAM 2, ni traverser le fond certain, ni
créer une composante.

## Références

1. Hauseux et collaborateurs,
   [papier sur l'extraction de fissures sans apprentissage par graphe de Frangi](../../../EUVIP/EUVIP_2026_Generalized_Frangi_Multimodality.pdf),
   2026.
2. Ravi et collaborateurs,
   [article présentant SAM 2](https://arxiv.org/abs/2408.00714), 2024.
3. Ge et collaborateurs,
   [article adaptant le premier SAM aux fissures](https://arxiv.org/abs/2312.04233),
   2023.
4. Fan et collaborateurs,
   [article sur les fissures de chaussée couplées aux ombres](https://www.ieee-jas.net/article/doi/10.1109/JAS.2023.123447),
   2023.
5. Hu et collaborateurs,
   [article présentant LoRA](https://arxiv.org/abs/2106.09685), 2021.
