# Piste principale — FrangiGraph-Residual

## Décision d'architecture

La similarité Frangi ne sera plus injectée comme un masque de segmentation.
Elle devient une source de candidats structuraux que les features de SAM peuvent
accepter, corriger ou ignorer.

La méthode cible s'écrit d'abord :

\[
z_0 = B(I), \qquad
z_1 = z_0 + H_\theta(F_{SAM}, z_0, R_G),
\]

- `B` est la baseline sans prompt ;
- `F_SAM` désigne les features Hiera multi-échelles ;
- `R_G` est une représentation raster et, plus tard, structurée du graphe ;
- `Hθ` est un petit réseau qui prédit seulement une correction ;
- la porte choisit ensuite **soit** `z1`, **soit** `z0`. Elle ne mélange pas
  les deux sorties : lorsqu'elle refuse la correction, le résultat est
  exactement celui de la baseline.

```text
                           ┌──────────────────────┐
Image ──► Hiera partagé ──►│ baseline sans prompt│──► z0 ─────────────┐
  │                        └──────────────────────┘                    │
  │                               │ features haute résolution        │
  ▼                               ▼                                  ▼
Frangi-graphe ──► cache v2 ──► adaptateur/vérificateur ──► Δz ──► abstention
                                                                    │
                                                                    ▼
                                                               masque final
```

## Pourquoi cette piste est principale

1. **Neutralité vérifiable** : dernière projection initialisée à zéro et
   fallback exact vers la baseline.
2. **Sémantique correcte** : zéro Frangi signifie « aucune évidence », pas
   « fond certain ».
3. **Résolution** : l'adaptateur peut exploiter les features haute résolution,
   importantes pour les structures minces.
4. **Progressivité** : on mesure d'abord la valeur de la similarité raster, puis
   l'apport propre du graphe.
5. **Coût maîtrisé** : une seule passe Hiera ; le correcteur et la porte restent
   petits devant le modèle SAM.

## Première version testable : correction par cartes Frangi

Le premier modèle reçoit une pyramide de cartes, sans logit :

- similarité `node_sim_max` brute ;
- support des nœuds valides ;
- magnitude Hessienne absolue normalisée par l'échelle, avant normalisation
  spatiale par maximum ;
- échelle gagnante ;
- orientation `sin(2θ)` et `cos(2θ)` ;
- distance au squelette rasterisé.

Le correcteur reçoit aussi le logit de la baseline et les cartes internes haute
résolution de SAM. Un petit réseau convolutionnel fusionne le tout et produit
uniquement `Δz`. Dans cette phase :

- backbone, prompt encoder, decoder et LoRA baseline sont gelés ;
- aucun Graph Transformer ;
- aucune porte spatiale ;
- prompt dropout et corruptions Frangi sont utilisés pour empêcher la
  dépendance systématique au prior.

Cette première version utilise le support et la distance au squelette issu du
MST, mais pas la centralité ni une liste explicite de nœuds et d'arêtes. Un gain
validerait donc l'utilité de ces cartes rasterisées, pas encore celle du graphe
complet. À capacité identique, il faut aussi entraîner une tête avec Frangi
neutralisé : sinon le petit réseau pourrait progresser grâce aux seules cartes
internes de SAM et au logit de la baseline.

La première porte de confiance est volontairement simple : une régression
logistique. Elle reçoit sept nombres calculables sans masque vrai :

1. incertitude moyenne de la baseline près du signal utile ;
2. surface de fissure prédite par la baseline ;
3. désaccord baseline/candidat près du signal utile ;
4. taille moyenne de la correction sur le support Frangi ;
5. augmentation ou diminution globale de la probabilité de fissure ;
6. similarité moyenne `node_sim_max` sur le support Frangi retenu ;
7. densité du support Frangi.

Elle produit la probabilité que le candidat améliore la baseline. Ses
coefficients sont appris sur quatre parts du jeu d'entraînement et le seuil
d'ouverture sur une cinquième part séparée. Si aucun seuil n'est suffisamment
fiable, la porte reste fermée. **Cette porte n'est pas un Transformer.** Le
Transformer de graphe décrit plus bas serait, un jour peut-être, un vérificateur
des arêtes ; c'est un autre composant.

## Extension : vérificateur de graphe

Le graphe complet n'est introduit qu'après cette première preuve. Pour chaque nœud :

- position, échelle, valeurs propres, magnitude absolue ;
- orientation, similarité, degré, endpoint/jonction ;
- composante, taille et persistance multi-échelle ;
- profil transverse vallée/marche ;
- features SAM et logit baseline échantillonnés à sa position.

Pour chaque arête :

- extrémités, longueur et similarité ;
- direction spatiale ;
- accord entre cette direction et les tangentes aux extrémités ;
- courbure, appartenance au MST et centralité ;
- identifiant de composante.

Le vérificateur prédit une confiance par nœud, arête et composante. Les arêtes
retenues sont rasterisées en squelette pondéré, orientation, largeur et distance
au squelette, puis fournies à la même tête résiduelle. Le graphe ne produit donc
jamais seul le masque final.

## Ombres : fiabilité douce

Une frontière d'ombre longue et cohérente peut être géométriquement plausible.
Les indices anti-ombre doivent rester des features, jamais des exclusions
irrévocables :

- symétrie d'un profil transverse de vallée ;
- asymétrie d'une marche ;
- déplacement normal du maximum entre échelles ;
- stabilité de l'orientation et de la largeur ;
- magnitude absolue avant normalisation relative ;
- accord sémantique avec les features SAM.

L'entraînement doit inclure des paires avant/après ombre synthétique, avec
recalcul complet du graphe. Il faut distinguer les ombres loin de la fissure de
celles qui la traversent.

## Invariants obligatoires

- `Frangi=0` ne doit jamais produire automatiquement un logit négatif fort ;
- à l'initialisation, le modèle doit reproduire la baseline à la tolérance
  numérique fixée ;
- avec abstention, la sortie doit être exactement la baseline ;
- la baseline reste évaluée par la même fonction et le même seuil ;
- le cache lie chaque graphe au SHA de l'image, aux paramètres Frangi, au split
  et au commit de l'implémentation ;
- toute porte est entraînée sur des prédictions résiduelles produites par un
  modèle qui n'a pas vu les images concernées pendant son propre entraînement ;
- le fold de calibration de la porte ne participe à aucun modèle producteur
  des lignes utilisées pour apprendre ses coefficients ;
- tant que la baseline historique n'est pas elle-même OOF, ce pilote est
  exploratoire et ne démontre pas à lui seul la fiabilité confirmatoire ;
- le test final n'est ouvert qu'après gel de l'architecture et des seuils.
