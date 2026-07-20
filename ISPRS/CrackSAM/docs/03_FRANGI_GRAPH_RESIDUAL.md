# Piste principale — FrangiGraph-Residual

## Décision d'architecture

La similarité Frangi ne sera plus injectée comme un masque de segmentation.
Elle devient une source de candidats structuraux que les features de SAM peuvent
accepter, corriger ou ignorer.

La méthode cible s'écrit :

\[
z_0 = B(I), \qquad
\Delta z = H_\theta(F_{SAM}, z_0, R_G), \qquad
z = z_0 + g\,\Delta z.
\]

- `B` est la baseline sans prompt ;
- `F_SAM` désigne les features Hiera multi-échelles ;
- `R_G` est une représentation raster et, plus tard, structurée du graphe ;
- `Hθ` est un petit adaptateur résiduel ;
- `g` est une confiance calibrée ; sous le seuil d'abstention, `z = z0`
  exactement.

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
5. **Coût maîtrisé** : une seule passe Hiera ; l'adaptateur et la gate restent
   petits devant le backbone.

## MVP : résidu raster

Le premier modèle reçoit une pyramide de cartes, sans logit :

- similarité `node_sim_max` brute ;
- support des nœuds valides ;
- magnitude Hessienne absolue normalisée par l'échelle, avant normalisation
  spatiale par maximum ;
- échelle gagnante ;
- orientation `sin(2θ)` et `cos(2θ)` ;
- distance au squelette rasterisé ;
- incertitude et gradient du logit baseline.

Un encodeur convolutionnel léger projette ces cartes vers les dimensions des
features Hiera. Une tête de fusion produit uniquement `Δz`. Dans cette phase :

- backbone, prompt encoder, decoder et LoRA baseline sont gelés ;
- aucun Graph Transformer ;
- aucune gate spatiale ;
- prompt dropout et corruptions Frangi sont utilisés pour empêcher la
  dépendance systématique au prior.

Une gate globale n'est ajoutée que si le candidat montre un oracle de validation
significatif **et** si les gains sont prédictibles out-of-fold.

## Extension : vérificateur de graphe

Le graphe complet est introduit après le MVP. Pour chaque nœud :

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
- toute gate est entraînée sur des prédictions out-of-fold ;
- le test final n'est ouvert qu'après gel de l'architecture et des seuils.
