# Question expérimentale et vocabulaire

## Question scientifique

La question n'est pas simplement « Frangi améliore-t-il SAM ? ». Elle est :

> Une similarité géométrique construite sur un graphe de réponses Hessiennes
> peut-elle apporter à un segmenter SAM une information de continuité utile,
> sans imposer ses faux positifs lorsque la géométrie est ambiguë ?

Cette formulation sépare trois capacités :

1. **détection géométrique** : Frangi propose des lignes et des connexions ;
2. **vérification sémantique** : SAM décide si elles ressemblent à des fissures ;
3. **segmentation sûre** : le système corrige la baseline seulement lorsque la
   correction est estimée fiable.

## Noms canoniques

| Nom | Définition | Statut |
|---|---|---|
| `baseline_sam2_lora` | SAM 2 Hiera-L, LoRA q/v, aucun prompt | réalisé |
| `frangi_dense_prompt_sam2_lora` | même famille de modèle, similarité transformée en pseudo-logits denses toujours actifs | réalisé, résultat négatif |
| `cracksam1_published` | résultats publiés de CrackSAM sur SAM 1 ViT-H | référence externe |
| `frangi_graph_residual_sam2` | baseline + adaptateur résiduel et abstention | piste principale |
| `frangi_graph_verifier_sam2` | extension structurée nœuds/arêtes/composantes | phase ultérieure |

## Ce qui a été testé

L'expérience historique utilise `node_sim_max`, c'est-à-dire le maximum de
similarité des arêtes incidentes à chaque nœud. Elle appelle l'extracteur avec
`compute_centrality=False` et s'arrête avant les composantes connexes, le MST et
la centralité. La carte est ensuite écrêtée, transformée par
`log(P / (1 - P))`, redimensionnée en `256 × 256` et passée comme `mask_input`.

La baseline et la variante Frangi ont été entraînées séparément. Le delta final
mélange donc l'effet direct du prompt et la coadaptation de deux jeux de LoRA
différents.

## Ce qui reste à démontrer

- effet immédiat de la carte Frangi avec des poids strictement identiques ;
- utilité d'une évidence positive qui n'assimile pas le zéro à du fond certain ;
- valeur ajoutée des magnitudes absolues, échelles et orientations ;
- valeur ajoutée propre du graphe au-delà des cartes raster ;
- prédictibilité des gains avant d'introduire une gate ;
- robustesse sur ombres naturelles et synthétiques appariées ;
- stabilité sur plusieurs graines d'entraînement et sur un test final intact.

Ces questions déterminent l'ordre de la [feuille de route](04_IMPLEMENTATION_ROADMAP.md).
