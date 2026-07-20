# Matrice causale du prompt Frangi

## Contrôle de restauration

La baseline sans prompt reproduit les résultats historiques dans la tolérance fixée.

| Jeu | IoU attendu | IoU observé | Écart |
|---|---:|---:|---:|
| khanhha_original | 0.623825 | 0.623804 | -0.000021 |
| khanhha_noisy1 | 0.567816 | 0.567812 | -0.000004 |
| khanhha_noisy2 | 0.513364 | 0.513344 | -0.000020 |
| road420 | 0.483509 | 0.483639 | +0.000130 |
| facade390 | 0.516411 | 0.516369 | -0.000042 |
| concrete3k | 0.699844 | 0.699823 | -0.000021 |

## Effets principaux

Les écarts sont des différences d'IoU candidat moins référence. Le macro donne le même poids aux quatre familles Khanhha, Road420, Facade390 et Concrete3k.

| Comparaison | Macro ΔIoU | IC 95 % | Khanhha | Road | Façade | Béton |
|---|---:|---:|---:|---:|---:|---:|
| baseline_frangi_vs_none | -0.0979 | [-0.1062, -0.0901] | -0.1049 | -0.1267 | -0.1196 | -0.0406 |
| baseline_zero_logit_vs_none | -0.1641 | [-0.1728, -0.1554] | -0.0946 | -0.3278 | -0.1601 | -0.0738 |
| baseline_permuted_vs_none | -0.3452 | [-0.3542, -0.3358] | -0.2397 | -0.3377 | -0.3391 | -0.4643 |
| baseline_shifted_vs_none | -0.3679 | [-0.3775, -0.3585] | -0.2627 | -0.3479 | -0.3620 | -0.4992 |
| matching_vs_permuted | +0.2473 | [+0.2366, +0.2582] | +0.1348 | +0.2109 | +0.2196 | +0.4238 |
| matching_vs_shifted | +0.2700 | [+0.2597, +0.2806] | +0.1578 | +0.2211 | +0.2424 | +0.4586 |
| frangi_epoch20_prompt_vs_none | +0.0012 | [-0.0005, +0.0030] | +0.0007 | +0.0070 | -0.0083 | +0.0053 |
| frangi_training_effect_without_prompt | -0.0146 | [-0.0201, -0.0091] | -0.0110 | -0.0161 | -0.0091 | -0.0223 |
| historical_joint_epoch20_vs_baseline | -0.0135 | [-0.0190, -0.0080] | -0.0103 | -0.0091 | -0.0175 | -0.0170 |
| frangi_best_prompt_vs_none | +0.0029 | [+0.0012, +0.0047] | +0.0013 | +0.0094 | -0.0051 | +0.0062 |
| historical_joint_best_vs_baseline | -0.0122 | [-0.0177, -0.0069] | -0.0091 | -0.0135 | -0.0164 | -0.0097 |

## Conclusion causale

1. Sur les poids baseline fixes, le prompt Frangi correct retire `0.0979` d'IoU macro. Un tenseur de logits nuls retire `0.1641` : `None` et un masque numériquement nul ne sont donc pas équivalents dans SAM 2.
2. Le bon alignement reste informatif : il bat le prompt d'une autre image de `+0.2473` et le prompt décalé de `+0.2700`. La géométrie est bien lue, mais l'interface dense la présente comme une hypothèse de masque trop contraignante.
3. Après l'entraînement historique Frangi, remettre le prompt apporte seulement `+0.0029` (IC 95 % `[+0.0012, +0.0047]`) et dégrade encore Façade. Les poids appris sans prompt perdent `0.0146`.
4. Au total, le meilleur système historique reste à `-0.0122` sous la baseline. Avec le prompt appliqué directement à la baseline, `3173` images sur `8895` perdent plus de 0,05 IoU.

**Décision :** abandonner `mask_input` pour Frangi. La suite doit garder la baseline gelée, traiter Frangi comme des cartes auxiliaires et n'appliquer qu'une correction résiduelle révocable par une porte de confiance simple.

## Règle de lecture

Si le contrôle de restauration échoue, les autres différences ne doivent pas être interprétées. Un prompt utile doit battre `None`, mais aussi le prompt permuté et le prompt décalé ; sinon le gain ne démontre pas l'usage de la bonne géométrie.
