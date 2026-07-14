# Protocole CrackSAM 2 — reprise Frangi du 14 juillet 2026

Statut : **note intermédiaire archivée**. L'expérience et les cinq évaluations
de jalon sont terminées ; les résultats définitifs, corrections de contrat et
illustrations se trouvent dans
[`frangi_milestone_report/RAPPORT_FRANGI_MILESTONES.md`](frangi_milestone_report/RAPPORT_FRANGI_MILESTONES.md).
Les métriques ci-dessous documentent uniquement l'état observé à l'époque 20.

## Exécution

- VM : `frangi-blackwell-spot`, `europe-west4-a`.
- Matériel observé : NVIDIA RTX PRO 6000 Blackwell (97 887 MiB), 48 vCPU,
  176 Gio de RAM, aucun swap.
- Variante : SAM 2 Hiera Large, LoRA `qv`, rang 4, prompt Frangi.
- Contrat : 70 époques configurées, batch 8, AdamW, LR `4e-4`, warm-up
  300 pas, décroissance polynomiale puissance 6, validation toutes les
  5 époques.
- Reprise exacte : checkpoint à l'époque interne 23, batch 607, pas global
  26 850. Le dernier calcul interrompu a donc repris au batch suivant, sans
  recommencer l'entraînement.

Le critère d'overfitting est fondé sur la validation, pas sur le test : on
poursuit si le Dice/IoU de validation ne présente pas une baisse persistante et
matérielle par rapport au meilleur jalon (seuil de vigilance : 0,01 absolu sur
deux validations consécutives). Les batteries de test restent des mesures de
rapport et ne servent pas à sélectionner le checkpoint.

## Validations rapides Frangi

| Époque | Précision | Rappel | Dice | IoU | Observation |
|---:|---:|---:|---:|---:|---|
| 5 | 0,734325 | 0,764927 | 0,730586 | 0,610169 |  |
| 10 | 0,760325 | 0,748532 | 0,735533 | 0,616242 |  |
| 15 | 0,764052 | 0,730904 | 0,732388 | 0,611873 |  |
| 20 | 0,759712 | 0,755151 | 0,742714 | 0,623721 | checkpoint de jalon |
| 25 | 0,759583 | 0,757655 | **0,744866** | **0,625429** | nouveau meilleur |

À l'époque 25, aucun signal d'overfitting n'est observé.

## Batterie complète à l'époque 20

| Jeu | Baseline IoU | Frangi IoU | CrackSAM publié IoU | Δ Frangi−baseline | Δ Frangi−publié |
|---|---:|---:|---:|---:|---:|
| Khanhha original | 0,623825 | 0,622104 | 0,6416 | −0,001721 | −0,019496 |
| Khanhha noisy 1 | 0,567816 | 0,567170 | 0,5782 | −0,000645 | −0,011030 |
| Khanhha noisy 2 | 0,513364 | 0,484845 | 0,4915 | −0,028519 | −0,006655 |
| Road420 | 0,483509 | 0,474534 | 0,6222 | −0,008975 | −0,147666 |
| Facade390 | 0,516411 | 0,498902 | 0,4544 | −0,017508 | +0,044502 |
| Concrete3k | 0,699844 | 0,682793 | 0,6798 | −0,017051 | +0,002993 |
| **Moyenne** | **0,567461** | **0,555058** | **0,577950** | **−0,012403** | **−0,022892** |

Sur Khanhha original, la comparaison complète est :

| Source | Précision | Rappel | Dice/F1 | IoU |
|---|---:|---:|---:|---:|
| Baseline SAM 2, époque 20 | 0,749300 | 0,771167 | 0,745335 | 0,623825 |
| Frangi SAM 2, époque 20 | 0,752582 | 0,764593 | 0,743397 | 0,622104 |
| CrackSAM_LoRA publié | 0,7620 | 0,7918 | 0,7639 | 0,6416 |

Les valeurs publiées sont celles de `CrackSAM_LoRA (qv, rank=4)` : Table 2
pour précision/rappel/F1 sur le test original et Table 6 pour les six IoU. Le
papier ne publie pas de Wasserstein, donc aucune valeur ne sera imputée.

## Wasserstein exact : contrôle de faisabilité

Le protocole demandé est le support complet des pixels actifs, coût euclidien
2D et `POT ot.emd2`, sans squelettisation ni sous-échantillonnage. Un benchmark
sur la VM mesure environ 46 octets de pic mémoire par arc dense ; le scheduler
réserve 48 octets par arc et limite BLAS à un thread par processus.

Le pré-scan reconstruit à l'époque 20 six cas connus au-dessus du budget de
140 Gio et un maximum estimé à 243 Gio. Ces cas ne peuvent pas tenir dans les
176 Gio physiques de la VM avec l'algorithme dense prescrit. Le calcul produira
donc :

- une valeur exacte pour chaque cas qui tient en mémoire ;
- un inventaire explicite des cas surdimensionnés ;
- un indicateur `wasserstein_complete` et le nombre de valeurs manquantes ;
- aucune moyenne partielle présentée comme une moyenne complète.

Les prédictions GPU et le scan réel des supports à partir des PNG sauvegardés
remplaceront cette estimation au premier jalon disponible.
