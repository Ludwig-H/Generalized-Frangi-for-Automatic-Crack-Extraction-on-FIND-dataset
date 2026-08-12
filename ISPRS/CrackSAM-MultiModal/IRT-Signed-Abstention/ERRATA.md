# Errata de la spécification — ce qui a dû être corrigé pour l'implémenter

La spécification [`README.md`](README.md) a été suivie à la lettre partout où elle
était exécutable. Ce document liste les points où elle **ne l'était pas**, avec la
vérification qui l'établit et la correction retenue. Chaque correction est
couverte par un test.

Deux points sont *bloquants* : sans eux le code ne peut pas s'écrire du tout.

<div align="center">

| # | Gravité | Point | Statut |
|:--:|:--:|:--|:--:|
| 1 | **bloquant** | `tau_mask` est vide quand `compute_centrality=False` | corrigé |
| 2 | **bloquant** | `from ISPRS.CrackSAM.cracksam2… ` n'est pas un import Python valide | corrigé |
| 3 | sérieux | le split officiel 358/90 n'est pas distribué avec le jeu | contourné, signalé |
| 4 | sérieux | l'identité bit-à-bit est **impossible** pour le bras A6 | prouvé, borné |
| 5 | sérieux | la résolution de travail n'est pas spécifiée | fixée et justifiée |
| 6 | mineur | l'encodeur ne reçoit pas de gradient **au premier pas** | mesuré, documenté |
| 7 | mineur | terme d'activité constant sur une tête sans abstention | neutralisé |
| 8 | mineur | sélection de checkpoint : stricte → tolérante 3 px | changé sur demande |
| 9 | — | « la carte utilisée est la deuxième valeur renvoyée » | **vérifié exact** |

</div>

---

## 1. `tau_mask` est un plan de zéros sans MST — bloquant

**Ce que dit la spécification.** §4.2 : « `support_union` est l'union des
`tau_mask` des deux polarités ». §4.3 : « Le MVP appelle deux fois l'extracteur
avec `compute_centrality=False` ».

**Ce qui est vrai.** Les deux exigences sont incompatibles. Sur la branche
`compute_centrality=False`, l'extracteur retourne des diagnostics factices :

```python
# ISPRS/src/graph_extraction.py:280-291
empty = np.zeros((H, W), dtype=np.float32)
return (max_S_global.cpu().numpy(), sim_img, empty.copy(), timings,
        {'tau_mask': empty.copy(), 'comp_mask': empty, **raster_diagnostics})
```

Le seuillage des nœuds qui remplit réellement `tau_mask` (`:294-331`) n'est
exécuté que sur la branche MST. Un support lu naïvement aurait donc été
**identiquement nul**, et le quatrième canal d'évidence aurait été un plan de
zéros — un contrôle silencieusement vide.

**Correction.** `thermal_residual.thermal_frangi.support_from_similarity`
réimplémente la règle de seuillage à partir de la réponse Frangi et de la carte
de similarité, toutes deux disponibles sans MST :

1. candidats = pixels où la réponse dépasse `0,01 × max` (le `τ₁` de `:150`) ;
2. nœuds = ceux dont la similarité a survécu au seuillage des arêtes ;
3. si les nœuds sont plus nombreux que `max(1, int(N·τ))`, seuls les `int(N·τ)`
   plus fortes similarités sont retenues.

**Vérification.** `tests/test_thermal_frangi.py::test_support_reproduit_tau_mask`
compare, sur trois images tirées au hasard, la reconstruction au `tau_mask` réel
obtenu avec `compute_centrality=True` : `np.array_equal` est vrai. Le test
assertionne aussi que la branche rapide renvoie bien un plan vide — si
l'extracteur venait à le remplir, le test casse et signale qu'il faut repasser à
une lecture directe.

## 2. L'import de la spécification est syntaxiquement impossible — bloquant

**Ce que dit la spécification.** §4.3 et §10.1 :
`from ISPRS.CrackSAM.cracksam2.frangi import extract_frangi_graph_gpu`.

**Ce qui est vrai.** `CrackSAM`, `CrackSAM-GeoLoRA` et `CrackSAM-MultiModal`
contiennent des tirets, interdits dans un identifiant Python : aucun de ces
dossiers ne peut être un composant de chemin d'import. (`ISPRS.src`, lui,
fonctionne — c'est d'ailleurs ce que fait `cracksam2/frangi.py:51` en repli.)

**Correction.** `thermal_residual/_repo.py` est le **seul** module qui touche à
`sys.path` ; il y insère `ISPRS/CrackSAM` et `ISPRS/CrackSAM-GeoLoRA`, après quoi
`from cracksam2.frangi import …` et `from geolora.losses import …` fonctionnent.
C'est le mécanisme déjà employé par les tests de `ISPRS/CrackSAM`. Un
`conftest.py` à la racine de l'étude fait la même chose pour pytest, de sorte que

```bash
python -m pytest "ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/tests"
```

marche depuis n'importe quel répertoire de travail.

## 3. Le split officiel 358/90 n'est pas distribué avec le jeu — sérieux

**Ce que dit la spécification.** §7.1 annonce « split canonique annoncé : 358
entraînement, 90 test » avec pour sources Zenodo 11624965 et
`lfangyu09/IR-Crack-detection`.

**Ce qui est vrai**, vérifié le 12 août 2026 :

- l'enregistrement Zenodo 11624965 contient **un seul fichier**, une archive ZIP
  de 618 Mo dont la description ne mentionne que quatre dossiers d'images
  (`01-Visible`, `02-Infrared`, `03-Fusion(50IRT)`, `04-Ground truth`) — aucune
  liste de split ;
- `lfangyu09/IR-Crack-detection` ne contient que `LICENSE` et `README.md` ;
- le split vit dans un dossier `00_List/{train_val.txt, test.txt}` distribué sur
  le **Google Drive** du benchmark IRFusionFormer (`Prepare.md`), lu par
  `IRFusionFormer/data/ircrack_dataset.py:38-44`.

**Correction.** `--official-split` accepte ce dossier `00_List` (ou un JSON
`{train, test}`) et n'est pas obligatoire. À défaut, `thermal_residual.splits`
construit un split **dérivé** de même effectif, déterministe, stratifié par
strate horaire puis par quartile de fraction de fissure, ordonné par
`sha256(sel|identifiant)` donc invariant à l'ordre d'énumération des fichiers.
Le split porte alors `origin: "derived"`, le script l'affiche en majuscules et le
rapport doit le dire : les chiffres publiés (IRFusionFormer IoU 0,818, etc.) ne
sont **pas** comparables au pixel près à un split dérivé.

## 4. L'identité bit-à-bit est impossible pour A6 — sérieux, et c'est un théorème

**Ce que dit la spécification.** §14 exige « la sortie initiale est bit-à-bit
égale à la baseline », et §8.4 définit le bras A6 par
`Δz = δ_max · σ(q) · S_max`.

**Ce qui est vrai.** Soit `f` la correction, non négative par construction. Si
`f(θ₀) = 0` alors `θ₀` est un **minimum global** de `f`, donc un point critique :
`∇f(θ₀) = 0`. Une correction non négative, nulle à l'initialisation et de
gradient non nul en ce point n'existe pas. Aucun choix d'initialisation ne
sauve A6 ; ce n'est pas un défaut d'implémentation.

Le cas signé, lui, s'en sort précisément parce qu'il **n'est pas** non négatif :
`π⁺ − π⁻` s'annule par symétrie sans être en un extremum, donc
`∂(π⁺−π⁻)/∂q ≠ 0`. C'est ce qui rend l'initialisation `bias = (−2, −2, +2)`
correcte — et elle l'est bien : les poids nuls rendent les logits d'action
constants, `π⁺` et `π⁻` sont alors **le même flottant**, donc leur différence
est exactement `0,0`.

**Correction.** A6 part de `bias = −8`, soit `σ(−8) ≈ 3,4·10⁻⁴` et
`|Δz| ≤ 1,3·10⁻³` — une identité à `10⁻³` près, mesurée par
`test_positive_only_est_proche_mais_pas_exact`. Le critère d'acceptation §14 est
lu comme portant sur les têtes signées (A1–A5), pour lesquelles il est vérifié
au bit près, avant **et après** entraînement.

## 5. La résolution de travail n'était pas spécifiée — sérieux

**Ce que dit la spécification.** §7.5 impose la « résolution originale » pour le
cache thermique, et ne dit rien de la résolution des logits.

**Ce qui est en jeu.** IRT-Crack est en `640×480` ; CrackSAM 2 a été entraîné sur
Khánh Hà en `448×448` (`cracksam2/data.py:637-642`, `cv2.INTER_CUBIC`). Trois
chaînes étaient possibles, et elles ne donnent pas les mêmes chiffres.

**Correction retenue**, et sa raison :

| Étape | Choix | Pourquoi |
|:--|:--|:--|
| entrée de SAM | `448×448`, bicubique | la baseline gelée doit voir la distribution sur laquelle elle a été apprise |
| sortie de SAM | logits `448²` | conséquence de l'entrée |
| cache | logits ré-échantillonnés en `480×640` | le seul rééchantillonnage porte sur des **logits lisses** |
| évidence, masque, métriques | `480×640` natif | **la vérité terrain n'est jamais rééchantillonnée** |

Sur des structures larges de quelques pixels, rééchantillonner l'annotation
coûterait plus que tous les écarts qu'on cherche à mesurer — le rapport GeoLoRA
mesure déjà `0,881` d'IoU d'un masque dilaté d'un seul pixel contre lui-même.

## 6. L'encodeur ne reçoit pas de gradient au premier pas — mineur, mais réel

**Ce que dit la spécification.** §5.4 : l'initialisation choisie « évite le
mauvais couplage […] qui peut empêcher tout signal d'apprentissage d'atteindre la
branche auxiliaire ».

**Ce qui est vrai.** C'est exact pour la **tête** : son gradient est non nul dès
le pas 0 (mesuré : 96 coefficients non nuls sur 96). Ce ne l'est pas pour
l'**encodeur** : avec `W = 0`, `∂q/∂h = Wᵀ = 0`, donc l'encodeur ne reçoit
strictement rien au pas 0. Mais `W` bouge au pas 0, donc l'encodeur apprend dès
le pas 1 (mesuré : 0 puis 2 016 coefficients non nuls).

**Différence avec l'échec GeoLoRA.** Là-bas, `gamma` et la projection étaient
nuls *tous les deux* : chacun annulait le gradient de l'autre, et le produit
restait nul indéfiniment — `gamma` est resté à `0,0000`. Ici un seul facteur est
nul, et il reçoit du gradient. C'est un retard d'un pas, pas un gel. Le test
`test_l_encodeur_recoit_du_gradient_des_le_second_pas` fixe cette propriété pour
qu'une régression ne la transforme pas silencieusement en gel.

## 7. Le terme d'activité est une constante sur une tête sans abstention — mineur

`L_active = moyenne(1 − π⁰)`. La tête à deux actions de A5 n'a pas de `π⁰` : le
terme y vaut `1` partout. Les gradients sont inchangés — une constante ne
contribue pas — mais les valeurs de perte deviennent incomparables entre bras.
`corrector_loss(..., has_abstention=False)` le neutralise pour A5 et A6.

## 8. Sélection de checkpoint : IoU stricte → IoU tolérante 3 px — changé sur demande

§7.6 dit « Le checkpoint est choisi sur l'IoU stricte de validation ». La
consigne de campagne étant « une baseline tol3 avec une tolérance de 3 pixels »,
la sélection se fait sur `iou_buffered_tol3` de la validation non augmentée.
C'est gelé dans les sept configurations, donc identique pour tous les bras, et
l'IoU stricte reste rapportée à côté dans tous les tableaux. `selection_metric`
est un champ de configuration : revenir à `iou` est un changement d'une ligne,
tracé dans `training.json`.

## 9. Points de la spécification vérifiés **exacts**

Il faut aussi dire ce qui a tenu.

- §4.3 « La carte utilisée est la deuxième valeur renvoyée, `similarity_img` » :
  **exact** — `cracksam2/frangi.py:174` documente le contrat à cinq valeurs et
  `:229-235` le respecte.
- « la similarité est identique avec ou sans centralité » : **exact**, vérifié
  par `np.array_equal` dans le test de reconstruction du support.
- §5.4, l'initialisation `zeros_(weight)` + `bias = (−2,−2,+2)` donne bien
  `Δz = 0` **exactement** et des gradients non nuls sur la tête.
- §8.2, le contrôle brut à quatre canaux `[T, 1−T, 2|T−0,5|, 1]` donne exactement
  la même capacité d'entrée que l'évidence Frangi : `test_configs.py` vérifie que
  A1, A2, A3 et A4 ont le **même** nombre de paramètres.
- §2.1, `implementation_notes.md` a raison sur le piège JET : sur un dégradé de
  256 niveaux encodé en JET, la conversion standard en gris est non monotone et
  le vert médian y dépasse le rouge maximal (`test_thermal_decode.py`), pour un
  écart moyen de `0,29` au décodage correct sur le faux jeu.

## 10. Deux remarques de méthode, hors correction

**La perte tolérante ne pénalise pas une rupture plus courte que `k`.** Le
rapport GeoLoRA écrit « Une rupture reste pénalisée à toutes les tolérances ».
Mesuré : une rupture de 4 px dans une ligne donne `iou_buffered = 1,000` à
`k = 5`, et `< 1` à `k = 0`. La propriété exacte est « une rupture **plus large
que `k`** reste pénalisée », ce qui reste la propriété utile — mais l'énoncé
absolu est faux, et `test_metrics.py` le fixe.

**La vérité terrain est en JPEG.** Les images sont des `.png`, les masques des
`.jpg` (`IRFusionFormer/data/ircrack_dataset.py:47`). Un masque binaire compressé
en JPEG porte des artefacts de bord. Le seuillage est fait à `> 0,5` après
division par 255, ce qui les absorbe, mais c'est une source de bruit d'annotation
supplémentaire à mentionner dans le rapport — et une raison de plus de conclure
à `k = 3` plutôt qu'à `k = 0`.
