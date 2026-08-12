# Errata de la spécification — ce qui a dû être corrigé pour l'implémenter

La spécification [`README.md`](README.md) a été suivie à la lettre partout où elle
était exécutable. Ce document liste les points où elle **ne l'était pas**, avec la
vérification qui l'établit et la correction retenue. Chaque correction est
couverte par un test.

Trois points sont *bloquants*. Les deux premiers empêchent d'écrire le code ; le
troisième l'aurait laissé s'écrire, tourner, et produire une campagne
**inconcluante** — c'est le plus dangereux des trois.

<div align="center">

| # | Gravité | Point | Statut |
|:--:|:--:|:--|:--:|
| 1 | **bloquant** | `tau_mask` est vide quand `compute_centrality=False` | corrigé |
| 2 | **bloquant** | les imports du §10.1 et les commandes du §12 sont incompatibles | corrigé |
| 3 | **bloquant** | `delta_max = 4` borne la fenêtre corrigeable à `\|z₀\| < 4` | porte de plafond ajoutée |
| 4 | sérieux | le split officiel 358/90 n'est pas distribué avec le jeu | contourné, signalé |
| 5 | sérieux | l'identité bit-à-bit est **impossible** pour le bras A6 | prouvé, borné |
| 6 | sérieux | la résolution de travail n'est pas spécifiée | fixée et justifiée |
| 7 | mineur | l'encodeur ne reçoit pas de gradient **au premier pas** | mesuré, documenté |
| 8 | mineur | terme d'activité constant sur une tête sans abstention | neutralisé |
| 9 | mineur | sélection de checkpoint : stricte → tolérante 3 px | changé sur demande |
| 10 | — | « la carte utilisée est la deuxième valeur renvoyée » | **vérifié exact** |

</div>

*Les numéros ci-dessous suivent ce tableau ; l'entrée 3 a été ajoutée après coup,
à la suite d'une contre-expertise adversariale de la spécification.*

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

## 2. Les imports du §10.1 et les commandes du §12 sont incompatibles — bloquant

**Ce que dit la spécification.** §4.3 et §10.1 :
`from ISPRS.CrackSAM.cracksam2.frangi import extract_frangi_graph_gpu`. §12 lance
les scripts depuis `IRT-Signed-Abstention/` (`python scripts/00_build_manifest.py`,
chemins relatifs `configs/`, `data/`).

**Ce qui est vrai**, vérifié :

- `from ISPRS.CrackSAM.cracksam2.frangi import …` **fonctionne** — `ISPRS` est un
  paquet-espace-de-noms PEP 420 (`ISPRS.__file__ is None`) et ni `ISPRS` ni
  `CrackSAM` ne contiennent de tiret. Mais cela exige **la racine du dépôt sur
  `sys.path`**, ce que les commandes du §12 ne fournissent pas : lancé depuis
  `IRT-Signed-Abstention/`, `sys.path[0]` vaut `…/scripts` ;
- `pip install -e .` à la racine n'aide pas : `setup.cfg` n'expose que
  `frangi_fusion` depuis `src/` ;
- les tirets mordent ailleurs, et durement : `geolora` vit sous
  `CrackSAM-GeoLoRA`, et `thermal_residual` sous **deux** dossiers à tirets.
  Aucun des deux ne peut être atteint par un chemin de paquet, jamais, quel que
  soit le `sys.path`.

**Correction.** `thermal_residual/_repo.py` est le **seul** module qui touche à
`sys.path` ; il y insère la racine du dépôt, `ISPRS/CrackSAM` et
`ISPRS/CrackSAM-GeoLoRA`, après quoi `from cracksam2.frangi import …` et
`from geolora.losses import …` fonctionnent quel que soit le répertoire de
travail. Chaque script insère en tête sa propre racine d'étude. C'est le
mécanisme déjà employé par les tests de `ISPRS/CrackSAM`. Un `conftest.py` à la
racine de l'étude fait la même chose pour pytest, de sorte que les deux formes
marchent :

```bash
python -m pytest "ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention/tests"   # depuis la racine
cd ISPRS/CrackSAM-MultiModal/IRT-Signed-Abstention && python -m pytest -q  # 116 passed
```

*Correction de correction : une première version de cet errata affirmait que
`CrackSAM` contenait un tiret. C'est faux, et le chemin d'import de la
spécification est valide en soi — c'est le répertoire de lancement qui le casse.*

## 3. `delta_max = 4` borne la fenêtre corrigeable — bloquant

**Ce que dit la spécification.** §5.1 normalise l'entrée par
`clip(z₀, −10, 10)/10`. §5.3 : « valeur initiale recommandée : `delta_max = 4.0` ».

**Ce qui est vrai.** Les deux phrases se contredisent. Le document clippe à ±10
*parce qu'il sait* que `|z₀|` dépasse 10 — puis borne la correction à 4. Or la
décision est `z₀ + Δz > 0` et `|Δz| ≤ δ_max`, donc

> un pixel n'est corrigeable que si `|z₀| < δ_max`,

c'est-à-dire, à `δ_max = 4`, si `p₀ ∈ (0,018 ; 0,982)`. **Tout faux négatif
confiant de CrackSAM est hors d'atteinte par construction**, quelle que soit la
qualité de l'évidence thermique. Vérifié sur des cas synthétiques à évidence
thermique parfaite : une fissure dont le logit baseline vaut `−6` reste manquée à
`δ_max = 4` (IoU oracle `0,000`) et est parfaitement retrouvée à `δ_max = 8`
(IoU oracle `1,000`).

Le danger n'est pas seulement de perdre du gain : c'est que la campagne
**devienne inconcluante**. Un résultat plat ne saurait pas distinguer « la
thermique n'aide pas » — la conclusion scientifique visée — de « la borne
d'amplitude est trop petite » — un défaut de réglage.

**Correction.** `scripts/08_correction_ceiling.py` est une **porte chiffrée**, à
franchir après le cache de logits et avant tout entraînement. Elle ne coûte ni
GPU ni entraînement, et elle mesure sur la **validation** :

* les quantiles de `|z₀|` — la spécification n'en donnait aucun ;
* la fraction des erreurs de la baseline hors de la fenêtre `±δ_max`, faux
  négatifs et faux positifs séparés ;
* l'**oracle borné** — `+δ_max` sur la vérité, `−δ_max` ailleurs — qui est la
  meilleure correction bornée possible, donc le plafond de toute la méthode, tous
  encodeurs et toutes évidences confondus.

C'est le pendant, pour la borne d'amplitude, de l'oracle de source de
CrackSAM-GFA : si la marge `oracle − baseline` n'excède pas nettement le plancher
de détection au `N` du test, la campagne ne peut pas conclure et `δ_max` doit
être relevé au q99 de `|z₀|` — **avant** le premier entraînement, jamais après
avoir vu les résultats.

## 4. Le split officiel 358/90 n'est pas distribué avec le jeu — sérieux

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

## 5. L'identité bit-à-bit est impossible pour A6 — sérieux, et c'est un théorème

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

## 6. La résolution de travail n'était pas spécifiée — sérieux

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

## 7. L'encodeur ne reçoit pas de gradient au premier pas — mineur, mais réel

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

## 8. Le terme d'activité est une constante sur une tête sans abstention — mineur

`L_active = moyenne(1 − π⁰)`. La tête à deux actions de A5 n'a pas de `π⁰` : le
terme y vaut `1` partout. Les gradients sont inchangés — une constante ne
contribue pas — mais les valeurs de perte deviennent incomparables entre bras.
`corrector_loss(..., has_abstention=False)` le neutralise pour A5 et A6.

## 9. Sélection de checkpoint : IoU stricte → IoU tolérante 3 px — changé sur demande

§7.6 dit « Le checkpoint est choisi sur l'IoU stricte de validation ». La
consigne de campagne étant « une baseline tol3 avec une tolérance de 3 pixels »,
la sélection se fait sur `iou_buffered_tol3` de la validation non augmentée.
C'est gelé dans les sept configurations, donc identique pour tous les bras, et
l'IoU stricte reste rapportée à côté dans tous les tableaux. `selection_metric`
est un champ de configuration : revenir à `iou` est un changement d'une ligne,
tracé dans `training.json`.

## 10. Points de la spécification vérifiés **exacts**

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

## 11. Deux remarques de méthode, hors correction

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
