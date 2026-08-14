# La hiérarchie oracle : ce qu'elle est, comment la calculer, et ce qu'elle apprend

> Question posée par Louis Hauseux le 14 août 2026. Elle sépare deux choses que la première
> version de l'audit confondait :
>
> - « la hiérarchie de Frangi est-elle assez bonne ? » — mesuré, non ([AUDIT §3.3](../AUDIT.md#33-larbre-mesuré-est-un-chemin-pas-une-hiérarchie), [§3.5](../AUDIT.md#35-retirer-lélagage-avant-le-mst--ce-que-cela-règle-et-ce-que-cela-aggrave)) ;
> - « **une** hiérarchie, quelle qu'elle soit, aiderait-elle ? » — c'est l'objet de ce document.
>
> Tout est reproductible en quelques minutes de CPU :
> [`experiments/04_oracle_hierarchy.py`](../experiments/04_oracle_hierarchy.py).

**Réponse en une phrase : la hiérarchie oracle est *sémantique*, pas géométrique — c'est
`{fissure, fond}` au sommet puis un découpage compact dans chaque partie — et elle ne demande
qu'une carte binaire, ni MST, ni composantes, ni centralité.**

---

## 1. Le raisonnement qui définit l'oracle

La *block constraint* de HSA impose `θ_ij = θ_AB` pour toutes les feuilles de deux sous-arbres
frères. Sa sémantique est donc :

> **les tokens d'un même sous-arbre sont interchangeables.**

Une bonne hiérarchie est celle dont les regroupements sont ceux que la tâche autorise à
confondre. Une mauvaise est celle qui **coupe** ce qu'il fallait garder ensemble.

D'où la métrique qui gouverne tout ce document. Quand `i` attend `j`, la valeur d'attention
est partagée par toutes les feuilles du plus haut ancêtre distinct de `j` : la clé et la
valeur de `j` sont **moyennées avec les `|B′| − 1` autres tokens de ce bloc**. On appelle
`|B′|` la **dilution** de la paire.

| dilution | signification |
|---:|---|
| `1` | attention intacte, comme sous Softmax |
| `181` | `j` est noyé dans 181 tokens : il ne reste qu'un vecteur moyen |
| `1024` | `j` est noyé dans un quart de l'image |

C'est la seule mesure comparable entre hiérarchies de profondeurs et d'arités différentes.
On la relève sur trois populations de paires :

- **locale** — tokens voisins tous deux dans la fissure : la continuité de proche en proche ;
- **longue portée** — tokens de fissure éloignés de plus de 16 tokens : **exactement ce que
  le graphe de Frangi prétend apporter et que la Softmax n'encode pas** ;
- fond — contraste : une forte dilution y est *souhaitable*, c'est là qu'on veut économiser.

## 2. Sept hiérarchies, toutes laminaires, toutes complètes

Toutes portent sur la grille de 64 × 64 tokens de SAM 2 — la seule résolution où une attention
globale existe — avec les 4 096 tokens aux feuilles.

| nom | construction |
|---|---|
| `semantic` | **niveau 1 = {fissure, fond}**, puis découpage compact dans chaque partie |
| `semantic_permuted` | la même, avec la fissure d'une **autre image** — contrôle causal |
| `spatial_mincut` | bipartition récursive **équilibrée** à coupe minimale, arêtes de fissure protégées (Fiedler du laplacien pondéré, coupure à la médiane) |
| `spatial_permuted` | la même, fissure d'une autre image |
| `crack_ordered` | tokens ordonnés le long du squelette (géodésique), coupure aux quartiles |
| `frangi_centroid` | **décomposition en centroïdes** du MST de Frangi non élagué |
| `quadtree` | découpage spatial pur, **aucune connaissance de la fissure** — le contrôle décisif |
| `random` | même forme, aucune structure — plancher |

> [!NOTE]
> `frangi_centroid` mérite d'être signalé : la décomposition en centroïdes **répare** le défaut
> mesuré au §3.3 de l'audit. Elle transforme la chenille (`b ≈ 1,15`, profondeur 157) en un
> arbre de profondeur 12 et d'arité 2,82, laminaire, feuilles aux tokens — donc pleinement
> compatible avec HSA. Le correctif que l'audit mentionnait sans le construire existe, il
> tient en trente lignes, et il est ici.

## 3. Résultat

Moyenne sur trois images synthétiques calibrées sur la géométrie de Khánh Hà
(448 px, fissures de 19 px ; 10,8 % des tokens).

| hiérarchie | prof. | arité | dilution **locale** | dilution **longue portée** |
|---|---:|---:|---:|---:|
| **`semantic`** | 10,3 | 3,17 | **2** | **181** |
| `semantic_permuted` | 10,0 | 3,23 | 2 | **808** |
| `spatial_mincut` | 9,3 | 2,75 | 3 | 768 |
| `spatial_permuted` | 9,3 | 2,79 | 2 | 1 024 |
| `crack_ordered` | 10,0 | 2,66 | 12 | 1 024 |
| `frangi_centroid` | 12,0 | 2,82 | 4 | 1 013 |
| `quadtree` | 7,0 | 4,00 | **2** | 1 024 |
| `random` | 7,0 | 4,00 | 1 024 | 1 024 |

Trois lectures, dans l'ordre d'importance.

### 3.1 Aucune hiérarchie *équilibrée* ne peut préserver la continuité à longue portée

Toutes les constructions équilibrées — quadtree, min-cut spectral qui *cherche pourtant* à
éviter la fissure, ordonnancement le long de la fissure, MST de Frangi rééquilibré — plafonnent
entre **768 et 1 024** de dilution à longue portée. Ce n'est pas un défaut de construction,
c'est une conséquence de l'équilibre : une coupe équilibrée au niveau 1 partage l'image en
deux moitiés, donc **coupe toute fissure qui la traverse**. Deux tokens de fissure éloignés se
retrouvent alors reliés par une unique valeur d'attention partagée avec un quart de l'image.

Or l'équilibre est précisément ce que HSA exige pour son `O(M·b²)` et sa profondeur
logarithmique. **La contrainte d'efficacité et la contrainte de continuité s'opposent
frontalement.**

### 3.2 La seule échappatoire est d'abandonner l'équilibre au sommet

`semantic` met **la fissure entière dans un sous-arbre** et le fond dans l'autre. Deux tokens
de fissure, si éloignés soient-ils, restent alors ensemble jusqu'au niveau 2 : la dilution
tombe à **181**, soit `|fissure| / arité`.

Et elle ne paie rien en local : dilution **2**, à égalité avec le quadtree, le meilleur.

Le contrôle permuté tranche : la même construction avec la fissure d'une **autre image** donne
**808**. Le gain est donc causalement dû au **bon masque**, pas à la forme de l'arbre —
c'est la séparation aligné-contre-permuté la plus nette de tout le dossier CrackSAM.

### 3.3 La compacité géométrique suffit pour le local ; seule la sémantique achète le long terme

Le `quadtree`, qui ne sait rien de la fissure, atteint la meilleure dilution **locale** (2).
La raison est géométrique : une cellule carrée a le plus petit périmètre à surface donnée,
donc coupe le moins de paires adjacentes. Aucune connaissance de la fissure n'améliore cela —
et `spatial_mincut`, qui essaie explicitement, fait *moins bien* (3), parce qu'il échange de la
compacité contre du contournement.

C'est la re-dérivation, dans notre cadre et depuis les premiers principes, de ce que l'annexe L
du papier HSA constatait sans l'expliquer : *« the choice of hierarchy is relatively
inconsequential »*. Pour tout ce qui est local, c'est vrai. Ce n'est faux que pour le
long terme, et seulement si l'on accepte de déséquilibrer l'arbre.

---

## 4. Ce que cela change pour le projet

**Ce que l'oracle demande réellement : une carte binaire fissure/fond.** Pas le MST, pas les
composantes, pas la centralité de betweenness. Autrement dit : **pas la partie du Frangi-Graphe
qui n'avait jamais été testée, et qui motivait ce dossier.** Ce qu'il faut, c'est
`node_sim_max` seuillé — la carte même qui a échoué comme *prompt dense* en juillet
(`−0,0979` d'IoU macro sur les poids gelés).

La question devient donc précise, et elle est ouverte :

> La même carte, injectée non plus comme **hypothèse de masque** mais comme **structure de
> blocs de l'attention**, aide-t-elle ?

C'est exactement ce que mesure le bras `block` de
[`02_attention_oracle.py`](../experiments/02_attention_oracle.py), qui applique la block
constraint avec la partition parfaite `{fissure, fond}`. Le script était déjà le bon test ;
on sait maintenant **pourquoi** c'est le bon test, et qu'il n'y a pas de hiérarchie plus riche
à chercher avant de l'avoir lancé.

Deux réserves à garder en tête au moment de le lancer :

1. **`181` reste énorme.** Sous Softmax plate, la dilution vaut 1. L'oracle sémantique est
   *moins mauvais* que les autres, il n'est pas *bon* : il réduit la fissure lointaine à trois
   ou quatre vecteurs moyens. Que ce résumé aide ou nuise est une question empirique, et c'est
   le GPU qui tranche.
2. **Ces constructions sont des heuristiques, pas des optima prouvés.** Quatre familles ont
   été essayées — compacte, min-cut évitant la fissure, ordonnée le long de la fissure,
   sémantique — et c'est la sémantique qui gagne sur la mesure qui compte. C'est un faisceau,
   pas une preuve d'optimalité.

---

## 5. Reproduire

```bash
# les sept hiérarchies, leurs formes et leurs dilutions
python ISPRS/CrackSAM-HierarchicalSelfAttention/experiments/04_oracle_hierarchy.py --n-images 3

# invariants (laminarité, feuilles = tokens, l'oracle coupe moins que le quadtree)
python ISPRS/CrackSAM-HierarchicalSelfAttention/experiments/04_oracle_hierarchy.py --self-test

# réaffichage sans recalcul
python ISPRS/CrackSAM-HierarchicalSelfAttention/experiments/04_oracle_hierarchy.py --report-only
```

Sortie : [`results/oracle_hierarchy.json`](../results/oracle_hierarchy.json).

> [!NOTE]
> Comme pour les autres mesures du dossier, les fissures sont **synthétiques**, calibrées sur
> trois grandeurs mesurées du jeu Khánh Hà. Les écarts rapportés ici sont larges — 181 contre
> 808 contre 1 024 — et stables sur trois images ; ils ne dépendent pas de détails de
> calibration. Rejouer sur les vraies images ne demande que de remplacer `synth_crack` par le
> chargement du masque de vérité terrain.
