# Ce que fait exactement HSA (NeurIPS 2025)

> Résumé technique de S. Amizadeh, S. Abdali, Y. Li et K. Koishida, *Hierarchical
> Self-Attention: Generalizing Neural Attention Mechanics to Multi-Scale Problems*,
> NeurIPS 2025 (Microsoft). PDF dans ce dossier.
>
> Objet de ce document : éviter que quiconque ait à relire 49 pages pour vérifier une
> affirmation de l'[audit](../AUDIT.md). Chaque point renvoie à l'équation ou à la section
> du papier.

## 1. L'objet d'entrée : `signal hierarchy`

Un *nested signal* (déf. 2.1) est un signal dont les valeurs sont elles-mêmes des signaux.
Sa représentation calculatoire est la **signal hierarchy** `h_x` (déf. 2.2) : un arbre

- **enraciné** ;
- **laminaire** : les nœuds internes sont des *regroupements* emboîtés ;
- dont les **feuilles sont les tokens** — un token n'est jamais un nœud interne ;
- dont chaque fratrie (*family*) partage un même plongement de position `ε_Ω`.

C'est la définition qui décide de tout le reste. Retenir : **les tokens sont aux feuilles**,
et les nœuds internes n'existent que pour regrouper.

## 2. Le mécanisme : la *block constraint*

Pour deux nœuds non apparentés `A` et `B`, l'énergie d'interaction (éq. 7) est

```
ψ_{A→B} = −ε(A′)ᵀ ε(B′) + (2√d·|ℓ(A)|·|ℓ(B)|)⁻¹ · Σ_{i∈ℓ(A)} Σ_{j∈ℓ(B)} ‖q_i − k_j‖²
```

c'est-à-dire une dissimilarité calculée **au niveau du sous-arbre**, sur les moyennes des
`q`, `k`, `v` de ses feuilles. Il en découle mécaniquement (§3.2) que la matrice d'attention
vérifie

```
θ_{i,j} = θ_{A,B}   pour tout i ∈ ℓ(A), j ∈ ℓ(B),  A et B frères
```

Tous les couples de feuilles entre deux sous-arbres frères partagent **une seule valeur
d'attention**. Les degrés de liberté passent de `O(|ℓ|²)` à `O(M·b²)`, `M` étant le nombre de
familles et `b` le facteur de branchement maximal. La matrice devient une *matrice
hiérarchique* au sens de Hackbusch.

> **Le point à ne pas manquer.** HSA ne fabrique pas d'information : il en **retire**. C'est
> un schéma de *compression* de la matrice d'attention, pas un mécanisme d'injection de
> connaissance. Le prior qu'il encode est la *scale separation* — « les feuilles d'un
> sous-arbre sont interchangeables ».

## 3. Le théorème d'optimalité, et ce qu'il ne dit pas

Théorème 3.2 : sous LayerNorm sur `Q` et `K`, la matrice `Θ̂` dérivée de la récurrence (9)
est **la plus proche, au sens de la divergence KL totale, de la Softmax plate** parmi toutes
les matrices stochastiques respectant la block constraint.

C'est un résultat d'**approximation**. Il borne l'écart à la Softmax ; il ne promet aucun
gain sur elle. Autrement dit : dans le meilleur des cas HSA vous rend l'attention Softmax,
jamais mieux — sauf par effet de régularisation en apprentissage.

## 4. Le coût

- Évaluation naïve de la récurrence (9) : `O(b²·M log_b M)`.
- Algorithme de programmation dynamique (algos 1–3, annexe E.1) : `O(M·b²)`, en deux
  passes — remontée des statistiques suffisantes (`ϕ`, `η`, `ρ_q`, `ρ_k`, `ρ_v`), puis
  descente des vecteurs d'attention.
- **Parallélisation GPU (annexe E.3)** : `ϕ(·)` et `ϑ(·)` ne se parallélisent que *par
  niveau de profondeur*. Il faut donc **`D` produits matrice creuse × vecteur séquentiels**,
  `D` étant la profondeur de `h_x`. La profondeur est un coût de premier ordre, pas un
  détail.
- Batching : impossible classiquement (arbres de formes différentes). Les auteurs
  concatènent les hiérarchies du batch sous une racine muette (*breadth-wise tree
  concatenation*).

## 5. Aucun paramètre apprenable

Annexes F et M, explicitement :

> « The proposed HSA mechanism does not introduce any trainable parameters on its own; it is
> simply an attention operation. »
>
> « our proposed framework does not introduce any additional learnable parameter across the
> hierarchy on top of the standard self-attention parameters. While this is a useful feature
> in certain usecases such as zero-shot HSA replacement post-training […] in some other
> scenarios, this would introduce a limitation in terms of the learning capacity of our
> framework. »

Les auteurs classent ce point parmi leurs **limitations**.

## 6. Ce que les expériences démontrent réellement

### 6.1 Entraînement *from scratch* — HSA gagne

| Jeu | Modèle | Word2Vec (Acc) | T5-small (Acc) |
|---|---|---:|---:|
| IMDB | Softmax plate | 0,6739 | 0,7577 |
| IMDB | **HSA** | **0,7469** | **0,8129** |
| Elec | Softmax plate | 0,7182 | 0,8212 |
| Elec | **HSA** | **0,7549** | **0,8521** |

Régime : modèles de **1,2 M paramètres entraînés de zéro**, sur des textes **longs** que
l'attention plate devait **tronquer**. Les auteurs attribuent le gain à (1) la
régularisation par scale separation, (2) l'absence de troncature. Ce n'est pas le régime
d'un modèle de fondation pré-entraîné de 224 M paramètres affiné en LoRA.

### 6.2 Remplacement *zero-shot* dans un modèle pré-entraîné — HSA perd partout

Table 3, RoBERTa-base (couches 7, 9, 11) et RoBERTa-large (couches 16–24 pour IMDB) :

| Jeu | RoBERTa original | HSA-RoBERTa | Δ Accuracy |
|---|---:|---:|---:|
| IMDB | 0,9558 | 0,9494 | −0,0064 |
| AGNEWS | 0,9469 | 0,9422 | −0,0047 |
| CoLA | 0,8150 | 0,7687 | −0,0463 |
| SST-2 | 0,9403 | 0,9025 | −0,0378 |
| MRPC | 0,9117 | 0,8553 | −0,0564 |
| RTE | 0,7833 | 0,7400 | −0,0433 |
| QNLI | 0,9267 | 0,5072 | **−0,4195** |

**7 dégradations sur 7.** La contrepartie annoncée est uniquement le nombre de FLOPs
d'attention. Les auteurs notent aussi que remplacer *toutes* les couches effondre le modèle,
et qu'il faut alterner couches HSA et couches Softmax.

### 6.3 La hiérarchie utilisée n'est pas sémantique — et cela ne change presque rien

Annexe L. Pour les expériences zero-shot, les auteurs choisissent délibérément des
hiérarchies **fixes**, par fenêtres glissantes non recouvrantes, **et non** la structure
sémantique du texte (phrases, paragraphes) :

> « The reason behind this choice is that semantic hierarchies […] are example dependant
> which means they would incur different number of FLOPs for different examples. »

Et sur quatre structures testées `(2,2,2,2)`, `(2,4,8,16)`, `(7,7,7,7)`, `(8,4,2)` :

- SST-2 : « the choice of hierarchy is relatively inconsequential » ;
- RTE : « Different hierarchy structures seem to have similar behavior » ;
- QNLI : « they do not exhibit any significant difference » ;
- MRPC : les faibles branchements en bas font mieux ;
- CoLA : les **forts** branchements en bas font mieux — soit l'inverse de MRPC.

Aucun signal structurel cohérent. Le facteur dominant est le *choix des couches remplacées*,
pas le contenu de la hiérarchie.

> C'est le résultat le plus gênant pour l'hypothèse « une meilleure hiérarchie donnera un
> meilleur modèle » : dans le papier lui-même, une hiérarchie arbitraire fait aussi bien
> qu'une hiérarchie sémantique.

## 7. Disponibilité du code

Checklist NeurIPS, question 5 (« Open access to data and code ») : *« code is not being
released at the moment until after submission. »* Aucun dépôt n'est référencé dans le
papier. Toute mise en œuvre part de zéro : algorithmes 1–3, la parallélisation par
profondeur de l'annexe E.3, la concaténation en largeur pour le batch, puis l'intégration
dans l'attention de Hiera.

## 8. Résumé en une phrase

> HSA est une **approximation hiérarchique, sans paramètre, de la Softmax**, prouvée
> KL-optimale sous contrainte de blocs, qui **échange de la résolution d'attention contre
> des FLOPs** ; elle améliore l'exactitude quand on entraîne de zéro un petit modèle sur des
> séquences trop longues, et la dégrade quand on l'injecte dans un modèle pré-entraîné.
