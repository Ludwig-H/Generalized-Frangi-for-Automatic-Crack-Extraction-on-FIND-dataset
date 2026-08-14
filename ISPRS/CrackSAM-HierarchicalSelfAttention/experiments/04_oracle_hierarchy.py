"""La hiérarchie oracle : à quoi ressemblerait la « bonne » hiérarchie, et comment la calculer.

Question posée par Louis Hauseux le 14 août 2026. Elle est décisive, parce qu'elle sépare deux
choses que l'audit confondait :

  (a) « la hiérarchie de Frangi est-elle assez bonne ? »  — mesuré, non (AUDIT §3.3, §3.5) ;
  (b) « une hiérarchie, quelle qu'elle soit, aiderait-elle ? » — jamais mesuré.

Ce script construit (b). Il produit quatre hiérarchies **directement comparables** sur la
grille de 64 x 64 tokens de SAM 2 — toutes laminaires, toutes avec les 4 096 tokens aux
feuilles, toutes de même arité — et les note.

Le raisonnement qui définit l'oracle
------------------------------------
La *block constraint* de HSA dit : `theta_ij = theta_AB` pour toutes les feuilles de deux
sous-arbres frères A et B. Sa sémantique est donc **« les tokens d'un même sous-arbre sont
interchangeables »**. Une bonne hiérarchie est celle dont les regroupements sont ceux que la
tâche autorise à confondre — et une mauvaise est celle qui **coupe** ce qu'il fallait garder
ensemble.

Pour une fissure, ce qu'il ne faut jamais couper tôt, c'est la fissure elle-même : deux tokens
voisins de la même fissure doivent rester dans le même sous-arbre le plus profondément
possible, sans quoi leur attention mutuelle est réduite à une seule valeur partagée avec tout
le fond. D'où :

> **La hiérarchie oracle est un arbre laminaire équilibré sur la grille de tokens, dont
> chaque coupe est choisie pour éviter la vérité terrain.**

C'est une bipartition récursive équilibrée à coupe minimale, sur le graphe de grille dont les
arêtes internes à la fissure sont rendues coûteuses. On la calcule par vecteur de Fiedler du
laplacien pondéré, avec coupure à la médiane pour forcer l'équilibre.

Les quatre hiérarchies comparées
--------------------------------
  `oracle`     bipartition récursive équilibrée évitant la vérité terrain (ci-dessus) ;
  `quadtree`   la même forme, sans aucune connaissance de la fissure — **le contrôle décisif** ;
  `permuted`   l'oracle d'une AUTRE image — le contrôle causal de la lignée CrackSAM ;
  `frangi`     décomposition en centroïdes du MST de Frangi non élagué, à la même résolution.

La métrique principale est la **survie de la fissure** : à chaque niveau, la fraction des
paires de tokens adjacents tous deux dans la fissure qui se retrouvent déjà séparés. Une
hiérarchie parfaite ne les sépare qu'au dernier niveau ; un quadtree les coupe tôt et
arbitrairement.

Usage:
    python ISPRS/CrackSAM-HierarchicalSelfAttention/experiments/04_oracle_hierarchy.py

Aucune donnée, aucun GPU : les fissures sont synthétisées comme dans `01`. Les hiérarchies
produites sont directement consommables par `02_attention_oracle.py --hierarchy ...`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GRID = 64  # blocs 23/33/43 de Hiera-L
ARITY = 4  # même arité pour oracle et quadtree, afin qu'ils soient comparables
PROTECT = 50.0  # poids d'une arête interne à la fissure : coûteuse à couper


# ======================================================================================
# Représentation : une hiérarchie est une liste de partitions emboîtées
# ======================================================================================
# `levels[d]` est un tableau de N entiers ; `levels[0]` vaut 0 partout (racine) et chaque
# niveau raffine le précédent. Le dernier niveau est la partition en singletons : les tokens
# sont donc bien aux feuilles, comme HSA l'exige.


def _refine(levels: list[np.ndarray], splitter) -> list[np.ndarray]:
    """Applique `splitter` à chaque cluster non singleton, jusqu'aux singletons."""
    n = len(levels[0])
    while True:
        cur = levels[-1]
        new = np.empty(n, dtype=np.int64)
        next_label = 0
        changed = False
        for c in np.unique(cur):
            idx = np.nonzero(cur == c)[0]
            if len(idx) == 1:
                new[idx] = next_label
                next_label += 1
                continue
            parts = splitter(idx)
            parts = [p for p in parts if len(p)]
            if len(parts) <= 1:  # découpage impossible : on éclate en singletons
                parts = [np.array([i]) for i in idx]
            changed = True
            for p in parts:
                new[p] = next_label
                next_label += 1
        levels.append(new)
        if not changed:
            levels.pop()
            break
        if next_label == n:
            break
    if not np.array_equal(levels[-1], np.arange(n)):
        levels.append(np.arange(n))
    return levels


def quadtree_hierarchy(grid: int = GRID) -> list[np.ndarray]:
    """Quadtree spatial pur : aucune connaissance de la fissure. Le contrôle décisif."""
    yy, xx = np.divmod(np.arange(grid * grid), grid)

    def splitter(idx):
        y, x = yy[idx], xx[idx]
        my, mx = np.median(y), np.median(x)
        return [
            idx[(y <= my) & (x <= mx)],
            idx[(y <= my) & (x > mx)],
            idx[(y > my) & (x <= mx)],
            idx[(y > my) & (x > mx)],
        ]

    return _refine([np.zeros(grid * grid, dtype=np.int64)], splitter)


def _grid_graph(grid: int, crack: np.ndarray, protect: float) -> csr_matrix:
    """Graphe de grille 4-connexe ; une arête interne à la fissure est `protect` fois plus
    coûteuse à couper qu'une arête de fond."""
    n = grid * grid
    rows, cols, vals = [], [], []
    for dy, dx in ((0, 1), (1, 0)):
        a = np.arange(n).reshape(grid, grid)
        b = np.roll(a, (-dy, -dx), axis=(0, 1))
        valid = np.ones((grid, grid), dtype=bool)
        if dy:
            valid[-1, :] = False
        if dx:
            valid[:, -1] = False
        i, j = a[valid], b[valid]
        w = np.where(crack[i] & crack[j], protect, 1.0)
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    return coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()


def crack_aware_hierarchy(
    crack: np.ndarray, grid: int = GRID, protect: float = PROTECT
) -> list[np.ndarray]:
    """**La hiérarchie oracle.** Bipartition récursive équilibrée, à coupe minimale, sur le
    graphe de grille dont les arêtes internes à la fissure sont protégées.

    Chaque coupe est le vecteur de Fiedler du laplacien pondéré, seuillé à la médiane : la
    médiane force l'équilibre (donc la profondeur logarithmique), le laplacien pondéré
    pousse la coupe à contourner la fissure plutôt qu'à la traverser.
    """
    n = grid * grid
    W = _grid_graph(grid, crack, protect)
    yy, xx = np.divmod(np.arange(n), grid)

    def quadrant(idx):
        """Découpage compact en quatre. Aux niveaux profonds c'est ce qui coupe le moins la
        fissure : une cellule carrée a le plus petit périmètre à surface donnée."""
        y, x = yy[idx], xx[idx]
        my, mx = np.median(y), np.median(x)
        return [
            idx[(y <= my) & (x <= mx)], idx[(y <= my) & (x > mx)],
            idx[(y > my) & (x <= mx)], idx[(y > my) & (x > mx)],
        ]

    def bisect(idx):
        sub = W[idx, :][:, idx]
        try:
            deg = np.asarray(sub.sum(axis=1)).ravel()
            lap = csr_matrix(np.diag(deg)) - sub
            _, vecs = eigsh(lap, k=2, sigma=-1e-6, which="LM")
            order = np.argsort(vecs[:, 1])
            h = len(idx) // 2
            return [idx[order[:h]], idx[order[h:]]]
        except Exception:
            return None

    def splitter(idx):
        # Spectral tant que le sous-problème le mérite ; sinon découpage compact, qui bat
        # toute coupe spectrale approchée sur de petites cellules.
        if len(idx) > 32:
            halves = bisect(idx)
            if halves is not None:
                out = []
                for h in halves:
                    q = bisect(h) if len(h) > 8 else None
                    out += q if q is not None else quadrant(h)
                return out
        return quadrant(idx)

    return _refine([np.zeros(n, dtype=np.int64)], splitter)


def crack_first_hierarchy(
    crack: np.ndarray, grid: int = GRID, arity: int = ARITY
) -> list[np.ndarray]:
    """**Second candidat oracle, construit à l'envers du premier.**

    Au lieu de couper l'espace en évitant la fissure, on ordonne d'abord la fissure, puis on
    lui rattache le fond. À chaque niveau :

      1. on ordonne les tokens de fissure du cluster le long du squelette (distance géodésique
         depuis une extrémité du diamètre) — deux tokens voisins de la fissure ont donc des
         positions voisines ;
      2. chaque token de fond hérite de la position de son token de fissure le plus proche ;
      3. on coupe aux quartiles de cette position.

    Les cellules obtenues sont **allongées le long de la fissure** au lieu d'être carrées, et
    l'équilibre est exact par construction (coupure aux quantiles). C'est la hiérarchie qu'on
    dessinerait à la main si l'on voulait maximiser la profondeur à laquelle la fissure est
    coupée.
    """
    n = grid * grid
    yy, xx = np.divmod(np.arange(n), grid)

    def geodesic_order(cr_idx):
        """Position le long du squelette, par double BFS sur les tokens de fissure."""
        m = len(cr_idx)
        pos = {int(v): k for k, v in enumerate(cr_idx)}
        rows, cols = [], []
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            for k, v in enumerate(cr_idx):
                y, x = yy[v] + dy, xx[v] + dx
                if 0 <= y < grid and 0 <= x < grid:
                    w = pos.get(int(y * grid + x))
                    if w is not None:
                        rows.append(k)
                        cols.append(w)
        if not rows:
            return np.zeros(m)
        g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(m, m)).tocsr()
        order, _ = breadth_first_order(g, i_start=0, directed=False,
                                       return_predecessors=True)
        far = order[-1]
        order2, _ = breadth_first_order(g, i_start=far, directed=False,
                                        return_predecessors=True)
        d = np.full(m, np.inf)
        d[order2] = np.arange(len(order2))
        d[~np.isfinite(d)] = d[np.isfinite(d)].max() + 1 if np.isfinite(d).any() else 0
        return d

    def splitter(idx):
        cr = idx[crack[idx]]
        if len(cr) < 2:  # pas de fissure ici : découpage spatial, le fond est interchangeable
            y, x = yy[idx], xx[idx]
            my, mx = np.median(y), np.median(x)
            return [
                idx[(y <= my) & (x <= mx)], idx[(y <= my) & (x > mx)],
                idx[(y > my) & (x <= mx)], idx[(y > my) & (x > mx)],
            ]
        pos_crack = geodesic_order(cr)
        # chaque token hérite de la position de son token de fissure le plus proche
        d2 = (yy[idx][:, None] - yy[cr][None, :]) ** 2 + (xx[idx][:, None] - xx[cr][None, :]) ** 2
        pos = pos_crack[np.argmin(d2, axis=1)]
        pos = pos + 1e-6 * np.argsort(np.argsort(d2.min(axis=1)))  # départage stable
        order = np.argsort(pos, kind="stable")
        return [p for p in np.array_split(idx[order], arity) if len(p)]

    return _refine([np.zeros(n, dtype=np.int64)], splitter)


def semantic_hierarchy(crack: np.ndarray, grid: int = GRID) -> list[np.ndarray]:
    """**Le vrai oracle** — et il est bien plus simple que les deux précédents.

    Les hiérarchies **équilibrées** ne peuvent pas préserver l'attention à longue portée le
    long d'une fissure : leur coupe de niveau 1 partage l'image en deux moitiés, donc coupe
    toute fissure qui la traverse. Deux tokens de fissure éloignés sont alors reliés par une
    unique valeur d'attention partagée avec ~1 000 tokens.

    Le seul remède est d'abandonner l'équilibre au sommet : mettre **la fissure entière dans
    un sous-arbre**, le fond dans l'autre. Deux tokens de fissure, si éloignés soient-ils,
    restent alors ensemble jusqu'au niveau 2, et leur dilution tombe à `|fissure| / arité`.

    Sous l'arbre, chaque partie est raffinée par découpage compact — c'est ce qui minimise la
    dilution locale (cf. `quadtree_hierarchy`).

    Remarquer ce que cette construction demande réellement : **une carte binaire
    fissure/fond**. Ni MST, ni composantes, ni centralité. C'est exactement ce que
    `node_sim_max` fournit déjà, et c'est ce que le bras `block` de
    `02_attention_oracle.py` mesure sur GPU.
    """
    n = grid * grid
    yy, xx = np.divmod(np.arange(n), grid)

    def quadrant(idx):
        y, x = yy[idx], xx[idx]
        my, mx = np.median(y), np.median(x)
        return [
            idx[(y <= my) & (x <= mx)], idx[(y <= my) & (x > mx)],
            idx[(y > my) & (x <= mx)], idx[(y > my) & (x > mx)],
        ]

    levels = [np.zeros(n, dtype=np.int64)]
    lvl1 = crack.astype(np.int64)  # niveau 1 : fissure d'un côté, fond de l'autre
    levels.append(lvl1)
    return _refine(levels, quadrant)



def centroid_hierarchy_from_tree(adj: csr_matrix) -> list[np.ndarray]:
    """Décomposition en centroïdes d'un arbre : la façon canonique de rendre **équilibré** un
    arbre qui ne l'est pas.

    À chaque étape on retire le centroïde — le nœud dont l'ablation laisse des composantes
    de taille au plus N/2 — et on récurse. La profondeur devient `O(log N)` quelle que soit
    la forme de l'arbre de départ. C'est le correctif que l'AUDIT §8.4 mentionnait sans le
    construire : appliqué au MST de Frangi, il donne à celui-ci sa meilleure chance.
    """
    n = adj.shape[0]

    def splitter(idx):
        sub = adj[idx, :][:, idx]
        ncomp, lab = connected_components(sub, directed=False)
        if ncomp > 1:  # forêt : chaque composante devient un enfant
            return [idx[lab == c] for c in range(ncomp)]
        # taille des sous-arbres, pour trouver le centroïde
        order, preds = breadth_first_order(sub, i_start=0, directed=False,
                                           return_predecessors=True)
        size = np.ones(len(idx), dtype=np.int64)
        for i in order[::-1]:
            if preds[i] >= 0:
                size[preds[i]] += size[i]
        total = len(idx)
        best, best_max = 0, total
        for v in range(total):
            children_max = 0
            for c in sub.indices[sub.indptr[v]:sub.indptr[v + 1]]:
                if preds[c] == v:
                    children_max = max(children_max, size[c])
            worst = max(children_max, total - size[v])
            if worst < best_max:
                best, best_max = v, worst
        keep = np.ones(total, dtype=bool)
        keep[best] = False
        rest = np.nonzero(keep)[0]
        parts = [idx[[best]]]
        if len(rest):
            sub2 = sub[rest, :][:, rest]
            _, lab2 = connected_components(sub2, directed=False)
            parts += [idx[rest[lab2 == c]] for c in np.unique(lab2)]
        return parts

    return _refine([np.zeros(n, dtype=np.int64)], splitter)


def random_balanced_hierarchy(n: int, seed: int = 0) -> list[np.ndarray]:
    """Contrôle : même forme, aucune structure spatiale."""
    rng = np.random.default_rng(seed)

    def splitter(idx):
        p = rng.permutation(len(idx))
        return np.array_split(idx[p], ARITY)

    return _refine([np.zeros(n, dtype=np.int64)], splitter)


# ======================================================================================
# Notation
# ======================================================================================
def hierarchy_stats(levels: list[np.ndarray]) -> dict:
    """Forme de l'arbre, et degrés de liberté que HSA laisserait à chaque token."""
    n = len(levels[0])
    depth = len(levels) - 1
    arities, dof = [], np.zeros(n)
    for d in range(depth):
        cur, nxt = levels[d], levels[d + 1]
        for c in np.unique(cur):
            idx = np.nonzero(cur == c)[0]
            k = len(np.unique(nxt[idx]))
            if k > 1:
                arities.append(k)
            dof[idx] += k - 1  # frères vus à ce niveau
    return dict(
        depth=depth,
        arity_mean=float(np.mean(arities)) if arities else 0.0,
        arity_max=int(np.max(arities)) if arities else 0,
        n_families=len(arities),
        dof_per_token_mean=float(dof.mean()),
        dof_vs_flat=float(dof.mean() / n),
    )


def crack_survival(levels: list[np.ndarray], crack: np.ndarray, grid: int = GRID) -> list[float]:
    """À chaque niveau, fraction des paires de tokens adjacents *tous deux fissure* déjà
    séparées. Bas et tardif = bon. C'est la mesure directe de « la hiérarchie coupe-t-elle
    ce qu'il fallait garder ensemble ? »"""
    a = np.arange(grid * grid).reshape(grid, grid)
    pairs = []
    for dy, dx in ((0, 1), (1, 0)):
        b = np.roll(a, (-dy, -dx), axis=(0, 1))
        valid = np.ones((grid, grid), dtype=bool)
        if dy:
            valid[-1, :] = False
        if dx:
            valid[:, -1] = False
        i, j = a[valid], b[valid]
        m = crack[i] & crack[j]
        pairs.append(np.stack([i[m], j[m]], axis=1))
    pairs = np.concatenate(pairs) if pairs else np.zeros((0, 2), dtype=int)
    if not len(pairs):
        return [0.0] * len(levels)
    return [float((lv[pairs[:, 0]] != lv[pairs[:, 1]]).mean()) for lv in levels]


def _dilution(levels: list[np.ndarray], pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Pour chaque paire, la taille du bloc dans lequel `j` est noyé vu depuis `i`."""
    depth = len(levels) - 1
    sep = np.full(len(pi), depth, dtype=np.int64)
    for d in range(depth, 0, -1):
        sep[levels[d][pi] != levels[d][pj]] = d
    dil = np.empty(len(pi), dtype=np.float64)
    for d in np.unique(sep):
        m = sep == d
        vals, counts = np.unique(levels[d], return_counts=True)
        sizes = np.zeros(int(levels[d].max()) + 1, dtype=np.int64)
        sizes[vals] = counts
        dil[m] = sizes[levels[d][pj[m]]]
    return dil


def tie_dilution(
    levels: list[np.ndarray], crack: np.ndarray, grid: int = GRID, seed: int = 0
) -> dict:
    """**La mesure comparable entre hiérarchies de profondeurs et d'arités différentes.**

    Sous la block constraint, quand `i` attend `j`, la valeur d'attention est partagée par
    toutes les feuilles du plus haut ancêtre distinct de `j` : la clé et la valeur de `j`
    sont donc **moyennées avec les `|B'| − 1` autres tokens de ce bloc**. On appelle `|B'|`
    la *dilution* — 1 = attention intacte, 1 000 = `j` est noyé dans un millier de tokens.

    Trois populations de paires, parce qu'elles ne disent pas la même chose :

      `crack_adj`  tokens voisins tous deux dans la fissure — la continuité **locale** ;
      `crack_far`  tokens de la fissure éloignés de plus de 16 tokens — la continuité **à
                   longue portée**, c'est-à-dire exactement ce que le graphe de Frangi
                   prétend apporter et que la Softmax ne sait pas encoder ;
      `background` paires de fond — contraste : une forte dilution y est *souhaitable*,
                   c'est là qu'on veut économiser.
    """
    n = grid * grid
    a = np.arange(n).reshape(grid, grid)
    pi, pj, adj_crack = [], [], []
    for dy, dx in ((0, 1), (1, 0)):
        b = np.roll(a, (-dy, -dx), axis=(0, 1))
        valid = np.ones((grid, grid), dtype=bool)
        if dy:
            valid[-1, :] = False
        if dx:
            valid[:, -1] = False
        i, j = a[valid], b[valid]
        pi.append(i)
        pj.append(j)
        adj_crack.append(crack[i] & crack[j])
    pi, pj = np.concatenate(pi), np.concatenate(pj)
    adj_crack = np.concatenate(adj_crack)

    out = {}
    d_adj = _dilution(levels, pi, pj)
    out["crack_adj_median"] = float(np.median(d_adj[adj_crack])) if adj_crack.any() else 0.0
    out["crack_adj_mean"] = float(d_adj[adj_crack].mean()) if adj_crack.any() else 0.0
    out["background_median"] = float(np.median(d_adj[~adj_crack]))
    out["n_crack_adj_pairs"] = int(adj_crack.sum())

    # paires de fissure éloignées : la continuité à longue portée
    cr = np.nonzero(crack)[0]
    yy, xx = np.divmod(cr, grid)
    rng = np.random.default_rng(seed)
    if len(cr) > 4:
        k = min(40000, len(cr) * 40)
        u = rng.integers(0, len(cr), k)
        v = rng.integers(0, len(cr), k)
        far = (np.abs(yy[u] - yy[v]) > 16) | (np.abs(xx[u] - xx[v]) > 16)
        u, v = u[far], v[far]
        if len(u):
            d_far = _dilution(levels, cr[u], cr[v])
            out["crack_far_median"] = float(np.median(d_far))
            out["crack_far_mean"] = float(d_far.mean())
            out["n_crack_far_pairs"] = int(len(u))
    out.setdefault("crack_far_median", 0.0)
    out.setdefault("crack_far_mean", 0.0)
    return out


def mean_separation_level(survival: list[float]) -> float:
    """Résumé de la courbe de survie : niveau moyen auquel une paire fissure-fissure est
    séparée. Plus c'est **haut**, plus l'attention fine entre tokens de la fissure est
    préservée profondément — donc mieux c'est."""
    inc = np.diff([0.0] + list(survival))
    lv = np.arange(len(survival))
    tot = inc.sum()
    return float((inc * lv).sum() / tot) if tot > 0 else 0.0


def level_ari(a: list[np.ndarray], b: list[np.ndarray]) -> list[float]:
    """Accord partition-à-partition, niveau par niveau (Rand ajusté)."""
    return [float(adjusted_rand_score(a[d], b[d])) for d in range(min(len(a), len(b)))]


# ======================================================================================
def build_all(seed: int = 1, verbose: bool = True) -> dict:
    import importlib

    mod = importlib.import_module("01_frangi_tree_shape")

    def token_crack(seed_):
        img, gt = mod.synth_crack(
            size=448, seed=seed_, n_branches=2, width=9, trunk_scale=1.1, wander=0.045
        )
        k = 448 // GRID
        img_t = img.reshape(GRID, k, GRID, k).mean(axis=(1, 3))
        gt_t = gt.reshape(GRID, k, GRID, k).any(axis=(1, 3))
        return img_t.astype(np.float32), gt_t.ravel()

    img_t, crack = token_crack(seed)
    _, crack_other = token_crack(seed + 100)  # pour le contrôle permuté

    hierarchies = {}
    if verbose:
        print(f"fissure : {100 * crack.mean():.1f} % des {GRID * GRID} tokens\n")

    hierarchies["spatial_mincut"] = crack_aware_hierarchy(crack)
    hierarchies["quadtree"] = quadtree_hierarchy()
    hierarchies["semantic"] = semantic_hierarchy(crack)
    hierarchies["semantic_permuted"] = semantic_hierarchy(crack_other)
    hierarchies["crack_ordered"] = crack_first_hierarchy(crack)
    hierarchies["spatial_permuted"] = crack_aware_hierarchy(crack_other)
    hierarchies["random"] = random_balanced_hierarchy(GRID * GRID)

    # Frangi : MST non élagué à la résolution des tokens, rendu équilibré par centroïdes
    g = mod.frangi_mst(img_t, prune=False)
    if g is not None and g["mst"].shape[0] == GRID * GRID:
        adj = (g["mst"] + g["mst"].T) > 0
        hierarchies["frangi_centroid"] = centroid_hierarchy_from_tree(adj.tocsr())
    elif verbose:
        print("(Frangi ignoré : le MST ne couvre pas tous les tokens)")

    out = {}
    for name, lv in hierarchies.items():
        out[name] = dict(
            **hierarchy_stats(lv),
            crack_survival=(cs := crack_survival(lv, crack)),
            mean_separation_level=mean_separation_level(cs),
            tie_dilution=tie_dilution(lv, crack),
            ari_vs_oracle=level_ari(hierarchies["semantic"], lv),
        )
    return dict(results=out, crack_token_pct=float(100 * crack.mean()), seed=seed)


ORDER = ("semantic", "semantic_permuted", "spatial_mincut", "spatial_permuted",
         "crack_ordered", "frangi_centroid", "quadtree", "random")


def _average(runs: list[dict]) -> dict:
    """Moyenne les mesures sur plusieurs images ; les listes sont moyennées terme à terme."""
    names = [n for n in ORDER if n in runs[0]["results"]]
    out = {}
    for name in names:
        acc = {}
        for key in runs[0]["results"][name]:
            vals = [r["results"][name][key] for r in runs if name in r["results"]]
            if isinstance(vals[0], dict):
                acc[key] = {k: float(np.mean([v[k] for v in vals])) for k in vals[0]}
            elif isinstance(vals[0], list):
                m = min(len(v) for v in vals)
                acc[key] = [float(np.mean([v[d] for v in vals])) for d in range(m)]
            else:
                acc[key] = float(np.mean(vals))
        out[name] = acc
    return dict(
        results=out,
        crack_token_pct=float(np.mean([r["crack_token_pct"] for r in runs])),
        n_images=len(runs),
        seeds=[r["seed"] for r in runs],
    )


def report(res: dict) -> None:
    r = res["results"]
    print(f"{'hiérarchie':<18}{'prof.':>7}{'arité':>8}{'familles':>10}"
          f"{'ARI/oracle':>12}{'dil. local':>12}{'dil. longue':>13}")
    print("-" * 80)
    for name in ORDER:
        if name not in r:
            continue
        s = r[name]
        ari = s["ari_vs_oracle"]
        mid = ari[min(3, len(ari) - 1)]
        t = s["tie_dilution"]
        print(f"{name:<18}{s['depth']:>7.1f}{s['arity_mean']:>8.2f}{s['n_families']:>10.0f}"
              f"{mid:>12.3f}{t['crack_adj_median']:>12.0f}"
              f"{t['crack_far_median']:>13.0f}")

    print("\n« dilution » = nombre de tokens moyennés avec j quand i l'attend ; 1 = intact.")
    print("« locale » = paires de fissure adjacentes ; « longue » = éloignées de > 16 tokens.")
    print(f"\nSurvie de la fissure — fraction des paires fissure–fissure déjà séparées")
    print(f"{'niveau':<18}" + "".join(f"{d:>8}" for d in range(1, 7)))
    print("-" * 80)
    for name in ORDER:
        if name not in r:
            continue
        cs = r[name]["crack_survival"]
        cells = "".join(f"{100 * cs[d]:>7.0f}%" if d < len(cs) else f"{'—':>8}"
                        for d in range(1, 7))
        print(f"{name:<18}{cells}")


def self_test() -> int:
    """Vérifie les invariants sans rien synthétiser de lourd."""
    n = GRID * GRID
    crack = np.zeros(n, dtype=bool)
    crack.reshape(GRID, GRID)[30, 8:56] = True  # un trait horizontal

    for name, lv in (
        ("quadtree", quadtree_hierarchy()),
        ("oracle", crack_aware_hierarchy(crack)),
        ("random", random_balanced_hierarchy(n)),
    ):
        assert len(lv[0]) == n and lv[0].max() == 0, f"{name}: niveau 0 doit être la racine"
        assert np.array_equal(np.sort(lv[-1]), np.arange(n)), f"{name}: feuilles = tokens"
        for d in range(len(lv) - 1):  # laminarité : chaque niveau raffine le précédent
            for c in np.unique(lv[d + 1]):
                assert len(np.unique(lv[d][lv[d + 1] == c])) == 1, f"{name}: non laminaire"
        print(f"  {name:<10} profondeur {len(lv) - 1}, arité {hierarchy_stats(lv)['arity_mean']:.2f}")

    o = crack_survival(crack_aware_hierarchy(crack), crack)
    q = crack_survival(quadtree_hierarchy(), crack)
    assert o[2] <= q[2] + 1e-9, "l'oracle doit couper la fissure moins tôt que le quadtree"
    print(f"  survie au niveau 2 : oracle {100*o[2]:.0f} % contre quadtree {100*q[2]:.0f} %")
    print("auto-test OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-images", type=int, default=1)
    ap.add_argument("--report-only", action="store_true",
                    help="réaffiche results/oracle_hierarchy.json sans recalculer")
    args = ap.parse_args()
    if args.self_test:
        return self_test()

    out = HERE.parent / "results" / "oracle_hierarchy.json"
    if args.report_only:
        report(json.loads(out.read_text()))
        return 0
    runs = [build_all(seed=args.seed + k, verbose=(k == 0)) for k in range(args.n_images)]
    res = _average(runs)
    report(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nÉcrit : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
