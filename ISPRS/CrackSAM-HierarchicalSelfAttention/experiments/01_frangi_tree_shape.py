"""L'arbre Frangi est-il une « signal hierarchy » au sens de HSA ?

HSA (NeurIPS 2025) opère sur une *signal hierarchy* : un arbre **enraciné**, **laminaire**,
dont les **feuilles sont les tokens** et dont les nœuds internes sont des *regroupements*.
Son coût est O(M b²) avec M le nombre de familles et b le facteur de branchement maximal,
et sa parallélisation GPU coûte D produits creux séquentiels, D étant la **profondeur**.

Le MST du graphe de Frangi est un arbre, mais pas celui-là : ses nœuds *sont* les pixels.
Ce script mesure, sur des fissures synthétiques dont on contrôle la géométrie, les quantités
qui décident de la compatibilité :

  1. le nombre de nœuds candidats N ;
  2. la profondeur D et le facteur de branchement b de l'arbre enraciné au maximum de
     centralité (l'enracinement que fait `extract_backbone_centrality`) ;
  3. la distribution des degrés — un squelette curviligne donne un arbre *chemin* ;
  4. l'empreinte de la fissure sur la grille de tokens 64x64 de SAM 2, seule résolution
     où une matrice d'attention globale existe.

Le point 4 est le plus important : il dit quelle fraction de la matrice d'attention que l'on
prétend contraindre est effectivement couverte par la géométrie de Frangi.

Usage:
    python ISPRS/CrackSAM-HierarchicalSelfAttention/experiments/01_frangi_tree_shape.py

Aucune donnée requise : les fissures sont synthétisées. Pour rejouer sur FIND, passer
--find-root ; la même instrumentation s'applique aux vraies images.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import (  # noqa: E402
    breadth_first_order,
    connected_components,
    minimum_spanning_tree,
)

from ISPRS.src.frangi_hessian import FrangiHessianGPU  # noqa: E402

# Paramètres FIND du papier EUVIP (§III-C) : ss=2, si=1/4, sa=1/8, Sigma={1,3,5,7}, R=3, tau=0.25
SIGMA = [1.0, 3.0, 5.0, 7.0]
RADIUS = 3
SS, SI, SA = 2.0, 0.25, 0.125
TAU = 0.25

SAM2_INPUT = 1024
SAM2_TOKEN_GRID = 64  # blocs 23/33/43 de Hiera-L : seule attention globale du tronc


# --------------------------------------------------------------------------------------
# Fissures synthétiques
# --------------------------------------------------------------------------------------
def synth_crack(
    size: int = 256,
    seed: int = 0,
    n_branches: int = 3,
    width: float = 1.6,
    trunk_scale: float = 2.0,
    wander: float = 0.12,
):
    """Marche aléatoire persistante + branches, épaissie puis floutée. Renvoie (image, masque)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=np.float32)

    def walk(y, x, theta, length, w):
        pts = []
        for _ in range(length):
            theta += rng.normal(0, wander)
            y += np.sin(theta)
            x += np.cos(theta)
            if not (2 <= y < size - 2 and 2 <= x < size - 2):
                break
            pts.append((y, x))
        for y_, x_ in pts:
            yy, xx = int(round(y_)), int(round(x_))
            r = int(np.ceil(w))
            mask[max(0, yy - r) : yy + r + 1, max(0, xx - r) : xx + r + 1] = np.maximum(
                mask[max(0, yy - r) : yy + r + 1, max(0, xx - r) : xx + r + 1], 1.0
            )
        return pts

    trunk = walk(size * 0.1, size * 0.15, 0.9, int(size * trunk_scale), width)
    for k in range(n_branches):
        if not trunk:
            break
        y0, x0 = trunk[rng.integers(len(trunk) // 5, max(len(trunk) // 5 + 1, len(trunk) - 1))]
        walk(y0, x0, rng.uniform(0, 2 * np.pi), int(size * 0.6), max(1.0, width - 0.5))

    # Fissure sombre sur fond texturé, comme une image d'intensité FIND
    bg = 0.65 + 0.05 * rng.standard_normal((size, size)).astype(np.float32)
    img = bg - 0.45 * mask
    img = np.clip(img, 0, 1)
    return img.astype(np.float32), (mask > 0)


# --------------------------------------------------------------------------------------
# Graphe de Frangi -> MST, instrumenté
# --------------------------------------------------------------------------------------
def frangi_mst(img: np.ndarray, device: str = "cpu"):
    """Reproduit `extract_frangi_graph_gpu` (K=1) mais **renvoie l'arbre** au lieu de le
    rastériser. C'est exactement le MST que consomme `extract_backbone_centrality`."""
    fh = FrangiHessianGPU(SIGMA, device=device)
    t = torch.from_numpy(img).to(device)

    scale_data, max_S = [], None
    for s in SIGMA:
        ixx, ixy, iyy = fh.compute_hessian(t, s)
        trace = ixx + iyy
        disc = torch.sqrt((ixx - iyy) ** 2 + 4 * ixy**2)
        max_norm = torch.max((torch.abs(trace) + disc) / 2.0) + 1e-8
        l1, l2, th = fh.compute_eigenvalues_and_vectors(ixx / max_norm, ixy / max_norm, iyy / max_norm)
        pos = l2 > 0
        rb = torch.zeros_like(l2)
        rb[pos] = torch.abs(l1[pos]) / (l2[pos] + 1e-8)
        sn = torch.zeros_like(l2)
        sn[pos] = l2[pos]
        max_S = sn.clone() if max_S is None else torch.max(max_S, sn)
        scale_data.append((rb, sn, th))

    H, W = max_S.shape
    cand = max_S > max_S.max() * 0.01
    coords = torch.nonzero(cand)
    N = int(coords.shape[0])
    if N == 0:
        return None

    idx_map = torch.full((H, W), -1, dtype=torch.long, device=device)
    idx_map[cand] = torch.arange(N, device=device)
    padded = torch.nn.functional.pad(
        idx_map.unsqueeze(0).unsqueeze(0).float(), (RADIUS,) * 4, value=-1
    ).long()[0, 0]
    patches = padded.unfold(0, 2 * RADIUS + 1, 1).unfold(1, 2 * RADIUS + 1, 1)
    ys, xs = torch.nonzero(cand, as_tuple=True)
    cand_patches = patches[ys, xs]

    dy, dx = torch.meshgrid(
        torch.arange(-RADIUS, RADIUS + 1, device=device),
        torch.arange(-RADIUS, RADIUS + 1, device=device),
        indexing="ij",
    )
    d2 = dx**2 + dy**2
    keep = (d2 <= RADIUS**2) & (d2 > 0) & ((dy > 0) | ((dy == 0) & (dx > 0)))
    dist = torch.sqrt(d2[keep].float())

    neigh = cand_patches[:, keep]
    src = torch.arange(N, device=device).unsqueeze(1).expand(-1, neigh.shape[1])
    ok = neigh != -1
    i_idx, j_idx = src[ok], neigh[ok]
    if len(i_idx) == 0:
        return None

    S = torch.zeros(len(i_idx), device=device)
    for rb, sn, th in scale_data:
        rb_c, s_c, t_c = rb[cand], sn[cand], th[cand]
        s_shape = torch.exp(-0.5 * ((rb_c[i_idx] + rb_c[j_idx]) / SS) ** 2)
        s_int = 1 - torch.exp(-0.5 * ((s_c[i_idx] * s_c[j_idx]) / SI) ** 2)
        s_align = torch.exp(-0.5 * (torch.sin(t_c[i_idx] - t_c[j_idx]) / SA) ** 2)
        S = torch.max(S, s_shape * s_int * s_align)

    d_ij = (1 - S) * dist.unsqueeze(0).expand(N, -1)[ok] + 1e-8

    # Seuillage arêtes puis nœuds, à tau
    n_e = len(S)
    k_e = max(1, int(n_e * TAU))
    thr = torch.kthvalue(S, n_e - k_e + 1).values.item() if n_e > k_e else -1.0
    m = (S >= thr).cpu().numpy()
    i_v, j_v = i_idx.cpu().numpy()[m], j_idx.cpu().numpy()[m]
    S_v, d_v = S.cpu().numpy()[m], d_ij.cpu().numpy()[m]

    node_sim = np.zeros(N, dtype=np.float32)
    np.maximum.at(node_sim, i_v, S_v)
    np.maximum.at(node_sim, j_v, S_v)

    k_n = max(1, int(N * TAU))
    cands = np.unique(np.concatenate([i_v, j_v]))
    if len(cands) > k_n:
        sims = node_sim[cands]
        thr_n = np.partition(sims, -k_n)[-k_n]
        valid = cands[sims >= thr_n]
    else:
        valid = cands

    keep_node = np.zeros(N, dtype=bool)
    keep_node[valid] = True
    fm = keep_node[i_v] & keep_node[j_v]
    i_v, j_v, S_v, d_v = i_v[fm], j_v[fm], S_v[fm], d_v[fm]
    if len(i_v) == 0:
        return None

    remap = np.full(N, -1, dtype=np.int64)
    remap[valid] = np.arange(len(valid))
    im, jm = remap[i_v], remap[j_v]
    nv = len(valid)
    coords_v = coords.cpu().numpy()[valid]

    dist_m = coo_matrix(
        (np.concatenate([d_v, d_v]), (np.concatenate([im, jm]), np.concatenate([jm, im]))),
        shape=(nv, nv),
    ).tocsr()
    sim_m = coo_matrix(
        (np.concatenate([S_v, S_v]), (np.concatenate([im, jm]), np.concatenate([jm, im]))),
        shape=(nv, nv),
    ).tocsr()

    # Plus grande composante (prior FIND : une seule structure d'intérêt)
    _, labels = connected_components(dist_m, directed=False)
    big = np.argmax(np.bincount(labels))
    sel = np.where(labels == big)[0]
    mst = minimum_spanning_tree(dist_m[sel, :][:, sel])
    return dict(
        mst=mst,
        sim=sim_m[sel, :][:, sel],
        coords=coords_v[sel],
        n_candidates=N,
        n_graph_nodes=nv,
        n_tree_nodes=len(sel),
        shape=(H, W),
    )


def tree_stats(mst, sim, coords, image_shape):
    """Enracine l'arbre au maximum de centralité (comme extract_backbone_centrality),
    puis mesure profondeur, branchement, et empreinte sur la grille de tokens SAM 2."""
    n = mst.shape[0]
    sym = mst + mst.T
    order, preds = breadth_first_order(sym, i_start=0, directed=False, return_predecessors=True)

    w = np.asarray(sim.max(axis=1).todense()).ravel().astype(np.float64)
    mass = w.copy()
    for i in order[::-1]:
        p = preds[i]
        if p >= 0:
            mass[p] += mass[i]
    total = mass[order[0]]
    cent = mass * (total - mass)
    root = int(np.argmax(cent))

    order, preds = breadth_first_order(sym, i_start=root, directed=False, return_predecessors=True)

    # Profondeur et branchement de l'arbre enraciné en `root`
    children = [[] for _ in range(n)]
    for i in order:
        p = preds[i]
        if p >= 0:
            children[p].append(i)
    depth = np.zeros(n, dtype=np.int64)
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for c in children[u]:
            depth[c] = depth[u] + 1
            dq.append(c)
    n_children = np.array([len(c) for c in children])
    internal = n_children > 0

    # --- Degrés de liberté d'une ligne de la matrice d'attention sous HSA ------------
    # HSA impose la « block constraint » : theta_ij = theta_AB pour tout i dans l(A),
    # j dans l(B), A et B frères. Le nombre de valeurs DISTINCTES que voit le token i
    # vaut donc  sum_{A ancêtre de i} |sib(A)| .
    # Conversion canonique du MST en signal hierarchy : chaque nœud v du MST devient un
    # nœud interne dont les enfants sont {feuille(v)} U {internes des enfants de v}.
    # Une famille a donc 1 + n_children(v) membres, et |sib| = n_children(parent(v)).
    family_size = n_children + 1  # +1 pour la feuille qui porte le token
    sib_count = np.zeros(n, dtype=np.float64)
    for i in order:
        p = preds[i]
        sib_count[i] = (family_size[p] - 1) if p >= 0 else 0.0
    dof = np.zeros(n, dtype=np.float64)
    for i in order:  # ordre BFS : le parent est toujours traité avant l'enfant
        p = preds[i]
        dof[i] = (dof[p] if p >= 0 else 0.0) + sib_count[i]
    dof += n_children  # + les frères de la feuille elle-même dans sa propre famille

    # Empreinte sur la grille de tokens 64x64 de SAM 2
    h, w_img = image_shape
    scale = SAM2_TOKEN_GRID / max(h, w_img)
    tok = np.zeros((SAM2_TOKEN_GRID, SAM2_TOKEN_GRID), dtype=bool)
    ty = np.clip((coords[:, 0] * scale).astype(int), 0, SAM2_TOKEN_GRID - 1)
    tx = np.clip((coords[:, 1] * scale).astype(int), 0, SAM2_TOKEN_GRID - 1)
    tok[ty, tx] = True
    n_tok = int(tok.sum())
    n_total_tok = SAM2_TOKEN_GRID**2

    return dict(
        n_tree_nodes=int(n),
        depth_max=int(depth.max()),
        depth_mean=float(depth.mean()),
        branching_max=int(n_children.max()),
        branching_mean_internal=float(n_children[internal].mean()) if internal.any() else 0.0,
        frac_nodes_with_1_child=float((n_children == 1).sum() / max(1, internal.sum())),
        n_leaves=int((n_children == 0).sum()),
        # Ce que HSA exigerait d'un arbre équilibré de même taille :
        depth_if_balanced_b8=float(np.log(max(n, 2)) / np.log(8)),
        # Valeurs d'attention distinctes vues par un token, sous la block constraint
        hsa_row_dof_mean=float(dof.mean()),
        hsa_row_dof_median=float(np.median(dof)),
        hsa_row_dof_vs_flat=float(dof.mean() / n),
        # Empreinte SAM 2
        sam2_crack_tokens=n_tok,
        sam2_total_tokens=n_total_tok,
        sam2_crack_token_pct=100.0 * n_tok / n_total_tok,
        sam2_attention_cells_covered_pct=100.0 * (n_tok / n_total_tok) ** 2,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=256, help="résolution native (FIND = 256)")
    ap.add_argument("--n-images", type=int, default=5)
    ap.add_argument("--width", type=float, default=1.6, help="demi-largeur de la fissure en px")
    ap.add_argument("--branches", type=int, default=3)
    ap.add_argument("--trunk-scale", type=float, default=2.0)
    ap.add_argument("--tag", type=str, default="default", help="suffixe du fichier de résultats")
    ap.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=None,
        help="échelles du Hessien (défaut : 1 3 5 7, les valeurs FIND du papier EUVIP)",
    )
    args = ap.parse_args()
    if args.sigma:
        global SIGMA
        SIGMA = list(args.sigma)

    rows = []
    for seed in range(args.n_images):
        img, gt = synth_crack(
            size=args.size,
            seed=seed,
            n_branches=args.branches,
            width=args.width,
            trunk_scale=args.trunk_scale,
        )
        g = frangi_mst(img)
        if g is None:
            print(f"seed {seed}: graphe vide, ignoré")
            continue
        st = tree_stats(g["mst"], g["sim"], g["coords"], g["shape"])
        st.update(
            seed=seed,
            n_candidates=g["n_candidates"],
            n_graph_nodes=g["n_graph_nodes"],
            gt_pixel_pct=100.0 * gt.mean(),
        )
        rows.append(st)
        print(
            f"seed {seed}: N_arbre={st['n_tree_nodes']:5d}  profondeur={st['depth_max']:4d}  "
            f"b_max={st['branching_max']}  b_moyen={st['branching_mean_internal']:.2f}  "
            f"1 seul enfant={100*st['frac_nodes_with_1_child']:.0f}%  "
            f"tokens SAM2={st['sam2_crack_tokens']:4d}/{st['sam2_total_tokens']} "
            f"({st['sam2_crack_token_pct']:.1f}%)"
        )

    if not rows:
        return 1

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    print("\n" + "=" * 78)
    print("MOYENNES")
    print("=" * 78)
    print(f"nœuds de l'arbre                        : {avg('n_tree_nodes'):.0f}")
    print(f"profondeur de l'arbre enraciné          : {avg('depth_max'):.0f}")
    print(f"profondeur d'un arbre équilibré (b=8)   : {avg('depth_if_balanced_b8'):.1f}")
    print(f"facteur de branchement moyen (internes) : {avg('branching_mean_internal'):.2f}")
    print(f"nœuds internes à UN SEUL enfant         : {100*avg('frac_nodes_with_1_child'):.0f} %")
    print(f"valeurs d'attention distinctes / token  : {avg('hsa_row_dof_mean'):.0f}"
          f"  (attention plate : {avg('n_tree_nodes'):.0f})")
    print(f"tokens SAM 2 touchés par la fissure     : {avg('sam2_crack_tokens'):.0f} / 4096"
          f"  ({avg('sam2_crack_token_pct'):.1f} %)")
    print(f"cellules de la matrice d'attention 4096²")
    print(f"  couvertes par la géométrie de Frangi  : {avg('sam2_attention_cells_covered_pct'):.2f} %")
    print(f"(couverture de la vérité terrain : {avg('gt_pixel_pct'):.1f} % des pixels)")

    out = Path(__file__).resolve().parents[1] / "results" / f"frangi_tree_shape_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(per_image=rows, params=dict(sigma=SIGMA, R=RADIUS, tau=TAU)), indent=2))
    print(f"\nÉcrit : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
