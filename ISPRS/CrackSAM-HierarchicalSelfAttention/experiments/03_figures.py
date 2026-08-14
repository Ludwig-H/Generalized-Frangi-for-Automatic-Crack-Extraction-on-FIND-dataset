"""Les quatre figures pédagogiques de l'audit.

Chaque figure répond à une question que l'on doit se poser avant d'écrire une ligne de code
d'attention hiérarchique. Elles sont toutes engendrées à partir de mesures, jamais dessinées
à la main.

    fig1 — Où sont les matrices d'attention de SAM 2, et combien coûtent-elles ?
    fig2 — À quoi ressemble vraiment l'arbre du Frangi-Graphe ?
    fig3 — Quelle part de la matrice d'attention la géométrie de Frangi couvre-t-elle ?
    fig4 — Que fait la « block constraint » de HSA à une matrice d'attention ?

Usage:
    python ISPRS/CrackSAM-HierarchicalSelfAttention/experiments/03_figures.py

Aucune donnée, aucun GPU. Réutilise `01_frangi_tree_shape.py` pour la géométrie.
"""

from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
sys.path.insert(0, str(HERE))

FG = "#1f2933"
ACCENT = "#c0392b"
COOL = "#2c7fb8"
MUTED = "#9aa5b1"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": FG,
        "text.color": FG,
        "xtick.color": FG,
        "ytick.color": FG,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


# --------------------------------------------------------------------------------------
def fig1_ou_est_lattention() -> None:
    """Carte des 48 blocs de Hiera-L et répartition des FLOPs."""
    data = json.loads((RESULTS / "sam2_attention_budget.json").read_text())
    blocks, flops = data["blocks"], data["flops"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.4, 1]}
    )

    tokens = [b["tokens"] for b in blocks]
    colors = [ACCENT if b["is_global"] else MUTED for b in blocks]
    ax1.bar(range(len(blocks)), tokens, color=colors, width=0.85)
    ax1.set_yscale("log")
    ax1.set_xlabel("indice du bloc dans le tronc Hiera-L")
    ax1.set_ylabel("tokens (échelle log)")
    ax1.set_title(
        "48 blocs, 3 attentions globales — et elles sont toutes au même étage",
        loc="left",
        weight="bold",
    )
    for b in blocks:
        if b["is_global"]:
            ax1.annotate(
                str(b["block"]),
                (b["block"], b["tokens"]),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                color=ACCENT,
                fontsize=8,
                weight="bold",
            )
    ax1.text(
        0.5,
        0.94,
        "gris : attention fenêtrée (déjà locale)      rouge : attention globale, 64×64 = 4 096 tokens",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color=FG,
    )

    parts = [
        ("MLP", flops["mlp_pct"], MUTED),
        ("projections", flops["projections_pct"], "#cbd2d9"),
        ("attention fenêtrée", flops["attention_matmuls_pct"] - flops["global_attention_pct"], COOL),
        ("attention globale", flops["global_attention_pct"], ACCENT),
    ]
    ax2.pie(
        [p[1] for p in parts],
        colors=[p[2] for p in parts],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
    )
    ax2.text(
        0,
        0.05,
        f"{flops['global_attention_pct']:.1f} %",
        ha="center",
        va="center",
        fontsize=17,
        weight="bold",
        color=ACCENT,
    )
    ax2.text(0, -0.22, "attention\nglobale", ha="center", va="center", fontsize=8, color=FG)
    ax2.set_title(
        f"Budget FLOPs du tronc ({flops['total_gflops']:.0f} G)", loc="center", weight="bold"
    )

    fig.suptitle(
        "Fig. 1 — Tout ce qu'une attention hiérarchique pourrait remplacer dans SAM 2",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig1_ou_est_lattention.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def _measure_tree():
    """Un arbre de Frangi réel, calibré sur la géométrie de Khánh Hà."""
    import importlib

    mod = importlib.import_module("01_frangi_tree_shape")
    img, gt = mod.synth_crack(size=448, seed=1, n_branches=2, width=9, trunk_scale=1.1, wander=0.045)
    g = mod.frangi_mst(img)
    if g is None:
        raise RuntimeError("graphe vide")

    from scipy.sparse.csgraph import breadth_first_order

    sym = g["mst"] + g["mst"].T
    n = sym.shape[0]
    sim = g["sim"]
    order, preds = breadth_first_order(sym, i_start=0, directed=False, return_predecessors=True)
    w = np.asarray(sim.max(axis=1).todense()).ravel().astype(np.float64)
    mass = w.copy()
    for i in order[::-1]:
        if preds[i] >= 0:
            mass[preds[i]] += mass[i]
    cent = mass * (mass[order[0]] - mass)
    root = int(np.argmax(cent))
    order, preds = breadth_first_order(sym, i_start=root, directed=False, return_predecessors=True)

    children = [[] for _ in range(n)]
    for i in order:
        if preds[i] >= 0:
            children[preds[i]].append(i)
    depth = np.zeros(n, dtype=int)
    dq = deque([root])
    while dq:
        u = dq.popleft()
        for c in children[u]:
            depth[c] = depth[u] + 1
            dq.append(c)
    return dict(
        n=n,
        n_children=np.array([len(c) for c in children]),
        depth=depth,
        coords=g["coords"],
        gt=gt,
        img=img,
    )


def fig2_forme_de_larbre(tree) -> None:
    """Distribution des degrés et profondeur — l'arbre est une chenille."""
    n_children, depth, n = tree["n_children"], tree["depth"], tree["n"]
    internal = n_children[n_children > 0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(11, 3.4))

    counts = np.bincount(internal, minlength=7)[:7]
    bars = ax1.bar(range(1, 7), counts[1:7], color=MUTED, width=0.7)
    bars[0].set_color(ACCENT)
    ax1.set_xlabel("nombre d'enfants d'un nœud interne")
    ax1.set_ylabel("nœuds")
    pct = 100 * counts[1] / internal.size
    ax1.set_title(f"{pct:.0f} % des nœuds internes\nn'ont qu'UN enfant", loc="left", weight="bold")
    ax1.annotate(
        "un nœud à un seul enfant\nne regroupe rien",
        (1, counts[1]),
        textcoords="offset points",
        xytext=(28, -18),
        fontsize=8,
        color=ACCENT,
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1),
    )

    ax2.hist(depth, bins=60, color=COOL)
    balanced = np.log(n) / np.log(8)
    ax2.axvline(balanced, color=ACCENT, lw=2)
    ax2.annotate(
        f"arbre équilibré (b=8)\nde même taille : {balanced:.1f}",
        (balanced, ax2.get_ylim()[1] * 0.72),
        textcoords="offset points",
        xytext=(30, 0),
        fontsize=8,
        color=ACCENT,
        arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1),
    )
    ax2.set_xlabel("profondeur d'un nœud")
    ax2.set_ylabel("nœuds")
    ax2.set_title(
        f"Profondeur max {depth.max()}\ncontre {balanced:.1f} si équilibré",
        loc="left",
        weight="bold",
    )

    ax3.axis("off")
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    def draw_tree(ax, cx, top, kind):
        if kind == "balanced":
            pos = {0: (cx, top)}
            edges = []
            k = 1
            for dx in (-0.22, 0.0, 0.22):
                pos[k] = (cx + dx, top - 0.13)
                edges.append((0, k))
                base = k
                k += 1
                for ddx in (-0.07, 0.07):
                    pos[k] = (cx + dx + ddx, top - 0.26)
                    edges.append((base, k))
                    k += 1
            col = COOL
        else:
            pos, edges = {}, []
            for i in range(8):
                pos[i] = (cx - 0.12 + 0.022 * i, top - 0.033 * i)
                if i:
                    edges.append((i - 1, i))
            for j, anchor in enumerate((2, 5)):
                pos[100 + j] = (pos[anchor][0] + 0.14, pos[anchor][1] - 0.025)
                edges.append((anchor, 100 + j))
            col = ACCENT
        for a, b in edges:
            ax.plot(*zip(pos[a], pos[b]), color=col, lw=1.2, zorder=1)
        for point in pos.values():
            ax.plot(*point, "o", ms=4.5, color=col, zorder=2)

    ax3.text(0.24, 0.97, "ce que HSA attend", ha="center", fontsize=9, color=COOL, weight="bold")
    ax3.text(0.76, 0.97, "ce que Frangi donne", ha="center", fontsize=9, color=ACCENT, weight="bold")
    draw_tree(ax3, 0.24, 0.86, "balanced")
    draw_tree(ax3, 0.76, 0.86, "caterpillar")
    ax3.text(0.24, 0.50, f"b ~ 8\nprofondeur ~ log8(N)", ha="center", va="top", fontsize=8.5, color=COOL)
    ax3.text(0.76, 0.50, f"b = {internal.mean():.2f}\nprofondeur = {depth.max()}", ha="center",
             va="top", fontsize=8.5, color=ACCENT)
    ax3.plot([0.5, 0.5], [0.36, 0.98], color=MUTED, lw=0.8, ls=":")
    ax3.text(
        0.5,
        0.28,
        "Le MST d'une structure curviligne est un chemin.\n"
        "La programmation dynamique de HSA y devient un\n"
        f"balayage sequentiel de profondeur {depth.max()},\n"
        "au lieu d'un seul produit matriciel.",
        ha="center",
        va="top",
        fontsize=8.5,
        color=FG,
    )

    fig.suptitle(
        "Fig. 2 — Le MST de Frangi n'est pas une hiérarchie : c'est un chemin",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig2_forme_de_larbre.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def fig3_couverture_attention(tree) -> None:
    """De l'image à la grille de tokens, puis à la matrice d'attention."""
    grid = 64
    coords, img, gt = tree["coords"], tree["img"], tree["gt"]
    h = img.shape[0]
    scale = grid / h
    tok = np.zeros((grid, grid), dtype=bool)
    ty = np.clip((coords[:, 0] * scale).astype(int), 0, grid - 1)
    tx = np.clip((coords[:, 1] * scale).astype(int), 0, grid - 1)
    tok[ty, tx] = True
    n_tok = int(tok.sum())
    pct_tok = 100 * n_tok / grid**2
    pct_cells = 100 * (n_tok / grid**2) ** 2

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.5))

    axes[0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0].contour(gt, levels=[0.5], colors=[ACCENT], linewidths=0.7)
    axes[0].set_title(f"1. Image {h}×{h}\nfissure : {100*gt.mean():.1f} % des pixels", loc="left")

    axes[1].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[1].scatter(coords[:, 1], coords[:, 0], s=0.05, color=COOL)
    axes[1].set_title(f"2. Nœuds du graphe de Frangi\n{len(coords)} nœuds", loc="left")

    axes[2].imshow(tok, cmap="Greys", interpolation="nearest")
    axes[2].set_title(
        f"3. Grille de tokens 64×64\n{n_tok} / 4096 touchés ({pct_tok:.1f} %)", loc="left"
    )

    ax = axes[3]
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#eef2f5", edgecolor=MUTED))
    side = np.sqrt(pct_cells / 100)
    ax.add_patch(Rectangle((0, 1 - side), side, side, facecolor=ACCENT, edgecolor="none"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"4. Matrice d'attention 4096²\n{pct_cells:.2f} % des cellules", loc="left"
    )
    ax.text(
        0.52,
        0.42,
        "les 99 % restants seraient\nstructurés par un\nregroupement inventé\n(grille, quadtree…)",
        ha="left",
        va="center",
        fontsize=8,
        color=FG,
    )

    for a in axes[:3]:
        a.set_xticks([])
        a.set_yticks([])

    fig.suptitle(
        "Fig. 3 — La géométrie de Frangi ne couvre qu'un pour cent de la matrice qu'on veut contraindre",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig3_couverture_attention.png", dpi=160)
    plt.close(fig)
    return dict(n_tok=n_tok, pct_tok=pct_tok, pct_cells=pct_cells)


# --------------------------------------------------------------------------------------
def fig4_block_constraint() -> None:
    """Ce que la block constraint fait à une matrice d'attention, sur un jouet."""
    rng = np.random.default_rng(3)
    n = 24
    # Une hiérarchie jouet : 3 familles de 8 feuilles, elles-mêmes en 2 sous-familles de 4.
    groups = [list(range(i, i + 4)) for i in range(0, n, 4)]
    parent = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}  # sous-famille -> famille

    q = rng.standard_normal((n, 8))
    k = rng.standard_normal((n, 8))
    logits = q @ k.T / np.sqrt(8)
    flat = np.exp(logits - logits.max(1, keepdims=True))
    flat /= flat.sum(1, keepdims=True)

    blocked = np.zeros_like(flat)
    for gi, gi_nodes in enumerate(groups):
        for gj, gj_nodes in enumerate(groups):
            if gi == gj:  # même famille : attention fine conservée
                blocked[np.ix_(gi_nodes, gj_nodes)] = flat[np.ix_(gi_nodes, gj_nodes)]
            elif parent[gi] == parent[gj]:  # frères : UNE valeur
                blocked[np.ix_(gi_nodes, gj_nodes)] = flat[np.ix_(gi_nodes, gj_nodes)].mean()
            else:  # ancêtres distincts plus haut : UNE valeur, plus grossière encore
                blocked[np.ix_(gi_nodes, gj_nodes)] = flat[np.ix_(gi_nodes, gj_nodes)].mean()
    blocked /= blocked.sum(1, keepdims=True)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    vmax = max(flat.max(), blocked.max())
    for ax, mat, title in (
        (axes[0], flat, "Softmax plate\nN² valeurs libres"),
        (axes[1], blocked, "Sous block constraint\nO(M·b²) valeurs"),
    ):
        ax.imshow(mat, cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
        for b in range(0, n + 1, 4):
            ax.axhline(b - 0.5, color="white", lw=0.6)
            ax.axvline(b - 0.5, color="white", lw=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, loc="left", weight="bold")

    ax = axes[2]
    ax.axis("off")
    ax.set_title("Ce que la contrainte fait", loc="left", weight="bold")
    ax.text(
        0.0,
        0.88,
        "Entre deux sous-arbres frères, TOUTES les paires\n"
        "de feuilles partagent une seule valeur d'attention :\n\n"
        "        theta_ij = theta_AB  pour i dans A, j dans B\n",
        fontsize=8.5,
        va="top",
        family="monospace",
    )
    ax.text(
        0.0,
        0.55,
        "Le théorème 3.2 dit que cette matrice est la plus\n"
        "proche possible de la Softmax, au sens KL, parmi\n"
        "celles qui respectent la contrainte.\n\n"
        "C'est un théorème d'APPROXIMATION : il borne la\n"
        "perte, il ne promet aucun gain.",
        fontsize=8.5,
        va="top",
    )
    ax.text(
        0.0,
        0.12,
        "HSA comprime l'attention.\nIl ne l'informe pas.",
        fontsize=10,
        va="top",
        color=ACCENT,
        weight="bold",
    )

    fig.suptitle(
        "Fig. 4 — La « block constraint » : ce que HSA fait vraiment à une matrice d'attention",
        x=0.01,
        ha="left",
        weight="bold",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig4_block_constraint.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if not (RESULTS / "sam2_attention_budget.json").exists():
        print("Lancer d'abord 00_sam2_attention_budget.py")
        return 1

    fig1_ou_est_lattention()
    print("fig1_ou_est_lattention.png")
    fig4_block_constraint()
    print("fig4_block_constraint.png")

    tree = _measure_tree()
    fig2_forme_de_larbre(tree)
    print("fig2_forme_de_larbre.png")
    stats = fig3_couverture_attention(tree)
    print(f"fig3_couverture_attention.png  ({stats['pct_cells']:.2f} % des cellules)")

    print(f"\nFigures dans {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
