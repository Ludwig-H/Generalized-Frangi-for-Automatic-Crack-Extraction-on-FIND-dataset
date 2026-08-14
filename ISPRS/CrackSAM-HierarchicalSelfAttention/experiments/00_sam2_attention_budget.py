"""Où peut-on brancher une contrainte d'attention dans SAM 2 Hiera-L, et pour quel gain ?

Ce script ne dépend d'aucune donnée : il instancie le tronc Hiera-L de SAM 2, relève la
résolution en tokens de chaque bloc, repère les blocs à attention *globale* (les seuls dont
la matrice d'attention est de taille N x N sur toute l'image) et chiffre leur part dans le
budget FLOPs de l'encodeur.

C'est l'étape 0 de l'audit HSA : avant de discuter d'une contrainte hiérarchique sur la
matrice d'attention, il faut savoir combien de matrices d'attention existent, à quelle
résolution, et ce que coûterait leur remplacement.

Usage:
    python ISPRS/CrackSAM-HierarchicalSelfAttention/experiments/00_sam2_attention_budget.py

Dépendances : torch, sam2 (aucun checkpoint requis, on n'instancie que l'architecture).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Configuration Hiera-L, recopiée de sam2/configs/sam2.1/sam2.1_hiera_l.yaml
HIERA_L = dict(
    embed_dim=144,
    num_heads=2,
    stages=[2, 6, 36, 4],
    global_att_blocks=[23, 33, 43],
    window_pos_embed_bkg_spatial_size=[7, 7],
    window_spec=[8, 4, 16, 8],
)

IMAGE_SIZE = 1024  # résolution d'entrée de SAM 2
PATCH_STRIDE = 4  # patch_embed = Conv2d(3, 144, kernel_size=7, stride=4, padding=3)


def build_trunk():
    from sam2.modeling.backbones.hieradet import Hiera

    return Hiera(**HIERA_L)


def block_table(trunk, image_size: int = IMAGE_SIZE):
    """Résolution, fenêtre et dimension de chaque bloc du tronc."""
    h = w = image_size // PATCH_STRIDE
    q_pool = set(getattr(trunk, "q_pool_blocks", [2, 8, 44]))
    rows = []
    for i, b in enumerate(trunk.blocks):
        if i in q_pool:
            h //= 2
            w //= 2
        rows.append(
            dict(
                block=i,
                grid=f"{h}x{w}",
                tokens=h * w,
                window=int(b.window_size),
                is_global=b.window_size == 0,
                dim_out=int(b.dim_out),
                heads=int(b.attn.num_heads),
                pixels_per_token=image_size // h,
            )
        )
    return rows


def flops_budget(trunk, image_size: int = IMAGE_SIZE):
    """FLOPs du tronc, séparés en produits d'attention / projections / MLP.

    Un multiply-accumulate compte pour 2 FLOPs. L'attention fenêtrée est comptée
    fenêtre par fenêtre (c'est ce qui la rend bon marché).
    """
    h = w = image_size // PATCH_STRIDE
    q_pool = set(getattr(trunk, "q_pool_blocks", [2, 8, 44]))
    attn_all = attn_global = proj = mlp = 0.0
    for i, b in enumerate(trunk.blocks):
        if i in q_pool:
            h //= 2
            w //= 2
        n_tokens, d, win = h * w, int(b.dim_out), int(b.window_size)
        if win == 0:
            n_windows, n_per_window = 1, n_tokens
        else:
            n_windows, n_per_window = (h // win) * (w // win), win * win
        # QK^T puis AV : deux produits de taille (n x n x d)
        attn = n_windows * 2 * (n_per_window**2 * d) * 2
        attn_all += attn
        if win == 0:
            attn_global += attn
        proj += 4 * n_tokens * d * d * 2  # q, k, v et projection de sortie
        mlp += 8 * n_tokens * d * d * 2  # expansion 4x, deux produits
    total = attn_all + proj + mlp
    return dict(
        total_gflops=total / 1e9,
        attention_matmuls_gflops=attn_all / 1e9,
        attention_matmuls_pct=100 * attn_all / total,
        global_attention_gflops=attn_global / 1e9,
        global_attention_pct=100 * attn_global / total,
        projections_pct=100 * proj / total,
        mlp_pct=100 * mlp / total,
    )


def resolution_sweep(trunk, sizes=(1024, 2048, 4096)):
    """L'attention globale devient-elle dominante si l'on monte en résolution ?

    C'est le seul angle où HSA pourrait rapporter ce que son papier démontre (des FLOPs).
    L'attention globale croît en O(N²) et le reste en O(N) : sa part augmente avec la
    résolution. Reste à savoir si le coût total reste finançable.
    """
    base = None
    out = []
    for s in sizes:
        b = flops_budget(trunk, s)
        b["image_size"] = s
        # Stage 3 (blocs 8-43, dont les 3 blocs globaux) : deux poolings après patch_embed
        grid3 = s // PATCH_STRIDE // 4
        b["tokens_stage3"] = grid3**2
        # Un token de stage 3 couvre toujours 16 px de l'entrée SAM 2 ; monter la
        # résolution d'entrée revient donc à échantillonner l'image native 2x plus fin.
        b["px_per_token_stage3"] = s // grid3
        b["native_px_per_token_find256"] = 256 / grid3
        if base is None:
            base = b["total_gflops"]
        b["cost_vs_1024"] = b["total_gflops"] / base
        # Coût si l'attention globale devenait gratuite (borne HSA la plus optimiste)
        b["cost_vs_1024_if_hsa_free"] = (
            b["total_gflops"] - b["global_attention_gflops"]
        ) / base
        out.append(b)
    return out


def main() -> int:
    try:
        trunk = build_trunk()
    except ImportError:
        print("sam2 n'est pas installé : pip install -r ISPRS/CrackSAM/requirements-sam2.txt")
        return 1

    rows = block_table(trunk)
    budget = flops_budget(trunk)
    globals_ = [r for r in rows if r["is_global"]]

    print(f"SAM 2 Hiera-L, entrée {IMAGE_SIZE}x{IMAGE_SIZE}\n")
    print(f"Blocs du tronc                     : {len(rows)}")
    print(f"Blocs à attention GLOBALE          : {len(globals_)} -> {[r['block'] for r in globals_]}")
    for r in globals_:
        print(
            f"  bloc {r['block']:2d} : grille {r['grid']} = {r['tokens']} tokens, "
            f"dim {r['dim_out']}, {r['heads']} têtes, {r['pixels_per_token']} px/token"
        )
    print("\nBudget FLOPs du tronc :")
    print(f"  total                            : {budget['total_gflops']:8.1f} GFLOPs")
    print(
        f"  produits d'attention (tous)      : {budget['attention_matmuls_gflops']:8.1f} GFLOPs"
        f"  ({budget['attention_matmuls_pct']:5.2f} %)"
    )
    print(
        f"  produits d'attention (globale)   : {budget['global_attention_gflops']:8.1f} GFLOPs"
        f"  ({budget['global_attention_pct']:5.2f} %)"
    )
    print(f"  projections linéaires            : {budget['projections_pct']:5.2f} %")
    print(f"  MLP                              : {budget['mlp_pct']:5.2f} %")
    print(
        f"\n=> Borne supérieure du gain FLOPs d'un remplacement de TOUTE l'attention globale "
        f"par un schéma O(M b^2) : {budget['global_attention_pct']:.2f} % du tronc."
    )

    sweep = resolution_sweep(trunk)
    print("\nMonter en résolution rend-il l'attention globale dominante ?")
    print(f"{'entrée':>8} {'tokens st.3':>12} {'px natifs/token':>16} {'total':>11} "
          f"{'att. glob.':>11} {'coût/1024':>10} {'idem si HSA gratuit':>20}")
    for b in sweep:
        print(
            f"{b['image_size']:>8} {b['tokens_stage3']:>12} "
            f"{b['native_px_per_token_find256']:>16.2f} "
            f"{b['total_gflops']:>9.0f} G {b['global_attention_pct']:>10.1f} % "
            f"{b['cost_vs_1024']:>9.1f}x {b['cost_vs_1024_if_hsa_free']:>19.1f}x"
        )
    print("(« px natifs/token » : pour une image FIND 256x256 rééchantillonnée à l'entrée)")
    print(
        "\n=> Même en rendant l'attention globale GRATUITE, monter en résolution reste "
        "beaucoup plus cher que 1024. HSA n'ouvre pas la haute résolution ; il en réduit "
        "la facture d'un facteur ~2 au mieux."
    )

    out = Path(__file__).resolve().parents[1] / "results" / "sam2_attention_budget.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(blocks=rows, flops=budget, resolution_sweep=sweep), indent=2))
    print(f"\nÉcrit : {out.relative_to(Path.cwd()) if out.is_relative_to(Path.cwd()) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
