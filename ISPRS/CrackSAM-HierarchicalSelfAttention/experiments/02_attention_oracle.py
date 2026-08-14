"""Oracle d'attention — le plafond de tout guidage géométrique de l'attention de SAM 2.

C'est l'expérience P0 de l'[audit](../AUDIT.md). Elle coûte ~2 h de GPU, n'entraîne rien,
et peut clore le dossier dans les deux sens.

Elle mesure deux choses distinctes, sur la baseline **gelée** :

  A. `bias` — **plafond du guidage.** On ajoute aux logits des trois blocs d'attention
     globale de Hiera-L (23, 33, 43) un biais dérivé de la **vérité terrain** :

         logits[i, j] += beta   si les tokens i et j sont du même côté (fissure/fond)
         logits[i, j] -= beta   sinon

     C'est l'oracle parfait : il connaît la vraie topologie, sans bruit et sans erreur
     d'échelle. Aucune contrainte issue de Frangi ne peut faire mieux. Si le gain est
     négligeable, toute la famille « contraindre l'attention par la géométrie » est close —
     HSA compris.

  B. `block` — **coût propre du mécanisme HSA.** On applique la *block constraint* de HSA
     avec la partition **parfaite** {fissure, fond} : à l'intérieur d'un bloc, Softmax
     normale ; entre les deux blocs, **une seule valeur d'attention** calculée sur les
     moyennes des clés et valeurs de l'autre bloc, exactement comme l'algorithme 3 (ligne 18)
     du papier NeurIPS 2025, terme `log |l(B)|` compris.

     Ce bras isole la question que l'audit juge décisive : *lier* les coefficients
     d'attention en blocs coûte-t-il quelque chose, même quand la partition est parfaite ?
     Si `block` dégrade ici, HSA dégradera a fortiori avec une hiérarchie bruitée.

Les deux bras partagent le même encodeur, la même baseline et la même métrique que
`evaluate_sam2.py` : les nombres sont directement comparables à `baseline`.

Usage (depuis ISPRS/CrackSAM, pour que `cracksam2` soit importable) :

    cd ISPRS/CrackSAM
    python ../CrackSAM-HierarchicalSelfAttention/experiments/02_attention_oracle.py \
        --data-root "$CRACKSAM2_DATA_ROOT" \
        --sam2-checkpoint "$SAM2_CHECKPOINT" \
        --adapter-checkpoint "$BASELINE_CHECKPOINT" \
        --output ../CrackSAM-HierarchicalSelfAttention/results/attention_oracle.json

Auto-test sans poids ni données (vérifie la mécanique d'attention par blocs) :

    python 02_attention_oracle.py --self-test

> [!WARNING]
> Les deux bras utilisent la **vérité terrain à l'inférence**. Ce sont des bornes
> supérieures, jamais des résultats publiables. Les reporter comme tels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

# Blocs à attention globale de Hiera-L (window_size == 0) — les seuls dont la matrice
# d'attention porte sur l'image entière. Vérifié par 00_sam2_attention_budget.py.
GLOBAL_BLOCKS = (23, 33, 43)
TOKEN_GRID = 64  # 1024 / 16


# --------------------------------------------------------------------------------------
# Les deux mécaniques d'attention
# --------------------------------------------------------------------------------------
def biased_attention(q, k, v, same_block, beta: float):
    """Softmax standard, plus un biais additif oracle sur les logits.

    q, k, v      : (B, nheads, N, C)
    same_block   : (B, N, N) booléen — i et j sont-ils du même côté de la partition ?
    """
    bias = torch.where(
        same_block.unsqueeze(1),
        torch.full_like(same_block, beta, dtype=q.dtype).unsqueeze(1),
        torch.full_like(same_block, -beta, dtype=q.dtype).unsqueeze(1),
    )
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)


def block_constrained_attention(q, k, v, labels):
    """La *block constraint* de HSA, avec une hiérarchie à deux niveaux.

    Hiérarchie : racine -> {sous-arbre 0, sous-arbre 1} -> tokens. Les deux sous-arbres
    sont frères, donc toutes les paires de feuilles entre eux partagent **une seule** valeur
    d'attention (§3.2 du papier, éq. 7 et 8).

    Pour un token i du bloc A, on met en concurrence dans un même Softmax :
      - les logits fins  q_i . k_j / sqrt(d)  pour tout j du bloc A ;
      - un unique logit grossier  q_i . kbar_B / sqrt(d) + log |B|,  où kbar_B est la
        moyenne des clés du bloc B. Le terme log |B| est celui de l'algorithme 3, ligne 18 :
        il rend compte de la multiplicité des feuilles agrégées.
    La sortie combine les v_j du bloc A et vbar_B, la moyenne des valeurs du bloc B.

    q, k, v : (B, nheads, N, C)   labels : (B, N) entiers dans {0, 1}
    """
    b, h, n, c = q.shape
    scale = 1.0 / math.sqrt(c)
    out = torch.empty_like(q)

    for bi in range(b):
        lab = labels[bi]
        for block_id in (0, 1):
            idx = torch.nonzero(lab == block_id, as_tuple=True)[0]
            other = torch.nonzero(lab != block_id, as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            q_a = q[bi, :, idx, :]  # (h, na, c)
            k_a, v_a = k[bi, :, idx, :], v[bi, :, idx, :]
            if other.numel() == 0:
                # Une seule famille : sans frère, la contrainte de blocs ne lie rien et
                # HSA se réduit exactement à la Softmax (annexe E.1 du papier).
                out[bi, :, idx, :] = F.scaled_dot_product_attention(
                    q_a.unsqueeze(0), k_a.unsqueeze(0), v_a.unsqueeze(0)
                ).squeeze(0)
                continue

            k_bar = k[bi, :, other, :].mean(dim=1, keepdim=True)  # (h, 1, c)
            v_bar = v[bi, :, other, :].mean(dim=1, keepdim=True)

            fine = torch.einsum("hnc,hmc->hnm", q_a, k_a) * scale  # (h, na, na)
            coarse = torch.einsum("hnc,hmc->hnm", q_a, k_bar) * scale  # (h, na, 1)
            coarse = coarse + math.log(float(other.numel()))

            weights = torch.softmax(torch.cat([fine, coarse], dim=-1), dim=-1)
            w_fine, w_coarse = weights[..., :-1], weights[..., -1:]
            out[bi, :, idx, :] = torch.einsum("hnm,hmc->hnc", w_fine, v_a) + w_coarse * v_bar

    return out


# --------------------------------------------------------------------------------------
# Branchement dans Hiera
# --------------------------------------------------------------------------------------
@contextmanager
def patched_global_attention(trunk, mode: str, state: dict, beta: float = 0.0):
    """Remplace temporairement l'attention des blocs globaux de Hiera.

    `state["labels"]` doit contenir, pendant la passe, un tenseur (B, N) de labels de
    tokens ; le contexte le lit à chaque appel pour rester compatible avec le batching.
    """
    from sam2.modeling.backbones.hieradet import do_pool

    originals = {}

    def make_forward(attn):
        def forward(x: torch.Tensor) -> torch.Tensor:
            b, h_, w_, _ = x.shape
            qkv = attn.qkv(x).reshape(b, h_ * w_, 3, attn.num_heads, -1)
            q, k, v = torch.unbind(qkv, 2)
            if attn.q_pool:  # jamais le cas pour 23/33/43, gardé par fidélité
                q = do_pool(q.reshape(b, h_, w_, -1), attn.q_pool)
                h_, w_ = q.shape[1:3]
                q = q.reshape(b, h_ * w_, attn.num_heads, -1)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, nheads, N, C)

            labels = state["labels"]
            if labels is None or labels.shape[-1] != q.shape[-2]:
                out = F.scaled_dot_product_attention(q, k, v)
            elif mode == "bias":
                same = labels.unsqueeze(2) == labels.unsqueeze(1)  # (B, N, N)
                out = biased_attention(q, k, v, same, beta)
            elif mode == "block":
                out = block_constrained_attention(q, k, v, labels)
            else:
                raise ValueError(f"mode inconnu : {mode}")

            out = out.transpose(1, 2).reshape(b, h_, w_, -1)
            return attn.proj(out)

        return forward

    try:
        for index in GLOBAL_BLOCKS:
            attn = trunk.blocks[index].attn
            originals[index] = attn.forward
            attn.forward = make_forward(attn)
        yield
    finally:
        for index, fn in originals.items():
            trunk.blocks[index].attn.forward = fn


def gt_token_labels(masks: torch.Tensor, grid: int = TOKEN_GRID) -> torch.Tensor:
    """Vérité terrain -> label par token. Un token vaut 1 s'il contient un pixel de fissure.

    masks : (B, 1, H, W) ou (B, H, W), valeurs dans {0, 1}.
    """
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    pooled = F.adaptive_max_pool2d(masks.float(), output_size=(grid, grid))
    return (pooled > 0.5).long().flatten(1)  # (B, grid*grid)


# --------------------------------------------------------------------------------------
# Auto-test : la mécanique, sans poids ni données
# --------------------------------------------------------------------------------------
def self_test() -> int:
    torch.manual_seed(0)
    b, h, n, c = 2, 4, 32, 16
    q, k, v = (torch.randn(b, h, n, c) for _ in range(3))

    # 1. Un biais nul rend exactement la Softmax standard.
    labels = torch.randint(0, 2, (b, n))
    same = labels.unsqueeze(2) == labels.unsqueeze(1)
    ref = F.scaled_dot_product_attention(q, k, v)
    assert torch.allclose(biased_attention(q, k, v, same, 0.0), ref, atol=1e-5), "biais nul"

    # 2. Une partition à un seul bloc rend exactement la Softmax standard : sans frère, la
    #    contrainte de blocs ne lie rien.
    single = torch.zeros(b, n, dtype=torch.long)
    got = block_constrained_attention(q, k, v, single)
    assert torch.allclose(got, ref, atol=1e-5), "bloc unique"

    # 3. La contrainte de blocs est bien une contrainte : avec deux blocs elle diffère de la
    #    Softmax, et chaque ligne reste une combinaison convexe (norme bornée par max|v|).
    two = block_constrained_attention(q, k, v, labels)
    assert not torch.allclose(two, ref, atol=1e-3), "deux blocs devraient contraindre"
    assert two.abs().max() <= v.abs().max() + 1e-4, "sortie hors enveloppe convexe"

    # 4. Les labels de tokens couvrent bien la fissure (max-pooling, pas de moyenne).
    m = torch.zeros(1, 1, 448, 448)
    m[0, 0, 200, 100:300] = 1.0  # un trait d'un pixel de haut
    lab = gt_token_labels(m, grid=TOKEN_GRID)
    assert lab.sum() > 0, "un trait fin ne doit pas disparaitre du pavage de tokens"
    print(f"auto-test OK — un trait de 1 px sur 448 occupe {int(lab.sum())} tokens sur "
          f"{TOKEN_GRID ** 2}")
    return 0


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true", help="vérifie la mécanique, sans GPU")
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--sam2-checkpoint", type=Path)
    ap.add_argument("--adapter-checkpoint", type=Path)
    ap.add_argument("--list-file", type=Path, help="défaut : la liste de test du protocole")
    ap.add_argument("--output", type=Path, default=Path("attention_oracle.json"))
    ap.add_argument("--betas", type=float, nargs="+", default=[0.0, 1.0, 2.0, 4.0, 8.0])
    ap.add_argument("--max-samples", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    for required in ("data_root", "sam2_checkpoint", "adapter_checkpoint"):
        if getattr(args, required) is None:
            ap.error(f"--{required.replace('_', '-')} est requis hors --self-test")

    # `cracksam2` vit à la racine de ISPRS/CrackSAM ; on l'ajoute au chemin pour que ce
    # script reste lançable depuis n'importe où.
    cracksam_root = Path(__file__).resolve().parents[2] / "CrackSAM"
    sys.path.insert(0, str(cracksam_root))
    from cracksam2.data import CrackSegmentationDataset, read_sample_list  # noqa: E402
    from cracksam2.metrics import segmentation_metrics  # noqa: E402
    from cracksam2.model import build_cracksam2, load_adapter_state_dict  # noqa: E402

    device = torch.device(args.device)
    model, _ = build_cracksam2(str(args.sam2_checkpoint), device=device)
    checkpoint = torch.load(args.adapter_checkpoint, map_location="cpu")
    load_adapter_state_dict(model, checkpoint["adapter"], strict=True)
    model.eval().to(device)

    list_file = args.list_file or (
        cracksam_root / "protocol" / "cracksam_paper" / "lists" / "test.txt"
    )
    names = read_sample_list(list_file)
    if args.max_samples:
        names = names[: args.max_samples]
    dataset = CrackSegmentationDataset(args.data_root, names)

    trunk = model.sam2.image_encoder.trunk
    state: dict[str, torch.Tensor | None] = {"labels": None}

    arms = [("baseline", "none", 0.0)]
    arms += [("bias", "bias", beta) for beta in args.betas if beta > 0]
    arms += [("block", "block", 0.0)]

    results = {}
    for arm_name, mode, beta in arms:
        scores = []
        with torch.inference_mode():
            for i in range(len(dataset)):
                sample = dataset[i]
                image = sample["image"].unsqueeze(0).to(device)
                target = sample["mask"].unsqueeze(0).to(device)
                state["labels"] = (
                    None if mode == "none" else gt_token_labels(target).to(device)
                )
                if mode == "none":
                    features = model.encode_images(image)
                else:
                    with patched_global_attention(trunk, mode, state, beta):
                        features = model.encode_images(image)
                logits = model.decode_features(features)["logits"]
                pred = (logits.sigmoid() > args.threshold).float()
                scores.append(segmentation_metrics(pred, target)["iou"])
        key = arm_name if beta == 0 else f"{arm_name}_beta{beta:g}"
        results[key] = float(sum(scores) / len(scores))
        print(f"{key:>16} : IoU {results[key]:.4f}")

    base = results["baseline"]
    summary = {
        "iou": results,
        "delta_vs_baseline": {k: v - base for k, v in results.items() if k != "baseline"},
        "n_samples": len(dataset),
        "global_blocks": list(GLOBAL_BLOCKS),
        "warning": "Les bras bias/block utilisent la vérité terrain à l'inférence : "
                   "bornes supérieures, jamais des résultats.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(f"\nÉcrit : {args.output}")

    best_bias = max(
        (v - base for k, v in results.items() if k.startswith("bias")), default=0.0
    )
    print(f"\nPlafond du guidage d'attention : {best_bias:+.4f} d'IoU")
    print(f"Coût propre de la contrainte de blocs : {results['block'] - base:+.4f}")
    if best_bias < 0.01:
        print("=> Sous +0,01 : la famille « contraindre l'attention » est close, HSA compris.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
