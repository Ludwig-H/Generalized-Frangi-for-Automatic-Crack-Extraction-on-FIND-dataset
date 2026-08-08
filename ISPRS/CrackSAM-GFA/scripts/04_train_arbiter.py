#!/usr/bin/env python3
"""Entraîne l'arbitre local, strictement hors pli, par scène physique.

Ne doit être exécuté que si la porte n° 1 est ``GO``. L'étiquette de chaque
fragment est l'action d'utilité marginale maximale calculée par
``03_run_oracles.py`` : « cette bande, seule, améliore-t-elle ``z0`` ? ».

Le découpage en plis se fait par **scène physique** via un hachage SHA-256 du
nom, jamais par crop : les variantes ``original/noisy1/noisy2`` d'une même
scène partagent le même pli. C'est le contrat de fuite du pilote existant.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gfa.arbiter import ABSTAIN, FragmentArbiter, arbiter_loss  # noqa: E402
from gfa.features import EVIDENCE_DIM, SAM_DIM  # noqa: E402
from gfa.fragments import Fragment  # noqa: E402
from gfa.features import sam_features  # noqa: E402

SEED = 3407
MAX_FRAGMENTS = 96


def scene_fold(name: str, folds: int) -> int:
    """Pli déterministe par scène physique, indépendant de l'ordre du split."""

    stem = Path(name).stem
    for suffix in ("_noisy1", "_noisy2"):
        stem = stem.replace(suffix, "")
    digest = hashlib.sha256(f"gfa-v1\0{stem}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(folds)


class FragmentDataset(Dataset):
    """Une image = un échantillon ; ses fragments forment la séquence."""

    def __init__(
        self,
        keys: list[str],
        features: dict[str, np.ndarray],
        labels: dict[str, np.ndarray],
        utilities: dict[str, np.ndarray],
    ) -> None:
        self.keys = [k for k in keys if features.get(k) is not None and len(features[k])]
        self.features = features
        self.labels = labels
        self.utilities = utilities

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        key = self.keys[index]
        features = self.features[key][:MAX_FRAGMENTS]
        labels = self.labels[key][:MAX_FRAGMENTS]
        utilities = self.utilities[key][:MAX_FRAGMENTS]
        return {
            "features": torch.from_numpy(np.ascontiguousarray(features)),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)).long(),
            "utilities": torch.from_numpy(np.ascontiguousarray(utilities)),
        }


def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    longest = max(int(item["features"].shape[0]) for item in batch)
    width = int(batch[0]["features"].shape[1])
    size = len(batch)
    features = torch.zeros(size, longest, width)
    labels = torch.full((size, longest), ABSTAIN, dtype=torch.long)
    utilities = torch.zeros(size, longest, 3)
    padding = torch.ones(size, longest, dtype=torch.bool)
    for index, item in enumerate(batch):
        count = int(item["features"].shape[0])
        features[index, :count] = item["features"]
        labels[index, :count] = item["labels"]
        utilities[index, :count] = item["utilities"]
        padding[index, :count] = False
    return {
        "features": features,
        "labels": labels,
        "utilities": utilities,
        "padding": padding,
    }


def load_tables(run_root: Path, split: str) -> tuple[dict, dict, dict]:
    """Charge features (évidence + SAM), étiquettes et utilités par image."""

    fragments_store = np.load(
        run_root / "evidence" / f"fragments_{split}.npz", allow_pickle=False
    )
    utilities_store = np.load(
        run_root / "oracles" / f"utilities_{split}.npz", allow_pickle=False
    )
    z0_dir = run_root / "baseline_analysis" / "z0"

    from gfa.evidence import SUPPORT_SOURCES

    features: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    utilities: dict[str, np.ndarray] = {}
    for name in utilities_store.files:
        if not name.endswith("/utilities"):
            continue
        key = name.split("/")[0]
        evidence_features = fragments_store[f"{key}/features"]
        if evidence_features.size == 0:
            continue
        logits_path = z0_dir / f"{key}.npy"
        if not logits_path.exists():
            continue
        z0 = np.load(logits_path).astype(np.float32)

        offsets = fragments_store[f"{key}/offsets"]
        polylines = fragments_store[f"{key}/polylines"]
        radii = fragments_store[f"{key}/radii"]
        flags = fragments_store[f"{key}/flags"]
        rows = []
        for index in range(len(offsets) - 1):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            fragment = Fragment(
                pixels=polylines[start:stop].astype(np.int32),
                radius=float(radii[index]),
                sources=tuple(
                    s for s, f in zip(SUPPORT_SOURCES, flags[index]) if f
                ),
            )
            rows.append(sam_features(z0, fragment))
        combined = np.concatenate(
            [evidence_features, np.stack(rows).astype(np.float32)], axis=1
        )
        features[key] = combined
        labels[key] = utilities_store[f"{key}/best_action"].astype(np.int64)
        utilities[key] = utilities_store[name].astype(np.float32)
    return features, labels, utilities


def train_fold(
    train_keys: list[str],
    valid_keys: list[str],
    tables: tuple[dict, dict, dict],
    epochs: int,
    device: str,
) -> tuple[FragmentArbiter, dict[str, float]]:
    features, labels, utilities = tables
    train_set = FragmentDataset(train_keys, features, labels, utilities)
    valid_set = FragmentDataset(valid_keys, features, labels, utilities)
    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True, collate_fn=collate, drop_last=False
    )
    valid_loader = DataLoader(
        valid_set, batch_size=16, shuffle=False, collate_fn=collate
    )

    model = FragmentArbiter(input_dim=EVIDENCE_DIM + SAM_DIM).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    best_state, best_score = None, -float("inf")
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            output = model(
                batch["features"].to(device), padding_mask=batch["padding"].to(device)
            )
            loss = arbiter_loss(
                output,
                batch["labels"].to(device),
                batch["utilities"].to(device),
                padding_mask=batch["padding"].to(device),
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        schedule.step()

        model.eval()
        realised, oracle = 0.0, 0.0
        with torch.no_grad():
            for batch in valid_loader:
                output = model(
                    batch["features"].to(device),
                    padding_mask=batch["padding"].to(device),
                )
                chosen = output.hard_actions().cpu()
                valid = ~batch["padding"]
                utility = batch["utilities"]
                taken = utility.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
                realised += float((taken * valid).sum())
                oracle += float((utility.max(dim=-1).values * valid).sum())
        # Utilité réalisée rapportée à l'utilité oracle : 1,0 signifie que
        # l'arbitre choisit partout l'action que le GT aurait choisie.
        score = realised / oracle if oracle > 1e-9 else 0.0
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"utility_ratio": float(best_score)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--min-utility",
        type=float,
        default=0.0,
        help=(
            "EXPLORATOIRE : n'entraîne que sur les fragments dont l'utilité "
            "marginale absolue dépasse ce seuil. Écart déclaré au "
            "pré-enregistrement, à ne pas présenter comme confirmatoire."
        ),
    )
    parser.add_argument("--suffix", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gate = json.loads(
        (args.run_root / "oracles" / f"source_oracle_{args.split}.json").read_text(
            encoding="utf-8"
        )
    )
    if gate["gate1_verdict"] != "GO":
        print("PORTE 1 non franchie : l'arbitre ne doit pas être entraîné.")
        return 3

    tables = load_tables(args.run_root, args.split)
    if args.min_utility > 0.0:
        features, labels, utilities = tables
        kept = {}
        for key in list(features):
            mask = np.abs(utilities[key]).max(axis=1) > args.min_utility
            if mask.any():
                kept[key] = (features[key][mask], labels[key][mask], utilities[key][mask])
        features = {k: v[0] for k, v in kept.items()}
        labels = {k: v[1] for k, v in kept.items()}
        utilities = {k: v[2] for k, v in kept.items()}
        tables = (features, labels, utilities)
        total = sum(len(v) for v in features.values())
        print(f'EXPLORATOIRE min-utility={args.min_utility}: {total} fragments retenus')
    keys = sorted(tables[0])
    if not keys:
        raise SystemExit("aucun fragment exploitable")
    print(f"{len(keys)} images avec fragments, dimension {tables[0][keys[0]].shape[1]}")

    output_dir = args.run_root / "arbiter"
    output_dir.mkdir(parents=True, exist_ok=True)
    scores: list[dict[str, float]] = []
    for fold in range(args.folds):
        valid_keys = [k for k in keys if scene_fold(k, args.folds) == fold]
        train_keys = [k for k in keys if scene_fold(k, args.folds) != fold]
        if not valid_keys or not train_keys:
            continue
        model, metrics = train_fold(
            train_keys, valid_keys, tables, args.epochs, device
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": EVIDENCE_DIM + SAM_DIM,
                "fold": fold,
                "metrics": metrics,
                "seed": SEED,
            },
            output_dir / f"fold_{fold}{args.suffix}.pt",
        )
        metrics["fold"] = fold
        metrics["validation_images"] = len(valid_keys)
        scores.append(metrics)
        print(f"pli {fold}: ratio d'utilité {metrics['utility_ratio']:.4f}")

    summary = {
        "folds": scores,
        "utility_ratio_mean": float(
            np.mean([s["utility_ratio"] for s in scores])
        )
        if scores
        else 0.0,
        "seed": SEED,
    }
    (output_dir / f"training_summary{args.suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
