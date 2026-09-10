#!/usr/bin/env python3
"""Repeat the primary MLP32 with another initialization, keeping every split fixed."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from compare_simple_classifiers import make_splits, fit_one
from analyze_features import CATEGORIES, encode_labels, file_hash, load_inputs, scores, write_json


HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output = HERE / "simple_models"
    contract = json.loads((output / "contract.json").read_text())
    if contract["script_sha256"] != file_hash(HERE / "compare_simple_classifiers.py"):
        raise ValueError("Training script changed since primary experiment")
    cases_path, features_path = HERE / "tables/categories.csv", HERE / "cache/extraction-b1/features.npz"
    if file_hash(cases_path) != contract["cases_sha256"] or file_hash(features_path) != contract["features_sha256"]:
        raise ValueError("Replication inputs differ from primary experiment")
    cases, features = load_inputs(cases_path, features_path)
    y = encode_labels(cases)
    # Inner partitions use the original seed. Only initialization/shuffling changes.
    splits = make_splits(cases, y, HERE / "results/folds.csv", contract["seed"])
    predictions = np.full((len(cases), 3), np.nan)
    records = []
    for split in splits:
        if split["evaluation"] != "grouped_cv":
            continue
        replicated = {**split, "seed": 123 + split["seed"] - contract["seed"]}
        print(f"Replication {split['name']}, initialization {replicated['seed']}", flush=True)
        probability, info = fit_one(features["mean"], y, replicated, "mlp32", contract["mlp"]["max_epochs"])
        predictions[split["test"]] = probability
        records.append({"split": split["name"], **info})
    if not np.isfinite(predictions).all():
        raise ValueError("Incomplete replicated predictions")
    table = cases[["image_id", "physical_group", "category"]].copy()
    for i, category in enumerate(CATEGORIES):
        table[f"p_{category}"] = predictions[:, i]
    table.to_csv(output / "replication_seed123_predictions.csv", index=False)
    result = {"status": "complete", "representation": "mean", "model": "mlp32", "seed": 123,
              "same_inner_and_outer_scenes_as_primary": True,
              "contract_sha256": file_hash(output / "contract.json"),
              "replication_script_sha256": file_hash(Path(__file__)),
              "metrics": scores(y, predictions, cases.delta_iou.to_numpy()), "runs": records}
    write_json(output / "replication_seed123.json", result)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
