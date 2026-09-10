#!/usr/bin/env python3
"""Grouped held-out comparison of logistic regression and one-hidden-layer MLPs."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from analyze_features import CATEGORIES, encode_labels, file_hash, load_inputs, scores, write_json


HERE = Path(__file__).resolve().parent
REPRESENTATIONS = ("mean", "multiscale_mean", "pre_global_mean")
MODELS = ("fixed_logistic", "tuned_logistic", "mlp8", "mlp32", "mlp64")
C_GRID = (.01, .1, 1., 10.)


def class_weights(y):
    counts = np.bincount(y, minlength=3)
    if len(counts) != 3 or (counts == 0).any():
        raise ValueError("All three categories must be present in training")
    return (len(y) / (3 * counts)).astype(np.float32)


def split_inner(cases, y, train, seed):
    """Return positions in the full table; the external test is inaccessible."""
    groups = cases.physical_group.to_numpy()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    inner, valid = next(splitter.split(np.zeros(len(train)), y[train], groups[train]))
    inner, valid = train[inner], train[valid]
    if set(groups[inner]) & set(groups[valid]):
        raise ValueError("Scene leakage in inner validation")
    class_weights(y[inner])
    if len(np.unique(y[valid])) != 3:
        raise ValueError("Inner validation lacks a category")
    return inner, valid


def make_splits(cases, y, folds_path, seed):
    recorded = pd.read_csv(folds_path).set_index("image_id").loc[cases.image_id]
    if not np.array_equal(recorded.physical_group.to_numpy(), cases.physical_group.to_numpy()):
        raise ValueError("Historical folds and cases differ")
    folds = recorded.fold.to_numpy(int)
    groups = cases.physical_group.to_numpy()
    splits = []
    for fold in map(int, sorted(set(folds))):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        if set(groups[train]) & set(groups[test]):
            raise ValueError("Scene leakage in external folds")
        inner, valid = split_inner(cases, y, train, seed + fold)
        splits.append(dict(name=f"cv{fold}", evaluation="grouped_cv", domain="all",
                           train=train, test=test, inner=inner, valid=valid, seed=seed + fold))
    for i, domain in enumerate(sorted(cases.domain.unique())):
        train = np.flatnonzero(cases.domain.ne(domain))
        test = np.flatnonzero(cases.domain.eq(domain))
        inner, valid = split_inner(cases, y, train, seed + 100 + i)
        splits.append(dict(name=f"domain_{domain}", evaluation="leave_domain_out", domain=domain,
                           train=train, test=test, inner=inner, valid=valid, seed=seed + 100 + i))
    return splits


def scaled_data(X, fit, *others):
    scaler = StandardScaler().fit(X[fit])
    return [np.ascontiguousarray(scaler.transform(X[index]), dtype=np.float32) for index in (fit, *others)]


def make_mlp(dimensions, width, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dimensions, width), nn.ReLU(), nn.Linear(width, 3))


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X)), dim=1).numpy()


def train_mlp(X, y, width, seed, epochs, validation=None, patience=30):
    """Train on supplied rows only; selection uses a separate grouped validation."""
    model = make_mlp(X.shape[1], width, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.001)
    loss_function = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights(y)))
    tx, ty = torch.from_numpy(X), torch.from_numpy(y.astype(np.int64))
    generator = torch.Generator().manual_seed(seed)
    best_score, best_epoch, trace = -np.inf, 0, []
    last_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        for indices in order.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(tx[indices]), ty[indices])
            loss.backward()
            optimizer.step()
        last_epoch = epoch
        if validation is not None and (epoch % 5 == 0 or epoch == epochs):
            vx, vy = validation
            score = float(balanced_accuracy_score(vy, predict_mlp(model, vx).argmax(1)))
            trace.append({"epoch": epoch, "balanced_accuracy": score})
            # Strict improvement keeps the earliest epoch when scores tie.
            if score > best_score + 1e-12:
                best_score, best_epoch = score, epoch
            if epoch - best_epoch >= patience:
                break
    info = {"best_epoch": best_epoch if validation is not None else epochs,
            "validation_balanced_accuracy": best_score if validation is not None else None,
            "epochs_run": last_epoch, "hit_epoch_limit": last_epoch == epochs,
            "validation_trace": trace}
    return model, info


def logistic(X, y, C):
    estimator = LogisticRegression(C=C, class_weight="balanced", solver="lbfgs", max_iter=2000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y)
    return estimator, [str(w.message) for w in caught]


def fit_one(X, y, split, model_name, epochs):
    start = time.monotonic()
    inner, valid, train, test = (split[k] for k in ("inner", "valid", "train", "test"))
    a, b = scaled_data(X, inner, valid)
    full, held_out = scaled_data(X, train, test)
    info = {"n_train": len(train), "n_inner_train": len(inner), "n_validation": len(valid),
            "n_test": len(test), "seed": split["seed"], "chosen_C": None, "best_epoch": None,
            "validation_balanced_accuracy": None, "hit_epoch_limit": False, "warnings": []}
    if model_name.endswith("logistic"):
        chosen_C = 1.
        if model_name == "tuned_logistic":
            candidates = []
            for C in C_GRID:
                model, messages = logistic(a, y[inner], C)
                info["warnings"].extend(messages)
                candidates.append({"C": C, "score": float(balanced_accuracy_score(y[valid], model.predict(b)))})
            # Grid order is ascending C; ties favor stronger regularization.
            chosen = max(candidates, key=lambda row: row["score"])
            chosen_C = chosen["C"]
            info.update(validation_balanced_accuracy=chosen["score"], candidates=candidates)
        model, messages = logistic(full, y[train], chosen_C)
        info["warnings"].extend(messages)
        probabilities, fitted = model.predict_proba(held_out), model.predict_proba(full)
        info["chosen_C"] = chosen_C
        info["n_parameters"] = X.shape[1] * 3 + 3
    else:
        width = int(model_name.removeprefix("mlp"))
        _, tuning = train_mlp(a, y[inner], width, split["seed"], epochs,
                              validation=(b, y[valid]))
        # Fresh refit on all outer training scenes, for the selected epoch count.
        model, _ = train_mlp(full, y[train], width, split["seed"], tuning["best_epoch"])
        probabilities, fitted = predict_mlp(model, held_out), predict_mlp(model, full)
        info.update(tuning)
        info["n_parameters"] = sum(p.numel() for p in model.parameters())
    info["train_balanced_accuracy"] = float(balanced_accuracy_score(y[train], fitted.argmax(1)))
    info["elapsed_seconds"] = time.monotonic() - start
    return probabilities, info


def bootstrap_predictions(cases, y, predictions, repeats, seed):
    """Paired stratified scene bootstrap, reusing the same draws for all models."""
    groups, codes = np.unique(cases.physical_group.to_numpy(), return_inverse=True)
    group_domains = cases.assign(_group=codes).groupby("_group").domain.first().reindex(range(len(groups))).to_numpy()
    rng = np.random.default_rng(seed)
    weights = np.zeros((repeats, len(groups)), dtype=np.float64)
    for domain in np.unique(group_domains):
        indices = np.flatnonzero(group_domains == domain)
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repeats)
    sizes = np.bincount(codes, minlength=len(groups))
    denominator = weights @ sizes
    intervals, draws = {}, {}
    delta = cases.delta_iou.to_numpy(float)
    for key, proba in predictions.items():
        predicted = proba.argmax(1)
        matrices = np.bincount(codes * 9 + y * 3 + predicted, minlength=len(groups) * 9).reshape(len(groups), 9)
        sampled = (weights @ matrices).reshape(repeats, 3, 3)
        recalls = sampled.diagonal(axis1=1, axis2=2) / np.maximum(sampled.sum(2), 1)
        ba = recalls.mean(1)
        gain_sums = np.bincount(codes, weights=np.where(predicted == 2, delta, 0), minlength=len(groups))
        gain = (weights @ gain_sums) / denominator
        draws[key] = {"balanced_accuracy": ba, "hard_gate_mean_delta_iou": gain}
        intervals[key] = {name: np.quantile(values, [.025, .975]).tolist() for name, values in draws[key].items()}
    primary = dict(representation="mean", candidate="mlp32", reference="tuned_logistic", repeats=repeats,
                   unit="physical scene", strata="domain", conditional_on_fixed_oof_predictions=True)
    candidate, reference = "mean/mlp32", "mean/tuned_logistic"
    for statistic, label in (("balanced_accuracy", "difference_balanced_accuracy"),
                             ("hard_gate_mean_delta_iou", "difference_gate_delta_iou")):
        difference = draws[candidate][statistic] - draws[reference][statistic]
        original = scores(y, predictions[candidate], delta)[statistic] - scores(y, predictions[reference], delta)[statistic]
        primary[label] = {"estimate": original, "ci95": np.quantile(difference, [.025, .975]).tolist()}
    return intervals, primary


def run(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cases, features = load_inputs(args.cases, args.features)
    y = encode_labels(cases)
    splits = make_splits(cases, y, args.folds, args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    contract = {"features_sha256": file_hash(args.features), "cases_sha256": file_hash(args.cases),
                "folds_sha256": file_hash(args.folds), "script_sha256": file_hash(Path(__file__)),
                "shared_scores_script_sha256": file_hash(HERE / "analyze_features.py"),
                "representations": list(REPRESENTATIONS), "models": list(MODELS), "seed": args.seed,
                "n_images": len(cases), "n_groups": int(cases.physical_group.nunique()),
                "inner_validation": "first StratifiedGroupKFold(5) split of outer training, shared across representations/models",
                "logistic_C_grid": list(C_GRID), "primary_comparison": "mean/mlp32 versus mean/tuned_logistic",
                "mlp": {"activation": "ReLU", "optimizer": "Adam", "learning_rate": .001,
                        "weight_decay": .001, "batch_size": 256, "max_epochs": args.max_epochs,
                        "validation_every_epochs": 5, "patience_epochs": 30,
                        "refit": "fresh model, full outer training, selected epoch count"},
                "bootstrap_repeats": args.bootstrap,
                "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "torch")}}
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Changed contract: choose a fresh output directory")
    write_json(path, contract)
    assignments = []
    for split in splits:
        for role in ("inner", "valid", "test"):
            part = cases.iloc[split[role]][["image_id", "physical_group", "domain"]].copy()
            part["split"], part["role"] = split["name"], role
            assignments.append(part)
    pd.concat(assignments).to_csv(output / "split_membership.csv", index=False)
    runs, results, predictions = [], [], {}
    raw_dir = output / "fits"
    raw_dir.mkdir(exist_ok=True)
    for representation in REPRESENTATIONS:
        X = features[representation]
        relevant = splits if representation == "mean" else [s for s in splits if s["evaluation"] == "grouped_cv"]
        for model_name in MODELS:
            key = f"{representation}/{model_name}"
            oof = np.full((len(cases), 3), np.nan)
            train_scores = []
            for split in relevant:
                stem = f"{representation}__{model_name}__{split['name']}"
                json_path, csv_path = raw_dir / f"{stem}.json", raw_dir / f"{stem}.csv"
                print(f"Fit {stem}", flush=True)
                if json_path.exists() and csv_path.exists():
                    info = json.loads(json_path.read_text())
                    table = pd.read_csv(csv_path)
                    if table.image_id.tolist() != cases.iloc[split["test"]].image_id.tolist():
                        raise ValueError("Cached prediction IDs differ")
                    probability = table[[f"p_{c}" for c in CATEGORIES]].to_numpy()
                else:
                    probability, info = fit_one(X, y, split, model_name, args.max_epochs)
                    table = pd.DataFrame({"image_id": cases.iloc[split["test"]].image_id.to_numpy()})
                    for i, c in enumerate(CATEGORIES):
                        table[f"p_{c}"] = probability[:, i]
                    table.to_csv(csv_path, index=False)
                    write_json(json_path, info)
                if not np.isfinite(probability).all() or not np.allclose(probability.sum(1), 1, atol=1e-5):
                    raise ValueError("Invalid probabilities")
                runs.append({"representation": representation, "model": model_name, "split": split["name"],
                             **{k: v for k, v in info.items() if k not in ("validation_trace", "candidates")}})
                if split["evaluation"] == "grouped_cv":
                    oof[split["test"]] = probability
                    train_scores.append(info["train_balanced_accuracy"])
                else:
                    results.append({"representation": representation, "model": model_name,
                                    "evaluation": "leave_domain_out", "held_out_domain": split["domain"],
                                    "train_balanced_accuracy": info["train_balanced_accuracy"],
                                    **scores(y[split["test"]], probability, cases.delta_iou.to_numpy()[split["test"]])})
                print(f"  train BA {info['train_balanced_accuracy']:.3f}; epoch={info['best_epoch']}, C={info['chosen_C']}", flush=True)
            if not np.isfinite(oof).all():
                raise ValueError("Incomplete external CV predictions")
            predictions[key] = oof
            result = {"representation": representation, "model": model_name, "evaluation": "grouped_cv",
                      "held_out_domain": "all", "train_balanced_accuracy": float(np.mean(train_scores)),
                      **scores(y, oof, cases.delta_iou.to_numpy())}
            results.append(result)
            print(f"OOF {key}: BA={result['balanced_accuracy']:.4f}, gate={result['hard_gate_mean_delta_iou']:.5f}", flush=True)
    print("Paired scene bootstrap", flush=True)
    intervals, primary = bootstrap_predictions(cases, y, predictions, args.bootstrap, args.seed)
    table = cases[["image_id", "physical_group", "dataset", "domain", "category", "delta_iou"]].copy()
    for key, probability in predictions.items():
        for i, category in enumerate(CATEGORIES):
            table[f"{key.replace('/', '__')}__p_{category}"] = probability[:, i]
    table.to_csv(output / "predictions.csv", index=False)
    pd.DataFrame(runs).to_csv(output / "training_runs.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in results]).to_csv(output / "metrics.csv", index=False)
    write_json(output / "summary.json", {"status": "complete", "contract": contract, "metrics": results,
               "cv_intervals": intervals, "primary_paired_bootstrap": primary, "class_order": list(CATEGORIES)})
    print(f"Complete: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "tables/categories.csv")
    parser.add_argument("--features", type=Path, default=HERE / "cache/extraction-b1/features.npz")
    parser.add_argument("--folds", type=Path, default=HERE / "results/folds.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "simple_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=1000)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
