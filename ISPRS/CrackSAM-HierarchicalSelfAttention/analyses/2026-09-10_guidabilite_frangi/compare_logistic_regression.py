#!/usr/bin/env python3
"""Multinomial logistic classification, with scene-nested selection and controls."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analyze_features import CATEGORIES, encode_labels, file_hash, load_inputs, rank_channels, scores, write_json


HERE = Path(__file__).resolve().parent
REPRESENTATIONS = ("mean", "multiscale_mean", "pre_global_mean")
C_GRID = (.001, .01, .1, 1., 10.)
K_GRID = (32, 128, "all")
STRATEGIES = {
    "balanced_ba": ("balanced", "balanced_accuracy"),
    "uniform_ba": ("uniform", "balanced_accuracy"),
    "uniform_accuracy": ("uniform", "accuracy"),
}


def make_outer_splits(cases, folds_path):
    recorded = pd.read_csv(folds_path).set_index("image_id").loc[cases.image_id]
    if not np.array_equal(recorded.physical_group, cases.physical_group):
        raise ValueError("Recorded scenes do not match cases")
    folds = recorded.fold.to_numpy(int)
    groups = cases.physical_group.to_numpy()
    splits = []
    for fold in map(int, sorted(set(folds))):
        splits.append(dict(name=f"cv{fold}", evaluation="grouped_cv", domain="all",
                           train=np.flatnonzero(folds != fold), test=np.flatnonzero(folds == fold), seed=42 + fold))
    for i, domain in enumerate(sorted(cases.domain.unique())):
        splits.append(dict(name=f"domain_{domain}", evaluation="leave_domain_out", domain=domain,
                           train=np.flatnonzero(cases.domain.ne(domain)),
                           test=np.flatnonzero(cases.domain.eq(domain)), seed=142 + i))
    for split in splits:
        if set(groups[split["train"]]) & set(groups[split["test"]]):
            raise ValueError("External scene leakage")
    return splits


def inner_splits(cases, y, train, seed):
    """Three folds built solely from outer-training labels and scene IDs."""
    groups = cases.physical_group.to_numpy()
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=seed)
    result = []
    for a, b in splitter.split(np.zeros(len(train)), y[train], groups[train]):
        fit, valid = train[a], train[b]
        if set(groups[fit]) & set(groups[valid]):
            raise ValueError("Internal scene leakage")
        if len(np.unique(y[fit])) != 3 or len(np.unique(y[valid])) != 3:
            raise ValueError("Every internal fold must contain three categories")
        result.append((fit, valid))
    return result


def prepare_design(X, y, train, test, k):
    """Rank and standardize only training rows; return the fitted transformation."""
    selected = rank_channels(X[train], y[train])[0][:int(k)] if k < X.shape[1] else np.arange(X.shape[1])
    scaler = StandardScaler().fit(X[train][:, selected])
    a = np.ascontiguousarray(scaler.transform(X[train][:, selected]))
    b = np.ascontiguousarray(scaler.transform(X[test][:, selected]))
    return a, b, selected, scaler


def fit_logistic(X, y, C, weighting, estimator=None):
    if estimator is None:
        estimator = LogisticRegression(solver="lbfgs", class_weight="balanced" if weighting == "balanced" else None,
                                       max_iter=2000, tol=1e-4, warm_start=True)
    estimator.set_params(C=C, max_iter=2000)
    retries = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        retries = 1
        estimator.set_params(max_iter=10000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X, y)
    if caught:
        raise RuntimeError("Logistic fitting warning: " + "; ".join(str(w.message) for w in caught))
    return estimator, retries


def matrix_metrics(matrix):
    matrix = np.asarray(matrix)
    return {"accuracy": float(matrix.trace() / matrix.sum()),
            "balanced_accuracy": float((matrix.diagonal() / matrix.sum(1)).mean())}


def fit_split(X, y, cases, split, c_grid=C_GRID, k_grid=K_GRID, checkpoint=None):
    """External labels are never accessed here. Return three held-out classifiers."""
    start = time.monotonic()
    train, test = split["train"], split["test"]
    ks = sorted(set(X.shape[1] if k == "all" else min(int(k), X.shape[1]) for k in k_grid))
    cs = sorted(map(float, c_grid))
    internal = inner_splits(cases, y, train, split["seed"])
    fold_results = []
    retries = 0
    for fold, (fit, valid) in enumerate(internal):
        path = checkpoint / f"inner{fold}.json" if checkpoint else None
        if path and path.exists():
            data = json.loads(path.read_text())
        else:
            rows = []
            for k in ks:
                a, b, _, _ = prepare_design(X, y, fit, valid, k)
                for weighting in ("balanced", "uniform"):
                    estimator = None
                    for C in cs:
                        estimator, retried = fit_logistic(a, y[fit], C, weighting, estimator)
                        matrix = confusion_matrix(y[valid], estimator.predict(b), labels=[0, 1, 2])
                        rows.append(dict(k=k, C=C, weighting=weighting, matrix=matrix.tolist(), retries=retried))
            data = {"fold": fold, "candidates": rows}
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json(path, data)
        fold_results.append(data["candidates"])
        retries += sum(row["retries"] for row in data["candidates"])
    candidates = []
    for rows in zip(*fold_results):
        keys = [(row["k"], row["C"], row["weighting"]) for row in rows]
        if len(set(keys)) != 1:
            raise ValueError("Internal candidate order mismatch")
        matrix = np.sum([row["matrix"] for row in rows], axis=0)
        candidates.append(dict(k=keys[0][0], C=keys[0][1], weighting=keys[0][2], **matrix_metrics(matrix)))
    outputs = {}
    for strategy, (weighting, objective) in STRATEGIES.items():
        eligible = [row for row in candidates if row["weighting"] == weighting]
        chosen = sorted(eligible, key=lambda row: (-row[objective], row["C"], row["k"]))[0]
        a, b, selected, scaler = prepare_design(X, y, train, test, chosen["k"])
        estimator, retried = fit_logistic(a, y[train], chosen["C"], weighting)
        train_matrix = confusion_matrix(y[train], estimator.predict(a), labels=[0, 1, 2])
        info = dict(chosen=chosen, objective=objective, selected_channels=selected.tolist(),
                    coefficients=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist(),
                    scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                    train_metrics=matrix_metrics(train_matrix), final_iterations=estimator.n_iter_.tolist(),
                    final_retry=retried)
        outputs[strategy] = (estimator.predict_proba(b), info)
    return outputs, dict(candidates=candidates, internal_convergence_retries=retries,
                         elapsed_seconds=time.monotonic() - start)


def paired_bootstrap(cases, y, predictions, repeats=1000, seed=42):
    """Conditional uncertainty from whole scenes, with shared draws across models."""
    groups, codes = np.unique(cases.physical_group, return_inverse=True)
    domains = cases.assign(code=codes).groupby("code").domain.first().reindex(range(len(groups))).to_numpy()
    rng = np.random.default_rng(seed)
    weights = np.zeros((repeats, len(groups)))
    for domain in np.unique(domains):
        indices = np.flatnonzero(domains == domain)
        weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=repeats)
    draws, estimates = {}, {}
    delta = cases.delta_iou.to_numpy(float)
    for key, proba in predictions.items():
        predicted = proba.argmax(1)
        matrices = np.bincount(codes * 9 + y * 3 + predicted, minlength=len(groups) * 9).reshape(len(groups), 9)
        sampled = (weights @ matrices).reshape(repeats, 3, 3)
        diagonal = sampled.diagonal(axis1=1, axis2=2)
        denominator = sampled.sum((1, 2))
        sums = np.bincount(codes, weights=np.where(predicted == 2, delta, 0), minlength=len(groups))
        draws[key] = {"accuracy": diagonal.sum(1) / denominator,
                      "balanced_accuracy": (diagonal / np.maximum(sampled.sum(2), 1)).mean(1),
                      "hard_gate_mean_delta_iou": (weights @ sums) / denominator}
        estimates[key] = scores(y, proba, delta)
    intervals = {key: {stat: np.quantile(values, [.025, .975]).tolist() for stat, values in stats.items()}
                 for key, stats in draws.items()}
    comparisons = []
    for candidate, reference in (("mean/uniform_accuracy", "always_neutral"),
                                  ("mean/balanced_ba", "previous_mean_tuned"),
                                  ("mean/uniform_ba", "mean/balanced_ba")):
        if candidate not in draws or reference not in draws:
            continue
        comparisons.append({"candidate": candidate, "reference": reference, "statistics": {
            stat: {"estimate": estimates[candidate][stat] - estimates[reference][stat],
                   "ci95": np.quantile(draws[candidate][stat] - draws[reference][stat], [.025, .975]).tolist()}
            for stat in draws[candidate]}})
    return {"intervals": intervals, "comparisons": comparisons, "repeats": repeats,
            "unit": "physical scene", "strata": "domain", "conditional_on_fixed_predictions": True}


def run(args):
    cases, features = load_inputs(args.cases, args.features)
    y = encode_labels(cases)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    contract = dict(features_sha256=file_hash(args.features), cases_sha256=file_hash(args.cases),
                    folds_sha256=file_hash(args.folds), script_sha256=file_hash(Path(__file__)),
                    shared_script_sha256=file_hash(HERE / "analyze_features.py"),
                    previous_predictions_sha256=file_hash(HERE / "simple_models/predictions.csv"),
                    representations=list(REPRESENTATIONS), strategies=STRATEGIES, C_grid=list(C_GRID),
                    k_grid=list(K_GRID), inner_folds=3, outer_folds=5, seed=42, solver="lbfgs", penalty="L2",
                    tolerance=1e-4, max_iter=2000, retry_max_iter=10000,
                    selection="pooled internal held-out confusion matrix; ties favor smaller C then fewer channels",
                    primary="mean/uniform_accuracy versus always neutral; mean/balanced_ba versus previous mean tuned",
                    n_images=len(cases), n_groups=int(cases.physical_group.nunique()),
                    bootstrap_repeats=args.bootstrap,
                    versions={p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy")})
    contract = json.loads(json.dumps(contract))
    path = output / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Contract changed; choose a fresh output directory")
    write_json(path, contract)
    splits = make_outer_splits(cases, args.folds)
    membership = []
    for split in splits:
        for fold, (fit, valid) in enumerate(inner_splits(cases, y, split["train"], split["seed"])):
            for role, indices in (("train", fit), ("validation", valid)):
                part = cases.iloc[indices][["image_id", "physical_group"]].copy()
                part["split"], part["inner_fold"], part["role"] = split["name"], fold, role
                membership.append(part)
        part = cases.iloc[split["test"]][["image_id", "physical_group"]].copy()
        part["split"], part["inner_fold"], part["role"] = split["name"], -1, "test"
        membership.append(part)
    pd.concat(membership).to_csv(output / "split_membership.csv", index=False)
    records, tables, predictions = [], [], {}
    for representation in REPRESENTATIONS:
        relevant = splits if representation == "mean" else [s for s in splits if s["evaluation"] == "grouped_cv"]
        oof = {strategy: np.full((len(cases), 3), np.nan) for strategy in STRATEGIES}
        training = {strategy: [] for strategy in STRATEGIES}
        for split in relevant:
            directory = output / "fits" / f"{representation}__{split['name']}"
            directory.mkdir(parents=True, exist_ok=True)
            complete = directory / "complete.json"
            if complete.exists():
                metadata = json.loads(complete.read_text())
                outputs = {}
                for strategy in STRATEGIES:
                    table = pd.read_csv(directory / f"{strategy}.csv").set_index("image_id").loc[cases.image_id.iloc[split["test"]]]
                    outputs[strategy] = (table[list(CATEGORIES)].to_numpy(), metadata["models"][strategy])
            else:
                outputs, search = fit_split(features[representation], y, cases, split, checkpoint=directory)
                metadata = dict(search=search, models={key: value[1] for key, value in outputs.items()})
                for strategy, (proba, info) in outputs.items():
                    table = pd.DataFrame(proba, columns=CATEGORIES)
                    table.insert(0, "image_id", cases.image_id.iloc[split["test"]].to_numpy())
                    table.to_csv(directory / f"{strategy}.csv", index=False)
                write_json(complete, metadata)
            for strategy, (proba, info) in outputs.items():
                if not np.isfinite(proba).all() or not np.allclose(proba.sum(1), 1):
                    raise ValueError("Invalid probabilities")
                if split["evaluation"] == "grouped_cv":
                    if np.isfinite(oof[strategy][split["test"]]).any():
                        raise ValueError("Repeated external predictions")
                    oof[strategy][split["test"]] = proba
                    training[strategy].append(info["train_metrics"])
                else:
                    records.append(dict(representation=representation, strategy=strategy, evaluation=split["evaluation"],
                                        held_out_domain=split["domain"], **scores(y[split["test"]], proba, cases.delta_iou.to_numpy()[split["test"]])))
                part = pd.DataFrame(proba, columns=CATEGORIES)
                part.insert(0, "image_id", cases.image_id.iloc[split["test"]].to_numpy())
                part["representation"], part["strategy"], part["split"] = representation, strategy, split["name"]
                tables.append(part)
            print(f"Completed {representation} {split['name']} ({metadata['search']['elapsed_seconds']:.1f}s)", flush=True)
        for strategy, proba in oof.items():
            if not np.isfinite(proba).all():
                raise ValueError("Incomplete out-of-fold predictions")
            predictions[f"{representation}/{strategy}"] = proba
            records.append(dict(representation=representation, strategy=strategy, evaluation="grouped_cv", held_out_domain="all",
                                train_accuracy=float(np.mean([row["accuracy"] for row in training[strategy]])),
                                train_balanced_accuracy=float(np.mean([row["balanced_accuracy"] for row in training[strategy]])),
                                **scores(y, proba, cases.delta_iou.to_numpy())))
    neutral = np.zeros((len(cases), 3))
    neutral[:, 1] = 1
    predictions["always_neutral"] = neutral
    previous = pd.read_csv(HERE / "simple_models/predictions.csv")
    predictions["previous_mean_tuned"] = previous.set_index("image_id").loc[cases.image_id, [f"mean__tuned_logistic__p_{c}" for c in CATEGORIES]].to_numpy()
    summary = dict(contract=contract, results=records, bootstrap=paired_bootstrap(cases, y, predictions, args.bootstrap),
                   always_neutral=scores(y, neutral, cases.delta_iou.to_numpy()))
    write_json(output / "summary.json", summary)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, (list, dict))} for row in records]).to_csv(output / "metrics.csv", index=False)
    pd.concat(tables).to_csv(output / "predictions.csv", index=False)
    print("All logistic models complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "tables/categories.csv")
    parser.add_argument("--features", type=Path, default=HERE / "cache/extraction-b1/features.npz")
    parser.add_argument("--folds", type=Path, default=HERE / "results/folds.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "logistic_regression")
    parser.add_argument("--bootstrap", type=int, default=1000)
    with threadpool_limits(limits=1):
        run(parser.parse_args())
