#!/usr/bin/env python3
"""Probe historical Frangi benefit from baseline SAM features, without UMAP labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


CATEGORIES = ("degraded", "neutral", "improved")
COLORS = ("#bc4b51", "#a0a7ae", "#168a78")
TITLES = ("Détérioration", "Neutre", "Amélioration")
REQUIRED = ("image_id", "dataset", "domain", "source_family", "physical_group", "category", "delta_iou")
FEATURE_KEYS = ("mean", "std", "grid2", "multiscale_mean", "pre_global_mean")


def load_inputs(cases_path: Path, features_path: Path):
    cases = pd.read_csv(cases_path, dtype={key: str for key in REQUIRED if key != "delta_iou"})
    missing = set(REQUIRED) - set(cases.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if cases.empty or cases[list(REQUIRED)].isna().any().any():
        raise ValueError("Cases must be nonempty, with no missing required values")
    if cases.image_id.duplicated().any():
        raise ValueError("Duplicate image_id in cases")
    unknown = set(cases.category) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")
    if not np.isfinite(cases.delta_iou.to_numpy(float)).all():
        raise ValueError("Non-finite delta_iou")
    if cases.groupby("physical_group").domain.nunique().max() != 1:
        raise ValueError("A physical_group spans multiple domains")
    with np.load(features_path, allow_pickle=False) as archive:
        if not {"ids", "mean"}.issubset(archive.files):
            raise ValueError("Feature archive requires ids and mean")
        ids = archive["ids"].astype(str)
        if ids.ndim != 1 or len(set(ids)) != len(ids):
            raise ValueError("Feature ids must be a unique one-dimensional array")
        index = {name: i for i, name in enumerate(ids)}
        absent = [name for name in cases.image_id if name not in index]
        if absent:
            raise ValueError(f"Missing features for {len(absent)} cases; first: {absent[0]}")
        order = np.array([index[name] for name in cases.image_id])
        features = {}
        for key in FEATURE_KEYS:
            if key not in archive.files:
                continue
            values = archive[key]
            if values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] == 0:
                raise ValueError(f"Invalid feature shape for {key}: {values.shape}")
            features[key] = np.asarray(values[order], dtype=np.float64)
            if not np.isfinite(features[key]).all():
                raise ValueError(f"Non-finite features in {key}")
    if "std" in features and features["std"].shape != features["mean"].shape:
        raise ValueError("std and mean must have the same shape")
    return cases.reset_index(drop=True), features


def encode_labels(cases):
    return np.array([CATEGORIES.index(value) for value in cases.category], dtype=int)


def group_splits(cases, y, seed=42, n_splits=5):
    groups = cases.physical_group.to_numpy()
    n_splits = min(n_splits, len(set(groups)))
    if n_splits < 2:
        raise ValueError("At least two physical groups are required")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("A physical scene leaked across folds")
    return splits


def rank_channels(X, y):
    """Descriptive ANOVA ranking; fitted on training rows only for top-k probes."""
    overall = X.mean(axis=0)
    between = np.zeros(X.shape[1])
    within = np.zeros(X.shape[1])
    classes = np.unique(y)
    for label in classes:
        values = X[y == label]
        center = values.mean(axis=0)
        between += len(values) * (center - overall) ** 2
        within += ((values - center) ** 2).sum(axis=0)
    scores = between / np.maximum(within, 1e-12)
    scores *= max(len(y) - len(classes), 1) / max(len(classes) - 1, 1)
    return np.argsort(-scores, kind="stable"), scores


def fit_predict(X, y, train, test, top_k=None, categorical=False, dummy=False):
    selected = np.arange(X.shape[1])
    if top_k is not None:
        selected = rank_channels(X[train], y[train])[0][:min(top_k, X.shape[1])]
    if categorical:
        transform = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        transform = StandardScaler()
    classifier = (
        DummyClassifier(strategy="most_frequent")
        if dummy or len(np.unique(y[train])) < 2
        else LogisticRegression(C=1.0, class_weight="balanced", max_iter=1500, solver="lbfgs")
    )
    model = make_pipeline(transform, classifier)
    model.fit(X[train][:, selected], y[train])
    raw = model.predict_proba(X[test][:, selected])
    proba = np.zeros((len(test), len(CATEGORIES)))
    proba[:, model.classes_.astype(int)] = raw
    return proba, selected, model


def out_of_fold(X, y, splits, **kwargs):
    proba = np.full((len(y), len(CATEGORIES)), np.nan)
    fold = np.full(len(y), -1)
    selected = []
    for number, (train, test) in enumerate(splits):
        if (fold[test] != -1).any():
            raise ValueError("Each case must be tested exactly once")
        values, channels, _ = fit_predict(X, y, train, test, **kwargs)
        proba[test] = values
        fold[test] = number
        selected.append(channels)
    if not np.isfinite(proba).all() or (fold < 0).any():
        raise ValueError("CV did not predict every case")
    return proba, fold, selected


def scores(y, proba, delta=None):
    predicted = proba.argmax(axis=1)
    matrix = confusion_matrix(y, predicted, labels=np.arange(3))
    support = matrix.sum(axis=1)
    recall = np.divide(matrix.diagonal(), support, out=np.zeros(3, float), where=support > 0)
    result = {
        "n": len(y),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1_score(y, predicted, labels=np.arange(3), average="macro", zero_division=0)),
        "accuracy": float((y == predicted).mean()),
        "chance_balanced_accuracy": float(1 / max((support > 0).sum(), 1)),
        "missing_classes": [CATEGORIES[i] for i in range(3) if support[i] == 0],
        "confusion_matrix": matrix.tolist(),
    }
    aucs = []
    for i, category in enumerate(CATEGORIES):
        target = (y == i)
        result[f"recall_{category}"] = float(recall[i]) if support[i] else None
        auc = float(roc_auc_score(target, proba[:, i])) if target.any() and not target.all() else None
        result[f"auroc_{category}"] = auc
        if auc is not None:
            aucs.append(auc)
    result["macro_auroc_ovr"] = float(np.mean(aucs)) if aucs else None
    if delta is not None:
        selected = predicted == CATEGORIES.index("improved")
        result.update({
            "frangi_activation_rate": float(selected.mean()),
            "hard_gate_mean_delta_iou": float(np.where(selected, delta, 0).mean()),
            "always_frangi_mean_delta_iou": float(np.mean(delta)),
            "oracle_positive_mean_delta_iou": float(np.maximum(delta, 0).mean()),
        })
    return result


def bootstrap_intervals(cases, y, proba, repeats, seed):
    """Resample complete scenes within each domain; fixed fitted OOF predictions."""
    rng = np.random.default_rng(seed)
    members = cases.groupby("physical_group", sort=True).indices
    domains = cases.drop_duplicates("physical_group").groupby("domain").physical_group.apply(list)
    keys = ("balanced_accuracy", "macro_f1", "auroc_improved", "macro_auroc_ovr", "hard_gate_mean_delta_iou")
    collected = {key: [] for key in keys}
    delta = cases.delta_iou.to_numpy(float)
    for _ in range(repeats):
        chosen = [group for groups in domains for group in rng.choice(groups, len(groups), replace=True)]
        indices = np.concatenate([members[group] for group in chosen])
        result = scores(y[indices], proba[indices], delta[indices])
        for key in keys:
            if result[key] is not None:
                collected[key].append(result[key])
    return {key: np.quantile(values, [.025, .975]).tolist() if values else None for key, values in collected.items()}


def representative_indices(cases):
    """One crop per scene, preferring clean Khanhha; selection never uses the target."""
    ordered = cases.assign(_noisy=cases.dataset.str.contains("noisy", regex=False))
    ordered = ordered.sort_values(["_noisy", "image_id"], kind="stable")
    return ordered.drop_duplicates("physical_group").index.to_numpy()


def permute_within_strata(y, strata, rng):
    result = y.copy()
    for stratum in np.unique(strata):
        positions = np.flatnonzero(strata == stratum)
        result[positions] = rng.permutation(y[positions])
    return result


def permutation_analysis(cases, X, y, repeats, seed):
    indices = representative_indices(cases)
    X, y = X[indices], y[indices]
    domains = cases.domain.to_numpy()[indices]
    strata = (cases.domain + "::" + cases.source_family).to_numpy()[indices]
    n_splits = min(5, len(indices))
    # Folds depend on domains/scenes, never on the benefit labels being permuted.
    if min(pd.Series(domains).value_counts()) >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, domains))
    else:
        splits = list(GroupKFold(n_splits=n_splits).split(X, groups=np.arange(len(X))))
    observed = scores(y, out_of_fold(X, y, splits)[0])["balanced_accuracy"]
    rng = np.random.default_rng(seed)
    null = []
    for number in range(repeats):
        permuted = permute_within_strata(y, strata, rng)
        probability = out_of_fold(X, permuted, splits)[0]
        null.append(scores(permuted, probability)["balanced_accuracy"])
        if (number + 1) % 25 == 0:
            print(f"Permutations: {number + 1}/{repeats}", flush=True)
    p_value = float((1 + np.sum(np.asarray(null) >= observed)) / (repeats + 1)) if repeats else None
    stratum_counts = pd.Series(strata).value_counts().to_dict()
    variable_strata = [stratum for stratum in np.unique(strata) if len(np.unique(y[strata == stratum])) > 1]
    return {
        "n_independent_representatives": len(indices),
        "observed_balanced_accuracy": observed,
        "p_value": p_value,
        "permutations": repeats,
        "stratification": "domain x source_family",
        "stratum_counts": stratum_counts,
        "n_strata": len(stratum_counts),
        "n_label_variable_strata": len(variable_strata),
        "n_permutable_representatives": int(np.isin(strata, variable_strata).sum()),
        "selection": "one deterministic crop per scene, clean version preferred; no label used",
        "fixed_folds": "label-independent, stratified by domain where feasible",
    }, null


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar_row(name, result):
    return {"representation": name, **{key: value for key, value in result.items() if isinstance(value, (int, float)) or value is None}}


def plot_projections(cases, X, output, seed, extra_seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap

    standardized = StandardScaler().fit_transform(X)
    y = encode_labels(cases)
    rng = np.random.default_rng(seed)
    sample = np.sort(rng.choice(len(X), min(2000, len(X)), replace=False))
    n_neighbors = min(30, len(X) - 1)
    diagnostics = {"standardization": "all images, descriptive visualizations only", "metric": "euclidean", "n_neighbors": n_neighbors, "min_dist": 0.1, "seed": seed}
    present, counts = np.unique(y[sample], return_counts=True)
    diagnostics["silhouette_original_standardized_mean_features"] = (
        float(silhouette_score(standardized[sample], y[sample]))
        if len(present) > 1 and len(present) < len(sample) and min(counts) >= 2 else None
    )
    diagnostics["diagnostic_subset_n"] = len(sample)
    embeddings = {}
    for dimensions in (2, 3):
        embedding = umap.UMAP(n_components=dimensions, n_neighbors=n_neighbors, min_dist=.1,
                              metric="euclidean", random_state=seed, n_jobs=1,
                              init="random" if len(X) <= dimensions + 1 else "spectral").fit_transform(standardized)
        embeddings[dimensions] = embedding
        table = cases[["image_id", "dataset", "physical_group", "category", "delta_iou"]].copy()
        for axis in range(dimensions):
            table[f"umap_{axis + 1}"] = embedding[:, axis]
        table.to_csv(output / f"umap_{dimensions}d.csv", index=False)
        k = min(10, (len(sample) - 1) // 2)
        diagnostics[f"trustworthiness_{dimensions}d"] = float(trustworthiness(standardized[sample], embedding[sample], n_neighbors=k)) if k else None
        if dimensions == 2:
            fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
            for i, title in enumerate(TITLES):
                chosen = y == i
                axes[0].scatter(*embedding[chosen].T, s=9, color=COLORS[i], alpha=.6, label=f"{title} ({chosen.sum()})", rasterized=True)
            axes[0].legend(fontsize=8, markerscale=2)
            axes[0].set_title("Catégories — sans supervision d’UMAP")
            for domain in sorted(cases.domain.unique()):
                chosen = cases.domain.eq(domain).to_numpy()
                axes[1].scatter(*embedding[chosen].T, s=9, alpha=.6, label=domain, rasterized=True)
            axes[1].legend(fontsize=8, markerscale=2)
            axes[1].set_title("Domaines — même projection")
            limit = max(float(np.quantile(np.abs(cases.delta_iou), .95)), .01)
            scatter = axes[2].scatter(*embedding.T, s=9, c=cases.delta_iou * 100, cmap="RdYlGn", vmin=-100*limit, vmax=100*limit, rasterized=True)
            fig.colorbar(scatter, ax=axes[2], label="ΔIoU (points)")
            axes[2].set_title("Gain du modèle guidé")
            for axis in axes:
                axis.set(xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
            fig.savefig(output / "umap_2d.png", dpi=170)
            plt.close(fig)
        else:
            fig = plt.figure(figsize=(8, 7), constrained_layout=True)
            axis = fig.add_subplot(projection="3d")
            for i, title in enumerate(TITLES):
                chosen = y == i
                axis.scatter(*embedding[chosen].T, s=8, color=COLORS[i], alpha=.6, depthshade=False, label=title)
            axis.set(xlabel="UMAP 1", ylabel="UMAP 2", zlabel="UMAP 3", title="UMAP 3D — catégories ajoutées après projection")
            axis.legend(loc="upper right")
            fig.savefig(output / "umap_3d.png", dpi=170)
            plt.close(fig)
            if importlib.util.find_spec("plotly") is not None:
                import plotly.express as px
                interactive = px.scatter_3d(table, x="umap_1", y="umap_2", z="umap_3", color="category",
                    color_discrete_map=dict(zip(CATEGORIES, COLORS)), hover_name="image_id",
                    hover_data=["dataset", "delta_iou"], title="UMAP non supervisée des features de la baseline")
                interactive.update_traces(marker={"size": 3, "opacity": .7})
                interactive.write_html(output / "umap_3d.html", include_plotlyjs=True, full_html=True)
                diagnostics["interactive_3d"] = True
            else:
                diagnostics["interactive_3d"] = False
    # A linear projection provides a visual comparison; no classifier uses this fit.
    pca = PCA(n_components=min(2, standardized.shape[1], len(X)), random_state=seed)
    coordinates = pca.fit_transform(standardized)
    if coordinates.shape[1] == 2:
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for i, title in enumerate(TITLES):
            axis.scatter(*coordinates[y == i].T, s=9, color=COLORS[i], alpha=.6, label=title)
        axis.legend(fontsize=9)
        axis.set(xlabel="PC 1", ylabel="PC 2", title="Projection linéaire PCA (comparaison)")
        fig.savefig(output / "pca_2d.png", dpi=160)
        plt.close(fig)
    diagnostics["pca_explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
    if extra_seeds:
        fig, axes = plt.subplots(1, len(extra_seeds), figsize=(6 * len(extra_seeds), 5), squeeze=False, constrained_layout=True)
        for axis, other_seed in zip(axes[0], extra_seeds):
            embedding = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=.1, random_state=other_seed, n_jobs=1).fit_transform(standardized)
            for i, title in enumerate(TITLES):
                axis.scatter(*embedding[y == i].T, s=8, color=COLORS[i], alpha=.6, label=title)
            axis.set(title=f"UMAP 2D — graine {other_seed}", xticks=[], yticks=[])
            axis.legend(fontsize=8)
        fig.savefig(output / "umap_seed_sensitivity.png", dpi=160)
        plt.close(fig)
    diagnostics["extra_seeds_2d"] = list(extra_seeds)
    return diagnostics


def plot_statistics(summary, output, null):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrix = np.asarray(summary["probes"]["mean"]["confusion_matrix"])
    rates = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    axis.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            axis.text(j, i, f"{matrix[i,j]}\n{rates[i,j]:.1%}", ha="center", va="center", color="white" if rates[i,j] > .55 else "black")
    axis.set(xticks=range(3), yticks=range(3), xticklabels=TITLES, yticklabels=TITLES,
             xlabel="Prédiction hors fold", ylabel="Catégorie observée", title="Sonde linéaire sur la moyenne de H")
    fig.savefig(output / "linear_probe_confusion.png", dpi=170)
    plt.close(fig)
    if null:
        fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.hist(null, bins=min(25, max(5, len(null) // 8)), color="#a0a7ae", edgecolor="white")
        observed = summary["permutation"]["observed_balanced_accuracy"]
        axis.axvline(observed, color=COLORS[2], linewidth=2, label=f"Observé : {observed:.3f}")
        axis.legend()
        axis.set(xlabel="Balanced accuracy", ylabel="Permutations", title="Une image par scène, permutation par domaine et famille source")
        fig.savefig(output / "permutation_test.png", dpi=170)
        plt.close(fig)


def channel_table(X, y, cases, splits):
    ranking, f_scores = rank_channels(X, y)
    _, _, fitted = fit_predict(X, y, np.arange(len(y)), np.arange(len(y)))
    classifier = fitted.steps[-1][1]
    coefficients = np.zeros((3, X.shape[1]))
    if hasattr(classifier, "coef_"):
        if len(classifier.classes_) == 2:
            coefficients[classifier.classes_[0]] = -classifier.coef_[0] / 2
            coefficients[classifier.classes_[1]] = classifier.coef_[0] / 2
        else:
            coefficients[classifier.classes_] = classifier.coef_
    count = np.zeros(X.shape[1], int)
    for train, _ in splits:
        count[rank_channels(X[train], y[train])[0][:min(10, X.shape[1])]] += 1
    correlations = []
    for channel in range(X.shape[1]):
        correlation = spearmanr(X[:, channel], cases.delta_iou).statistic if np.ptp(X[:, channel]) > 0 else 0.0
        correlations.append(float(correlation) if np.isfinite(correlation) else 0.0)
    return pd.DataFrame({
        "channel": np.arange(X.shape[1]), "anova_score_descriptive": f_scores,
        "spearman_delta_descriptive": correlations,
        "coefficient_improved_minus_degraded_descriptive": coefficients[2] - coefficients[0],
        "mean_absolute_standardized_coefficient_descriptive": np.abs(coefficients).mean(axis=0),
        "top10_training_fold_frequency": count / len(splits),
    }).iloc[ranking]


def analyze(args):
    cases, features = load_inputs(args.cases, args.features)
    if not args.skip_umap and importlib.util.find_spec("umap") is None:
        raise RuntimeError("Install umap-learn to generate the requested 2D/3D projections")
    if len(cases) < 4 or cases.category.nunique() < 2:
        raise ValueError("Analysis requires at least four cases and two observed categories")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    y = encode_labels(cases)
    delta = cases.delta_iou.to_numpy(float)
    splits = group_splits(cases, y, args.seed)
    folds = cases[["image_id", "physical_group", "domain"]].copy()
    summary = {
        "n_images": len(cases), "n_physical_groups": cases.physical_group.nunique(),
        "category_order": CATEGORIES, "category_counts": cases.category.value_counts().to_dict(),
        "seed": args.seed, "grouped_folds": len(splits),
        "probe": "StandardScaler fitted on training rows; balanced L2 logistic regression, C=1, lbfgs",
        "primary_representation": "mean: spatial mean of baseline SAM features before the prompt",
        "bootstrap": {"repeats": args.bootstrap, "unit": "physical scene", "strata": "domain", "conditional_on_fixed_oof_predictions": True},
        "features_path": str(args.features), "cases_path": str(args.cases),
        "features_sha256": file_hash(args.features),
        "cases_sha256": file_hash(args.cases),
        "analysis_script_sha256": file_hash(Path(__file__)),
        "versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "umap-learn", "plotly")},
        "probes": {},
    }
    representations = {key: values for key, values in features.items() if key != "std"}
    if "std" in features:
        representations["mean_std"] = np.concatenate([features["mean"], features["std"]], axis=1)
    predictions = cases[["image_id", "dataset", "domain", "physical_group", "category", "delta_iou"]].copy()
    mean_proba = None
    for name, X in representations.items():
        print(f"Linear probe: {name} ({X.shape[1]} dimensions)", flush=True)
        proba, fold, _ = out_of_fold(X, y, splits)
        summary["probes"][name] = {"dimensions": X.shape[1], **scores(y, proba, delta)}
        predictions[f"predicted_{name}"] = np.asarray(CATEGORIES)[proba.argmax(axis=1)]
        if name == "mean":
            mean_proba = proba
            folds["fold"] = fold
            for i, category in enumerate(CATEGORIES):
                predictions[f"p_{category}"] = proba[:, i]
            summary["primary_group_bootstrap_ci95"] = bootstrap_intervals(cases, y, proba, args.bootstrap, args.seed)
    for name, columns, dummy in (("domain_only", ["domain"], False), ("domain_source_only", ["domain", "source_family"], False), ("majority", ["domain"], True)):
        proba, _, _ = out_of_fold(cases[columns].to_numpy(), y, splits, categorical=True, dummy=dummy)
        summary["probes"][name] = scores(y, proba, delta)
    for count in (1, 5, 10, 25):
        proba, _, _ = out_of_fold(features["mean"], y, splits, top_k=count)
        summary["probes"][f"mean_top{count}_train_selection"] = scores(y, proba, delta)
    if "baseline_iou" in cases:
        baseline_mean = float(cases.baseline_iou.mean())
        summary["historical_model_mean_iou"] = {
            "baseline": baseline_mean,
            "always_frangi": baseline_mean + float(delta.mean()),
            "oracle_positive": baseline_mean + float(np.maximum(delta, 0).mean()),
        }
        for result in summary["probes"].values():
            result["hard_gate_mean_iou"] = baseline_mean + result["hard_gate_mean_delta_iou"]
    rows = []
    for column in ("domain", "dataset"):
        for value in sorted(cases[column].unique()):
            chosen = cases[column].eq(value).to_numpy()
            rows.append({"grouping": column, "cohort": value, **scalar_row("mean_grouped_oof", scores(y[chosen], mean_proba[chosen], delta[chosen]))})
    # Historical training/validation scene overlap is an explanatory audit, not a new split.
    if {"historical_train_group", "historical_validation_group"}.issubset(cases.columns):
        exposed = np.zeros(len(cases), bool)
        for column in ("historical_train_group", "historical_validation_group"):
            exposed |= cases[column].astype(str).str.lower().isin(("true", "1")).to_numpy()
        khanhha = cases.domain.str.lower().eq("khanhha").to_numpy()
        clean = cases.dataset.eq("khanhha_original").to_numpy()
        for name, chosen in (
            ("absent_historical_train_and_validation", ~exposed),
            ("historical_group_overlap", exposed),
            ("khanhha_original_unseen_historical_scene", khanhha & clean & ~exposed),
            ("khanhha_all_conditions_unseen_historical_scene", khanhha & ~exposed),
            ("external_domains", ~khanhha),
        ):
            if chosen.any():
                rows.append({"grouping": "history", "cohort": name, **scalar_row("mean_grouped_oof", scores(y[chosen], mean_proba[chosen], delta[chosen]))})
    leave_domain = []
    for domain in sorted(cases.domain.unique()):
        train = np.flatnonzero(cases.domain.ne(domain))
        test = np.flatnonzero(cases.domain.eq(domain))
        if not len(train):
            continue
        proba, _, _ = fit_predict(features["mean"], y, train, test)
        leave_domain.append({"held_out_domain": domain, **scores(y[test], proba, delta[test])})
    summary["leave_one_domain_out"] = leave_domain
    print("Permutation test on one deterministic representative per scene", flush=True)
    summary["permutation"], null = permutation_analysis(cases, features["mean"], y, args.permutations, args.seed)
    folds.to_csv(output / "folds.csv", index=False)
    predictions.to_csv(output / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame([scalar_row(name, result) for name, result in summary["probes"].items()]).to_csv(output / "probe_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "cohort_metrics.csv", index=False)
    pd.DataFrame([scalar_row(row["held_out_domain"], row) for row in leave_domain]).to_csv(output / "leave_one_domain_out.csv", index=False)
    pd.DataFrame({"permutation": np.arange(1, len(null)+1), "balanced_accuracy": null}).to_csv(output / "permutation_distribution.csv", index=False)
    channel_table(features["mean"], y, cases, splits).to_csv(output / "channel_ranking.csv", index=False)
    write_json(output / "statistics.json", summary)
    plot_statistics(summary, output, null)
    if not args.skip_umap:
        print("Unsupervised UMAP 2D and 3D", flush=True)
        summary["versions"]["umap-learn"] = version("umap-learn")
        if importlib.util.find_spec("plotly") is not None:
            summary["versions"]["plotly"] = version("plotly")
        summary["projections"] = plot_projections(cases, features["mean"], output, args.seed, args.umap_extra_seeds)
    else:
        summary["projections"] = {"skipped": True}
    write_json(output / "statistics.json", summary)
    print(f"Completed: {output / 'statistics.json'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=Path("tables/categories.csv"))
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-extra-seeds", type=int, nargs="*", default=[])
    parser.add_argument("--skip-umap", action="store_true", help="Statistics only; explicitly records that projections were skipped")
    args = parser.parse_args()
    if args.permutations < 0 or args.bootstrap < 0:
        parser.error("--permutations and --bootstrap must be nonnegative")
    return args


if __name__ == "__main__":
    # BLAS oversubscription otherwise dominates these small linear probes.
    with threadpool_limits(limits=1):
        analyze(parse_args())
