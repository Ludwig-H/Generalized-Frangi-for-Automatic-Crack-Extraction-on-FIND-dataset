#!/usr/bin/env python3
"""Compare fixed SAM feature summaries; labels never enter UMAP or preprocessing."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
from threadpoolctl import threadpool_limits
import umap

from analyze_features import (
    CATEGORIES, COLORS, TITLES, encode_labels, file_hash, load_inputs,
    representative_indices, scores, write_json,
)
from feature_variants import build_variants


HERE = Path(__file__).resolve().parent
DOMAIN_COLORS = {"concrete3k": "#397db4", "facade390": "#d99b26",
                 "khanhha": "#7966a8", "road420": "#283b43"}


def transform_values(spec, train, test=None):
    """All fitted statistics depend only on train; this function has no labels."""
    X = spec["values"]
    if spec["transform"] == "l2":
        a = normalize(X[train])
        b = normalize(X[test]) if test is not None else None
    else:
        scaler = StandardScaler().fit(X[train])
        a = scaler.transform(X[train])
        b = scaler.transform(X[test]) if test is not None else None
        if spec["transform"] == "balanced":
            weights = np.concatenate([np.full(width, 1 / np.sqrt(width)) for width in spec["blocks"]])
            a *= weights
            if b is not None:
                b *= weights
        elif spec["transform"] == "pca32":
            pca = PCA(n_components=min(32, a.shape[1], len(a)), svd_solver="full")
            a = pca.fit_transform(a)
            if b is not None:
                b = pca.transform(b)
        elif spec["transform"] != "standard":
            raise ValueError(f"Unknown transform: {spec['transform']}")
    return np.asarray(a, dtype=np.float32), None if b is None else np.asarray(b, dtype=np.float32)


def saved_splits(cases, path):
    recorded = pd.read_csv(path).set_index("image_id").loc[cases.image_id]
    if not np.array_equal(recorded.physical_group.to_numpy(), cases.physical_group.to_numpy()):
        raise ValueError("Historical fold groups do not match current cases")
    folds = recorded.fold.to_numpy(int)
    if pd.DataFrame({"group": cases.physical_group.to_numpy(), "fold": folds}).groupby("group").fold.nunique().max() != 1:
        raise ValueError("Physical scene leakage in folds")
    return [(np.flatnonzero(folds != k), np.flatnonzero(folds == k)) for k in sorted(set(folds))]


def probe(spec, y, delta, splits):
    proba = np.full((len(y), 3), np.nan)
    for train, test in splits:
        a, b = transform_values(spec, train, test)
        model = LogisticRegression(C=1, class_weight="balanced", solver="lbfgs", max_iter=1500)
        model.fit(a, y[train])
        proba[np.ix_(test, model.classes_)] = model.predict_proba(b)
    if not np.isfinite(proba).all():
        raise ValueError("Incomplete out-of-fold probabilities")
    return scores(y, proba, delta), proba


def macro_neighbor_agreement(X, y, domains=None, metric="euclidean", k=15):
    """One row per scene; optionally search only within each query's domain."""
    agreement = np.full(len(X), np.nan)
    strata = [np.arange(len(X))] if domains is None else [np.flatnonzero(domains == d) for d in sorted(set(domains))]
    for subset in strata:
        n_neighbors = min(k + 1, len(subset))
        if n_neighbors < 2:
            continue
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric=metric).fit(X[subset]).kneighbors(X[subset], return_distance=False)
        for i, candidates in enumerate(neighbors):
            others = candidates[candidates != i][:k]
            agreement[subset[i]] = np.mean(y[subset[others]] == y[subset[i]])
    return float(np.nanmean([np.nanmean(agreement[y == label]) for label in np.unique(y)]))


def diagnostics(X, embedding, y, domains, metric):
    return {
        "silhouette_features": float(silhouette_score(X, y, metric=metric)),
        "silhouette_umap": float(silhouette_score(embedding, y)),
        "silhouette_domain_umap": float(silhouette_score(embedding, domains)),
        "neighbors_features": macro_neighbor_agreement(X, y, metric=metric),
        "neighbors_umap": macro_neighbor_agreement(embedding, y),
        "neighbors_features_within_domain": macro_neighbor_agreement(X, y, domains, metric),
        "neighbors_umap_within_domain": macro_neighbor_agreement(embedding, y, domains),
        "trustworthiness": float(trustworthiness(X, embedding, n_neighbors=10, metric=metric)),
    }


def projection(X, dimensions, seed, metric):
    return umap.UMAP(n_components=dimensions, n_neighbors=30, min_dist=.1,
                     metric=metric, random_state=seed, n_jobs=1).fit_transform(X)


def category_scatter(axis, embedding, y, order, size=2):
    # Mix drawing order independently of labels, identically in every panel.
    axis.scatter(*embedding[order].T, c=np.asarray(COLORS)[y[order]], s=size, alpha=.55,
                 linewidths=0, rasterized=True)
    axis.set(xticks=[], yticks=[])


def legend_handles():
    return [Line2D([], [], marker="o", linestyle="", color=c, label=t, markersize=6)
            for c, t in zip(COLORS, TITLES)]


def plot_gallery(variants, embeddings, metrics, cases, output, order):
    y = encode_labels(cases)
    rows = int(np.ceil(len(variants) / 4))
    for coloring in ("categories", "domains"):
        fig, axes = plt.subplots(rows, 4, figsize=(17, 3.4 * rows), constrained_layout=True)
        for axis, (name, spec) in zip(axes.flat, variants.items()):
            embedding = embeddings[name]
            if coloring == "categories":
                category_scatter(axis, embedding, y, order)
                subtitle = f"silhouette : {metrics[name]['silhouette_umap']:.3f}"
            else:
                colors = cases.domain.map(DOMAIN_COLORS).to_numpy()[order]
                axis.scatter(*embedding[order].T, c=colors, s=2, alpha=.55, linewidths=0, rasterized=True)
                axis.set(xticks=[], yticks=[])
                subtitle = f"silhouette domaines : {metrics[name]['silhouette_domain_umap']:.3f}"
            axis.set_title(f"{spec['title']}\n{subtitle}", fontsize=10)
        for axis in list(axes.flat)[len(variants):]:
            axis.set_visible(False)
        handles = legend_handles() if coloring == "categories" else [
            Line2D([], [], marker="o", linestyle="", color=c, label=d) for d, c in DOMAIN_COLORS.items()]
        fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
        fig.suptitle("Même protocole UMAP — catégories ajoutées après projection" if coloring == "categories"
                     else "Mêmes projections — couleurs selon la collection d’images", fontsize=15)
        fig.savefig(output / f"comparison_{coloring}.png", dpi=170)
        plt.close(fig)


def main(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cases, features = load_inputs(args.cases, args.features)
    variants = build_variants(features)
    y = encode_labels(cases)
    domains = cases.domain.to_numpy()
    independent = representative_indices(cases)
    splits = saved_splits(cases, args.folds)
    order = np.random.default_rng(20260910).permutation(len(cases))
    contract = {
        "features_sha256": file_hash(args.features), "cases_sha256": file_hash(args.cases),
        "folds_sha256": file_hash(args.folds), "script_sha256": file_hash(Path(__file__)),
        "variants_sha256": file_hash(HERE / "feature_variants.py"),
        "versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "umap-learn", "matplotlib", "plotly")},
        "n_images": len(cases), "n_independent_diagnostic_scenes": len(independent),
        "seed": 42, "n_neighbors": 30, "min_dist": .1,
        "labels_used_for_umap_or_preprocessing": False,
        "diagnostic_sampling": "one deterministic image per physical scene, clean preferred",
        "selection": "largest silhouette of categories in 2D; exploratory among all published variants",
        "variants": {k: {a: b for a, b in v.items() if a != "values"} for k, v in variants.items()},
    }
    contract_path = output / "contract.json"
    contract = json.loads(json.dumps(contract))
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Contract changed: use a fresh output directory")
    write_json(contract_path, contract)
    metrics, embeddings = {}, {}
    predictions = cases[["image_id", "physical_group", "category", "delta_iou"]].copy()
    for name, spec in variants.items():
        print(f"UMAP + probe: {name}, {spec['values'].shape[1]} dimensions", flush=True)
        X, _ = transform_values(spec, np.arange(len(cases)))
        coordinate_path = output / f"{name}_2d.csv"
        detail_path = output / f"{name}_metrics.json"
        if coordinate_path.exists() and detail_path.exists():
            coordinates = pd.read_csv(coordinate_path)
            if coordinates.image_id.tolist() != cases.image_id.tolist():
                raise ValueError("Resume coordinate IDs differ")
            embedding = coordinates[["umap_1", "umap_2"]].to_numpy()
            result = json.loads(detail_path.read_text())
            probabilities = coordinates[[f"p_{c}" for c in CATEGORIES]].to_numpy()
        else:
            embedding = projection(X, 2, 42, spec["metric"])
            result = {"representation": name, "dimensions": X.shape[1],
                      **diagnostics(X[independent], embedding[independent], y[independent], domains[independent], spec["metric"])}
            linear, probabilities = probe(spec, y, cases.delta_iou.to_numpy(), splits)
            result["probe"] = linear
            coordinates = cases[["image_id", "category", "domain", "physical_group", "delta_iou"]].copy()
            coordinates["umap_1"], coordinates["umap_2"] = embedding.T
            for i, category in enumerate(CATEGORIES):
                coordinates[f"p_{category}"] = probabilities[:, i]
            coordinates.to_csv(coordinate_path, index=False)
            write_json(detail_path, result)
        metrics[name], embeddings[name] = result, embedding
        predictions[f"predicted_{name}"] = np.asarray(CATEGORIES)[probabilities.argmax(1)]
        print(f"  silhouette {result['silhouette_umap']:.4f}, BA {result['probe']['balanced_accuracy']:.4f}", flush=True)
    plot_gallery(variants, embeddings, metrics, cases, output, order)
    best = max(metrics, key=lambda k: metrics[k]["silhouette_umap"])
    print(f"Exploratory best silhouette: {best}; checking seeds and 3D", flush=True)
    selected = list(dict.fromkeys(["mean", best]))
    robustness = []
    fig, axes = plt.subplots(len(selected), 3, figsize=(14, 4 * len(selected)), squeeze=False, constrained_layout=True)
    for row, name in enumerate(selected):
        X, _ = transform_values(variants[name], np.arange(len(cases)))
        for column, seed in enumerate((42, 7, 123)):
            path = output / f"{name}_2d_seed{seed}.csv"
            if seed == 42:
                embedding = embeddings[name]
            elif path.exists():
                embedding = pd.read_csv(path)[["umap_1", "umap_2"]].to_numpy()
            else:
                print(f"Stability: {name}, seed {seed}", flush=True)
                embedding = projection(X, 2, seed, variants[name]["metric"])
                pd.DataFrame({"image_id": cases.image_id, "umap_1": embedding[:, 0], "umap_2": embedding[:, 1]}).to_csv(path, index=False)
            silhouette = float(silhouette_score(embedding[independent], y[independent]))
            robustness.append({"representation": name, "seed": seed, "silhouette_2d": silhouette})
            category_scatter(axes[row, column], embedding, y, order, size=3)
            axes[row, column].set_title(f"{variants[name]['title']} — graine {seed}\nsilhouette : {silhouette:.3f}")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "seed_comparison.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(robustness).to_csv(output / "seed_metrics.csv", index=False)
    fig = plt.figure(figsize=(8 * len(selected), 7), constrained_layout=True)
    three_d = {}
    for column, name in enumerate(selected):
        X, _ = transform_values(variants[name], np.arange(len(cases)))
        path = output / f"{name}_3d.csv"
        if path.exists():
            embedding = pd.read_csv(path)[["umap_1", "umap_2", "umap_3"]].to_numpy()
        else:
            print(f"3D: {name}", flush=True)
            embedding = projection(X, 3, 42, variants[name]["metric"])
            table = cases[["image_id", "category", "domain", "delta_iou"]].copy()
            for k in range(3):
                table[f"umap_{k+1}"] = embedding[:, k]
            table.to_csv(path, index=False)
        three_d[name] = float(silhouette_score(embedding[independent], y[independent]))
        axis = fig.add_subplot(1, len(selected), column + 1, projection="3d")
        axis.scatter(*embedding[order].T, c=np.asarray(COLORS)[y[order]], s=3, alpha=.6, depthshade=False)
        axis.set(title=f"{variants[name]['title']}\nsilhouette 3D : {three_d[name]:.3f}", xlabel="UMAP 1", ylabel="UMAP 2", zlabel="UMAP 3")
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(output / "comparison_3d.png", dpi=160)
    plt.close(fig)
    # One self-contained export, with a selector between baseline and selected variant.
    import plotly.graph_objects as go
    interactive = go.Figure()
    for j, name in enumerate(selected):
        table = pd.read_csv(output / f"{name}_3d.csv")
        for i, category in enumerate(CATEGORIES):
            chosen = table.category.eq(category)
            interactive.add_trace(go.Scatter3d(x=table.loc[chosen, "umap_1"], y=table.loc[chosen, "umap_2"],
                z=table.loc[chosen, "umap_3"], mode="markers", name=TITLES[i],
                text=table.loc[chosen, "image_id"], marker={"color": COLORS[i], "size": 2, "opacity": .6}, visible=j == 0))
    buttons = [{"label": variants[name]["title"], "method": "update", "args": [
        {"visible": [j == i // 3 for i in range(3 * len(selected))]}, {"title": variants[name]["title"]}]} for j, name in enumerate(selected)]
    interactive.update_layout(title=variants[selected[0]]["title"], updatemenus=[{"buttons": buttons}],
                              scene={"xaxis_title": "UMAP 1", "yaxis_title": "UMAP 2", "zaxis_title": "UMAP 3"})
    interactive.write_html(output / "comparison_3d.html", include_plotlyjs=True)
    rows = []
    for name, result in metrics.items():
        rows.append({k: v for k, v in result.items() if k != "probe"} | {
            "balanced_accuracy": result["probe"]["balanced_accuracy"],
            "auroc_improved": result["probe"]["auroc_improved"],
            "gate_delta_iou": result["probe"]["hard_gate_mean_delta_iou"]})
    pd.DataFrame(rows).to_csv(output / "comparison_metrics.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    write_json(output / "summary.json", {"best_visual": best, "metrics": metrics,
               "stability": robustness, "silhouettes_3d": three_d, "status": "complete"})
    print(f"Complete: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=HERE / "tables/categories.csv")
    parser.add_argument("--features", type=Path, default=HERE / "cache/extraction-b1/features.npz")
    parser.add_argument("--folds", type=Path, default=HERE / "results/folds.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "feature_comparison")
    with threadpool_limits(limits=1):
        main(parser.parse_args())
