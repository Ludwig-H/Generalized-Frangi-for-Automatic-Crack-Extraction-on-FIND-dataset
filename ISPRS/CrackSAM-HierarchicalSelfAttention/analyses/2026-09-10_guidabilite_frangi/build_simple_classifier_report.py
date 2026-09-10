#!/usr/bin/env python3
"""Render completed simple-classifier comparisons without fitting any model."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
MODELS = ("fixed_logistic", "tuned_logistic", "mlp8", "mlp32", "mlp64")
REPRESENTATIONS = ("mean", "multiscale_mean", "pre_global_mean")
MODEL_TITLES = {
    "fixed_logistic": "Logistique C = 1",
    "tuned_logistic": "Logistique C réglé",
    "mlp8": "MLP, 8 neurones",
    "mlp32": "MLP, 32 neurones",
    "mlp64": "MLP, 64 neurones",
}
FEATURE_TITLES = {
    "mean": "Moyenne H (256)",
    "multiscale_mean": "Trois résolutions (352)",
    "pre_global_mean": "Avant attention globale (576)",
}
CLASS_TITLES = ("Détériore", "Neutre", "Améliore")


def number(value, digits=1, signed=False):
    """Format a finite scalar for the French report."""
    if value is None or not np.isfinite(value):
        return "—"
    return (f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}").replace(".", ",")


def percentage(value):
    return number(100 * value) + " %"


def interval(bounds, scale=100, digits=1):
    return "[" + " ; ".join(number(scale * value, digits) for value in bounds) + "]"


def load_results(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("The experiment must be complete before rendering the report")
    for name, key in (("compare_simple_classifiers.py", "script_sha256"),
                      ("analyze_features.py", "shared_scores_script_sha256")):
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != summary["contract"][key]:
            raise ValueError("The recorded training/scoring code differs from the source")
    replication = json.loads((directory / "replication_seed123.json").read_text())
    if replication["status"] != "complete" or not replication["same_inner_and_outer_scenes_as_primary"]:
        raise ValueError("The initialization replication is incomplete")
    if replication["contract_sha256"] != hashlib.sha256((directory / "contract.json").read_bytes()).hexdigest():
        raise ValueError("Replication and main experiment have different contracts")
    summary["replication"] = replication
    if summary["class_order"] != ["degraded", "neutral", "improved"]:
        raise ValueError("Unexpected class order")
    rows = summary["metrics"]
    cv = {(row["representation"], row["model"]): row for row in rows if row["evaluation"] == "grouped_cv"}
    if len(cv) != 15 or set(cv) != {(r, m) for r in REPRESENTATIONS for m in MODELS}:
        raise ValueError("Expected all fifteen grouped-CV comparisons")
    if len([row for row in rows if row["evaluation"] == "grouped_cv"]) != len(cv):
        raise ValueError("Duplicate grouped-CV result")
    for key, row in cv.items():
        confusion = np.asarray(row["confusion_matrix"])
        if row["n"] != 8895 or confusion.shape != (3, 3) or confusion.sum() != row["n"]:
            raise ValueError(f"Incomplete confusion matrix or unexpected cohort: {key}")
        bounds = summary["cv_intervals"]["/".join(key)]
        for field in ("balanced_accuracy", "hard_gate_mean_delta_iou"):
            if len(bounds[field]) != 2 or not np.isfinite(bounds[field]).all():
                raise ValueError(f"Missing bootstrap interval: {key}, {field}")
    paired = summary["primary_paired_bootstrap"]
    if (paired["representation"], paired["candidate"], paired["reference"]) != ("mean", "mlp32", "tuned_logistic"):
        raise ValueError("The declared primary comparison changed")
    lodo = {(row["held_out_domain"], row["model"]): row for row in rows
            if row["evaluation"] == "leave_domain_out" and row["representation"] == "mean"}
    expected_lodo = {(domain, model) for domain in ("concrete3k", "facade390", "khanhha", "road420") for model in MODELS}
    if set(lodo) != expected_lodo:
        raise ValueError("Expected all twenty domain-exclusion comparisons on mean H")
    summary["neutral_control"] = evaluate_neutral_control(directory)
    return summary, cv, lodo


def evaluate_neutral_control(directory):
    """Explicit constant-label baseline; no fitting or model selection."""
    from analyze_features import CATEGORIES, scores
    table = pd.read_csv(directory / "predictions.csv")
    y = np.array([CATEGORIES.index(category) for category in table.category])
    neutral = np.zeros((len(table), 3))
    neutral[:, CATEGORIES.index("neutral")] = 1
    predictions = {"always_neutral": neutral}
    for model in ("fixed_logistic", "tuned_logistic", "mlp32"):
        predictions[model] = table[[f"mean__{model}__p_{category}" for category in CATEGORIES]].to_numpy()
    cohorts = [("all", "all", np.ones(len(table), dtype=bool))]
    for column in ("domain", "dataset"):
        cohorts.extend((column, value, table[column].eq(value).to_numpy()) for value in sorted(table[column].unique()))
    rows = []
    for scope, cohort, selected in cohorts:
        for model, probability in predictions.items():
            result = scores(y[selected], probability[selected], table.delta_iou.to_numpy()[selected])
            rows.append({"scope": scope, "cohort": cohort, "model": model,
                         **{key: value for key, value in result.items() if not isinstance(value, (list, dict))}})
    pd.DataFrame(rows).to_csv(directory / "neutral_control.csv", index=False)
    return rows


def plot_performance(summary, cv, directory):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True, constrained_layout=True)
    colors = ("#c8cccf", "#6c7985", "#397db4", "#65ada2", "#168a78", "#284d48")
    for axis, representation in zip(axes, REPRESENTATIONS):
        values = np.array([1/3] + [cv[(representation, model)]["balanced_accuracy"] for model in MODELS]) * 100
        bounds = np.array([[1/3, 1/3]] + [summary["cv_intervals"][f"{representation}/{model}"]["balanced_accuracy"] for model in MODELS]) * 100
        positions = np.arange(len(MODELS) + 1)
        axis.bar(positions, values, color=colors, width=.65)
        # Percentile intervals need not contain the point estimate: draw their
        # endpoints directly rather than inventing nonnegative error lengths.
        axis.vlines(positions, bounds[:, 0], bounds[:, 1], color="#202b33", linewidth=1.3)
        axis.hlines(bounds[:, 0], positions - .08, positions + .08, color="#202b33", linewidth=1.3)
        axis.hlines(bounds[:, 1], positions - .08, positions + .08, color="#202b33", linewidth=1.3)
        for position, value, upper in zip(positions, values, bounds[:, 1]):
            axis.text(position, max(value, upper) + 1, f"{value:.1f}", ha="center", fontsize=9)
        axis.axhline(100 / 3, color="#7f8589", linestyle="--", linewidth=1)
        axis.set(xticks=positions, xticklabels=("Toujours\nneutre", "LR\nC=1", "LR\nC réglé", "MLP\n8", "MLP\n32", "MLP\n64"),
                 title=FEATURE_TITLES[representation], ylim=(0, 100))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Balanced accuracy hors fold (%)")
    fig.suptitle("Prédire trois catégories sur des scènes laissées de côté\nIC 95 % par scènes ; trait pointillé : 33,3 %", fontsize=12)
    fig.savefig(directory / "grouped_cv_performance.png", dpi=170)
    plt.close(fig)


def plot_confusions(cv, directory):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for axis, model in zip(axes, ("fixed_logistic", "tuned_logistic", "mlp32")):
        matrix = np.asarray(cv[("mean", model)]["confusion_matrix"])
        rates = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        axis.imshow(rates, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                axis.text(j, i, f"{matrix[i, j]}\n{rates[i, j]:.1%}", ha="center", va="center", fontsize=9,
                          color="white" if rates[i, j] > .55 else "black")
        axis.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_TITLES, yticklabels=CLASS_TITLES,
                 title=MODEL_TITLES[model], xlabel="Catégorie prédite")
    axes[0].set_ylabel("Catégorie observée")
    fig.suptitle("Moyenne de H : toutes les prédictions sont hors fold", fontsize=12)
    fig.savefig(directory / "mean_confusions.png", dpi=170)
    plt.close(fig)


def report_document(summary, cv, lodo):
    reference = cv[("mean", "tuned_logistic")]
    candidate = cv[("mean", "mlp32")]
    paired = summary["primary_paired_bootstrap"]
    difference = paired["difference_balanced_accuracy"]
    gate_difference = paired["difference_gate_delta_iou"]
    if difference["ci95"][0] > 0:
        verdict = "**Une couche cachée améliore la prédiction par rapport à la logistique réglée.**"
    elif difference["ci95"][1] < 0:
        verdict = "**La petite couche cachée fait moins bien que la logistique réglée.**"
    else:
        verdict = "**Sur la comparaison principale, la couche cachée n’apporte pas d’avantage établi sur la logistique réglée.**"
    selected = np.asarray(candidate["confusion_matrix"])[:, 2]
    precision = selected[2] / selected.sum() if selected.sum() else 0
    controls = {(row["scope"], row["cohort"], row["model"]): row for row in summary["neutral_control"]}
    lines = [
        "# Un petit réseau peut-il prédire le bénéfice de Frangi ?",
        "Tests du 10 septembre 2026 — mêmes **8 895 observations**, **2 122 scènes** et features SAM que le "
        "[rapport initial](RAPPORT.md). Calcul CPU local, sans réentraînement de SAM ni GCP.",
        "**Le signal est réel, mais la performance reste faible pour décider de faire confiance à Frangi.** "
        "Les meilleurs résultats restent autour de 55 % de balanced accuracy ; beaucoup de choix du modèle Frangi sont défavorables.",
        verdict + f" Sur la moyenne de H, la balanced accuracy atteint **{percentage(candidate['balanced_accuracy'])}** "
        f"avec 32 neurones, contre **{percentage(reference['balanced_accuracy'])}** avec la logistique réglée. "
        "Les matrices ci-dessous montrent les erreurs qui subsistent entre les trois catégories.",
        "**Une seule couche affine suivie d’un softmax est déjà une régression logistique multinomiale.** "
        "Nous testons donc aussi une **couche cachée ReLU**, suivie de trois sorties softmax, pour ajouter une non-linéarité.",
        "$p(y\\mid h)=\\mathrm{softmax}(Wh+b)$ pour la logistique ; "
        "$p(y\\mid h)=\\mathrm{softmax}(W_2\\,\\mathrm{ReLU}(W_1h+b_1)+b_2)$ pour le petit réseau.",
        "## Le témoin indispensable : toujours prédire « neutre »",
        "L’**accuracy classique** est la proportion totale de bonnes réponses. La **balanced accuracy (BA)** "
        "donne le même poids aux trois catégories : leur rappel moyen. Prédire toujours « neutre » donne "
        "100 % de rappel aux neutres, 0 % aux deux autres catégories, soit **33,3 % de BA**.",
        "| Ensemble | Accuracy toujours neutre | Accuracy MLP 32 | BA toujours neutre | BA MLP 32 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scope, cohort, title in (("all", "all", "Toutes les observations"),
                                 ("domain", "khanhha", "Khanhha, trois conditions"),
                                 ("dataset", "khanhha_original", "Khanhha propre")):
        dummy = controls[(scope, cohort, "always_neutral")]
        network = controls[(scope, cohort, "mlp32")]
        lines.append(f"| {title} | {percentage(dummy['accuracy'])} | {percentage(network['accuracy'])} | "
                     f"{percentage(dummy['balanced_accuracy'])} | {percentage(network['balanced_accuracy'])} |")
    lines.extend([
        "**Sur Khanhha propre, toujours prédire neutre obtient même davantage de bonnes réponses globales "
        "que le réseau.** La BA indique que le réseau reconnaît aussi les améliorations et détériorations ; "
        "elle ne rend pas ses décisions suffisamment fiables. [Contrôles détaillés par jeu](simple_models/neutral_control.csv).",
        "## Résultats sur de nouvelles scènes",
        "Les modèles voient les features originales, sans UMAP. La balanced accuracy moyenne les rappels des trois classes "
        "(hasard : 33,3 %). La colonne entraînement permet de repérer la mémorisation. "
        "Le ΔIoU choisit le checkpoint Frangi uniquement lorsque la classe prédite est « amélioration ».",
        "![Comparaison des quinze modèles](simple_models/grouped_cv_performance.png)",
        "| Features | Modèle | BA entraînement | BA hors fold | Macro-F1 | ΔIoU sélection (points) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ])
    for representation in REPRESENTATIONS:
        for model in MODELS:
            row = cv[(representation, model)]
            lines.append(f"| {FEATURE_TITLES[representation]} | {MODEL_TITLES[model]} | "
                         f"{percentage(row['train_balanced_accuracy'])} | {percentage(row['balanced_accuracy'])} | "
                         f"{number(row['macro_f1'], 3)} | {number(100 * row['hard_gate_mean_delta_iou'], 2, True)} |")
    lines.extend([
        f"Sur les features **avant attention globale**, le réseau à 32 neurones atteint "
        f"{percentage(cv[('pre_global_mean', 'mlp32')]['balanced_accuracy'])}, contre "
        f"{percentage(cv[('pre_global_mean', 'tuned_logistic')]['balanced_accuracy'])} pour la logistique réglée. "
        "C’est un résultat secondaire intéressant pour le futur guidage, à confirmer. Augmenter la largeur "
        "améliore surtout la performance d’entraînement ; le gain sur de nouvelles scènes reste limité.",
        "## Comparaison principale : 32 neurones contre logistique réglée",
        f"Différence de balanced accuracy : **{number(100 * difference['estimate'], 2, True)} points**, "
        f"IC 95 % **{interval(difference['ci95'], digits=2)}**. "
        f"Différence de gain IoU entre les deux politiques de sélection : **{number(100 * gate_difference['estimate'], 2, True)} point**, "
        f"IC 95 % **{interval(gate_difference['ci95'], digits=2)}**. "
        "Ce sont des différences appariées : chaque rééchantillonnage utilise les mêmes scènes pour les deux modèles.",
        f"**Autre initialisation, mêmes scènes :** avec la graine 123, le réseau à 32 neurones atteint "
        f"{percentage(summary['replication']['metrics']['balanced_accuracy'])} et "
        f"{number(100 * summary['replication']['metrics']['hard_gate_mean_delta_iou'], 2, True)} point d’IoU de sélection. "
        "Les deux exécutions sont conservées ; aucune graine n’est choisie pour améliorer le score publié.",
        f"Avec le réseau à 32 neurones, Frangi est choisi pour **{percentage(candidate['frangi_activation_rate'])}** des images. "
        f"Parmi ces choix, **{percentage(precision)}** améliorent l’IoU de plus d’un point ; "
        f"**{format(int(selected[0]), ',').replace(',', ' ')}** le détériorent de plus d’un point.",
        "![Matrices de confusion sur la moyenne de H](simple_models/mean_confusions.png)",
        "## Transfert vers un domaine absent de l’entraînement",
        "Chaque domaine est entièrement exclu, y compris les trois versions Khanhha ensemble. "
        "Les réglages restent choisis dans les domaines d’entraînement. Résultats sur la moyenne de H :",
        "| Domaine exclu | BA logistique réglée | BA MLP 32 | ΔIoU logistique (points) | ΔIoU MLP 32 (points) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for domain in ("khanhha", "road420", "facade390", "concrete3k"):
        left, right = lodo[(domain, "tuned_logistic")], lodo[(domain, "mlp32")]
        lines.append(f"| {domain} | {percentage(left['balanced_accuracy'])} | {percentage(right['balanced_accuracy'])} | "
                     f"{number(100 * left['hard_gate_mean_delta_iou'], 2, True)} | "
                     f"{number(100 * right['hard_gate_mean_delta_iou'], 2, True)} |")
    lines.extend([
        "**Le transfert reste insuffisant :** sur un domaine inconnu, le réseau à 32 neurones perd de l’IoU "
        "sur Khanhha, Road420 et Concrete3k ; le gain sur Façade390 est presque nul.",
        "## Protocole et portée",
        "**Cinq folds externes identiques à l’analyse précédente**, groupés par scène ; recadrages et versions bruitées "
        "restent ensemble. Chaque entraînement réserve environ 20 % de ses scènes à une validation interne. "
        "Normalisation et poids équilibrant les classes sont calculés uniquement sur l’entraînement concerné.",
        "La logistique utilise C = 1, ou choisit C parmi 0,01 ; 0,1 ; 1 ; 10 sur la validation interne. "
        "Les réseaux comportent 8, 32 ou 64 neurones ; Adam, taux d’apprentissage 0,001, pénalisation des poids 0,001, au plus 300 époques. "
        "L’époque retenue vient de la validation interne, puis le modèle est réentraîné sur tout l’entraînement externe "
        "pendant ce nombre d’époques. La couche de 32 neurones sur 256 features contient **8 323 paramètres**.",
        "La comparaison principale est fixée à l’avance : moyenne H, MLP 32 contre logistique réglée. "
        "Les autres largeurs et représentations sont secondaires. Les IC rééchantillonnent 1 000 fois les scènes par domaine, "
        "conditionnellement aux prédictions hors fold ; ils ne mesurent pas la variabilité d’un nouvel entraînement de SAM.",
        "Les classes décrivent toujours **la différence entre deux checkpoints historiques** : amélioration au-delà "
        "de +1 point d’IoU, détérioration sous −1 point, neutralité sinon. Les chevauchements historiques entre scènes "
        "d’entraînement de SAM et de test subsistent. Une meilleure classification préparerait une porte de confiance ; "
        "elle ne validerait pas encore un biais d’attention hiérarchique Frangi-graphe.",
        "Les classes sont équilibrées pendant l’apprentissage : les sorties softmax ne sont pas des probabilités "
        "de fiabilité calibrées.",
        "Depuis ce sous-dossier, avec PyTorch et les dépendances de `requirements-analysis.txt` :",
        "```bash\npython compare_simple_classifiers.py\npython check_simple_model_seed.py\npython build_simple_classifier_report.py\n"
        "python -m pytest test_simple_models.py -q\n```",
        "[Mesures et intervalles complets](simple_models/summary.json) · "
        "[Prédictions hors fold](simple_models/predictions.csv) · "
        "[Réglages et époques par entraînement](simple_models/training_runs.csv) · "
        "[Réplication avec une autre initialisation](simple_models/replication_seed123.json) · "
        "[Contrat et versions](simple_models/contract.json).",
    ])
    return "\n\n".join(lines).replace("|\n\n|", "|\n|") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=HERE / "simple_models")
    parser.add_argument("--output", type=Path, default=HERE / "MODELES_SIMPLES.md")
    args = parser.parse_args()
    summary, cv, lodo = load_results(args.results_dir)
    plot_performance(summary, cv, args.results_dir)
    plot_confusions(cv, args.results_dir)
    args.output.write_text(report_document(summary, cv, lodo), encoding="utf-8")
    print(f"Simple-classifier report written: {args.output}")


if __name__ == "__main__":
    main()
