#!/usr/bin/env python3
"""Build a logistic-only report from the complete saved held-out predictions."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_features import CATEGORIES, encode_labels, file_hash, scores


HERE = Path(__file__).resolve().parent
DIRECTORY = HERE / "logistic_regression"
REPRESENTATIONS = {"mean": "Moyenne H (256)", "multiscale_mean": "Trois résolutions (352)",
                   "pre_global_mean": "Avant attention (576)"}
STRATEGIES = {"balanced_ba": "Équilibrée / BA", "uniform_ba": "Uniforme / BA",
              "uniform_accuracy": "Uniforme / accuracy"}


def percent(value):
    return f"{100 * value:.1f}".replace(".", ",") + " %"


def points(value):
    return f"{100 * value:+.2f}".replace(".", ",")


def interval(values):
    return "[" + " ; ".join(f"{100 * value:.2f}".replace(".", ",") for value in values) + "]"


def validate():
    summary = json.loads((DIRECTORY / "summary.json").read_text())
    contract = summary["contract"]
    for key, path in (("script_sha256", HERE / "compare_logistic_regression.py"),
                      ("shared_script_sha256", HERE / "analyze_features.py"),
                      ("cases_sha256", HERE / "tables/categories.csv"),
                      ("folds_sha256", HERE / "results/folds.csv"),
                      ("previous_predictions_sha256", HERE / "simple_models/predictions.csv")):
        if file_hash(path) != contract[key]:
            raise ValueError(f"Changed input: {path}")
    cases = pd.read_csv(HERE / "tables/categories.csv")
    predictions = pd.read_csv(DIRECTORY / "predictions.csv")
    if len(summary["results"]) != 21 or len(list((DIRECTORY / "fits").glob("*/complete.json"))) != 19:
        raise ValueError("Incomplete experiment")
    proba = {}
    for result in summary["results"]:
        part = predictions.loc[(predictions.representation == result["representation"]) &
                               (predictions.strategy == result["strategy"])]
        if result["evaluation"] == "grouped_cv":
            subset = cases
            part = part.loc[part.split.str.startswith("cv")]
        else:
            subset = cases.loc[cases.domain == result["held_out_domain"]]
            part = part.loc[part.split == "domain_" + result["held_out_domain"]]
        if part.image_id.duplicated().any() or set(part.image_id) != set(subset.image_id):
            raise ValueError("Invalid prediction coverage")
        values = part.set_index("image_id").loc[subset.image_id, list(CATEGORIES)].to_numpy()
        actual = scores(encode_labels(subset), values, subset.delta_iou.to_numpy())
        for stat in ("accuracy", "balanced_accuracy", "macro_f1", "hard_gate_mean_delta_iou"):
            np.testing.assert_allclose(actual[stat], result[stat], rtol=0, atol=1e-12)
        if result["evaluation"] == "grouped_cv":
            proba[f"{result['representation']}/{result['strategy']}"] = values
    return summary, cases, proba


def build():
    summary, cases, predictions = validate()
    cv = {(row["representation"], row["strategy"]): row for row in summary["results"] if row["evaluation"] == "grouped_cv"}
    y = encode_labels(cases)
    neutral = np.zeros((len(cases), 3))
    neutral[:, 1] = 1
    cohorts = {"Toutes les observations": np.ones(len(cases), bool),
               "Khanhha, trois conditions": cases.domain.eq("khanhha").to_numpy(),
               "Khanhha propre": cases.dataset.eq("khanhha_original").to_numpy()}
    cohort_rows = []
    cohort_predictions = {"always_neutral": neutral, **predictions}
    for cohort, mask in cohorts.items():
        for model, proba in cohort_predictions.items():
            cohort_rows.append(dict(cohort=cohort, model=model, **scores(y[mask], proba[mask], cases.delta_iou.to_numpy()[mask])))
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, list)} for row in cohort_rows]).to_csv(DIRECTORY / "cohort_metrics.csv", index=False)
    by_cohort = {(row["cohort"], row["model"]): row for row in cohort_rows}

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.2), constrained_layout=True)
    names = ("Toujours neutre", "Logistique équilibrée\nréglée sur BA", "Logistique uniforme\nréglée sur accuracy")
    keys = ("always_neutral", "mean/balanced_ba", "mean/uniform_accuracy")
    labels = ("Détériore", "Neutre", "Améliore")
    for ax, key, name in zip(axes, keys, names):
        row = by_cohort[("Toutes les observations", key)]
        matrix = np.asarray(row["confusion_matrix"])
        normalized = matrix / matrix.sum(1, keepdims=True)
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{100 * normalized[i, j]:.0f} %\n({matrix[i,j]})", ha="center", va="center",
                        color="white" if normalized[i,j] > .6 else "#172538")
        ax.set(xticks=range(3), xticklabels=labels, yticks=range(3), yticklabels=labels,
               xlabel="Catégorie prédite", title=f"{name}\nAccuracy {percent(row['accuracy'])} · BA {percent(row['balanced_accuracy'])}")
    axes[0].set_ylabel("Catégorie observée")
    fig.suptitle("Moyenne des features H — scènes exclues de l’apprentissage du classifieur", fontsize=12)
    fig.savefig(DIRECTORY / "confusion_matrices.png", dpi=180, bbox_inches="tight", pad_inches=.2)
    plt.close(fig)

    selections = []
    for path in sorted((DIRECTORY / "fits").glob("*/complete.json")):
        representation, split = path.parent.name.split("__")
        data = json.loads(path.read_text())
        for strategy, model in data["models"].items():
            selections.append(dict(representation=representation, strategy=strategy, split=split,
                                   **model["chosen"], train_accuracy=model["train_metrics"]["accuracy"],
                                   train_balanced_accuracy=model["train_metrics"]["balanced_accuracy"],
                                   final_iterations=model["final_iterations"][0], final_retry=model["final_retry"]))
    pd.DataFrame(selections).to_csv(DIRECTORY / "selected_settings.csv", index=False)
    balanced = cv[("mean", "balanced_ba")]
    uniform = cv[("mean", "uniform_accuracy")]
    comparisons = summary["bootstrap"]["comparisons"]
    previous = next(row for row in comparisons if row["reference"] == "previous_mean_tuned")
    ba_difference = previous["statistics"]["balanced_accuracy"]
    clean_neutral = by_cohort[("Khanhha propre", "always_neutral")]
    clean_uniform = by_cohort[("Khanhha propre", "mean/uniform_accuracy")]
    lines = [
        "# Classer les trois catégories par régression logistique",
        "",
        "10 septembre 2026 — **8 895 observations, 2 122 scènes**. Features du SAM 2 + LoRA de référence, calcul CPU local.",
        "",
        f"**La logistique obtient {percent(uniform['accuracy'])} de bonnes classifications sur la moyenne de H**, contre {percent(summary['always_neutral']['accuracy'])} en prédisant toujours « neutre ». Avec équilibrage des classes, le rappel moyen atteint **{percent(balanced['balanced_accuracy'])}**. Les erreurs restent nombreuses : ces features ne permettent pas une séparation nette des trois catégories avec les logistiques testées.",
        "",
        f"La variante uniforme ne retrouve que **{percent(uniform['recall_improved'])} des améliorations**, contre {percent(balanced['recall_improved'])} avec équilibrage. Le score global doit se lire avec ces rappels.",
        "",
        "## Modèle utilisé",
        "",
        "**Régression logistique multinomiale** : une combinaison linéaire des canaux pour chaque catégorie, puis softmax. Entraînement avec `sklearn.linear_model.LogisticRegression`, solveur L-BFGS, pénalisation L2. [Documentation de l’implémentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).",
        "",
        r"$p(y=c\mid h)=\frac{\exp(w_c^T h+b_c)}{\sum_{r=1}^{3}\exp(w_r^T h+b_r)}$.",
        "",
        "Les trois classes sont **détérioration**, **neutralité** et **amélioration**, avec une zone neutre de ±1 point d’IoU. Les features originales sont utilisées directement, avant toute UMAP.",
        "",
        "## Deux objectifs à distinguer",
        "",
        "L’**accuracy** compte toutes les bonnes réponses ; la **BA** moyenne les rappels des trois catégories. Trois réglages sont évalués : poids équilibrés et choix par BA ; poids uniformes et choix par BA ; poids uniformes et choix par accuracy. La variante intermédiaire distingue l’effet des poids de celui du critère de sélection.",
        "",
        "| Features | Poids / critère de réglage | Accuracy | BA | Rappel détériore | Rappel neutre | Rappel améliore |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| Témoin | Toujours neutre | {percent(summary['always_neutral']['accuracy'])} | 33,3 % | 0 % | 100 % | 0 % |",
    ]
    for representation, label in REPRESENTATIONS.items():
        for strategy, name in STRATEGIES.items():
            row = cv[(representation, strategy)]
            values = [row[k] for k in ("accuracy", "balanced_accuracy", "recall_degraded", "recall_neutral", "recall_improved")]
            lines.append(f"| {label} | {name} | " + " | ".join(map(percent, values)) + " |")
    lines += ["", "![Matrices de confusion, moyenne de H](logistic_regression/confusion_matrices.png)", "",
              "Les cellules indiquent le pourcentage de chaque catégorie réelle et le nombre d’images. Pondérer les classes change le compromis entre les trois rappels ; cela ne garantit pas des probabilités calibrées.",
              "", "## Le témoin neutre dépend du jeu", "",
              "| Ensemble | N | Accuracy toujours neutre | Accuracy logistique uniforme | BA logistique uniforme |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for cohort in cohorts:
        control = by_cohort[(cohort, "always_neutral")]
        row = by_cohort[(cohort, "mean/uniform_accuracy")]
        lines.append(f"| {cohort} | {row['n']} | {percent(control['accuracy'])} | {percent(row['accuracy'])} | {percent(row['balanced_accuracy'])} |")
    lines += ["", f"Sur Khanhha propre, la logistique uniforme obtient {percent(clean_uniform['accuracy'])}, contre {percent(clean_neutral['accuracy'])} pour le témoin. Les résultats globaux mélangent plusieurs domaines et niveaux de bruit. [Tous les modèles par sous-ensemble](logistic_regression/cohort_metrics.csv).", "",
              "## Ce qu’apporte ce nouveau réglage", "",
              f"Sur H moyen, différence de BA entre la nouvelle logistique équilibrée et la logistique équilibrée précédemment réglée : **{points(ba_difference['estimate'])} point**, IC 95 % **{interval(ba_difference['ci95'])}**. Cet intervalle compare les mêmes scènes et inclut zéro : **aucun gain établi avec ce nouveau réglage**.", "",
              "Le nombre de canaux est lui aussi choisi en validation interne : 32, 128 ou tous. Canaux retenus dans les cinq folds externes, pour la logistique équilibrée :", "",
              "| Features | Nombre de canaux retenus, folds 0 à 4 |",
              "| --- | --- |"]
    for representation, label in REPRESENTATIONS.items():
        rows = [row for row in selections if row["representation"] == representation and row["strategy"] == "balanced_ba" and row["split"].startswith("cv")]
        rows.sort(key=lambda row: row["split"])
        lines.append(f"| {label} | " + ", ".join(str(row["k"]) for row in rows) + " |")
    lines += ["", "La validation retient **tous les canaux** pour les trois représentations avec équilibrage des classes. Les [réglages](logistic_regression/selected_settings.csv), numéros des canaux, poids appris et normalisations sont enregistrés dans [les fits](logistic_regression/fits/). Ils ne donnent pas, à eux seuls, une interprétation physique des canaux.", "",
              "## Domaine entièrement absent de l’apprentissage de la logistique", "",
              "Test complémentaire sur H moyen ; les trois conditions Khanhha sont exclues ensemble.", "",
              "| Domaine exclu | BA équilibrée | Accuracy toujours neutre | Accuracy uniforme | BA uniforme | ΔIoU sélection uniforme (points) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    lodo = {(row["held_out_domain"], row["strategy"]): row for row in summary["results"] if row["evaluation"] == "leave_domain_out"}
    for domain in sorted(cases.domain.unique()):
        b, u = lodo[(domain, "balanced_ba")], lodo[(domain, "uniform_accuracy")]
        neutral_accuracy = float(cases.loc[cases.domain == domain, "category"].eq("neutral").mean())
        lines.append(f"| {domain} | {percent(b['balanced_accuracy'])} | {percent(neutral_accuracy)} | {percent(u['accuracy'])} | {percent(u['balanced_accuracy'])} | {points(u['hard_gate_mean_delta_iou'])} |")
    lines += ["", "La sélection utilise le checkpoint Frangi lorsque la classe prédite est « amélioration ». Les scores de domaines exclus évaluent le transfert du classifieur, au-delà de la reconnaissance de nouvelles scènes dans des domaines déjà présents.", "",
              "## Protocole et limites", "",
              "Cinq folds externes identiques aux analyses précédentes, séparés par scène : recadrages et versions bruitées restent ensemble. À l’intérieur de chaque entraînement, trois folds groupés règlent C ∈ {0,001 ; 0,01 ; 0,1 ; 1 ; 10} et le nombre de canaux. Classement ANOVA des canaux, normalisation et poids des classes sont recalculés sur chaque entraînement interne, puis sur l’entraînement externe complet. Les scènes externes ne servent jamais au réglage.", "",
              "Les IC rééchantillonnent 1 000 fois les scènes par domaine, avec les mêmes tirages pour tous les modèles. Ils sont conditionnels aux prédictions sauvegardées. Les comparaisons principales portent sur H moyen. Ces analyses successives explorent les mêmes données ; une évaluation indépendante reste nécessaire.", "",
              "Les classes comparent **deux checkpoints historiques distincts**. Les chevauchements historiques entre scènes d’entraînement de SAM et scènes évaluées subsistent ; séparer les scènes pour la logistique ne les efface pas. L’essai ancien utilise un prompt Frangi-similarité : il ne mesure pas encore l’intérêt d’une hiérarchie Frangi-graphe. H final est disponible après l’encodeur ; les 576 features préattention sont disponibles avant l’insertion envisagée.", "",
              "## Reproduire", "", "Depuis ce sous-dossier, avec `requirements-analysis.txt` et le cache de features :", "", "```bash",
              "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python compare_logistic_regression.py",
              "python build_logistic_report.py", "python -m pytest test_logistic_regression.py -q", "```", "",
              "[Mesures et intervalles](logistic_regression/summary.json) · [Prédictions](logistic_regression/predictions.csv) · [Contrat et versions](logistic_regression/contract.json) · [Étude précédente](MODELES_SIMPLES.md).", ""]
    (HERE / "REGRESSION_LOGISTIQUE.md").write_text("\n".join(lines))
    print("Validated 21 held-out result sets; wrote REGRESSION_LOGISTIQUE.md")


if __name__ == "__main__":
    build()
