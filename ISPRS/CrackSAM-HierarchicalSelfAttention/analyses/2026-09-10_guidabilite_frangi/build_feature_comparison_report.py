#!/usr/bin/env python3
"""Render the completed feature comparison without refitting any projection."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "feature_comparison"
COLORS = {"degraded": "#bc4b51", "neutral": "#a0a7ae", "improved": "#168a78"}
TITLES = {"degraded": "Détérioration", "neutral": "Neutre", "improved": "Amélioration"}
DOMAIN_COLORS = {"concrete3k": "#397db4", "facade390": "#d99b26", "khanhha": "#7966a8", "road420": "#283b43"}


def number(value, digits=3):
    return f"{value:.{digits}f}".replace(".", ",")


def main():
    summary = json.loads((OUTPUT / "summary.json").read_text())
    contract = json.loads((OUTPUT / "contract.json").read_text())
    assert summary["status"] == "complete"
    for name, key in (("compare_feature_umaps.py", "script_sha256"), ("feature_variants.py", "variants_sha256")):
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == contract[key]
    assert len(summary["metrics"]) == len(contract["variants"]) == 14
    best = summary["best_visual"]
    specs, results = contract["variants"], summary["metrics"]
    selected = list(dict.fromkeys(["mean", best]))
    fig, axes = plt.subplots(2, len(selected), figsize=(7 * len(selected), 10), squeeze=False, constrained_layout=True)
    for column, name in enumerate(selected):
        table = pd.read_csv(OUTPUT / f"{name}_2d.csv")
        assert len(table) == 8895 and not table.image_id.duplicated().any()
        order = np.random.default_rng(20260910).permutation(len(table))
        xy = table[["umap_1", "umap_2"]].to_numpy()[order]
        for row, (variable, palette) in enumerate((("category", COLORS), ("domain", DOMAIN_COLORS))):
            axis = axes[row, column]
            axis.scatter(*xy.T, c=table[variable].map(palette).to_numpy()[order], s=5, alpha=.6, linewidths=0)
            axis.set(xticks=[], yticks=[], title=specs[name]["title"] + (" — catégories" if row == 0 else " — domaines"))
            handles = [Line2D([], [], marker="o", linestyle="", color=color, label=TITLES.get(label, label)) for label, color in palette.items()]
            axis.legend(handles=handles, fontsize=8, loc="lower left", framealpha=.9)
    fig.savefig(OUTPUT / "reference_vs_selected.png", dpi=170)
    plt.close(fig)
    baseline, chosen = results["mean"], results[best]
    best_predictive = max(results, key=lambda k: results[k]["probe"]["balanced_accuracy"])
    best_title = specs[best]["title"][:1].lower() + specs[best]["title"][1:]
    predictive_title = specs[best_predictive]["title"][:1].lower() + specs[best_predictive]["title"][1:]
    report = [
        "# Quelles features séparent le mieux les trois catégories avec UMAP ?",
        "Comparaison du 10 septembre 2026 — **14 représentations**, mêmes 8 895 images et catégories que le "
        "[rapport initial](RAPPORT.md). Les features SAM sont réutilisées ; aucun nouveau calcul GCP ni entraînement de SAM.",
        "**Aucune des 14 variantes ne fait apparaître trois catégories nettement séparées.** "
        "Les silhouettes UMAP sont toutes négatives. Certains amas changent de forme ou de position, "
        "mais les cas améliorés et détériorés restent fortement mélangés.",
        f"La meilleure silhouette 2D est obtenue avec **{best_title}** : "
        f"**{number(chosen['silhouette_umap'])}**, contre **{number(baseline['silhouette_umap'])}** pour la moyenne de H. "
        "La sélection est exploratoire : les quatorze variantes sont publiées, y compris celles qui fonctionnent moins bien.",
        f"Le faible avantage visuel des écarts-types ne correspond pas à une meilleure prédiction : "
        f"**{number(100 * results['std']['probe']['balanced_accuracy'], 1)} %** de balanced accuracy, "
        f"contre **{number(100 * baseline['probe']['balanced_accuracy'], 1)} %** pour la moyenne. "
        "Avec ces features et ces réglages, je ne retiendrais donc pas les écarts-types comme une amélioration convaincante.",
        "![Référence et variante sélectionnée, catégories et domaines](feature_comparison/reference_vs_selected.png)",
        "## Ce qui a été testé",
        "Moyenne, variabilité spatiale, moyenne + variabilité, grille 2 × 2, variabilité entre et dans les quadrants, "
        "cartes de haute résolution séparées ou réunies, trois résolutions, features avant attention globale. "
        "Trois variantes de prétraitement complètent la comparaison : distance cosinus, poids égaux par résolution et PCA à 32 dimensions.",
        "La variabilité **entre quadrants** décrit les différences entre quatre grandes zones de l’image. "
        "La variabilité **dans les quadrants** conserve le reste de la variance spatiale. "
        "Cette décomposition utilise les moments déjà extraits ; elle ne désigne pas directement les ombres ou les fissures.",
        "**Protocole identique :** UMAP non supervisée, 30 voisins, `min_dist=0.1`, graine 42. "
        "Les canaux sont standardisés ; seule la variante cosinus utilise les vecteurs bruts normalisés en norme L2. "
        "L’équilibrage divise chaque bloc standardisé par la racine de son nombre de canaux. "
        "Les catégories ne participent ni au calcul des features, ni à PCA, ni à UMAP. "
        "L’ordre de dessin des points est mélangé indépendamment des catégories et identique entre figures.",
        "![Les quatorze représentations, colorées par catégorie](feature_comparison/comparison_categories.png)",
        "<details>\n<summary>Mêmes projections, colorées par domaine</summary>\n\n"
        "![Comparaison par domaine](feature_comparison/comparison_domains.png)\n\n</details>",
        "Les hautes résolutions séparent surtout les collections d’images. À l’inverse, la variabilité entre "
        "quadrants atténue fortement ces regroupements, sans révéler les trois catégories. "
        "Atténuer l’effet de provenance ne suffit donc pas à faire apparaître une séparation.",
        "## Comparaison chiffrée",
        "La silhouette compare la compacité des catégories à leur éloignement : une valeur proche de zéro indique "
        "un fort recouvrement. Elle est calculée avant et après UMAP sur **une image par scène physique**, "
        "soit 2 122 représentants déterministes, version propre préférée. Les figures montrent les 8 895 observations.",
        "La **balanced accuracy** mesure la prédiction des catégories dans les features, avec les cinq folds "
        "historiques regroupés par scène. Normalisation et PCA sont ajustées uniquement sur l’entraînement de chaque fold. "
        "Le ΔIoU choisit le modèle Frangi lorsque la sonde prédit « amélioration ». "
        "Les sondes utilisent une régression logistique L2, `C=1`, sans réglage par variante ; le changement d’échelle "
        "modifie donc aussi la régularisation effective. L’écart-type et la normalisation L2 ne sont pas linéaires dans H.",
        "Toutes les sondes, référence comprise, sont recalculées dans l’environnement CPU documenté ; "
        "de faibles écarts numériques avec le rapport initial sont possibles.",
        "| Représentation | Dim. | Silhouette features | Silhouette UMAP 2D | BA | ΔIoU sélection (points) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, result in results.items():
        title = specs[name]["title"]
        if name == best:
            title = f"**{title}**"
        report.append(f"| {title} | {result['dimensions']} | {number(result['silhouette_features'])} | "
                      f"{number(result['silhouette_umap'])} | {number(100 * result['probe']['balanced_accuracy'], 1)} % | "
                      f"{number(100 * result['probe']['hard_gate_mean_delta_iou'], 2)} |")
    report.extend([
        f"La meilleure BA descriptive est celle de **{predictive_title}** "
        f"({number(100 * results[best_predictive]['probe']['balanced_accuracy'], 1)} %). "
        "L’apparence de la carte et la prédiction sur de nouvelles scènes sont donc évaluées séparément.",
        "[Toutes les mesures](feature_comparison/comparison_metrics.csv) incluent aussi la silhouette des domaines, "
        "la fidélité des voisinages et l’accord de catégorie parmi les 15 voisins, avec ou sans restriction au même domaine. "
        "Pour ce dernier contrôle, l’accord macro attendu par tirage aléatoire au sein du domaine est d’environ **39,8 %**, "
        "à cause des proportions différentes des catégories entre domaines.",
        "## Vérification avec d’autres graines et en 3D",
        "La référence et la variante retenue sont recalculées avec les graines 7 et 123, sans changer les paramètres.",
        "![Vérification selon trois graines](feature_comparison/seed_comparison.png)",
        "| Représentation | Graine 42 | Graine 7 | Graine 123 | 3D, graine 42 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for name in selected:
        values = {v["seed"]: v["silhouette_2d"] for v in summary["stability"] if v["representation"] == name}
        report.append(f"| {specs[name]['title']} | " + " | ".join(number(values[s]) for s in (42, 7, 123))
                      + f" | {number(summary['silhouettes_3d'][name])} |")
    report.extend([
        "![UMAP 3D : référence et variante retenue](feature_comparison/comparison_3d.png)",
        "Le petit avantage de l’écart-type se retrouve sur les trois graines en 2D, mais disparaît en 3D. "
        "Dans tous les cas, les silhouettes restent négatives et les catégories mélangées.",
        "[Vue 3D interactive autonome](feature_comparison/comparison_3d.html) : télécharger le fichier HTML et "
        "utiliser son menu pour changer de représentation.",
        "## Limites et reproduction",
        "Cette recherche compare des résumés des features disponibles, pas tous les tokens ni toutes les couches de SAM. "
        "Découper un gain continu d’IoU en trois catégories ne garantit pas trois groupes géométriques dans les features. "
        "La sélection de la meilleure carte ne constitue pas un nouveau test indépendant. Les catégories restent une "
        "comparaison entre deux checkpoints historiques ; les chevauchements antérieurs entre scènes d’entraînement "
        "et de test restent ceux documentés dans le rapport initial. Aucun gain du futur biais hiérarchique n’est démontré ici.",
        "Depuis ce sous-dossier, avec les dépendances de `requirements-analysis.txt` :",
        "```bash\npython compare_feature_umaps.py\npython build_feature_comparison_report.py\n"
        "python -m pytest test_feature_variants.py test_compare_feature_umaps.py -q\n```",
        "[Contrat, versions et SHA-256](feature_comparison/contract.json) · "
        "[Définition des représentations](feature_variants.py) · [Résultats complets](feature_comparison/summary.json). "
        "Le calcul reprend les variantes terminées seulement si le contrat est inchangé.",
    ])
    document = "\n\n".join(report).replace("|\n\n|", "|\n|") + "\n"
    (HERE / "COMPARAISON_FEATURES.md").write_text(document, encoding="utf-8")
    print("Comparison report written.")


if __name__ == "__main__":
    main()
