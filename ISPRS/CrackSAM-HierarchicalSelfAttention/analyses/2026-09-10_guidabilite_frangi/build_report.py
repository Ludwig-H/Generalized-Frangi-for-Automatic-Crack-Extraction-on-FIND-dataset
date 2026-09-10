#!/usr/bin/env python3
"""Build the French report from complete, mutually consistent real artifacts.

This script performs no feature extraction, fitting, metric estimation or
selection of successful experiments. It fails before writing if required
results, projections or baseline verification records are missing.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
ORDER = ("improved", "neutral", "degraded")
TITLES = {"improved": "Amélioration", "neutral": "Neutre", "degraded": "Détérioration"}
DOMAINS = {"khanhha": "Khanhha, trois conditions", "road420": "Road420", "facade390": "Façade390", "concrete3k": "Concrete3k"}
DATASETS = {"khanhha_original": "Khanhha propre", "khanhha_noisy1": "Khanhha bruit 1", "khanhha_noisy2": "Khanhha bruit 2", **{key: value for key, value in DOMAINS.items() if key != "khanhha"}}
PROBES = (
    ("mean", "Moyenne de H"),
    ("pre_global_mean", "Moyenne avant la dernière attention globale"),
    ("grid2", "Moyennes par grille 2 × 2"),
    ("multiscale_mean", "Moyennes aux trois résolutions"),
    ("mean_std", "Moyenne + écart-type"),
    ("mean_top1_train_selection", "1 canal, sélection dans chaque entraînement"),
    ("mean_top5_train_selection", "5 canaux, sélection dans chaque entraînement"),
    ("mean_top10_train_selection", "10 canaux, sélection dans chaque entraînement"),
    ("mean_top25_train_selection", "25 canaux, sélection dans chaque entraînement"),
    ("domain_only", "Témoin : domaine seul"),
    ("domain_source_only", "Témoin : domaine + famille source"),
    ("majority", "Témoin : classe majoritaire"),
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, description):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite result: {description}")
    return value


def same(actual, expected, description, tolerance=1e-10):
    if not math.isclose(finite(actual, description), finite(expected, description), abs_tol=tolerance, rel_tol=tolerance):
        raise ValueError(f"Inconsistent {description}: {actual} versus {expected}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, digits=3, signed=False):
    return format(finite(value, "formatted number"), f"{'+' if signed else ''}.{digits}f").replace(".", ",")


def integer(value):
    return f"{int(value):,}".replace(",", " ")


def percent(value, digits=1, signed=False):
    return number(100 * float(value), digits, signed)


def interval(values, scale=1, digits=3):
    require(isinstance(values, list) and len(values) == 2, "Missing confidence interval")
    return f"[{number(scale * values[0], digits)} ; {number(scale * values[1], digits)}]"


def markdown_table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows),
    ])


def load_verified_artifacts(analysis, results, metadata_path):
    cases_path = analysis / "tables/categories.csv"
    cases = read_csv(cases_path)
    indexed = {row["image_id"]: row for row in cases}
    require(len(cases) == len(indexed) == 8895, "Expected all 8895 unique historical observations")
    counts = Counter(row["category"] for row in cases)
    require(set(counts) == set(ORDER), "All three categories must be present")
    statistics = read_json(results / "statistics.json")
    metadata = read_json(metadata_path)
    require(statistics["n_images"] == len(cases), "Statistics do not cover every case")
    require(statistics["n_physical_groups"] == len({row["physical_group"] for row in cases}), "Statistics group count differs")
    require(statistics["category_counts"] == dict(counts), "Statistics use different categories")
    require(statistics["cases_sha256"] == sha256(cases_path), "Statistics categories SHA256 mismatch")
    require(metadata["status"] == "complete" and metadata["n_images"] == len(cases), "Incomplete feature extraction")
    require(metadata["features_sha256"] == statistics["features_sha256"], "Statistics and extraction use different features")
    require(metadata["contract"]["cases_sha256"] == sha256(cases_path), "Extraction categories SHA256 mismatch")
    require(metadata["contract"]["adapter_sha256"] == "d154d60a82ec2a0af4540559785a483818612319350c04a1b06035053b6f6a04", "Unexpected historical LoRA checkpoint")
    require(metadata["contract"]["foundation_sha256"] == "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b", "Unexpected SAM 2 foundation checkpoint")
    require(bool(metadata["verified"]), "Baseline predictions were not verified")
    verification = metadata["verification"]
    require(len(verification) == len(cases) and {row["image_id"] for row in verification} == set(indexed), "Full per-image baseline verification is required")
    tolerance = finite(metadata["contract"]["verification"]["absolute_iou_tolerance"], "IoU verification tolerance")
    require(tolerance <= 0.005, "Baseline verification tolerance exceeds 0.005")
    for row in verification:
        same(row["historical_iou"], indexed[row["image_id"]]["baseline_iou"], "Historical baseline IoU")
        error = abs(finite(row["observed_iou"], "Verified IoU") - float(row["historical_iou"]))
        same(row["absolute_error"], error, "Baseline verification error")
        require(error <= tolerance, f"Baseline verification failed: {row['image_id']}")
    require(metadata["contract"]["pre_global"] is not None, "Pre-global feature extraction is required")

    probes_table = {row["representation"]: row for row in read_csv(results / "probe_metrics.csv")}
    for name, _ in PROBES:
        require(name in statistics["probes"] and name in probes_table, f"Missing probe: {name}")
        result = statistics["probes"][name]
        require(result["n"] == len(cases), f"Incomplete probe: {name}")
        for key in ("balanced_accuracy", "macro_f1", "auroc_improved", "hard_gate_mean_delta_iou"):
            same(result[key], probes_table[name][key], f"Probe CSV/JSON {name}/{key}")
    for key in ("balanced_accuracy", "auroc_improved", "hard_gate_mean_delta_iou"):
        interval(statistics["primary_group_bootstrap_ci95"][key])
    permutation = statistics["permutation"]
    require(permutation["permutations"] > 0 and permutation["p_value"] is not None, "Permutation test is incomplete")
    require(permutation["stratification"] == "domain x source_family", "Permutation test must control domain and source family")
    projections = statistics.get("projections", {})
    require(projections and not projections.get("skipped", False), "UMAP projections have not been computed")
    require(projections.get("interactive_3d") is True, "The interactive 3D projection is missing")
    for dimensions in (2, 3):
        coordinates = read_csv(results / f"umap_{dimensions}d.csv")
        require(len(coordinates) == len(cases) and {row["image_id"] for row in coordinates} == set(indexed), f"Incomplete UMAP {dimensions}D coordinates")
        for row in coordinates:
            require(row["category"] == indexed[row["image_id"]]["category"], "UMAP colors use different categories")
            for axis in range(1, dimensions + 1):
                finite(row[f"umap_{axis}"], f"UMAP {dimensions}D coordinate")
        finite(projections[f"trustworthiness_{dimensions}d"], "UMAP trustworthiness")
    for name in ("umap_2d.png", "umap_3d.png", "pca_2d.png", "linear_probe_confusion.png", "permutation_test.png"):
        require((results / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), f"Missing or invalid plot: {name}")
    require("<html>" in (results / "umap_3d.html").read_text(encoding="utf-8")[:200].lower(), "Invalid interactive HTML")
    if projections.get("extra_seeds_2d"):
        require((results / "umap_seed_sensitivity.png").is_file(), "UMAP seed sensitivity plot is missing")

    summary = read_csv(analysis / "tables/summary.csv")
    all_summary = next(row for row in summary if row["scope"] == "all")
    require(int(all_summary["n_images"]) == len(cases), "Preparation summary is incomplete")
    for category in ORDER:
        require(int(all_summary[category]) == counts[category], "Preparation category count mismatch")
    sensitivity = read_csv(analysis / "tables/sensitivity.csv")
    require({float(row["tolerance_iou"]) for row in sensitivity} == {0.005, 0.01, 0.02}, "Sensitivity table is incomplete")
    representatives = read_csv(analysis / "tables/representatives.csv")
    require(Counter(row["category"] for row in representatives) == Counter({key: 2 for key in ORDER}), "Expected two representatives per category")
    for row in representatives:
        require(row["image_id"] in indexed and row["category"] == indexed[row["image_id"]]["category"], "Representative classification mismatch")
        require(sha256(analysis / row["panel"]) == row["panel_sha256"], "Representative panel SHA256 mismatch")
    cohorts = read_csv(results / "cohort_metrics.csv")
    require({row["cohort"] for row in cohorts if row["grouping"] == "domain"} == set(DOMAINS), "Domain metrics are incomplete")
    ranking = read_csv(results / "channel_ranking.csv")
    require(len(ranking) == 256 and len({row["channel"] for row in ranking}) == 256, "Expected rankings for all 256 mean channels")
    return cases, summary, sensitivity, representatives, statistics, cohorts, ranking, metadata


def render(analysis, results, metadata_path, destination, data):
    cases, summaries, sensitivity, representatives, stats, cohorts, ranking, metadata = data
    all_summary = next(row for row in summaries if row["scope"] == "all")
    mean = stats["probes"]["mean"]
    improvement_index = stats["category_order"].index("improved")
    selected_counts = {
        category: mean["confusion_matrix"][index][improvement_index]
        for index, category in enumerate(stats["category_order"])
    }
    selected_n = sum(selected_counts.values())
    reference = stats["probes"]["domain_source_only"]
    ci = stats["primary_group_bootstrap_ci95"]
    projections = stats["projections"]
    permutation = stats["permutation"]
    relative = lambda path: Path(os.path.relpath(path, destination.parent)).as_posix()
    link = lambda title, path: f"[{title}]({relative(path)})"
    figure = lambda title, path: f"![{title}]({relative(path)})"
    n = len(cases)
    max_error = max(row["absolute_error"] for row in metadata["verification"])
    domain_cohorts = {row["cohort"]: row for row in cohorts if row["grouping"] == "domain"}
    unseen = next(row for row in cohorts if row["cohort"] == "khanhha_original_unseen_historical_scene")
    leave_out = {row["held_out_domain"]: row for row in stats["leave_one_domain_out"]}

    sections = [
        "# Les features de SAM 2 prédisent-elles le bénéfice de Frangi ?",
        "Analyse du 10 septembre 2026 — SAM 2 + LoRA historique, sans nouvel entraînement de SAM.",
        "[Complément : comparaison de 14 représentations des features avec UMAP](COMPARAISON_FEATURES.md).",
        "[Tests complémentaires : logistique et réseaux à une couche cachée](MODELES_SIMPLES.md).",
        "**Oui, partiellement : H contient un signal prédictif, mais les trois catégories ne sont pas "
        "facilement séparables.** Les projections mélangent amélioration et détérioration ; leurs amas "
        "reflètent surtout les collections d’images. Le signal suffit à un petit gain de sélection dans "
        "les domaines représentés à l’entraînement de la sonde. Son transfert à un domaine entièrement "
        "nouveau échoue généralement.",
        f"**Résultat de la sonde principale : balanced accuracy {percent(mean['balanced_accuracy'])} % "
        f"(IC 95 % {interval(ci['balanced_accuracy'], 100, 1)}), contre {percent(reference['balanced_accuracy'])} % "
        f"avec le seul domaine et la famille source.** Sélectionner le modèle Frangi lorsque cette sonde prédit "
        f"« amélioration » donne {percent(mean['hard_gate_mean_delta_iou'], 2, True)} point d’IoU en moyenne "
        f"hors fold (IC 95 % {interval(ci['hard_gate_mean_delta_iou'], 100, 2)}). Ces nombres évaluent une "
        "sélection entre deux modèles historiques ; ils ne valident pas encore le guidage hiérarchique.",
        "## 1. Comparaison et catégories",
        "Nous reprenons **baseline best, époque 20**, contre **Frangi-similarité best, époque 25**, "
        "tous deux choisis par le Dice de validation. Frangi entrait comme **prompt de masque dense**, "
        "après conversion de la similarité en pseudo-logits. Il ne s’agissait pas d’un biais d’attention.",
        f"Les {integer(n)} observations proviennent de {integer(stats['n_physical_groups'])} scènes regroupées "
        "par le parseur historique. Les recadrages et les trois versions Khanhha restent dans le même fold. "
        "Les moyennes ci-dessous pondèrent également les observations, pas les jeux de données.",
        r"$\Delta_i=\mathrm{IoU}_{\mathrm{Frangi},i}-\mathrm{IoU}_{\mathrm{baseline},i}$.",
        "**Amélioration** si ΔIoU > 0,01 ; **détérioration** si ΔIoU < −0,01 ; **neutre** sinon. "
        "Cette marge d’un point d’IoU est une tolérance pratique, pas un seuil de significativité.",
        markdown_table(["Jeu", "Observations", "Améliore", "Neutre", "Détériore"], [
            [DATASETS[row["name"]], integer(row["n_images"]), integer(row["improved"]), integer(row["neutral"]), integer(row["degraded"])]
            for row in summaries if row["scope"] == "dataset"
        ] + [["**Total**", integer(n), *(integer(all_summary[key]) for key in ORDER)]]),
        figure("Répartition des catégories", analysis / "figures/cases/category_counts.png"),
        "Sensibilité à la marge de neutralité :",
        markdown_table(["Marge (points d’IoU)", "Améliore", "Neutre", "Détériore"], [
            [f"±{percent(row['tolerance_iou'], 1)}", *(integer(row[key]) for key in ORDER)]
            for row in sensitivity if row["scope"] == "all"
        ]),
        "<details>\n<summary>Distribution complète des écarts</summary>\n\n"
        + figure("Distribution des deltas IoU", analysis / "figures/cases/delta_iou_histogram.png") + "\n\n</details>",
        "## 2. Six illustrations vérifiées",
        "Les panneaux sont repris à l’identique de l’expérience archivée. Les gains et pertes sont ses "
        "exemples extrêmes ; les neutres sont ses cas médians. Ils illustrent les catégories sans constituer "
        "un échantillon aléatoire. Leur choix ne dépend ni de H ni d’UMAP.",
    ]
    for category in ORDER:
        members = [row for row in representatives if row["category"] == category]
        for position, row in enumerate(members):
            caption = f"**{TITLES[category]} — {DATASETS[row['dataset']]}** : `{row['case_name']}`, " \
                f"IoU {number(row['baseline_iou'], 4)} → {number(row['frangi_iou'], 4)}, " \
                f"Δ = {percent(row['delta_iou'], 2, True)} points."
            content = caption + "\n\n" + figure(TITLES[category], analysis / row["panel"])
            if position:
                content = f"<details>\n<summary>Second exemple : {TITLES[category].lower()}</summary>\n\n{content}\n\n</details>"
            sections.append(content)
    sections.extend([
        "## 3. UMAP en deux et trois dimensions",
        "H est extrait de la **baseline SAM 2 + LoRA, avant tout prompt Frangi**. La représentation "
        "principale moyenne spatialement ses 256 canaux ; aucune annotation ni statistique de performance "
        "n’entre dans ces features. UMAP utilise les features standardisées et ne reçoit pas les catégories.",
        figure("UMAP 2D : catégories, domaines et delta IoU", results / "umap_2d.png"),
        figure("UMAP 3D non supervisée", results / "umap_3d.png"),
        link("Ouvrir la vue 3D interactive, autonome", results / "umap_3d.html") + ". "
        "Télécharger le fichier HTML pour le consulter depuis GitHub.",
        f"Fidélité locale (*trustworthiness*, {integer(projections['diagnostic_subset_n'])} observations) : "
        f"**{number(projections['trustworthiness_2d'])} en 2D**, **{number(projections['trustworthiness_3d'])} en 3D**. "
        f"Silhouette des catégories dans les features originales standardisées : "
        f"**{number(projections['silhouette_original_standardized_mean_features'])}**. "
        "Cette silhouette proche de zéro ne montre pas trois amas compacts distincts. "
        "La fidélité mesure la conservation des voisinages ; elle ne mesure pas la séparabilité des catégories. "
        "Une séparation visuelle après UMAP ne démontre pas une séparation linéaire dans H.",
        "<details>\n<summary>Comparaison avec une projection linéaire PCA</summary>\n\n"
        + figure("PCA des mêmes features", results / "pca_2d.png") + "\n\n</details>",
    ])
    if projections.get("extra_seeds_2d"):
        sections.append("<details>\n<summary>UMAP : stabilité selon la graine</summary>\n\n"
                        + figure("Sensibilité UMAP à la graine", results / "umap_seed_sensitivity.png") + "\n\n</details>")
    probe_rows = []
    for name, title in PROBES:
        result = stats["probes"][name]
        dimensions = result.get("dimensions")
        if name.startswith("mean_top"):
            dimensions = name.removeprefix("mean_top").split("_", 1)[0]
        if dimensions is None:
            dimensions = "Constant" if name == "majority" else "Catégoriel"
        probe_rows.append([title, dimensions, f"{percent(result['balanced_accuracy'])} %", number(result["macro_f1"]),
                           number(result["auroc_improved"]), percent(result["hard_gate_mean_delta_iou"], 2, True)])
    sections.extend([
        "## 4. Séparabilité dans les features originales",
        f"Régression logistique L2, classes équilibrées, C = 1, {stats['grouped_folds']} folds groupés. "
        "Normalisation et sélection des canaux utilisent exclusivement l’entraînement de chaque fold. "
        "La balanced accuracy est la moyenne des rappels des trois classes ; elle vaut 33,3 % au hasard. "
        "Le ΔIoU mesure la politique fixe : utiliser Frangi seulement lorsque la catégorie prédite est « amélioration ». "
        "L’ajout de l’écart-type est une variante secondaire, non linéaire dans H.",
        markdown_table(["Sonde", "Dimensions", "Balanced accuracy", "Macro-F1", "AUROC amélioration", "ΔIoU sélection (points)"], probe_rows),
        f"Pour la moyenne de H, l’AUROC « amélioration contre reste » est {number(mean['auroc_improved'])}, "
        f"IC 95 % {interval(ci['auroc_improved'])}. Frangi est choisi pour {percent(mean['frangi_activation_rate'])} % "
        f"des observations. Les IC rééchantillonnent {integer(stats['bootstrap']['repeats'])} fois les scènes "
        "au sein des domaines, conditionnellement aux prédictions hors fold déjà calculées.",
        f"**La décision image par image reste incertaine :** sur les {integer(selected_n)} sélections de Frangi, "
        f"{integer(selected_counts['improved'])} améliorent de plus d’un point, "
        f"{integer(selected_counts['neutral'])} sont neutres et "
        f"{integer(selected_counts['degraded'])} détériorent de plus d’un point. "
        f"Seules {percent(selected_counts['improved'] / selected_n)} % des sélections appartiennent donc "
        "à la catégorie amélioration, malgré le gain moyen positif.",
        figure("Matrice de confusion hors fold", results / "linear_probe_confusion.png"),
        f"**Permutation contrôlant domaine et famille source** : p = {number(permutation['p_value'], 4)} "
        f"sur {integer(permutation['permutations'])} permutations, avec une image déterministe par scène "
        f"({integer(permutation['n_independent_representatives'])} représentants, dont "
        f"{integer(permutation['n_permutable_representatives'])} dans des strates où la catégorie varie). "
        f"La balanced accuracy observée sur ce sous-échantillon vaut {percent(permutation['observed_balanced_accuracy'])} %. "
        + (f"La p-valeur atteint la résolution minimale de ces {integer(permutation['permutations'])} permutations. "
           if math.isclose(permutation['p_value'], 1 / (permutation['permutations'] + 1)) else "")
        +
        "Ce test contrôle l’association aux collections ; il ne mesure pas le gain d’une future porte hiérarchique.",
        "<details>\n<summary>Distribution sous permutation</summary>\n\n"
        + figure("Test par permutation", results / "permutation_test.png") + "\n\n</details>",
        "### Généralisation entre domaines",
        "« Domaine exclu » entraîne la sonde uniquement sur les trois autres domaines. "
        "Cette mesure distingue la prédiction de nouvelles scènes du transfert à une nouvelle collection.",
        markdown_table(["Domaine testé", "BA, CV groupée", "BA, domaine exclu", "ΔIoU, CV groupée (points)", "ΔIoU, domaine exclu (points)"], [
            [DOMAINS[domain], f"{percent(domain_cohorts[domain]['balanced_accuracy'])} %", f"{percent(leave_out[domain]['balanced_accuracy'])} %",
             percent(domain_cohorts[domain]["hard_gate_mean_delta_iou"], 2, True), percent(leave_out[domain]["hard_gate_mean_delta_iou"], 2, True)]
            for domain in DOMAINS
        ]),
        "**Le changement de domaine est le principal échec :** sans exemples de la collection cible "
        "pour entraîner la sonde, la sélection perd de l’IoU sur Khanhha, Road420 et Concrete3k ; "
        "elle est presque neutre sur Façade390. Le résultat global positif ne justifie donc pas "
        "une confiance universelle prédite par H.",
        f"Sur les **{integer(unseen['n'])} images Khanhha propres de scènes absentes du train et de la validation "
        f"historiques**, la sonde hors fold atteint {percent(unseen['balanced_accuracy'])} % de balanced accuracy "
        f"et {percent(unseen['hard_gate_mean_delta_iou'], 2, True)} point d’IoU de sélection.",
        "### Plafond d’une sélection parfaite",
        markdown_table(["Politique", "IoU moyenne", "ΔIoU contre baseline (points)"], [
            ["Baseline pour toutes les images", number(all_summary["baseline_iou"], 4), "0,00"],
            ["Frangi pour toutes les images", number(all_summary["frangi_iou"], 4), percent(all_summary["mean_delta_iou"], 2, True)],
            ["Sélection par moyenne de H, hors fold", number(mean["hard_gate_mean_iou"], 4), percent(mean["hard_gate_mean_delta_iou"], 2, True)],
            ["Oracle utilisant la vérité terrain", number(all_summary["oracle_iou"], 4), percent(all_summary["oracle_gain_iou"], 2, True)],
        ]),
        "L’oracle prend, pour chaque observation, la meilleure des deux sorties déjà disponibles. "
        "Il indique une réserve de gain pour cette sélection particulière ; il n’est pas utilisable en pratique.",
        "## 5. Certains canaux sont-ils plus informatifs ?",
        "Les cinq premiers canaux selon le classement ANOVA global sont donnés ci-dessous. "
        "ρ désigne la corrélation de Spearman avec ΔIoU ; le coefficient compare les classes amélioration et "
        "détérioration dans la régression standardisée. La fréquence indique la présence parmi les dix premiers "
        "canaux sélectionnés dans les entraînements des folds.",
        markdown_table(["Canal (index à partir de 0)", "ρ avec ΔIoU", "Coefficient descriptif", "Fréquence top 10"], [
            [row["channel"], number(row["spearman_delta_descriptive"]),
             number(row["coefficient_improved_minus_degraded_descriptive"]), f"{percent(row['top10_training_fold_frequency'], 0)} %"]
            for row in ranking[:5]
        ]),
        "Ces rangs globaux sont descriptifs. Les canaux sont corrélés et ne correspondent pas nécessairement "
        "à une propriété nommable, comme « ombre ». Les performances top-k du tableau précédent utilisent leur "
        "propre sélection dans chaque entraînement ; elles ne réutilisent pas ce classement global.",
        f"**Pas de canal isolé suffisant :** un canal atteint seulement "
        f"{percent(stats['probes']['mean_top1_train_selection']['balanced_accuracy'])} % de balanced accuracy ; "
        f"25 canaux atteignent {percent(stats['probes']['mean_top25_train_selection']['balanced_accuracy'])} %, "
        "avec un gain d’IoU presque nul. L’information utile paraît distribuée entre plusieurs canaux.",
        "## 6. Portée pour le guidage hiérarchique",
        "La comparaison renseigne la préférence entre **deux checkpoints distincts** : elle mélange l’effet "
        "du prompt et celui de l’apprentissage LoRA. Elle ne donne pas directement la confiance optimale dans "
        "la hiérarchie Frangi-graphe. **730 groupes Khanhha** du test historique apparaissent aussi dans le train ; "
        "la CV de la sonde ne supprime pas cette exposition antérieure de SAM. Les groupes reposent sur les noms "
        "d’origine, sans garantie contre tous les doublons visuels.",
        "Une porte calculée avec la moyenne finale de H est disponible après l’encodage. Pour modifier une "
        "attention durant cette même passe, la variante extraite **avant la dernière attention globale** est "
        f"plus directement utilisable : ses 576 canaux atteignent "
        f"{percent(stats['probes']['pre_global_mean']['balanced_accuracy'])} % de balanced accuracy et "
        f"{percent(stats['probes']['pre_global_mean']['hard_gate_mean_delta_iou'], 2, True)} point d’IoU de sélection. "
        "C’est un indice favorable à l’essai d’un petit module de confiance, pas une validation de ce module. "
        "La faible séparation des moyennes ne prouve pas l’absence d’information dans toutes les features spatiales.",
        "**Suite proposée :** calculer un coefficient de confiance à partir des features disponibles avant "
        "l’attention guidée, et apprendre ce coefficient avec LoRA via la perte de segmentation. "
        "Comparer au même biais hiérarchique avec un coefficient global, puis à SAM gelé + LoRA seul, "
        "sur des scènes séparées dès l’entraînement de SAM. Les étiquettes historiques de cette analyse "
        "servent au diagnostic ; elles ne sont pas une vérité terrain de fiabilité de la hiérarchie.",
        "## Traçabilité et reproduction",
        f"Les SHA-256 des poids historiques et des entrées sont contrôlés. Les logits baseline ont été décodés "
        f"et confrontés aux IoU archivées pour **{integer(n)} observations** ; erreur absolue maximale : "
        f"**{number(max_error, 6)} IoU**, sous la tolérance {number(metadata['contract']['verification']['absolute_iou_tolerance'], 3)}. "
        f"Extraction sur {metadata['contract']['device_name']}, précision {metadata['contract']['amp_dtype']}.",
        " · ".join([
            link("Commandes", analysis / "README.md"), link("Protocole et références", analysis / "methods.md"),
            link("Catégories image par image", analysis / "tables/categories.csv"), link("Statistiques complètes", results / "statistics.json"),
            link("Métadonnées d’extraction", metadata_path), link("Classement des 256 canaux", results / "channel_ranking.csv"),
            link("Calcul G4 et arrêt vérifié", results / "gcp_execution.json"),
        ]),
    ])
    return "\n\n".join(sections) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=HERE)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--extraction-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analysis = args.analysis_dir.resolve()
    results = (args.results_dir or analysis / "results").resolve()
    metadata = args.extraction_metadata
    if metadata is None:
        published = results / "extraction_metadata.json"
        metadata = published if published.is_file() else analysis / "cache/extraction-b1/metadata.json"
    metadata = metadata.resolve()
    destination = (args.output or analysis / "RAPPORT.md").resolve()
    data = load_verified_artifacts(analysis, results, metadata)
    report = render(analysis, results, metadata, destination, data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(f"Verified report written: {destination}")


if __name__ == "__main__":
    main()
