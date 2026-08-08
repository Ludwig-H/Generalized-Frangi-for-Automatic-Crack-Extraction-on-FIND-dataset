#!/usr/bin/env python3
"""Valide les sorties de l'étude anti-ombre sans relire les données brutes.

Le validateur contrôle le contrat de la campagne par défaut de ``run_study.py`` :
24 scènes Khánh Hà sous trois conditions, dont les 16 ancres historiques, six
scènes externes, seize cartes par cas et trois ombres synthétiques appliquées aux
16 ancres. Il ne suit jamais ``data_root_runtime`` et n'accède qu'au dossier
passé avec ``--output``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import struct
import sys
from typing import Iterable, Mapping, Sequence


CONDITIONS = ("original", "noisy1", "noisy2")
MAPS = (
    "frangi_similarity_historical",
    "frangi_relative",
    "black_tophat",
    "derivative_pair",
    "paired_profile",
    "line_step_bic",
    "phase_symmetry",
    "ofs",
    "ofs_reflectance",
    "fusion_ofs_profile",
    "fusion_precision",
    "fusion_reflectance",
    "verified_frangi_ofs",
    "verified_frangi_bic",
    "verified_frangi_v2",
    "verified_frangi_consensus",
)
SHADOW_KINDS = ("hard", "soft", "curved")

ANCHOR_CASES = frozenset(
    {
        "CRACK500_20160222_115828_641_1.jpg",
        "CRACK500_20160222_115843_1281_361.jpg",
        "CRACK500_20160222_115847_641_361.jpg",
        "CRACK500_20160308_073532_1_361.jpg",
        "CRACK500_20160316_143445_1281_361.jpg",
        "CRACK500_20160326_142354_641_1081.jpg",
        "CRACK500_20160328_154318_641_1.jpg",
        "CRACK500_20160329_093924_1921_721.jpg",
        "CRACK500_20160329_094010_1281_361.jpg",
        "DeepCrack_11231-3.jpg",
        "Sylvie_Chambon_319.jpg",
        "Volker_DSC01646_226_19_1273_1645.jpg",
        "cracktree200_6266.jpg",
        "noncrack_noncrack_concrete_wall_28_0.jpg.jpg",
        "noncrack_noncrack_concrete_wall_43_50.jpg.jpg",
        "noncrack_noncrack_concrete_wall_81_4.jpg.jpg",
    }
)

PRESENTATION_CASES = frozenset(
    {
        ("khanhha", "Sylvie_Chambon_319.jpg", "original"),
        (
            "khanhha",
            "CRACK500_20160329_093924_1921_721.jpg",
            "noisy1",
        ),
        (
            "khanhha",
            "Volker_DSC01646_226_19_1273_1645.jpg",
            "noisy2",
        ),
        ("khanhha", "CRACK500_20160308_073532_1_361.jpg", "noisy2"),
    }
)

EXTERNAL_CASES: Mapping[tuple[str, str, str], str] = {
    ("road420", "2023_11_01_20_33_IMG_6353.jpg", "original"): "gain_frangi_ombre",
    ("road420", "2023_10_30_16_44_IMG_6033.jpg", "original"): "perte_frangi",
    ("road420", "2023_11_05_21_38_IMG_6516.jpg", "original"): "deux_bons",
    ("facade390", "DJ_Wall_66.JPG", "original"): "gain_frangi",
    ("facade390", "DJ_Wall_231.JPG", "original"): "perte_frangi",
    ("facade390", "DJ_Wall_343.JPG", "original"): "deux_faibles",
}

EXPECTED_CASES = 78
EXPECTED_KHANHHA_CASES = 72
EXPECTED_EXTERNAL_CASES = 6
EXPECTED_METRIC_ROWS = EXPECTED_CASES * len(MAPS)
EXPECTED_SHADOW_ROWS = len(ANCHOR_CASES) * len(SHADOW_KINDS) * len(MAPS)

PER_CASE_REQUIRED_COLUMNS = (
    "dataset",
    "case_name",
    "condition",
    "role",
    "qualitative",
    "map",
    "target_fraction",
    "score_mean",
    "score_q99",
    "score_max",
    "active_fraction_010",
    "response_mass_near_gt",
    "ap_exact",
    "ap_tol2",
    "precision_top_p005",
    "skeleton_recall_top_p005",
    "precision_top_p01",
    "skeleton_recall_top_p01",
    "precision_top_p02",
    "skeleton_recall_top_p02",
    "precision_top_p05",
    "skeleton_recall_top_p05",
)

SUMMARY_METRICS = (
    "ap_exact",
    "ap_tol2",
    "precision_top_p01",
    "skeleton_recall_top_p01",
    "response_mass_near_gt",
    "active_fraction_010",
)

SHADOW_REQUIRED_COLUMNS = (
    "case_name",
    "shadow_kind",
    "map",
    "shadow_fraction",
    "boundary_fraction",
    "clean_crack_response",
    "shadow_crack_response",
    "crack_retention",
    "clean_boundary_response",
    "shadow_boundary_response",
    "boundary_response_increase",
)

GLOBAL_FIGURES = (
    "atlas_presentation_khanhha.png",
    "atlas_presentation_external.png",
    "atlas_ombres_synthetiques.png",
    "metric_overview.png",
)


class Findings:
    """Accumule les erreurs pour produire un diagnostic complet en une passe."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.notes: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="Dossier de sortie à valider (défaut : dossier courant).",
    )
    return parser.parse_args(argv)


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _describe_key(key: Iterable[object]) -> str:
    return "/".join(str(part) for part in key)


def _duplicates(values: Iterable[tuple[str, ...]]) -> list[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    repeated: set[tuple[str, ...]] = set()
    for value in values:
        if value in seen:
            repeated.add(value)
        seen.add(value)
    return sorted(repeated)


def _load_manifest(output: Path, findings: Findings) -> dict[str, object] | None:
    path = output / "study_manifest.json"
    if not path.is_file():
        findings.error(f"manifeste absent : {path}")
        return None
    if path.stat().st_size == 0:
        findings.error(f"manifeste vide : {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        findings.error(f"manifeste JSON illisible ({path}) : {exc}")
        return None
    if not isinstance(value, dict):
        findings.error(f"le manifeste doit être un objet JSON : {path}")
        return None

    def require_integer(name: str, expected: int | None = None, minimum: int | None = None) -> None:
        raw = value.get(name)
        if not _is_integer(raw):
            findings.error(f"manifest.{name} doit être un entier, valeur={raw!r}")
            return
        integer = int(raw)
        if expected is not None and integer != expected:
            findings.error(
                f"manifest.{name}={integer}, valeur attendue={expected} pour la campagne par défaut"
            )
        if minimum is not None and integer < minimum:
            findings.error(f"manifest.{name}={integer}, minimum attendu={minimum}")

    require_integer("schema_version", expected=1)
    require_integer("image_size", minimum=96)
    require_integer("random_cases", expected=8)
    require_integer("shadow_cases", expected=len(ANCHOR_CASES))
    require_integer("cases", expected=EXPECTED_CASES)
    require_integer("metric_rows", expected=EXPECTED_METRIC_ROWS)
    require_integer("shadow_rows", expected=EXPECTED_SHADOW_ROWS)
    require_integer("seed")
    require_integer("bootstrap", minimum=1)

    if value.get("conditions") != list(CONDITIONS):
        findings.error(
            f"manifest.conditions={value.get('conditions')!r}, attendu={list(CONDITIONS)!r}"
        )
    if value.get("maps") != list(MAPS):
        findings.error(
            f"manifest.maps incohérent : attendu exactement {list(MAPS)!r}"
        )
    if value.get("missing_external_cases") != []:
        findings.error(
            "manifest.missing_external_cases doit être vide pour les six cas externes, "
            f"valeur={value.get('missing_external_cases')!r}"
        )
    data_root = value.get("data_root_runtime")
    if not isinstance(data_root, str) or not data_root.strip():
        findings.error("manifest.data_root_runtime doit être une chaîne non vide")
    elapsed = value.get("elapsed_seconds")
    if (
        not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        findings.error(
            f"manifest.elapsed_seconds doit être fini et positif ou nul, valeur={elapsed!r}"
        )
    return value


def _load_csv(
    path: Path,
    label: str,
    required_columns: Iterable[str],
    findings: Findings,
) -> list[dict[str, str]] | None:
    if not path.is_file():
        findings.error(f"CSV {label} absent : {path}")
        return None
    if path.stat().st_size == 0:
        findings.error(f"CSV {label} vide : {path}")
        return None
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            if fields is None:
                findings.error(f"CSV {label} sans en-tête : {path}")
                return None
            if any(field is None or not field.strip() for field in fields):
                findings.error(f"CSV {label} contient une colonne sans nom : {fields!r}")
            duplicate_columns = sorted(
                {field for field in fields if fields.count(field) > 1 and field is not None}
            )
            if duplicate_columns:
                findings.error(
                    f"CSV {label} contient des colonnes dupliquées : {duplicate_columns}"
                )
            missing_columns = sorted(set(required_columns) - set(fields))
            if missing_columns:
                findings.error(
                    f"CSV {label} colonnes manquantes : {missing_columns}; présentes={fields}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        findings.error(f"CSV {label} illisible ({path}) : {exc}")
        return None

    for line_number, row in enumerate(rows, start=2):
        if None in row:
            findings.error(
                f"CSV {label}, ligne {line_number} : valeurs supplémentaires hors en-tête"
            )
        if not any((value or "").strip() for key, value in row.items() if key is not None):
            findings.error(f"CSV {label}, ligne {line_number} entièrement vide")
    return rows


def _parse_bool(raw: str | None, context: str, findings: Findings) -> bool | None:
    normalized = (raw or "").strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    findings.error(f"{context} doit être booléen, valeur={raw!r}")
    return None


def _parse_number(
    raw: str | None,
    context: str,
    findings: Findings,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    allow_blank: bool = False,
) -> float | None:
    text = (raw or "").strip()
    if not text:
        if not allow_blank:
            findings.error(f"{context} est vide")
        return None
    try:
        value = float(text)
    except ValueError:
        findings.error(f"{context} n'est pas numérique, valeur={raw!r}")
        return None
    if not math.isfinite(value):
        findings.error(f"{context} doit être fini, valeur={raw!r}")
        return None
    if minimum is not None and value < minimum - 1e-9:
        findings.error(f"{context}={value} est inférieur à {minimum}")
    if maximum is not None and value > maximum + 1e-9:
        findings.error(f"{context}={value} est supérieur à {maximum}")
    return value


def _parse_csv_integer(
    raw: str | None,
    context: str,
    findings: Findings,
    *,
    minimum: int = 0,
) -> int | None:
    text = (raw or "").strip()
    try:
        value = int(text)
    except ValueError:
        findings.error(f"{context} doit être un entier, valeur={raw!r}")
        return None
    if value < minimum:
        findings.error(f"{context}={value}, minimum attendu={minimum}")
    return value


def _validate_cases(
    output: Path,
    findings: Findings,
) -> tuple[list[dict[str, str]], dict[tuple[str, str, str], tuple[str, bool]]]:
    path = output / "tables/generated/cases_manifest.csv"
    rows = _load_csv(
        path,
        "cases_manifest",
        ("dataset", "case_name", "condition", "role", "qualitative"),
        findings,
    )
    if rows is None:
        return [], {}
    if len(rows) != EXPECTED_CASES:
        findings.error(
            f"cases_manifest contient {len(rows)} lignes, attendu={EXPECTED_CASES} "
            f"({EXPECTED_KHANHHA_CASES} Khánh Hà + {EXPECTED_EXTERNAL_CASES} externes)"
        )

    keys = [
        (
            (row.get("dataset") or "").strip(),
            (row.get("case_name") or "").strip(),
            (row.get("condition") or "").strip(),
        )
        for row in rows
    ]
    for duplicate in _duplicates(keys):
        findings.error(f"cases_manifest : clé dupliquée {_describe_key(duplicate)}")

    metadata: dict[tuple[str, str, str], tuple[str, bool]] = {}
    khanhha_rows = 0
    khanhha_names: set[str] = set()
    external_keys: set[tuple[str, str, str]] = set()
    for line_number, (row, key) in enumerate(zip(rows, keys), start=2):
        dataset, case_name, condition = key
        context = f"cases_manifest ligne {line_number} ({_describe_key(key)})"
        role = (row.get("role") or "").strip()
        qualitative = _parse_bool(row.get("qualitative"), f"{context}.qualitative", findings)
        if not dataset or not case_name or not condition:
            findings.error(f"{context} : dataset, case_name et condition doivent être non vides")
        if qualitative is not None:
            metadata[key] = (role, qualitative)

        if dataset == "khanhha":
            khanhha_rows += 1
            khanhha_names.add(case_name)
            if condition not in CONDITIONS:
                findings.error(f"{context} : condition Khánh Hà inconnue {condition!r}")
            expected_role = "anchor" if case_name in ANCHOR_CASES else "hash_sample"
            expected_qualitative = case_name in ANCHOR_CASES
            if role != expected_role:
                findings.error(
                    f"{context}.role={role!r}, attendu={expected_role!r}"
                )
            if qualitative is not None and qualitative != expected_qualitative:
                findings.error(
                    f"{context}.qualitative={qualitative}, attendu={expected_qualitative}"
                )
        else:
            external_keys.add(key)
            expected_role = EXTERNAL_CASES.get(key)
            if expected_role is None:
                findings.error(f"{context} : cas externe inattendu")
            elif role != expected_role:
                findings.error(
                    f"{context}.role={role!r}, attendu={expected_role!r}"
                )
            if qualitative is not None and not qualitative:
                findings.error(f"{context}.qualitative doit être True")

    if khanhha_rows != EXPECTED_KHANHHA_CASES:
        findings.error(
            f"cases_manifest : {khanhha_rows} lignes Khánh Hà, "
            f"attendu={EXPECTED_KHANHHA_CASES}"
        )
    if len(khanhha_names) != 24:
        findings.error(
            f"cases_manifest : {len(khanhha_names)} scènes Khánh Hà uniques, attendu=24"
        )
    anchor_names = khanhha_names & ANCHOR_CASES
    missing_anchors = sorted(ANCHOR_CASES - anchor_names)
    if missing_anchors:
        findings.error(f"cases_manifest : ancres Khánh Hà manquantes : {missing_anchors}")
    non_anchor_names = khanhha_names - ANCHOR_CASES
    if len(non_anchor_names) != 8:
        findings.error(
            f"cases_manifest : {len(non_anchor_names)} scènes hash_sample uniques, attendu=8"
        )
    for name in sorted(khanhha_names):
        actual = {condition for dataset, case, condition in keys if dataset == "khanhha" and case == name}
        if actual != set(CONDITIONS):
            findings.error(
                f"cases_manifest : conditions de {name}={sorted(actual)}, "
                f"attendu={list(CONDITIONS)}"
            )

    if external_keys != set(EXTERNAL_CASES):
        missing = sorted(set(EXTERNAL_CASES) - external_keys)
        extra = sorted(external_keys - set(EXTERNAL_CASES))
        if missing:
            findings.error(f"cases_manifest : cas externes manquants : {missing}")
        if extra:
            findings.error(f"cases_manifest : cas externes supplémentaires : {extra}")

    key_set = set(keys)
    missing_presentation = sorted(PRESENTATION_CASES - key_set)
    if missing_presentation:
        findings.error(
            "cases_manifest : quatre cas/conditions de présentation incomplets, "
            f"manquants={missing_presentation}"
        )
    return rows, metadata


def _validate_per_case_metrics(
    output: Path,
    case_metadata: Mapping[tuple[str, str, str], tuple[str, bool]],
    findings: Findings,
) -> tuple[list[dict[str, str]], dict[tuple[str, str, str], float]]:
    path = output / "tables/generated/per_case_metrics.csv"
    rows = _load_csv(path, "per_case_metrics", PER_CASE_REQUIRED_COLUMNS, findings)
    if rows is None:
        return [], {}
    if len(rows) != EXPECTED_METRIC_ROWS:
        findings.error(
            f"per_case_metrics contient {len(rows)} lignes, attendu={EXPECTED_METRIC_ROWS} "
            f"({EXPECTED_CASES} cas × {len(MAPS)} cartes)"
        )

    keys: list[tuple[str, str, str, str]] = []
    target_values: dict[tuple[str, str, str], list[float]] = {}
    for line_number, row in enumerate(rows, start=2):
        case_key = (
            (row.get("dataset") or "").strip(),
            (row.get("case_name") or "").strip(),
            (row.get("condition") or "").strip(),
        )
        map_name = (row.get("map") or "").strip()
        key = (*case_key, map_name)
        keys.append(key)
        context = f"per_case_metrics ligne {line_number} ({_describe_key(key)})"
        if case_key not in case_metadata:
            findings.error(f"{context} : cas absent de cases_manifest")
        else:
            expected_role, expected_qualitative = case_metadata[case_key]
            role = (row.get("role") or "").strip()
            qualitative = _parse_bool(
                row.get("qualitative"), f"{context}.qualitative", findings
            )
            if role != expected_role:
                findings.error(
                    f"{context}.role={role!r}, cases_manifest={expected_role!r}"
                )
            if qualitative is not None and qualitative != expected_qualitative:
                findings.error(
                    f"{context}.qualitative={qualitative}, "
                    f"cases_manifest={expected_qualitative}"
                )
        if map_name not in MAPS:
            findings.error(f"{context} : carte inconnue {map_name!r}")

        target = _parse_number(
            row.get("target_fraction"),
            f"{context}.target_fraction",
            findings,
            minimum=0.0,
            maximum=1.0,
        )
        if target is not None:
            target_values.setdefault(case_key, []).append(target)
        score_q99 = None
        score_max = None
        for column in (
            "score_mean",
            "score_q99",
            "score_max",
            "active_fraction_010",
            "response_mass_near_gt",
            "precision_top_p005",
            "precision_top_p01",
            "precision_top_p02",
            "precision_top_p05",
        ):
            parsed = _parse_number(
                row.get(column),
                f"{context}.{column}",
                findings,
                minimum=0.0,
                maximum=1.0,
            )
            if column == "score_q99":
                score_q99 = parsed
            elif column == "score_max":
                score_max = parsed
        if score_q99 is not None and score_max is not None and score_q99 > score_max + 1e-7:
            findings.error(
                f"{context} : score_q99={score_q99} dépasse score_max={score_max}"
            )

        target_empty = target is not None and target <= 1e-15
        for column in (
            "ap_exact",
            "ap_tol2",
            "skeleton_recall_top_p005",
            "skeleton_recall_top_p01",
            "skeleton_recall_top_p02",
            "skeleton_recall_top_p05",
        ):
            parsed = _parse_number(
                row.get(column),
                f"{context}.{column}",
                findings,
                minimum=0.0,
                maximum=1.0,
                allow_blank=target_empty,
            )
            if target_empty and parsed is not None:
                findings.error(
                    f"{context}.{column} doit être vide lorsque target_fraction=0, "
                    f"valeur={parsed}"
                )

    for duplicate in _duplicates(keys):
        findings.error(f"per_case_metrics : clé dupliquée {_describe_key(duplicate)}")

    expected_keys = {
        (*case_key, map_name)
        for case_key in case_metadata
        for map_name in MAPS
    }
    actual_keys = set(keys)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        findings.error(
            f"per_case_metrics : {len(missing)} combinaisons cas×carte manquantes, "
            f"premières={missing[:8]}"
        )
    if extra:
        findings.error(
            f"per_case_metrics : {len(extra)} combinaisons cas×carte inattendues, "
            f"premières={extra[:8]}"
        )

    target_by_case: dict[tuple[str, str, str], float] = {}
    for case_key, values in target_values.items():
        if values and max(values) - min(values) > 1e-12:
            findings.error(
                f"per_case_metrics : target_fraction varie selon la carte pour "
                f"{_describe_key(case_key)} : min={min(values)}, max={max(values)}"
            )
        if values:
            target_by_case[case_key] = values[0]
    return rows, target_by_case


def _validate_summary_metrics(
    output: Path,
    metric_rows: Sequence[Mapping[str, str]],
    findings: Findings,
) -> None:
    required = ["condition", "map", "n", "nonempty_targets"]
    for metric in SUMMARY_METRICS:
        required.extend(
            (
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
            )
        )
    path = output / "tables/generated/summary_metrics.csv"
    rows = _load_csv(path, "summary_metrics", required, findings)
    if rows is None:
        return
    expected_rows = len(CONDITIONS) * len(MAPS)
    if len(rows) != expected_rows:
        findings.error(
            f"summary_metrics contient {len(rows)} lignes, attendu={expected_rows}"
        )

    actual_metric_counts: dict[tuple[str, str], int] = {}
    actual_nonempty_counts: dict[tuple[str, str], int] = {}
    for row in metric_rows:
        key = ((row.get("condition") or "").strip(), (row.get("map") or "").strip())
        actual_metric_counts[key] = actual_metric_counts.get(key, 0) + 1
        raw_target = (row.get("target_fraction") or "").strip()
        try:
            nonempty = math.isfinite(float(raw_target)) and float(raw_target) > 0.0
        except ValueError:
            nonempty = False
        actual_nonempty_counts[key] = actual_nonempty_counts.get(key, 0) + int(nonempty)

    keys: list[tuple[str, str]] = []
    for line_number, row in enumerate(rows, start=2):
        key = (
            (row.get("condition") or "").strip(),
            (row.get("map") or "").strip(),
        )
        keys.append(key)
        context = f"summary_metrics ligne {line_number} ({_describe_key(key)})"
        condition, map_name = key
        if condition not in CONDITIONS:
            findings.error(f"{context} : condition inconnue")
        if map_name not in MAPS:
            findings.error(f"{context} : carte inconnue")
        row_count = _parse_csv_integer(row.get("n"), f"{context}.n", findings)
        nonempty = _parse_csv_integer(
            row.get("nonempty_targets"), f"{context}.nonempty_targets", findings
        )
        expected_count = actual_metric_counts.get(key)
        if row_count is not None and expected_count is not None and row_count != expected_count:
            findings.error(
                f"{context}.n={row_count}, per_case_metrics contient {expected_count} lignes"
            )
        expected_nonempty = actual_nonempty_counts.get(key)
        if (
            nonempty is not None
            and expected_nonempty is not None
            and nonempty != expected_nonempty
        ):
            findings.error(
                f"{context}.nonempty_targets={nonempty}, attendu={expected_nonempty}"
            )
        if row_count is not None and nonempty is not None and nonempty > row_count:
            findings.error(f"{context} : nonempty_targets={nonempty} dépasse n={row_count}")

        for metric in SUMMARY_METRICS:
            values: dict[str, float | None] = {}
            for statistic in ("mean", "median", "ci95_low", "ci95_high"):
                column = f"{metric}_{statistic}"
                values[statistic] = _parse_number(
                    row.get(column),
                    f"{context}.{column}",
                    findings,
                    minimum=0.0,
                    maximum=1.0,
                )
            low = values["ci95_low"]
            high = values["ci95_high"]
            if low is not None and high is not None and low > high + 1e-12:
                findings.error(
                    f"{context}.{metric} : CI95 inversé, low={low}, high={high}"
                )

    for duplicate in _duplicates(keys):
        findings.error(f"summary_metrics : clé dupliquée {_describe_key(duplicate)}")
    expected_keys = {(condition, map_name) for condition in CONDITIONS for map_name in MAPS}
    missing = sorted(expected_keys - set(keys))
    extra = sorted(set(keys) - expected_keys)
    if missing:
        findings.error(f"summary_metrics : clés manquantes : {missing}")
    if extra:
        findings.error(f"summary_metrics : clés inattendues : {extra}")


def _validate_shadow_rows(
    output: Path,
    target_by_case: Mapping[tuple[str, str, str], float],
    findings: Findings,
) -> list[dict[str, str]]:
    path = output / "tables/generated/shadow_stress_per_case.csv"
    rows = _load_csv(path, "shadow_stress_per_case", SHADOW_REQUIRED_COLUMNS, findings)
    if rows is None:
        return []
    if len(rows) != EXPECTED_SHADOW_ROWS:
        findings.error(
            f"shadow_stress_per_case contient {len(rows)} lignes, "
            f"attendu={EXPECTED_SHADOW_ROWS} "
            f"({len(ANCHOR_CASES)} ancres × {len(SHADOW_KINDS)} ombres × {len(MAPS)} cartes)"
        )

    keys: list[tuple[str, str, str]] = []
    fractions: dict[tuple[str, str], list[tuple[float, float]]] = {}
    clean_responses: dict[tuple[str, str], list[float]] = {}
    for line_number, row in enumerate(rows, start=2):
        case_name = (row.get("case_name") or "").strip()
        shadow_kind = (row.get("shadow_kind") or "").strip()
        map_name = (row.get("map") or "").strip()
        key = (case_name, shadow_kind, map_name)
        keys.append(key)
        context = f"shadow_stress_per_case ligne {line_number} ({_describe_key(key)})"
        if case_name not in ANCHOR_CASES:
            findings.error(f"{context} : l'image n'est pas l'une des 16 ancres")
        if shadow_kind not in SHADOW_KINDS:
            findings.error(f"{context} : type d'ombre inconnu")
        if map_name not in MAPS:
            findings.error(f"{context} : carte inconnue")

        shadow_fraction = _parse_number(
            row.get("shadow_fraction"),
            f"{context}.shadow_fraction",
            findings,
            minimum=0.0,
            maximum=1.0,
        )
        boundary_fraction = _parse_number(
            row.get("boundary_fraction"),
            f"{context}.boundary_fraction",
            findings,
            minimum=0.0,
            maximum=1.0,
        )
        if shadow_fraction is not None and shadow_fraction <= 0.0:
            findings.error(f"{context}.shadow_fraction doit être strictement positive")
        if boundary_fraction is not None and boundary_fraction <= 0.0:
            findings.error(f"{context}.boundary_fraction doit être strictement positive")
        if shadow_fraction is not None and boundary_fraction is not None:
            fractions.setdefault((case_name, shadow_kind), []).append(
                (shadow_fraction, boundary_fraction)
            )

        target = target_by_case.get(("khanhha", case_name, "original"))
        target_empty = target is not None and target <= 1e-15
        for column in ("clean_crack_response", "shadow_crack_response"):
            parsed = _parse_number(
                row.get(column),
                f"{context}.{column}",
                findings,
                minimum=0.0,
                maximum=1.0,
                allow_blank=target_empty,
            )
            if target_empty and parsed is not None:
                findings.error(
                    f"{context}.{column} doit être vide pour une ancre sans fissure"
                )
            if column == "clean_crack_response" and parsed is not None:
                clean_responses.setdefault((case_name, map_name), []).append(parsed)
        retention = _parse_number(
            row.get("crack_retention"),
            f"{context}.crack_retention",
            findings,
            minimum=0.0,
            allow_blank=target_empty,
        )
        if target_empty and retention is not None:
            findings.error(f"{context}.crack_retention doit être vide pour une ancre sans fissure")
        for column in ("clean_boundary_response", "shadow_boundary_response"):
            _parse_number(
                row.get(column),
                f"{context}.{column}",
                findings,
                minimum=0.0,
                maximum=1.0,
            )
        _parse_number(
            row.get("boundary_response_increase"),
            f"{context}.boundary_response_increase",
            findings,
            minimum=-1.0,
            maximum=1.0,
        )

    for duplicate in _duplicates(keys):
        findings.error(f"shadow_stress_per_case : clé dupliquée {_describe_key(duplicate)}")
    expected_keys = {
        (case_name, shadow_kind, map_name)
        for case_name in ANCHOR_CASES
        for shadow_kind in SHADOW_KINDS
        for map_name in MAPS
    }
    missing = sorted(expected_keys - set(keys))
    extra = sorted(set(keys) - expected_keys)
    if missing:
        findings.error(
            f"shadow_stress_per_case : {len(missing)} combinaisons manquantes, "
            f"premières={missing[:8]}"
        )
    if extra:
        findings.error(
            f"shadow_stress_per_case : {len(extra)} combinaisons inattendues, "
            f"premières={extra[:8]}"
        )

    for key, values in fractions.items():
        shadow_values = [value[0] for value in values]
        boundary_values = [value[1] for value in values]
        if shadow_values and max(shadow_values) - min(shadow_values) > 1e-12:
            findings.error(
                f"shadow_stress_per_case : shadow_fraction varie selon la carte pour "
                f"{_describe_key(key)}"
            )
        if boundary_values and max(boundary_values) - min(boundary_values) > 1e-12:
            findings.error(
                f"shadow_stress_per_case : boundary_fraction varie selon la carte pour "
                f"{_describe_key(key)}"
            )
    for key, values in clean_responses.items():
        if values and max(values) - min(values) > 1e-7:
            findings.error(
                f"shadow_stress_per_case : clean_crack_response varie selon l'ombre pour "
                f"{_describe_key(key)}"
            )
    return rows


def _validate_shadow_summary(
    output: Path,
    shadow_rows: Sequence[Mapping[str, str]],
    findings: Findings,
) -> None:
    required = (
        "shadow_kind",
        "map",
        "n",
        "crack_retention_median",
        "boundary_response_increase_mean",
        "shadow_boundary_response_mean",
    )
    path = output / "tables/generated/shadow_stress_summary.csv"
    rows = _load_csv(path, "shadow_stress_summary", required, findings)
    if rows is None:
        return
    expected_count = len(SHADOW_KINDS) * len(MAPS)
    if len(rows) != expected_count:
        findings.error(
            f"shadow_stress_summary contient {len(rows)} lignes, attendu={expected_count}"
        )

    actual_counts: dict[tuple[str, str], int] = {}
    for row in shadow_rows:
        key = (
            (row.get("shadow_kind") or "").strip(),
            (row.get("map") or "").strip(),
        )
        actual_counts[key] = actual_counts.get(key, 0) + 1

    keys: list[tuple[str, str]] = []
    for line_number, row in enumerate(rows, start=2):
        key = (
            (row.get("shadow_kind") or "").strip(),
            (row.get("map") or "").strip(),
        )
        keys.append(key)
        context = f"shadow_stress_summary ligne {line_number} ({_describe_key(key)})"
        if key[0] not in SHADOW_KINDS:
            findings.error(f"{context} : type d'ombre inconnu")
        if key[1] not in MAPS:
            findings.error(f"{context} : carte inconnue")
        count = _parse_csv_integer(row.get("n"), f"{context}.n", findings)
        expected_n = actual_counts.get(key)
        if count is not None and expected_n is not None and count != expected_n:
            findings.error(f"{context}.n={count}, attendu={expected_n}")
        _parse_number(
            row.get("crack_retention_median"),
            f"{context}.crack_retention_median",
            findings,
            minimum=0.0,
        )
        _parse_number(
            row.get("boundary_response_increase_mean"),
            f"{context}.boundary_response_increase_mean",
            findings,
            minimum=-1.0,
            maximum=1.0,
        )
        _parse_number(
            row.get("shadow_boundary_response_mean"),
            f"{context}.shadow_boundary_response_mean",
            findings,
            minimum=0.0,
            maximum=1.0,
        )

    for duplicate in _duplicates(keys):
        findings.error(f"shadow_stress_summary : clé dupliquée {_describe_key(duplicate)}")
    expected_keys = {(kind, map_name) for kind in SHADOW_KINDS for map_name in MAPS}
    missing = sorted(expected_keys - set(keys))
    extra = sorted(set(keys) - expected_keys)
    if missing:
        findings.error(f"shadow_stress_summary : clés manquantes : {missing}")
    if extra:
        findings.error(f"shadow_stress_summary : clés inattendues : {extra}")


def _slug(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe[:180]


def _validate_png(path: Path, label: str, findings: Findings) -> None:
    if not path.is_file():
        findings.error(f"figure {label} absente : {path}")
        return
    size = path.stat().st_size
    if size < 24:
        findings.error(f"figure {label} vide ou tronquée ({size} octets) : {path}")
        return
    try:
        with path.open("rb") as source:
            header = source.read(24)
    except OSError as exc:
        findings.error(f"figure {label} illisible ({path}) : {exc}")
        return
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        findings.error(f"figure {label} n'est pas un PNG valide : {path}")
        return
    width, height = struct.unpack(">II", header[16:24])
    if width == 0 or height == 0:
        findings.error(f"figure {label} a des dimensions nulles : {path}")


def _validate_figures(
    output: Path,
    case_metadata: Mapping[tuple[str, str, str], tuple[str, bool]],
    findings: Findings,
) -> None:
    figures = output / "figures/generated"
    for name in GLOBAL_FIGURES:
        _validate_png(figures / name, name, findings)

    expected_panels = {
        case_key
        for case_key, (_, qualitative) in case_metadata.items()
        if qualitative
    }
    canonical_expected = {
        ("khanhha", case_name, condition)
        for case_name in ANCHOR_CASES
        for condition in CONDITIONS
    } | set(EXTERNAL_CASES)
    if expected_panels != canonical_expected:
        missing = sorted(canonical_expected - expected_panels)
        extra = sorted(expected_panels - canonical_expected)
        if missing:
            findings.error(f"panneaux qualitatifs attendus mais non déclarés : {missing}")
        if extra:
            findings.error(f"panneaux qualitatifs déclarés en trop : {extra}")

    for dataset, case_name, condition in sorted(canonical_expected):
        path = (
            figures
            / "cases"
            / dataset
            / condition
            / f"{_slug(case_name)}.png"
        )
        _validate_png(
            path,
            f"panneau {_describe_key((dataset, case_name, condition))}",
            findings,
        )


def _validate_optional_reports(output: Path, findings: Findings) -> None:
    reports = sorted(path for path in output.glob("RAPPORT*.md") if path.is_file())
    if not reports:
        findings.note("aucun RAPPORT*.md trouvé (rapport optionnel)")
        return
    for path in reports:
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            findings.error(f"rapport Markdown illisible ({path}) : {exc}")
            continue
        if not content.strip():
            findings.error(f"rapport Markdown vide : {path}")


def _cross_check_manifest(
    manifest: Mapping[str, object] | None,
    case_rows: Sequence[Mapping[str, str]],
    metric_rows: Sequence[Mapping[str, str]],
    shadow_rows: Sequence[Mapping[str, str]],
    findings: Findings,
) -> None:
    if manifest is None:
        return
    for field, rows in (
        ("cases", case_rows),
        ("metric_rows", metric_rows),
        ("shadow_rows", shadow_rows),
    ):
        declared = manifest.get(field)
        if _is_integer(declared) and int(declared) != len(rows):
            findings.error(
                f"manifest.{field}={declared}, mais le CSV correspondant contient {len(rows)} lignes"
            )


def validate(output: Path) -> Findings:
    findings = Findings()
    output = output.expanduser().resolve()
    if not output.is_dir():
        findings.error(f"dossier de sortie absent ou non répertoire : {output}")
        return findings

    manifest = _load_manifest(output, findings)
    case_rows, case_metadata = _validate_cases(output, findings)
    metric_rows, target_by_case = _validate_per_case_metrics(
        output, case_metadata, findings
    )
    _validate_summary_metrics(output, metric_rows, findings)
    shadow_rows = _validate_shadow_rows(output, target_by_case, findings)
    _validate_shadow_summary(output, shadow_rows, findings)
    _validate_figures(output, case_metadata, findings)
    _validate_optional_reports(output, findings)
    _cross_check_manifest(
        manifest, case_rows, metric_rows, shadow_rows, findings
    )
    return findings


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output.expanduser().resolve()
    findings = validate(output)
    for note in findings.notes:
        print(f"[INFO] {note}")
    if findings.errors:
        for error in findings.errors:
            print(f"[ERREUR] {error}", file=sys.stderr)
        print(
            f"[ÉCHEC] {len(findings.errors)} erreur(s) dans {output}",
            file=sys.stderr,
        )
        return 1
    print(
        f"[OK] Sorties valides : {EXPECTED_CASES} cas, "
        f"{EXPECTED_METRIC_ROWS} mesures et {EXPECTED_SHADOW_ROWS} tests d'ombre dans {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
