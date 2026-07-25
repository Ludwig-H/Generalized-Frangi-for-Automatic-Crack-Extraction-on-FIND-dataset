from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from assemble_logistic_gate_data import (  # noqa: E402
    EXPECTED_FOLDS,
    OOF_MANIFEST_SCHEMA,
    assemble_oof_gate_data,
    load_and_validate_oof_manifest,
    local_file_identity,
    parse_fold_directories,
    validate_manifest_output_csv,
)
from cracksam2.gating import (  # noqa: E402
    DEFAULT_GATE_FEATURES,
    LogisticConfidenceGate,
)
from cracksam2.oof import strict_oof_training_contract  # noqa: E402
from train_logistic_gate import main as train_gate_main  # noqa: E402


def _recorded_identity(path: Path) -> dict[str, object]:
    identity = local_file_identity(path)
    return {name: identity[name] for name in ("name", "bytes", "sha256")}


def _names_sha256(names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_gate_rows(path: Path, fold: str, *, shared_group: str | None = None) -> None:
    role = "gate_calibration" if fold == "4" else "gate_fit"
    fieldnames = [
        "case_name",
        "source_group",
        "dataset",
        "role",
        "fold",
        "delta_iou",
        *DEFAULT_GATE_FEATURES,
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row_index, delta in enumerate((-0.02, 0.03)):
            signal = float(int(fold) * 2 + row_index + 1) / 10.0
            row: dict[str, object] = {
                "case_name": f"case-{fold}-{row_index}.png",
                "source_group": shared_group or f"source-{fold}-{row_index}",
                "dataset": "KhanhhaTrain",
                "role": role,
                "fold": fold,
                "delta_iou": delta,
            }
            row.update(
                {
                    name: signal + feature_index / 100.0
                    for feature_index, name in enumerate(DEFAULT_GATE_FEATURES)
                }
            )
            writer.writerow(row)


def _write_oof_source_tree(tmp_path: Path) -> dict[str, Path]:
    shared = tmp_path / "shared"
    shared.mkdir(parents=True)
    sam2 = shared / "sam2.pt"
    baseline = shared / "baseline.pt"
    group_assignments = shared / "train_group_folds.csv"
    dataset_list = shared / "train.txt"
    cache_root = shared / "graph-cache"
    cache_root.mkdir()
    cache_manifest = cache_root / ".cracksam2-frangi-graph-v2.json"
    sam2.write_bytes(b"sam2-foundation")
    baseline.write_bytes(b"baseline-adapter")
    group_assignments.write_text("group-safe-folds\n", encoding="utf-8")
    dataset_list.write_text("full-training-list\n", encoding="utf-8")
    cache_manifest.write_text('{"status":"complete"}\n', encoding="utf-8")

    directories: dict[str, Path] = {}
    for fold in EXPECTED_FOLDS:
        role = "gate_calibration" if fold == "4" else "gate_fit"
        directory = tmp_path / f"evaluation-fold-{fold}"
        directory.mkdir()
        csv_path = directory / "per_image.csv"
        _write_gate_rows(csv_path, fold)
        residual = shared / f"residual-fold-{fold}.pt"
        residual.write_bytes(f"residual:{fold}".encode("ascii"))
        names = [f"case-{fold}-{row_index}.png" for row_index in range(2)]
        contract = {
            "schema": "cracksam2.frangigraph-residual-evaluation",
            "schema_version": 1,
            "dataset": {
                "name": "KhanhhaTrain",
                "role": role,
                "fold": fold,
                "root": str((shared / "dataset").resolve()),
                "list": _recorded_identity(dataset_list),
                "split": "train",
                "noise": "original",
                "image_size": [448, 448],
                "selected_samples": len(names),
                "selected_sample_names_sha256": _names_sha256(names),
                "group_assignments": _recorded_identity(group_assignments),
            },
            "checkpoints": {
                "sam2": _recorded_identity(sam2),
                "baseline": _recorded_identity(baseline),
                "residual": _recorded_identity(residual),
            },
            "graph_cache": {
                "root": str(cache_root.resolve()),
                "manifest": _recorded_identity(cache_manifest),
                "extractor_sha256": "a" * 64,
                "frangi": {"scales": [1.0, 2.0, 4.0]},
                "channels": [f"channel-{index}" for index in range(7)],
                "verify_cache_hashes": True,
                "verify_data_hashes": True,
            },
            "residual": {
                "raster_channels": 7,
                "high_resolution_channels": [32, 64],
                "hidden_channels": 8,
                "raster_preprocessing": {"fold": fold},
                "training_raster_condition": "correct",
                "evaluation_raster_condition": "correct",
                "causal_raster_override": False,
                "checkpoint_held_out_fold": int(fold),
                "checkpoint_oof_training": strict_oof_training_contract(
                    int(fold)
                ),
            },
            "segmentation_threshold": 0.5,
            "label_minimum_gain": 0.0,
            "gate_policy": {
                "feature_rows_only": True,
                "threshold_selected_by_this_command": False,
                "threshold_may_later_be_calibrated_from_this_role": fold == "4",
                "historical_tests_forbidden_for_threshold_selection": True,
            },
        }
        (directory / "evaluation_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        directories[fold] = directory
    return directories


def test_fold_cli_requires_exactly_one_directory_for_each_fold(
    tmp_path: Path,
) -> None:
    directories = _write_oof_source_tree(tmp_path)
    values = [f"{fold}={directories[fold]}" for fold in EXPECTED_FOLDS]
    assert parse_fold_directories(values) == {
        fold: path.resolve() for fold, path in directories.items()
    }
    with pytest.raises(ValueError, match="exactly five"):
        parse_fold_directories(values[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        parse_fold_directories([*values[:-1], values[0]])


def test_assembler_binds_five_contracts_checkpoints_and_atomic_outputs(
    tmp_path: Path,
) -> None:
    directories = _write_oof_source_tree(tmp_path)
    output = tmp_path / "assembled"

    manifest = assemble_oof_gate_data(directories, output)
    restored, manifest_sha = load_and_validate_oof_manifest(
        output / "oof_manifest.json"
    )

    assert restored == manifest
    assert len(manifest_sha) == 64
    assert manifest["schema"] == OOF_MANIFEST_SCHEMA
    assert manifest["outputs"]["gate_fit_csv"]["rows"] == 8
    assert manifest["outputs"]["gate_fit_csv"]["folds"] == ["0", "1", "2", "3"]
    assert manifest["outputs"]["gate_calibration_csv"]["rows"] == 2
    assert manifest["outputs"]["gate_calibration_csv"]["folds"] == ["4"]
    assert len(
        {
            manifest["folds"][fold]["residual_checkpoint"]["sha256"]
            for fold in EXPECTED_FOLDS
        }
    ) == 5
    for fold in ("0", "1", "2", "3"):
        assert manifest["folds"][fold]["oof_training"]["training_folds"] == [
            candidate
            for candidate in range(4)
            if candidate != int(fold)
        ]
        assert manifest["folds"][fold]["oof_training"][
            "additional_excluded_training_folds"
        ] == [4]
    assert manifest["folds"]["4"]["oof_training"]["training_folds"] == [
        0,
        1,
        2,
        3,
    ]
    assert manifest["artifacts"]["graph_cache"]["extractor_sha256"] == "a" * 64
    validate_manifest_output_csv(manifest, "gate_fit_csv", output / "gate_fit.csv")

    with (output / "gate_fit.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert {row["role"] for row in rows} == {"gate_fit"}
    assert {row["fold"] for row in rows} == {"0", "1", "2", "3"}

    with (output / "gate_fit.csv").open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="does not match oof_manifest"):
        validate_manifest_output_csv(
            manifest, "gate_fit_csv", output / "gate_fit.csv"
        )


def test_assembler_rejects_cross_fold_group_leakage_before_publishing(
    tmp_path: Path,
) -> None:
    directories = _write_oof_source_tree(tmp_path)
    fold_one_csv = directories["1"] / "per_image.csv"
    content = fold_one_csv.read_text(encoding="utf-8")
    fold_one_csv.write_text(
        content.replace("source-1-0", "source-0-0"), encoding="utf-8"
    )
    output = tmp_path / "not-published"

    with pytest.raises(ValueError, match="source_group collision"):
        assemble_oof_gate_data(directories, output)

    assert not (output / "gate_fit.csv").exists()
    assert not (output / "oof_manifest.json").exists()


@pytest.mark.parametrize("fold", ["0", "4"])
def test_assembler_rejects_causal_override_rows_for_both_gate_stages(
    tmp_path: Path, fold: str
) -> None:
    directories = _write_oof_source_tree(tmp_path)
    contract_path = directories[fold] / "evaluation_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["analytical_only"] = True
    contract["residual"]["evaluation_raster_condition"] = "no_evidence"
    contract["residual"]["causal_raster_override"] = True
    contract["gate_policy"]["eligible_for_later_gate_fit"] = False
    contract["gate_policy"][
        "threshold_may_later_be_calibrated_from_this_role"
    ] = False
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    output = tmp_path / "causal-must-not-be-assembled"

    with pytest.raises(ValueError, match="input-ablation raster override evaluation"):
        assemble_oof_gate_data(directories, output)

    assert not (output / "gate_fit.csv").exists()
    assert not (output / "gate_calibration.csv").exists()
    assert not (output / "oof_manifest.json").exists()


def test_assembler_rejects_checkpoint_fold_or_shared_artifact_drift(
    tmp_path: Path,
) -> None:
    directories = _write_oof_source_tree(tmp_path)
    contract_path = directories["3"] / "evaluation_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["residual"]["checkpoint_held_out_fold"] = 2
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint held out on fold 2"):
        assemble_oof_gate_data(directories, tmp_path / "held-out-drift")

    directories = _write_oof_source_tree(tmp_path / "second")
    contract_path = directories["2"] / "evaluation_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["checkpoints"]["baseline"]["sha256"] = "b" * 64
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="different baseline"):
        assemble_oof_gate_data(directories, tmp_path / "baseline-drift")


def test_assembler_rejects_fold_four_leakage_even_with_self_consistent_rows(
    tmp_path: Path,
) -> None:
    directories = _write_oof_source_tree(tmp_path)
    contract_path = directories["1"] / "evaluation_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    oof_training = contract["residual"]["checkpoint_oof_training"]
    oof_training["training_folds"] = [0, 2, 3, 4]
    oof_training["additional_excluded_training_folds"] = []
    oof_training["all_excluded_training_folds"] = [1]
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    with pytest.raises(ValueError, match="strict OOF training contract"):
        assemble_oof_gate_data(directories, tmp_path / "leaky")

    assert not (tmp_path / "leaky" / "gate_fit.csv").exists()


def test_oof_manifest_rejects_embedded_guard_drift_without_source_reads(
    tmp_path: Path,
) -> None:
    directories = _write_oof_source_tree(tmp_path)
    output = tmp_path / "assembled"
    assemble_oof_gate_data(directories, output)
    manifest_path = output / "oof_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["folds"]["0"]["oof_training"]["training_folds"].append(4)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="strict OOF training contract"):
        load_and_validate_oof_manifest(manifest_path, verify_sources=False)


def test_training_consumes_only_manifest_locked_csvs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directories = _write_oof_source_tree(tmp_path)
    assembled = tmp_path / "assembled"
    manifest = assemble_oof_gate_data(directories, assembled)
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_logistic_gate.py",
            "--train-csv",
            str(assembled / "gate_fit.csv"),
            "--calibration-csv",
            str(assembled / "gate_calibration.csv"),
            "--oof-manifest",
            str(assembled / "oof_manifest.json"),
            "--output",
            str(output),
            "--git-commit",
            "c" * 40,
            "--label-minimum-gain",
            "0",
            "--minimum-precision",
            "0.5",
            "--minimum-selected",
            "1",
            "--minimum-selected-groups",
            "1",
            "--minimum-mean-delta-iou",
            "-1",
            "--maximum-severe-loss-rate",
            "1",
        ],
    )

    train_gate_main()

    gate = LogisticConfidenceGate.load_json(output)
    summary = json.loads(capsys.readouterr().out)
    assert gate.feature_names == DEFAULT_GATE_FEATURES
    assert gate.provenance is not None
    assert gate.provenance.oof_manifest_sha256 == local_file_identity(
        assembled / "oof_manifest.json"
    )["sha256"]
    assert gate.provenance.frangi_extractor_sha256 == "a" * 64
    assert gate.provenance.baseline_checkpoint_sha256 == manifest["artifacts"][
        "baseline_checkpoint"
    ]["sha256"]
    assert summary["train_folds"] == ["0", "1", "2", "3"]
    assert summary["calibration_folds"] == ["4"]

    fit_path = assembled / "gate_fit.csv"
    fit_path.write_text(fit_path.read_text(encoding="utf-8") + "tamper\n")
    with pytest.raises(ValueError, match="does not match oof_manifest"):
        train_gate_main()
