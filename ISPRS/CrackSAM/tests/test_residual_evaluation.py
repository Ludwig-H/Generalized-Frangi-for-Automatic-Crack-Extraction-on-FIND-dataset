from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cracksam2.gating import DEFAULT_GATE_FEATURES
from cracksam2.data import sample_names_sha256
from cracksam2.graph_cache import (
    GRAPH_CACHE_MANIFEST,
    graph_cache_record,
    save_graph_raster_atomic,
    sha256_file,
    write_json_atomic,
)
from cracksam2.graph_types import (
    FRANGI_RASTER_CHANNEL_INDEX,
    FRANGI_RASTER_CHANNELS,
    FRANGI_RASTER_SCHEMA_VERSION,
    FrangiRasterSample,
)
from cracksam2.oof import strict_oof_training_contract
from cracksam2.residual_evaluation import (
    GraphRasterEvaluationDataset,
    ROW_FIELDS,
    SELECTOR_DIAGNOSTIC_FIELDS,
    RasterPreprocessing,
    append_progress_batch,
    build_evaluation_row,
    build_selector_diagnostic,
    ensure_evaluation_contract,
    evaluate_residual_loader,
    load_group_assignments,
    load_safe_torch_checkpoint,
    read_progress_rows,
    read_selector_diagnostics,
    resolve_raster_condition,
    summarize_rows,
    validate_residual_checkpoint,
    write_rows_csv_atomic,
)
from cracksam2.residual_data import FrangiRasterNormalization
from evaluate_frangi_graph_residual import (  # noqa: E402
    _evaluation_usage_policy,
    _ordered_finalize,
    _selector_metadata_from_checkpoint,
    parse_args,
)


def _identity_preprocessing() -> RasterPreprocessing:
    channels = FRANGI_RASTER_CHANNELS
    return RasterPreprocessing(
        channel_names=channels,
        log1p_channels=(),
        offset=(0.0,) * len(channels),
        scale=(1.0,) * len(channels),
    )


def test_evaluation_help_explains_checkpoint_bound_feature_cell_units() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluate_frangi_graph_residual.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "profile radii and evidence dilation" in help_text
    assert "first SAM 2/Hiera high-resolution feature grid" in help_text
    assert "evaluation never overrides them" in help_text


def _row(target: torch.Tensor | None = None) -> dict[str, object]:
    baseline = torch.tensor([[[-2.0, 2.0], [-2.0, -2.0]]])
    candidate = torch.tensor([[[2.0, 2.0], [-2.0, -2.0]]])
    support = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    if target is None:
        target = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    return build_evaluation_row(
        case_name="case.png",
        source_group="wall_001",
        dataset="synthetic",
        role="gate_fit",
        fold="0",
        baseline_logits=baseline,
        candidate_logits=candidate,
        target=target,
        frangi_stats={"similarity": 0.75, "density": 0.25},
        frangi_support=support,
    )


def _selector_diagnostic(
    case_name: str = "case.png",
    *,
    score_source: str = "evidence_score_fusion",
) -> dict[str, object]:
    support = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
    score = torch.tensor([[[0.8, 0.2], [0.0, 0.0]]])
    selected = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    envelope_fusion = selected.clone()
    envelope_output = torch.zeros(1, 4, 4)
    envelope_output[:, :2, :2] = 1.0
    residual = torch.zeros(1, 4, 4)
    residual[:, :2, :2] = 2.0
    target = torch.zeros(1, 4, 4)
    target[:, 0, 0] = 1.0
    target[:, 0, 3] = 1.0
    return build_selector_diagnostic(
        case_name=case_name,
        evidence_score_source=score_source,
        gate_spatial_support_source="accepted_local_selector_output",
        evidence_support_fusion=support,
        evidence_score_fusion=score,
        evidence_selected_fusion=selected,
        correction_envelope_fusion=envelope_fusion,
        correction_envelope_output=envelope_output,
        residual_logits=residual,
        target=target,
    )


def test_gate_features_are_exactly_seven_and_do_not_depend_on_ground_truth() -> None:
    positive_target = _row()
    negative_target = _row(torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]))

    assert tuple(name for name in ROW_FIELDS if name in DEFAULT_GATE_FEATURES) == (
        DEFAULT_GATE_FEATURES
    )
    assert {name: positive_target[name] for name in DEFAULT_GATE_FEATURES} == {
        name: negative_target[name] for name in DEFAULT_GATE_FEATURES
    }
    assert positive_target["candidate_iou_gain"] > 0
    assert negative_target["candidate_iou_gain"] < 0
    assert positive_target["delta_iou"] == positive_target["candidate_iou_gain"]
    assert positive_target["candidate_better"] == 1


def test_empty_frangi_support_produces_a_finite_zero_support_correction() -> None:
    row = build_evaluation_row(
        case_name="empty.png",
        source_group="empty",
        dataset="synthetic",
        role="development",
        fold="",
        baseline_logits=torch.zeros(1, 2, 2),
        candidate_logits=torch.ones(1, 2, 2),
        target=torch.zeros(1, 2, 2),
        frangi_stats={"similarity": 0.0, "density": 0.0},
        frangi_support=torch.zeros(1, 2, 2),
    )
    assert row["support_correction_probability_mean"] == 0.0


def test_progress_is_resumable_and_repairs_only_a_truncated_last_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.jsonl"
    append_progress_batch(path, dataset="synthetic", rows=[_row()])
    with path.open("ab") as stream:
        stream.write(b'{"format_version":')

    rows = read_progress_rows(path, expected_dataset="synthetic")

    assert list(rows) == ["case.png"]
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError, match="Duplicate"):
        append_progress_batch(path, dataset="synthetic", rows=[_row()])
        read_progress_rows(path, expected_dataset="synthetic")


def test_selector_diagnostic_uses_fusion_cells_and_output_residual_amplitude() -> None:
    diagnostic = _selector_diagnostic()

    assert tuple(diagnostic) == SELECTOR_DIAGNOSTIC_FIELDS
    assert diagnostic["fusion_height"] == 2
    assert diagnostic["fusion_width"] == 2
    assert diagnostic["spatial_cells"] == 4
    assert diagnostic["selector_support_fraction"] == 0.5
    assert diagnostic["selector_accepted_fraction_on_support"] == 0.5
    assert diagnostic["evidence_score_support_mean"] == pytest.approx(0.5)
    assert diagnostic["evidence_score_support_p05"] == pytest.approx(0.23)
    assert diagnostic["evidence_score_support_p50"] == pytest.approx(0.5)
    assert diagnostic["evidence_score_support_p95"] == pytest.approx(0.77)
    assert diagnostic["correction_envelope_fraction_on_fusion_grid"] == 0.25
    assert diagnostic["residual_absolute_mean_inside_output_envelope"] == 2.0
    assert diagnostic["residual_absolute_mean_outside_output_envelope"] == 0.0
    assert (
        diagnostic[
            "annotation_overlap_precision_on_selected_support"
        ]
        == 1.0
    )
    assert (
        diagnostic["annotation_overlap_recall_on_supported_target"]
        == 0.5
    )
    assert _selector_diagnostic(
        score_source="evidence_probability"
    )["evidence_score_source"] == "evidence_probability"


def test_old_progress_schema_resumes_and_finalizes_partial_selector_diagnostics(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress.jsonl"
    old_row = dict(_row())
    old_row["case_name"] = "old.png"
    append_progress_batch(progress, dataset="synthetic", rows=[old_row])
    old_entry = json.loads(progress.read_text(encoding="utf-8").splitlines()[0])
    assert set(old_entry) == {"format_version", "dataset", "rows"}

    new_row = dict(_row())
    new_row["case_name"] = "new.png"
    append_progress_batch(
        progress,
        dataset="synthetic",
        rows=[new_row],
        selector_diagnostics=[_selector_diagnostic("new.png")],
    )

    assert list(read_progress_rows(progress, expected_dataset="synthetic")) == [
        "old.png",
        "new.png",
    ]
    assert list(
        read_selector_diagnostics(progress, expected_dataset="synthetic")
    ) == ["new.png"]
    _ordered_finalize(
        output=tmp_path,
        dataset_name="synthetic",
        selected_names=["old.png", "new.png"],
        role="gate_fit",
        causal_raster_override=False,
    )

    artifact = json.loads(
        (tmp_path / "selector_diagnostics.json").read_text(encoding="utf-8")
    )
    assert artifact["complete"] is False
    assert artifact["selected_cases"] == 2
    assert artifact["diagnostic_cases"] == 1
    assert artifact["missing_case_names"] == ["old.png"]
    assert artifact["row_fields"] == list(SELECTOR_DIAGNOSTIC_FIELDS)
    assert [row["case_name"] for row in artifact["rows"]] == ["new.png"]
    with (tmp_path / "per_image.csv").open(newline="", encoding="utf-8") as stream:
        assert tuple(csv.DictReader(stream).fieldnames or ()) == ROW_FIELDS


def test_contract_and_final_csv_are_atomic_and_strict(tmp_path: Path) -> None:
    contract = {"schema": "test", "selected": ["case.png"]}
    ensure_evaluation_contract(tmp_path, contract)
    ensure_evaluation_contract(tmp_path, contract)
    with pytest.raises(RuntimeError, match="another immutable contract"):
        ensure_evaluation_contract(tmp_path, {"schema": "other"})

    write_rows_csv_atomic(tmp_path / "per_image.csv", [_row()])
    with (tmp_path / "per_image.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames or ()) == ROW_FIELDS
        assert next(reader)["source_group"] == "wall_001"
    summary = summarize_rows([_row()])
    assert summary["gate_threshold_selected"] is False


def test_causal_override_summary_is_analytical_and_gate_ineligible(
    tmp_path: Path,
) -> None:
    row = dict(_row())
    row["role"] = "gate_calibration"
    row["fold"] = "4"
    append_progress_batch(
        tmp_path / "progress.jsonl", dataset="synthetic", rows=[row]
    )

    usage = _evaluation_usage_policy(
        role="gate_calibration", causal_raster_override=True
    )
    assert usage == {
        "analytical_only": True,
        "eligible_for_later_gate_fit": False,
        "eligible_for_later_gate_threshold_calibration": False,
    }
    assert _evaluation_usage_policy(
        role="gate_fit", causal_raster_override=False
    ) == {
        "analytical_only": False,
        "eligible_for_later_gate_fit": True,
        "eligible_for_later_gate_threshold_calibration": False,
    }

    summary = _ordered_finalize(
        output=tmp_path,
        dataset_name="synthetic",
        selected_names=["case.png"],
        role="gate_calibration",
        causal_raster_override=True,
    )

    assert summary["analytical_only"] is True
    assert summary["eligible_for_later_gate_fit"] is False
    assert summary["eligible_for_later_gate_threshold_calibration"] is False
    stored_summary = json.loads(
        (tmp_path / "summary.json").read_text(encoding="utf-8")
    )
    assert stored_summary == summary


def test_raster_preprocessing_is_explicit_and_reproducible() -> None:
    channels = FRANGI_RASTER_CHANNELS
    magnitude_index = FRANGI_RASTER_CHANNEL_INDEX["hessian_magnitude"]
    offset = [0.0] * len(channels)
    scale = [1.0] * len(channels)
    offset[magnitude_index] = 1.0
    scale[magnitude_index] = 2.0
    preprocessing = RasterPreprocessing.from_mapping(
        {
            "channel_names": list(channels),
            "log1p_channels": ["hessian_magnitude"],
            "offset": offset,
            "scale": scale,
        }
    )
    raster = torch.zeros(1, len(channels), 1, 1)
    raster[:, magnitude_index] = np.e**3 - 1

    transformed = preprocessing.apply(raster)

    torch.testing.assert_close(
        transformed[:, magnitude_index], torch.tensor([[[1.0]]]), atol=1e-6, rtol=0
    )


def test_evaluator_preprocessing_matches_training_preprocessing() -> None:
    scale = [1.0] * len(FRANGI_RASTER_CHANNELS)
    offset = [0.0] * len(FRANGI_RASTER_CHANNELS)
    magnitude_index = FRANGI_RASTER_CHANNEL_INDEX["hessian_magnitude"]
    scale_index = FRANGI_RASTER_CHANNEL_INDEX["winning_scale"]
    offset[magnitude_index] = 0.3
    scale[magnitude_index] = 0.7
    scale[scale_index] = 15.0
    training = FrangiRasterNormalization(
        channel_names=FRANGI_RASTER_CHANNELS,
        log1p_channels=("hessian_magnitude",),
        offset=tuple(offset),
        scale=tuple(scale),
        fit_sample_count=3,
        fit_sample_names_sha256="a" * 64,
        graph_manifest_sha256="b" * 64,
    )
    raster = np.zeros((len(FRANGI_RASTER_CHANNELS), 2, 3), dtype=np.float32)
    raster[FRANGI_RASTER_CHANNEL_INDEX["similarity"]] = 0.4
    raster[FRANGI_RASTER_CHANNEL_INDEX["support"], 0, 0] = 1.0
    raster[magnitude_index] = np.asarray([[0.0, 1.0, 4.0], [2.0, 0.5, 8.0]])
    raster[FRANGI_RASTER_CHANNEL_INDEX["distance_to_skeleton"]] = 1.0

    expected = training.transform(raster)
    evaluator = RasterPreprocessing.from_mapping(training.preprocessing_contract())
    observed = evaluator.apply(torch.from_numpy(raster).unsqueeze(0))[0].numpy()

    np.testing.assert_allclose(observed, expected, rtol=2e-6, atol=2e-7)


def _residual_payload(baseline_sha: str, manifest: dict[str, object]):
    scale = [1.0] * len(FRANGI_RASTER_CHANNELS)
    preprocessing = RasterPreprocessing(
        channel_names=FRANGI_RASTER_CHANNELS,
        log1p_channels=("hessian_magnitude",),
        offset=(0.0,) * len(FRANGI_RASTER_CHANNELS),
        scale=tuple(scale),
        format_version=1,
        fit_sample_count=4,
        fit_sample_names_sha256="e" * 64,
        graph_manifest_sha256=str(manifest["manifest_sha256"]),
        winning_scale_divisor=1.0,
    )
    run_contract = {
        "contract_version": 2,
        "raster_condition": "correct",
        "held_out_fold": 2,
        "oof_training": strict_oof_training_contract(2),
        "excluded_training_folds": [4],
        "all_folds_excluded_from_training": [2, 4],
    }
    run_contract_sha256 = hashlib.sha256(
        json.dumps(
            run_contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return {
        "format_version": 1,
        "training_state": "complete",
        "residual_adapter": {"output_projection.bias": torch.zeros(1)},
        "residual": {
            "raster_channels": 7,
            "high_resolution_channels": [32, 64],
            "hidden_channels": 8,
        },
        "baseline_adapter": {"sha256": baseline_sha},
        "graph_cache": {
            "extractor_sha256": manifest["extractor_sha256"],
            "frangi": manifest["frangi"],
            "channels": manifest["channels"],
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "raster_preprocessing": preprocessing.as_dict(),
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
    }


def test_checkpoint_loading_is_safe_and_bound_to_baseline_and_cache(
    tmp_path: Path,
) -> None:
    manifest = {
        "extractor_sha256": "a" * 64,
        "frangi": {"scales": [1.0]},
        "channels": list(FRANGI_RASTER_CHANNELS),
        "manifest_sha256": "d" * 64,
    }
    payload = _residual_payload("b" * 64, manifest)
    path = tmp_path / "residual.pt"
    torch.save(payload, path)

    loaded = load_safe_torch_checkpoint(path)
    spec = validate_residual_checkpoint(
        loaded,
        baseline_checkpoint_sha256="b" * 64,
        graph_cache_manifest=manifest,
    )

    assert spec.raster_channels == 7
    assert spec.training_raster_condition == "correct"
    assert spec.held_out_fold == 2
    assert spec.oof_training["training_folds"] == [0, 1, 3]
    assert spec.training_state == "complete"
    assert spec.adapter_mode == "legacy_raster_v1"
    assert spec.profile_radii == ()
    assert (
        _selector_metadata_from_checkpoint(
            payload,
            adapter_mode=spec.adapter_mode,
            profile_radii=spec.profile_radii,
            evidence_dilation=spec.evidence_dilation,
        )
        == {}
    )
    with pytest.raises(ValueError, match="another baseline"):
        validate_residual_checkpoint(
            loaded,
            baseline_checkpoint_sha256="c" * 64,
            graph_cache_manifest=manifest,
        )


def test_checkpoint_validates_verified_local_architecture() -> None:
    manifest = {
        "extractor_sha256": "a" * 64,
        "frangi": {"scales": [1.0]},
        "channels": list(FRANGI_RASTER_CHANNELS),
        "manifest_sha256": "d" * 64,
    }
    payload = _residual_payload("b" * 64, manifest)
    verified_architecture = {
        "adapter_mode": "verified_local_v1",
        "profile_radii": [1.5, 3.0],
        "evidence_dilation": 2,
        "evidence_threshold": 0.5,
    }
    payload["residual"].update(verified_architecture)
    payload["run_contract"]["residual"] = dict(verified_architecture)
    payload["run_contract_sha256"] = hashlib.sha256(
        json.dumps(
            payload["run_contract"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    spec = validate_residual_checkpoint(
        payload,
        baseline_checkpoint_sha256="b" * 64,
        graph_cache_manifest=manifest,
    )

    assert spec.adapter_mode == "verified_local_v1"
    assert spec.profile_radii == (1.5, 3.0)
    assert spec.evidence_dilation == 2
    assert spec.evidence_threshold == 0.5

    selector_grid = {
        "source": "SAM2ImageFeatures.high_resolution_features[0]",
        "parameter_unit": "hiera_high_resolution_feature_cells",
        "input_image_size_pixels": [448, 448],
        "fusion_grid_size_feature_cells": [256, 224],
        "effective_stride_input_pixels_per_feature_cell": [1.75, 2.0],
    }
    payload["residual"].update(
        {
            "profile_radii_feature_cells": [1.5, 3.0],
            "evidence_dilation_feature_cells": 2,
            "selector_grid": selector_grid,
        }
    )
    selector_metadata = _selector_metadata_from_checkpoint(
        payload,
        adapter_mode=spec.adapter_mode,
        profile_radii=spec.profile_radii,
        evidence_dilation=spec.evidence_dilation,
    )
    assert selector_metadata == {
        "profile_radii_feature_cells": [1.5, 3.0],
        "evidence_dilation_feature_cells": 2,
        "selector_grid": selector_grid,
    }

    payload["residual"]["selector_grid"][
        "effective_stride_input_pixels_per_feature_cell"
    ] = [2.0, 2.0]
    with pytest.raises(ValueError, match="effective stride"):
        _selector_metadata_from_checkpoint(
            payload,
            adapter_mode=spec.adapter_mode,
            profile_radii=spec.profile_radii,
            evidence_dilation=spec.evidence_dilation,
        )
    payload["residual"].pop("selector_grid")

    payload["residual"]["evidence_threshold"] = 1.0
    with pytest.raises(ValueError, match="evidence_threshold"):
        validate_residual_checkpoint(
            payload,
            baseline_checkpoint_sha256="b" * 64,
            graph_cache_manifest=manifest,
        )


def test_checkpoint_refuses_missing_or_incoherent_strict_oof_guard() -> None:
    manifest = {
        "extractor_sha256": "a" * 64,
        "frangi": {"scales": [1.0]},
        "channels": list(FRANGI_RASTER_CHANNELS),
        "manifest_sha256": "d" * 64,
    }
    payload = _residual_payload("b" * 64, manifest)
    run_contract = dict(payload["run_contract"])
    oof_training = dict(run_contract["oof_training"])
    oof_training["training_folds"] = [0, 1, 3, 4]
    run_contract["oof_training"] = oof_training
    payload["run_contract"] = run_contract
    payload["run_contract_sha256"] = hashlib.sha256(
        json.dumps(
            run_contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="strict OOF training contract is inconsistent"):
        validate_residual_checkpoint(
            payload,
            baseline_checkpoint_sha256="b" * 64,
            graph_cache_manifest=manifest,
        )

    legacy = _residual_payload("b" * 64, manifest)
    legacy_contract = dict(legacy["run_contract"])
    legacy_contract["contract_version"] = 1
    legacy["run_contract"] = legacy_contract
    legacy["run_contract_sha256"] = hashlib.sha256(
        json.dumps(
            legacy_contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="predates the strict OOF leakage guard"):
        validate_residual_checkpoint(
            legacy,
            baseline_checkpoint_sha256="b" * 64,
            graph_cache_manifest=manifest,
        )


def test_checkpoint_oof_guard_is_bound_to_requested_evaluation_fold_and_role() -> None:
    manifest = {
        "extractor_sha256": "a" * 64,
        "frangi": {"scales": [1.0]},
        "channels": list(FRANGI_RASTER_CHANNELS),
        "manifest_sha256": "d" * 64,
    }
    payload = _residual_payload("b" * 64, manifest)

    with pytest.raises(ValueError, match="held-out fold differs from evaluation"):
        validate_residual_checkpoint(
            payload,
            baseline_checkpoint_sha256="b" * 64,
            graph_cache_manifest=manifest,
            expected_oof_fold=3,
            expected_oof_role="gate_fit",
        )
    with pytest.raises(ValueError, match="role differs from evaluation role"):
        validate_residual_checkpoint(
            payload,
            baseline_checkpoint_sha256="b" * 64,
            graph_cache_manifest=manifest,
            expected_oof_fold=2,
            expected_oof_role="gate_calibration",
        )


def test_incomplete_checkpoint_is_forbidden_for_gate_rows() -> None:
    manifest = {
        "extractor_sha256": "a" * 64,
        "frangi": {"scales": [1.0]},
        "channels": list(FRANGI_RASTER_CHANNELS),
        "manifest_sha256": "d" * 64,
    }
    payload = _residual_payload("b" * 64, manifest)
    payload["training_state"] = "incomplete"

    spec = validate_residual_checkpoint(
        payload,
        baseline_checkpoint_sha256="b" * 64,
        graph_cache_manifest=manifest,
    )
    assert spec.training_state == "incomplete"
    with pytest.raises(ValueError, match="incomplete"):
        validate_residual_checkpoint(
            payload,
            baseline_checkpoint_sha256="b" * 64,
            graph_cache_manifest=manifest,
            require_complete=True,
        )

    del payload["training_state"]
    with pytest.raises(ValueError, match="training_state"):
        validate_residual_checkpoint(
            payload,
            baseline_checkpoint_sha256="b" * 64,
            graph_cache_manifest=manifest,
        )


class _FakeBaseDataset(Dataset):
    def __init__(self, image_path: Path, mask_path: Path) -> None:
        self.image_path = image_path
        self.mask_path = mask_path
        self.sample_names = ["case.png"]
        self.image_size = (4, 5)
        self.noise_mode = "original"

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        assert index == 0
        return {
            "image": torch.zeros(3, 4, 5),
            "mask": torch.zeros(1, 4, 5),
            "case_name": "case.png",
        }

    def _paths(self, name: str):
        assert name == "case.png"
        return self.image_path, self.mask_path


def test_graph_dataset_verifies_cache_and_exposes_raw_gate_statistics(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.bin"
    mask_path = tmp_path / "mask.bin"
    image_path.write_bytes(b"image")
    mask_path.write_bytes(b"mask")
    raster = np.zeros((len(FRANGI_RASTER_CHANNELS), 4, 5), dtype=np.float32)
    raster[FRANGI_RASTER_CHANNEL_INDEX["similarity"], 0, 0] = 0.75
    raster[FRANGI_RASTER_CHANNEL_INDEX["support"], 0, 0] = 1.0
    raster[FRANGI_RASTER_CHANNEL_INDEX["hessian_magnitude"], 0, 0] = 2.0
    raster[FRANGI_RASTER_CHANNEL_INDEX["distance_to_skeleton"]] = 1.0
    sample = FrangiRasterSample(
        case_name="case.png",
        raster=raster,
        image_sha256=sha256_file(image_path),
        mask_sha256=sha256_file(mask_path),
        elapsed_seconds=0.1,
    )
    cache_root = tmp_path / "cache"
    cache_path = cache_root / "case.png.npz"
    save_graph_raster_atomic(cache_path, sample)
    record = graph_cache_record(cache_root, cache_path, sample)
    manifest = {
        "schema_version": FRANGI_RASTER_SCHEMA_VERSION,
        "status": "complete",
        "sample_count": 1,
        "sample_names_sha256": sample_names_sha256(["case.png"]),
        "image_size": [4, 5],
        "noise": "original",
        "channels": list(FRANGI_RASTER_CHANNELS),
        "extractor_sha256": "a" * 64,
        "frangi": {"scales": [1.0]},
        "entries": [record],
    }
    write_json_atomic(cache_root / GRAPH_CACHE_MANIFEST, manifest)
    dataset = GraphRasterEvaluationDataset(
        _FakeBaseDataset(image_path, mask_path), cache_root
    )

    loaded = dataset[0]

    assert loaded["frangi_raster"].shape == (7, 4, 5)
    assert float(loaded["frangi_similarity"]) == pytest.approx(0.75)
    assert float(loaded["frangi_density"]) == pytest.approx(1 / 20)


class _SyntheticResidualDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        raster = torch.zeros(len(FRANGI_RASTER_CHANNELS), 2, 2)
        raster[FRANGI_RASTER_CHANNEL_INDEX["similarity"], 0, 0] = 0.8
        raster[FRANGI_RASTER_CHANNEL_INDEX["support"], 0, 0] = 1.0
        target = torch.zeros(1, 2, 2)
        target[0, 0, 0] = 1.0
        return {
            "case_name": f"case{index}.png",
            "image": torch.zeros(3, 2, 2),
            "mask": target,
            "frangi_raster": raster,
            "frangi_similarity": torch.tensor(0.8),
            "frangi_density": torch.tensor(0.25),
        }


class _FakeResidual(torch.nn.Module):
    def forward(self, images, rasters, output_size, accept_residual):
        batch = images.shape[0]
        baseline = torch.full((batch, 1, *output_size), -2.0, device=images.device)
        candidate = baseline.clone()
        candidate[:, :, 0, 0] = 2.0
        return {"baseline_logits": baseline, "candidate_logits": candidate}


class _FakeVerifiedResidual(torch.nn.Module):
    def forward(self, images, rasters, output_size, accept_residual):
        batch = images.shape[0]
        baseline = torch.full((batch, 1, *output_size), -2.0, device=images.device)
        residual = torch.zeros_like(baseline)
        residual[:, :, 0, 1] = 4.0
        candidate = baseline + residual
        support_fusion = torch.zeros_like(baseline)
        support_fusion[:, :, 0, :2] = 1.0
        score_fusion = torch.zeros_like(baseline)
        score_fusion[:, :, 0, 0] = 0.8
        score_fusion[:, :, 0, 1] = 0.2
        selected_fusion = torch.zeros_like(baseline)
        selected_fusion[:, :, 0, 1] = 1.0
        envelope_fusion = selected_fusion.clone()
        return {
            "baseline_logits": baseline,
            "candidate_logits": candidate,
            "residual_logits": residual,
            "evidence_score_fusion": score_fusion,
            "evidence_support_fusion": support_fusion,
            "evidence_selected_fusion": selected_fusion,
            "correction_envelope_fusion": envelope_fusion,
            "evidence_selected": selected_fusion,
            "correction_envelope": envelope_fusion,
        }


def test_loader_pipeline_runs_without_sam2_and_writes_gate_ready_rows(
    tmp_path: Path,
) -> None:
    loader = DataLoader(_SyntheticResidualDataset(), batch_size=2, shuffle=False)
    progress = tmp_path / "progress.jsonl"

    completed = evaluate_residual_loader(
        _FakeResidual(),
        loader,
        preprocessing=_identity_preprocessing(),
        source_groups={"case0.png": "wall0", "case1.png": "wall1"},
        dataset="synthetic",
        role="gate_calibration",
        fold="4",
        progress_path=progress,
        device="cpu",
        amp_dtype="none",
        show_progress=False,
    )
    rows = read_progress_rows(progress, expected_dataset="synthetic")

    assert completed == 2
    assert list(rows) == ["case0.png", "case1.png"]
    assert all(row["candidate_better"] == 1 for row in rows.values())
    assert all(row["fold"] == "4" for row in rows.values())
    assert all(set(DEFAULT_GATE_FEATURES).issubset(row) for row in rows.values())


def test_verified_loader_journals_and_finalizes_complete_selector_diagnostics(
    tmp_path: Path,
) -> None:
    loader = DataLoader(_SyntheticResidualDataset(), batch_size=2, shuffle=False)
    progress = tmp_path / "progress.jsonl"

    completed = evaluate_residual_loader(
        _FakeVerifiedResidual(),
        loader,
        preprocessing=_identity_preprocessing(),
        source_groups={"case0.png": "wall0", "case1.png": "wall1"},
        dataset="synthetic",
        role="gate_calibration",
        fold="4",
        progress_path=progress,
        device="cpu",
        amp_dtype="none",
        show_progress=False,
    )
    rows = read_progress_rows(progress, expected_dataset="synthetic")
    diagnostics = read_selector_diagnostics(
        progress, expected_dataset="synthetic"
    )

    assert completed == 2
    assert list(diagnostics) == ["case0.png", "case1.png"]
    assert all(
        diagnostic["evidence_score_source"] == "evidence_score_fusion"
        for diagnostic in diagnostics.values()
    )
    assert all(
        diagnostic["gate_spatial_support_source"]
        == "accepted_local_selector_output"
        for diagnostic in diagnostics.values()
    )
    # The local accepted support contains the only correction.  This verifies
    # that gate spatial summaries no longer use the raw cache support by default.
    assert all(
        row["support_correction_probability_mean"] > 0.0 for row in rows.values()
    )

    _ordered_finalize(
        output=tmp_path,
        dataset_name="synthetic",
        selected_names=["case0.png", "case1.png"],
        role="gate_calibration",
        causal_raster_override=False,
    )
    artifact = json.loads(
        (tmp_path / "selector_diagnostics.json").read_text(encoding="utf-8")
    )
    assert artifact["complete"] is True
    assert artifact["diagnostic_cases"] == 2
    assert artifact["missing_case_names"] == []
    assert not list(tmp_path.glob(".selector_diagnostics.json.*.tmp"))


def test_no_evidence_is_canonical_and_override_is_one_way(tmp_path: Path) -> None:
    assert resolve_raster_condition("correct", None, allow_causal_override=False) == (
        "correct",
        False,
    )
    assert resolve_raster_condition(
        "correct", "no_evidence", allow_causal_override=True
    ) == ("no_evidence", True)
    with pytest.raises(ValueError, match="must never receive"):
        resolve_raster_condition("no_evidence", "correct", allow_causal_override=True)
    with pytest.raises(ValueError, match="allow-input-ablation-raster-override"):
        resolve_raster_condition("correct", "no_evidence", allow_causal_override=False)

    progress = tmp_path / "no_evidence.jsonl"
    evaluate_residual_loader(
        _FakeResidual(),
        DataLoader(_SyntheticResidualDataset(), batch_size=2),
        preprocessing=_identity_preprocessing(),
        source_groups={"case0.png": "wall0", "case1.png": "wall1"},
        dataset="synthetic_no_evidence",
        role="development",
        fold="",
        progress_path=progress,
        device="cpu",
        amp_dtype="none",
        raster_condition="no_evidence",
        show_progress=False,
    )
    rows = read_progress_rows(progress, expected_dataset="synthetic_no_evidence")
    for row in rows.values():
        assert row["frangi_similarity_support_mean"] == 0.0
        assert row["frangi_density"] == 0.0
        assert row["support_correction_probability_mean"] == 0.0


@pytest.mark.parametrize(
    "override_flag",
    (
        "--allow-input-ablation-raster-override",
        "--allow-causal-raster-override",
    ),
)
def test_evaluation_cli_accepts_canonical_input_ablation_flag_and_legacy_alias(
    monkeypatch: pytest.MonkeyPatch,
    override_flag: str,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_frangi_graph_residual.py",
            "--data-root",
            "/data",
            "--list-file",
            "/data/list.txt",
            "--split",
            "train",
            "--dataset-name",
            "synthetic",
            "--graph-cache",
            "/cache",
            "--sam2-checkpoint",
            "/weights/sam2.pt",
            "--baseline-checkpoint",
            "/weights/baseline.pt",
            "--residual-checkpoint",
            "/weights/residual.pt",
            "--output",
            "/output",
            "--role",
            "development",
            override_flag,
        ],
    )

    assert parse_args().allow_causal_raster_override is True


def test_group_assignments_verify_physical_fold(tmp_path: Path) -> None:
    path = tmp_path / "groups.csv"
    path.write_text(
        "sample_name,physical_group,oof_fold\ncase.png,wall_01,4\n",
        encoding="utf-8",
    )
    assert load_group_assignments(
        path, selected_names=["case.png"], expected_fold="4"
    ) == {"case.png": "wall_01"}
    with pytest.raises(ValueError, match="not all in held-out fold"):
        load_group_assignments(path, selected_names=["case.png"], expected_fold="3")
