from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch
import torch.nn as nn

CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

import train_frangi_graph_residual as training  # noqa: E402
from cracksam2.graph_cache import (  # noqa: E402
    GRAPH_CACHE_MANIFEST,
    aggregate_graph_cache_records,
    build_graph_cache_contract,
    graph_cache_record,
    graph_cache_relative_path,
    save_graph_raster_atomic,
    sha256_file,
    write_json_atomic,
)
from cracksam2.graph_types import (  # noqa: E402
    FRANGI_RASTER_CHANNEL_INDEX,
    FRANGI_RASTER_CHANNELS,
    FrangiRasterSample,
)
from cracksam2.oof import (  # noqa: E402
    strict_oof_training_contract,
    validate_oof_run_contract,
)
from cracksam2.residual_data import (  # noqa: E402
    FrangiGraphRasterDataset,
    fit_frangi_raster_normalization,
    load_graph_cache_index,
    load_group_safe_fold,
    load_or_fit_frangi_raster_normalization,
)


def _write_training_fixture(tmp_path: Path, magnitudes: list[float]):
    data = tmp_path / "data"
    image_dir = data / "images"
    mask_dir = data / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    names = [f"sample_{index}.png" for index in range(len(magnitudes))]
    split = tmp_path / "train.txt"
    split.write_text("\n".join(names) + "\n", encoding="utf-8")
    cache = tmp_path / "cache"
    records = []
    for index, (name, magnitude) in enumerate(zip(names, magnitudes)):
        image_path = image_dir / name
        mask_path = mask_dir / name
        image = np.full((8, 8, 3), 20 + index, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:, :4] = 255
        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)

        raster = np.zeros((len(FRANGI_RASTER_CHANNELS), 8, 8), dtype=np.float32)
        raster[FRANGI_RASTER_CHANNEL_INDEX["similarity"]] = 0.25
        raster[FRANGI_RASTER_CHANNEL_INDEX["support"]] = 1.0
        raster[FRANGI_RASTER_CHANNEL_INDEX["hessian_magnitude"]] = magnitude
        raster[FRANGI_RASTER_CHANNEL_INDEX["winning_scale"]] = 3.0
        raster[FRANGI_RASTER_CHANNEL_INDEX["orientation_cos2"]] = 1.0
        raster[FRANGI_RASTER_CHANNEL_INDEX["distance_to_skeleton"]] = 0.5
        sample = FrangiRasterSample(
            case_name=name,
            raster=raster,
            image_sha256=sha256_file(image_path),
            mask_sha256=sha256_file(mask_path),
            elapsed_seconds=0.01,
        )
        cache_path = cache / graph_cache_relative_path(name)
        save_graph_raster_atomic(cache_path, sample)
        records.append(graph_cache_record(cache, cache_path, sample))

    contract = build_graph_cache_contract(
        names,
        image_size=(8, 8),
        noise_mode="original",
        frangi_parameters={"scales": [1.0, 3.0], "R": 3, "K": 1},
        extractor_sha256="e" * 64,
    )
    manifest = {
        **contract,
        "status": "complete",
        "entries": records,
        "aggregate": aggregate_graph_cache_records(records),
    }
    write_json_atomic(cache / GRAPH_CACHE_MANIFEST, manifest)

    folds = tmp_path / "folds.csv"
    with folds.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=("sample_name", "source_family", "physical_group", "oof_fold"),
        )
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow(
                {
                    "sample_name": name,
                    "source_family": "synthetic",
                    "physical_group": f"group_{index}",
                    "oof_fold": index % 5,
                }
            )
    return data, split, folds, cache, names


def test_fold_selection_and_train_only_raster_normalization(tmp_path: Path) -> None:
    data, split, folds, cache, names = _write_training_fixture(
        tmp_path, [0.0, 1.0, 3.0, 7.0, 10_000.0]
    )
    selected = load_group_safe_fold(folds, names, held_out_fold=4)
    assert selected.validation_names == ("sample_4.png",)
    assert not set(selected.train_groups).intersection(selected.validation_groups)

    index = load_graph_cache_index(
        cache, names, image_size=(8, 8), noise_mode="original"
    )
    normalization = fit_frangi_raster_normalization(index, selected.train_names)
    expected_train_mean = float(np.mean(np.log1p([0.0, 1.0, 3.0, 7.0])))
    magnitude_index = FRANGI_RASTER_CHANNEL_INDEX["hessian_magnitude"]
    assert normalization.offset[magnitude_index] == pytest.approx(expected_train_mean)
    assert normalization.max_scale == 3.0
    assert normalization.fit_sample_count == 4
    assert normalization.preprocessing_contract()["log1p_channels"] == [
        "hessian_magnitude"
    ]
    assert len(normalization.offset) == len(normalization.scale) == 7
    assert all(value > 0.0 for value in normalization.scale)

    normalization_path = tmp_path / "normalization.json"
    persisted = load_or_fit_frangi_raster_normalization(
        normalization_path, index, selected.train_names
    )
    assert persisted == normalization
    dataset = FrangiGraphRasterDataset(
        data,
        split,
        selected.validation_names,
        index,
        normalization,
        verify_source_files=True,
    )
    sample = dataset[0]
    raster = sample["frangi_raster"]
    assert raster.shape == (7, 8, 8)
    assert torch.isfinite(raster).all()
    assert float(raster[FRANGI_RASTER_CHANNEL_INDEX["winning_scale"]].max()) == 1.0
    # A huge held-out value was not allowed to alter the training-fold moments.
    assert float(raster[magnitude_index].mean()) > 5.0


def test_fold_selection_can_reserve_an_external_training_fold(
    tmp_path: Path,
) -> None:
    _, _, folds, _, names = _write_training_fixture(tmp_path, [1.0] * 5)

    selected = load_group_safe_fold(
        folds,
        names,
        held_out_fold=0,
        exclude_training_folds=(4,),
    )

    assert selected.validation_names == ("sample_0.png",)
    assert selected.train_names == (
        "sample_1.png",
        "sample_2.png",
        "sample_3.png",
    )
    assert selected.excluded_training_folds == (4,)
    assert selected.excluded_training_names == ("sample_4.png",)
    assert selected.excluded_training_groups == ("group_4",)
    assert not set(selected.train_groups).intersection(
        selected.excluded_training_groups
    )

    # Omitting the option preserves the original four-fold training split.
    default = load_group_safe_fold(folds, names, held_out_fold=0)
    assert default.train_names == tuple(names[1:])
    assert default.excluded_training_folds == ()
    assert default.excluded_training_names == ()


@pytest.mark.parametrize(
    ("held_out", "excluded", "message"),
    [
        (0, (4, 4), "duplicate"),
        (0, (0,), "already excluded"),
        (0, (1, 2, 3, 4), "training split is empty"),
    ],
)
def test_external_fold_exclusions_reject_invalid_or_empty_splits(
    tmp_path: Path,
    held_out: int,
    excluded: tuple[int, ...],
    message: str,
) -> None:
    _, _, folds, _, names = _write_training_fixture(tmp_path, [1.0] * 5)
    with pytest.raises(ValueError, match=message):
        load_group_safe_fold(
            folds,
            names,
            held_out_fold=held_out,
            exclude_training_folds=excluded,
        )


@pytest.mark.parametrize(
    ("held_out_fold", "expected_training", "expected_additional_exclusions"),
    [
        (0, [1, 2, 3], [4]),
        (1, [0, 2, 3], [4]),
        (2, [0, 1, 3], [4]),
        (3, [0, 1, 2], [4]),
        (4, [0, 1, 2, 3], []),
    ],
)
def test_strict_oof_contract_reserves_fold_four_from_gate_fit_training(
    held_out_fold: int,
    expected_training: list[int],
    expected_additional_exclusions: list[int],
) -> None:
    contract = strict_oof_training_contract(held_out_fold)

    assert contract["training_folds"] == expected_training
    assert (
        contract["additional_excluded_training_folds"]
        == expected_additional_exclusions
    )
    assert contract["evaluation_role"] == (
        "gate_calibration" if held_out_fold == 4 else "gate_fit"
    )


def test_exclude_training_fold_cli_is_automatic_and_cannot_override_policy() -> None:
    required = [
        "--data-root",
        "data",
        "--fold",
        "0",
        "--graph-cache",
        "cache",
        "--sam2-checkpoint",
        "sam.pt",
        "--baseline-checkpoint",
        "baseline.pt",
        "--output",
        "output",
    ]
    args = training.parse_args(required)
    training._validate_args(args)
    assert args.exclude_training_folds == [4]

    explicit = training.parse_args([*required, "--exclude-training-fold", "4"])
    training._validate_args(explicit)
    assert explicit.exclude_training_folds == [4]

    duplicate = training.parse_args(
        [*required, "--exclude-training-fold", "4", "--exclude-training-fold", "4"]
    )
    with pytest.raises(ValueError, match="same fold"):
        training._validate_args(duplicate)

    override = training.parse_args([*required, "--exclude-training-fold", "3"])
    with pytest.raises(ValueError, match="cannot override"):
        training._validate_args(override)

    fold_four = training.parse_args(
        [
            *required[:3],
            "4",
            *required[4:],
        ]
    )
    training._validate_args(fold_four)
    assert fold_four.exclude_training_folds == []

    fold_four_override = training.parse_args(
        [
            *required[:3],
            "4",
            *required[4:],
            "--exclude-training-fold",
            "0",
        ]
    )
    with pytest.raises(ValueError, match=r"requires \[\]"):
        training._validate_args(fold_four_override)


def test_verified_adapter_cli_refuses_structurally_empty_training() -> None:
    required = [
        "--data-root",
        "data",
        "--fold",
        "0",
        "--graph-cache",
        "cache",
        "--sam2-checkpoint",
        "sam.pt",
        "--baseline-checkpoint",
        "baseline.pt",
        "--output",
        "output",
    ]
    verified = training.parse_args(required)
    training._validate_args(verified)
    assert verified.adapter_mode == "verified_local_v1"
    assert verified.evidence_loss_weight == 0.25

    explicit_units = training.parse_args(
        [
            *required,
            "--profile-radii-feature-cells",
            "2.0",
            "4.0",
            "--evidence-dilation-feature-cells",
            "3",
        ]
    )
    training._validate_args(explicit_units)
    assert explicit_units.profile_radii == [2.0, 4.0]
    assert explicit_units.evidence_dilation == 3

    legacy_aliases = training.parse_args(
        [
            *required,
            "--profile-radii",
            "2.0",
            "4.0",
            "--evidence-dilation",
            "3",
        ]
    )
    training._validate_args(legacy_aliases)
    assert legacy_aliases.profile_radii == explicit_units.profile_radii
    assert legacy_aliases.evidence_dilation == explicit_units.evidence_dilation

    empty = training.parse_args([*required, "--raster-condition", "no_evidence"])
    with pytest.raises(ValueError, match="cannot be trained with no_evidence"):
        training._validate_args(empty)

    legacy = training.parse_args(
        [
            *required,
            "--adapter-mode",
            "legacy_raster_v1",
            "--evidence-loss-weight",
            "0",
            "--raster-condition",
            "no_evidence",
        ]
    )
    training._validate_args(legacy)


def test_selector_grid_contract_records_feature_cell_geometry() -> None:
    contract = training._selector_grid_contract(
        input_image_size=(448, 448),
        fusion_grid_size=(256, 224),
    )

    assert contract == {
        "source": "SAM2ImageFeatures.high_resolution_features[0]",
        "parameter_unit": "hiera_high_resolution_feature_cells",
        "input_image_size_pixels": [448, 448],
        "fusion_grid_size_feature_cells": [256, 224],
        "effective_stride_input_pixels_per_feature_cell": [1.75, 2.0],
    }


def test_training_help_names_selector_feature_cell_units() -> None:
    result = subprocess.run(
        [sys.executable, str(CRACKSAM_ROOT / "train_frangi_graph_residual.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    help_text = " ".join(result.stdout.split())
    assert "--profile-radii-feature-cells" in help_text
    assert "--evidence-dilation-feature-cells" in help_text
    assert "first SAM 2/Hiera high-resolution feature grid" in help_text


def test_output_reuse_and_resume_config_are_guarded(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    training._prepare_output_directory(fresh, resume=False)
    assert fresh.is_dir()
    (fresh / "partial.csv").write_text("partial\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        training._prepare_output_directory(fresh, resume=False)
    with pytest.raises(FileNotFoundError, match="config.json"):
        training._prepare_output_directory(fresh, resume=True)

    config_path = fresh / "config.json"
    write_json_atomic(config_path, {"contract": "original"})
    training._prepare_output_directory(fresh, resume=True)
    training._publish_or_validate_config(
        config_path, {"contract": "original"}, resume=True
    )
    before = config_path.read_bytes()
    with pytest.raises(RuntimeError, match="incompatible"):
        training._publish_or_validate_config(
            config_path, {"contract": "changed"}, resume=True
        )
    assert config_path.read_bytes() == before


def test_fold_assignment_rejects_a_physical_group_crossing_parts(
    tmp_path: Path,
) -> None:
    _, _, folds, _, names = _write_training_fixture(tmp_path, [1.0] * 5)
    rows = list(csv.DictReader(folds.open(encoding="utf-8")))
    rows[1]["physical_group"] = rows[0]["physical_group"]
    with folds.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="crosses folds"):
        load_group_safe_fold(folds, names, held_out_fold=4)


class _TinyResidual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.baseline = nn.Conv2d(3, 1, 1)
        self.baseline.requires_grad_(False)
        self.adapter = nn.Conv2d(7, 1, 1)

    def train(self, mode: bool = True):
        super().train(mode)
        self.baseline.eval()
        return self

    def forward(self, images, rasters, output_size):
        del output_size
        with torch.no_grad():
            baseline_logits = self.baseline(images)
        residual = self.adapter(rasters)
        return {
            "baseline_logits": baseline_logits,
            "candidate_logits": baseline_logits + residual,
        }


def test_cpu_training_and_atomic_resume_checkpoint_smoke(tmp_path: Path) -> None:
    _, _, _, cache, names = _write_training_fixture(tmp_path, [1.0] * 5)
    index = load_graph_cache_index(
        cache, names, image_size=(8, 8), noise_mode="original"
    )
    normalization = fit_frangi_raster_normalization(index, names[:4])
    model = _TinyResidual().train()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    baseline_before = {
        name: value.detach().clone()
        for name, value in model.baseline.state_dict().items()
    }
    adapter_before = model.adapter.weight.detach().clone()
    batch = {
        "image": torch.rand(2, 3, 8, 8),
        "mask": torch.randint(0, 2, (2, 1, 8, 8), dtype=torch.float32),
        "frangi_raster": torch.rand(2, 7, 8, 8),
    }
    values = training.train_batch(
        model,
        batch,
        optimizer,
        scaler,
        device=torch.device("cpu"),
        amp_dtype="none",
        loss_parameters={
            "ce_weight": 0.2,
            "topology_weight": 0.1,
            "safety_weight": 1.0,
            "safety_margin": 0.0,
            "skeleton_iterations": 2,
        },
        gradient_clip=1.0,
    )
    assert np.isfinite(values["loss"])
    assert not torch.equal(adapter_before, model.adapter.weight)
    for name, value in model.baseline.state_dict().items():
        torch.testing.assert_close(value, baseline_before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.baseline.parameters())

    oof_training = strict_oof_training_contract(4)
    contract = {
        "contract_version": 2,
        "held_out_fold": 4,
        "oof_training": oof_training,
        "excluded_training_folds": [],
        "all_folds_excluded_from_training": [4],
    }
    assert validate_oof_run_contract(contract) == oof_training
    contract_sha = training._json_sha256(contract)
    payload = training.residual_checkpoint_payload(
        model,
        high_resolution_channels=(3,),
        hidden_channels=1,
        baseline_checkpoint_sha256="b" * 64,
        graph_cache=index,
        normalization=normalization,
        run_contract=contract,
        run_contract_sha256=contract_sha,
        optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict(),
        epoch=2,
        next_batch=3,
        global_step=11,
        best_validation_iou=0.4,
    )
    assert payload["format_version"] == 1
    assert payload["training_state"] == "incomplete"
    assert set(payload["residual_adapter"]) == set(model.adapter.state_dict())
    assert all(not key.startswith("adapter.") for key in payload["residual_adapter"])
    assert payload["residual"]["raster_channels"] == 7
    assert payload["graph_cache"]["channels"] == list(FRANGI_RASTER_CHANNELS)
    assert payload["raster_preprocessing"]["log1p_channels"] == ["hessian_magnitude"]
    destination = tmp_path / "latest.pt"
    training._atomic_torch_save(payload, destination)
    saved_adapter = {
        name: value.clone() for name, value in payload["residual_adapter"].items()
    }
    with torch.no_grad():
        model.adapter.weight.add_(10.0)
    state = training.restore_residual_checkpoint(
        destination,
        model,
        optimizer,
        scaler,
        expected_run_contract=contract,
        expected_run_contract_sha256=contract_sha,
    )
    assert state["epoch"] == 2
    for name, value in model.adapter.state_dict().items():
        torch.testing.assert_close(value, saved_adapter[name])
    assert list(tmp_path.glob(".latest.pt.*.tmp")) == []


def test_verified_checkpoint_records_explicit_selector_grid_units(tmp_path: Path) -> None:
    _, _, _, cache, names = _write_training_fixture(tmp_path, [1.0] * 5)
    index = load_graph_cache_index(
        cache, names, image_size=(8, 8), noise_mode="original"
    )
    normalization = fit_frangi_raster_normalization(index, names[:4])
    model = _TinyResidual()
    model.adapter_mode = "verified_local_v1"
    model.adapter.profile_radii = (1.5, 3.0)
    model.adapter.evidence_dilation = 2
    model.adapter.evidence_threshold = 0.5
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    selector_grid = training._selector_grid_contract(
        input_image_size=(448, 448), fusion_grid_size=(256, 256)
    )
    contract = {
        "contract_version": 2,
        "held_out_fold": 4,
        "oof_training": strict_oof_training_contract(4),
        "excluded_training_folds": [],
        "all_folds_excluded_from_training": [4],
        "residual": {
            "profile_radii_feature_cells": [1.5, 3.0],
            "evidence_dilation_feature_cells": 2,
            "selector_grid": selector_grid,
        },
    }
    payload = training.residual_checkpoint_payload(
        model,
        high_resolution_channels=(3,),
        hidden_channels=1,
        baseline_checkpoint_sha256="b" * 64,
        graph_cache=index,
        normalization=normalization,
        run_contract=contract,
        run_contract_sha256=training._json_sha256(contract),
        optimizer_state=optimizer.state_dict(),
        scaler_state=scaler.state_dict(),
        epoch=1,
        next_batch=0,
        global_step=1,
        best_validation_iou=0.0,
    )

    assert payload["format_version"] == 1
    assert payload["residual"]["profile_radii"] == [1.5, 3.0]
    assert payload["residual"]["profile_radii_feature_cells"] == [1.5, 3.0]
    assert payload["residual"]["evidence_dilation"] == 2
    assert payload["residual"]["evidence_dilation_feature_cells"] == 2
    assert payload["residual"]["selector_grid"] == selector_grid
