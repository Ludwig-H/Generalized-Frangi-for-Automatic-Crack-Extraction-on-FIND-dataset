from __future__ import annotations

import sys
from pathlib import Path


CRACKSAM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CRACKSAM_ROOT))

from protocol.build_next_protocol import (  # noqa: E402
    assign_group_folds,
    physical_source_group,
)


def test_known_crop_names_map_to_physical_source() -> None:
    assert (
        physical_source_group("CRACK500_20160222_080850_1281_361.jpg")
        == "CRACK500_20160222_080850"
    )
    assert (
        physical_source_group("Rissbilder_for_Florian_9S6A2782_20_1_2_3.jpg")
        == "Rissbilder_for_Florian_9S6A2782"
    )
    assert (
        physical_source_group("noncrack_noncrack_concrete_wall_9_27.jpg.jpg")
        == "noncrack_noncrack_concrete_wall_9"
    )
    assert physical_source_group("CFD_002.jpg") == "CFD_002"
    assert physical_source_group("017_24.jpg") == "017"


def test_group_fold_assignment_is_stable_and_never_splits_a_source() -> None:
    names = [
        "CRACK500_20160222_080850_1_1.jpg",
        "CRACK500_20160222_080850_2_2.jpg",
        "CRACK500_20160222_080851_1_1.jpg",
        "CRACK500_20160222_080852_1_1.jpg",
        "CFD_001.jpg",
        "CFD_002.jpg",
        "CFD_003.jpg",
    ]
    first = assign_group_folds(names, folds=3, seed=3407)
    repeated = assign_group_folds(reversed(names), folds=3, seed=3407)
    assert first == repeated
    assert len(first) == 6
    assert set(first.values()) == {0, 1, 2}
