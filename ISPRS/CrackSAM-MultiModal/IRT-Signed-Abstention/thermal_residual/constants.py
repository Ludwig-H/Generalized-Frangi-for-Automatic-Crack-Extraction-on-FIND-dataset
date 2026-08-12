"""Constantes gelées du protocole IRT-Signed-Abstention.

Tout ce qui est figé avant le premier run vit ici : ordre des actions, ordre des
canaux d'évidence, versions de schéma des caches, noms canoniques des bras.
Changer une de ces valeurs invalide les caches et doit s'accompagner d'une montée
de version de schéma.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------- #
# Actions du correcteur — §5.2 de la spécification
# --------------------------------------------------------------------------- #

ACTION_REINFORCE: Final[int] = 0
ACTION_SUPPRESS: Final[int] = 1
ACTION_ABSTAIN: Final[int] = 2

ACTION_NAMES: Final[tuple[str, ...]] = ("reinforce", "suppress", "abstain")

# --------------------------------------------------------------------------- #
# Canaux d'évidence thermique — §4.2
# --------------------------------------------------------------------------- #

#: Ordre canonique des quatre canaux d'évidence. Il est écrit dans chaque cache
#: et vérifié à la relecture.
EVIDENCE_CHANNELS: Final[tuple[str, ...]] = (
    "similarity_dark",
    "similarity_bright",
    "similarity_max",
    "support_union",
)

#: Canaux dérivés de la baseline, concaténés devant l'évidence — §5.1.
BASELINE_CHANNELS: Final[tuple[str, ...]] = (
    "logit_scaled",
    "probability",
    "entropy",
)

#: Nombre total de canaux d'entrée du correcteur.
CORRECTOR_INPUT_CHANNELS: Final[int] = len(BASELINE_CHANNELS) + len(EVIDENCE_CHANNELS)

#: Borne utilisée pour normaliser le logit baseline : ``clip(z0,-10,10)/10``.
LOGIT_CLIP: Final[float] = 10.0

# --------------------------------------------------------------------------- #
# Versions de schéma
# --------------------------------------------------------------------------- #

DATASET_MANIFEST_VERSION: Final[int] = 1
BASELINE_CACHE_VERSION: Final[int] = 1
THERMAL_CACHE_VERSION: Final[int] = 1
CHECKPOINT_FORMAT_VERSION: Final[int] = 1

# --------------------------------------------------------------------------- #
# Jeu de données
# --------------------------------------------------------------------------- #

#: Sous-dossiers d'IRT-Crack tels que distribués sur Zenodo 11624965, avec les
#: variantes de nommage rencontrées dans les ré-hébergements (Google Drive du
#: benchmark IRFusionFormer, qui renomme avec des tirets bas).
VISIBLE_DIRECTORY_CANDIDATES: Final[tuple[str, ...]] = (
    "01-Visible images",
    "01_Visible_Image",
    "00_Visible_Image",
    "01-Visible image",
    "Visible",
)
INFRARED_DIRECTORY_CANDIDATES: Final[tuple[str, ...]] = (
    "02-Infrared images",
    "02_Infrared_Image",
    "02-Infrared image",
    "Infrared",
)
FUSION_DIRECTORY_CANDIDATES: Final[tuple[str, ...]] = (
    "03-Fusion(50IRT) images",
    "03_Fusion_Image",
    "03-Fusion images",
    "Fusion",
)
GROUND_TRUTH_DIRECTORY_CANDIDATES: Final[tuple[str, ...]] = (
    "04-Ground truth",
    "04_Ground_Truth",
    "04-Ground Truth",
    "GroundTruth",
)

#: Extensions d'image acceptées, en minuscules.
IMAGE_SUFFIXES: Final[tuple[str, ...]] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

#: Résolution native annoncée par le papier (H, W).
NATIVE_SIZE: Final[tuple[int, int]] = (480, 640)

#: Taille d'entrée de CrackSAM 2 sur Khánh Hà : les images y sont redimensionnées
#: à 448×448 avant l'encodeur (``cracksam2/data.py``). La reproduire garantit que
#: la baseline gelée voit la même distribution qu'à l'entraînement.
BASELINE_INPUT_SIZE: Final[tuple[int, int]] = (448, 448)

SPLIT_TRAIN: Final[str] = "train"
SPLIT_VALIDATION: Final[str] = "validation"
SPLIT_TEST: Final[str] = "test"
SPLITS: Final[tuple[str, ...]] = (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_TEST)

#: Strates horaires admissibles. ``unknown`` est la valeur par défaut : la
#: spécification interdit de deviner la strate à partir de la luminosité.
TIME_STRATA: Final[tuple[str, ...]] = ("morning", "noon", "evening", "unknown")

# --------------------------------------------------------------------------- #
# Bras d'ablation — §8
# --------------------------------------------------------------------------- #

ARM_BASELINE: Final[str] = "baseline"
ARM_RGB_RECALIBRATION: Final[str] = "rgb_recalibration"
ARM_FRANGI_SIGNED_ABSTENTION: Final[str] = "frangi_signed_abstention"
ARM_FRANGI_PERMUTED: Final[str] = "frangi_permuted"
ARM_RAW_THERMAL: Final[str] = "raw_thermal"
ARM_FRANGI_NO_ABSTENTION: Final[str] = "frangi_no_abstention"
ARM_POSITIVE_ONLY: Final[str] = "positive_only"
ARM_FRANGI_WEIGHTED: Final[str] = "frangi_difficulty_weighted"
ARM_FRANGI_WEIGHTED_PERMUTED: Final[str] = "frangi_difficulty_weighted_permuted"

#: Identifiants courts du tableau de résultats, dans l'ordre du protocole.
ARM_IDENTIFIERS: Final[dict[str, str]] = {
    "A0": ARM_BASELINE,
    "A1": ARM_RGB_RECALIBRATION,
    "A2": ARM_FRANGI_SIGNED_ABSTENTION,
    "A3": ARM_FRANGI_PERMUTED,
    "A4": ARM_RAW_THERMAL,
    "A5": ARM_FRANGI_NO_ABSTENTION,
    "A6": ARM_POSITIVE_ONLY,
    "A7": ARM_FRANGI_WEIGHTED,
    "A8": ARM_FRANGI_WEIGHTED_PERMUTED,
}

#: Source des canaux d'évidence donnés au correcteur.
EVIDENCE_SOURCE_FRANGI: Final[str] = "frangi"
EVIDENCE_SOURCE_ZEROS: Final[str] = "zeros"
EVIDENCE_SOURCE_RAW_THERMAL: Final[str] = "raw_thermal"
EVIDENCE_SOURCES: Final[tuple[str, ...]] = (
    EVIDENCE_SOURCE_FRANGI,
    EVIDENCE_SOURCE_ZEROS,
    EVIDENCE_SOURCE_RAW_THERMAL,
)

#: Familles de tête de décision.
HEAD_SIGNED_ABSTENTION: Final[str] = "signed_abstention"
HEAD_SIGNED: Final[str] = "signed"
HEAD_POSITIVE_ONLY: Final[str] = "positive_only"
HEADS: Final[tuple[str, ...]] = (
    HEAD_SIGNED_ABSTENTION,
    HEAD_SIGNED,
    HEAD_POSITIVE_ONLY,
)

SCOPE_GLOBAL: Final[str] = "global"
SCOPE_EVIDENCE_UNION: Final[str] = "evidence_union"
SCOPES: Final[tuple[str, ...]] = (SCOPE_GLOBAL, SCOPE_EVIDENCE_UNION)

# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #

#: Tolérances rapportées. ``3`` est la tolérance primaire choisie pour cette
#: campagne : c'est celle de la perte, et le rapport GeoLoRA a montré qu'à
#: tolérance nulle l'IoU mesure surtout un biais de largeur de frontière.
TOLERANCES: Final[tuple[int, ...]] = (0, 1, 2, 3, 5, 8)
PRIMARY_TOLERANCE: Final[int] = 3

#: Seuil de binarisation gelé, identique pour tous les bras.
DECISION_THRESHOLD: Final[float] = 0.5

BOOTSTRAP_REPLICATES: Final[int] = 10_000

__all__ = [name for name in dir() if not name.startswith("_")]
