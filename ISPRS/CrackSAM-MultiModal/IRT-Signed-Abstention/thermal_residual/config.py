"""Chargement et validation des configurations YAML d'un bras.

Un fichier de configuration décrit **un bras** et rien d'autre : sa source
d'évidence, sa tête, sa portée, et les hyperparamètres partagés. Les
hyperparamètres sont volontairement répétés à l'identique dans chaque fichier
plutôt qu'hérités : un réglage par bras se verrait alors dans le diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .constants import (
    ARM_IDENTIFIERS,
    EVIDENCE_SOURCES,
    HEADS,
    SCOPES,
)
from .losses import LossWeights
from .provenance import sha256_json
from .training import ArmSpecification, TrainingConfig


class ConfigError(ValueError):
    """Une configuration de bras est incomplète ou incohérente."""


@dataclass(frozen=True)
class ArmConfig:
    """Une configuration complète, prête à être exécutée."""

    identifier: str
    arm: ArmSpecification
    training: TrainingConfig
    loss: LossWeights
    thermal: dict[str, Any]
    source: Path | None = None

    def digest(self) -> str:
        return sha256_json(self.to_json())

    def to_json(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "arm": {
                "name": self.arm.name,
                "evidence_source": self.arm.evidence_source,
                "permuted": self.arm.permuted,
                "trained": self.arm.trained,
                "difficulty_weighted": self.arm.difficulty_weighted,
                "model": dict(self.arm.model),
            },
            "training": self.training.to_json(),
            "loss": self.loss.to_json(),
            "thermal": dict(self.thermal),
        }


def load_arm_config(path: str | Path) -> ArmConfig:
    """Lit un YAML de bras et rejette toute valeur hors domaine."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return parse_arm_config(payload, source=source)


def parse_arm_config(payload: Mapping[str, Any], *, source: Path | None = None) -> ArmConfig:
    identifier = str(payload.get("identifier", "")).strip()
    if identifier not in ARM_IDENTIFIERS:
        raise ConfigError(
            f"identifiant de bras inconnu : {identifier!r} (attendu {sorted(ARM_IDENTIFIERS)})"
        )
    expected_name = ARM_IDENTIFIERS[identifier]
    name = str(payload.get("name", expected_name))
    if name != expected_name:
        raise ConfigError(
            f"le bras {identifier} doit s'appeler {expected_name!r}, pas {name!r}"
        )

    arm_payload = dict(payload.get("arm", {}) or {})
    evidence_source = str(arm_payload.get("evidence_source", "frangi"))
    if evidence_source not in EVIDENCE_SOURCES:
        raise ConfigError(f"source d'évidence inconnue : {evidence_source!r}")

    model = dict(arm_payload.get("model", {}) or {})
    head = str(model.get("head", "signed_abstention"))
    if head not in HEADS:
        raise ConfigError(f"tête inconnue : {head!r}")
    scope = str(model.get("correction_scope", "global"))
    if scope not in SCOPES:
        raise ConfigError(f"portée inconnue : {scope!r}")

    specification = ArmSpecification(
        name=name,
        evidence_source=evidence_source,
        permuted=bool(arm_payload.get("permuted", False)),
        trained=bool(arm_payload.get("trained", True)),
        difficulty_weighted=bool(arm_payload.get("difficulty_weighted", False)),
        model=model,
    )
    return ArmConfig(
        identifier=identifier,
        arm=specification,
        training=TrainingConfig.from_mapping(payload.get("training")),
        loss=LossWeights.from_mapping(payload.get("loss")),
        thermal=dict(payload.get("thermal", {}) or {}),
        source=source,
    )


def load_ablation_matrix(path: str | Path) -> dict[str, Any]:
    """Lit ``configs/ablation_matrix.yaml``."""

    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    arms = payload.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ConfigError(f"{source} ne déclare aucun bras")
    resolved: list[dict[str, Any]] = []
    for entry in arms:
        config_path = (source.parent / str(entry["config"])).resolve()
        if not config_path.is_file():
            raise ConfigError(f"configuration de bras introuvable : {config_path}")
        resolved.append({"identifier": str(entry["identifier"]), "config": config_path})
    return {
        "arms": resolved,
        "seeds": [int(value) for value in payload.get("seeds", (13, 37, 73))],
        "comparisons": [str(value) for value in payload.get("comparisons", ())],
    }


__all__ = ["ArmConfig", "ConfigError", "load_ablation_matrix", "load_arm_config", "parse_arm_config"]
