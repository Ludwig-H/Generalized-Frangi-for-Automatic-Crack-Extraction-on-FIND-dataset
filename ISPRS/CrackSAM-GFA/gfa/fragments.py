"""Proposition de fragments courts orientés à partir de l'évidence.

Contrainte C4 du pré-enregistrement : le support candidat est l'**union** des
supports de plusieurs sources, chacune seuillée par un seuil **absolu** gelé, et
il peut être vide. Ce n'est jamais le seul support Frangi — sur ``DJ_Wall_66``
l'historique est dominé par le linteau sombre et manque la fissure que la voie
sans Frangi retrouve (``AP2=0,198``, rappel ``0,868``).

Un fragment est l'unité de décision de l'arbitre. C'est la plus petite entité
qui possède une tangente, une longueur et une cohérence d'orientation, donc la
seule à laquelle l'action « s'abstenir » soit vérifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .evidence import Evidence, SUPPORT_SOURCES

#: Longueur géodésique cible d'un fragment, en pixels à 448.
TARGET_LENGTH: Final[int] = 20
#: En deçà, une branche isolée est du bruit de squelette et n'est pas proposée.
MIN_LENGTH: Final[int] = 6
#: Composantes de support plus petites que cela : supprimées avant squelette.
MIN_SUPPORT_AREA: Final[int] = 12
#: Bornes du rayon de corridor, en pixels.
MIN_RADIUS: Final[float] = 1.5
MAX_RADIUS: Final[float] = 6.0

_NEIGHBOURS: Final[np.ndarray] = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int64,
)


@dataclass
class Fragment:
    """Une bande courte orientée, avec son corridor fixé **sans** GT."""

    pixels: np.ndarray  # (n, 2) int32, ordonnés le long de l'axe
    radius: float
    sources: tuple[str, ...]  # sources dont le support a produit ce fragment

    _cached_corridor: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def length(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        return self.pixels[0], self.pixels[-1]

    def tangent(self) -> np.ndarray:
        """Tangente moyenne unitaire ``(dy, dx)``, orientée par les extrémités."""

        if self.length < 2:
            return np.array([0.0, 0.0], dtype=np.float32)
        delta = (self.pixels[-1] - self.pixels[0]).astype(np.float32)
        norm = float(np.hypot(delta[0], delta[1]))
        if norm < 1e-6:
            return np.array([0.0, 0.0], dtype=np.float32)
        return (delta / norm).astype(np.float32)

    def straightness(self) -> float:
        """Distance des extrémités divisée par la longueur géodésique.

        Vaut ``1`` pour un segment droit et diminue avec la courbure. Une
        frontière d'ombre est typiquement longue et très droite ; une fissure
        l'est moins. C'est une feature, pas une décision.
        """

        if self.length < 2:
            return 1.0
        delta = (self.pixels[-1] - self.pixels[0]).astype(np.float64)
        chord = float(np.hypot(delta[0], delta[1]))
        return float(np.clip(chord / max(self.length - 1, 1), 0.0, 1.0))

    def corridor(self, shape: tuple[int, int]) -> np.ndarray:
        """Masque booléen du corridor : axe dilaté par ``radius``."""

        if self._cached_corridor is not None:
            return self._cached_corridor
        radius = float(np.clip(self.radius, MIN_RADIUS, MAX_RADIUS))
        pad = int(np.ceil(radius)) + 1

        # La transformée de distance est plus isotrope qu'une dilatation par
        # élément structurant pour un rayon fractionnaire, mais la calculer sur
        # l'image entière pour un fragment de ~20 px coûte deux ordres de
        # grandeur de trop. On la restreint à la boîte englobante dilatée.
        row_min = max(int(self.pixels[:, 0].min()) - pad, 0)
        row_max = min(int(self.pixels[:, 0].max()) + pad + 1, shape[0])
        col_min = max(int(self.pixels[:, 1].min()) - pad, 0)
        col_max = min(int(self.pixels[:, 1].max()) + pad + 1, shape[1])

        local = np.zeros((row_max - row_min, col_max - col_min), dtype=bool)
        local[self.pixels[:, 0] - row_min, self.pixels[:, 1] - col_min] = True
        distance = ndi.distance_transform_edt(~local)

        corridor = np.zeros(shape, dtype=bool)
        corridor[row_min:row_max, col_min:col_max] = distance <= radius
        object.__setattr__(self, "_cached_corridor", corridor)
        return corridor


def build_support(
    evidence: Evidence,
    thresholds: Mapping[str, float],
    sources: Sequence[str] = SUPPORT_SOURCES,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Union des supports seuillés **absolument**, source par source.

    Retourne le support et le détail par source, afin qu'un fragment puisse
    déclarer de quelles sources il provient — information dont l'arbitre a
    besoin et qui serait perdue par une fusion précoce.
    """

    per_source: dict[str, np.ndarray] = {}
    union = np.zeros(evidence.shape, dtype=bool)
    for name in sources:
        if name not in thresholds:
            raise KeyError(f"seuil absolu manquant pour la source {name!r}")
        mask = evidence.plane(name) >= float(thresholds[name])
        per_source[name] = mask
        union |= mask

    if union.any():
        labels, count = ndi.label(union, structure=np.ones((3, 3), dtype=int))
        if count:
            areas = np.bincount(labels.ravel())
            areas[0] = 0
            keep = np.isin(labels, np.flatnonzero(areas >= MIN_SUPPORT_AREA))
            union = union & keep
    return union, per_source


def _degree_map(skeleton: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    counts = ndi.convolve(
        skeleton.astype(np.uint8), kernel, mode="constant", cval=0
    )
    return np.where(skeleton, counts, 0).astype(np.int8)


def _walk_branches(skeleton: np.ndarray) -> list[np.ndarray]:
    """Découpe le squelette en branches entre extrémités et jonctions."""

    degree = _degree_map(skeleton)
    nodes = skeleton & ((degree != 2) | (degree == 0))
    visited_edges: set[tuple[int, int, int, int]] = set()
    branches: list[np.ndarray] = []
    height, width = skeleton.shape

    def neighbours(row: int, col: int) -> list[tuple[int, int]]:
        out = []
        for delta_row, delta_col in _NEIGHBOURS:
            next_row, next_col = row + int(delta_row), col + int(delta_col)
            if 0 <= next_row < height and 0 <= next_col < width:
                if skeleton[next_row, next_col]:
                    out.append((next_row, next_col))
        return out

    def trace(start: tuple[int, int], first: tuple[int, int]) -> np.ndarray | None:
        edge = (*start, *first)
        if edge in visited_edges:
            return None
        path = [start, first]
        visited_edges.add(edge)
        visited_edges.add((*first, *start))
        previous, current = start, first
        while not nodes[current]:
            candidates = [n for n in neighbours(*current) if n != previous]
            if not candidates:
                break
            # Préférer la continuation 4-connexe pour éviter les zigzags.
            candidates.sort(
                key=lambda n: abs(n[0] - current[0]) + abs(n[1] - current[1])
            )
            nxt = candidates[0]
            edge = (*current, *nxt)
            if edge in visited_edges:
                break
            visited_edges.add(edge)
            visited_edges.add((*nxt, *current))
            path.append(nxt)
            previous, current = current, nxt
        return np.asarray(path, dtype=np.int32)

    for row, col in np.argwhere(nodes):
        for neighbour in neighbours(int(row), int(col)):
            traced = trace((int(row), int(col)), neighbour)
            if traced is not None and traced.shape[0] >= 2:
                branches.append(traced)

    # Cycles purs : aucun nœud de degré != 2. On les ouvre arbitrairement.
    remaining = skeleton & ~nodes
    if remaining.any():
        labels, count = ndi.label(remaining, structure=np.ones((3, 3), dtype=int))
        for label in range(1, count + 1):
            pixels = np.argwhere(labels == label)
            if pixels.shape[0] < MIN_LENGTH:
                continue
            start = tuple(int(v) for v in pixels[0])
            adjacency = neighbours(*start)
            if adjacency:
                traced = trace(start, adjacency[0])
                if traced is not None and traced.shape[0] >= 2:
                    branches.append(traced)
    return branches


def _split_branch(branch: np.ndarray, target: int = TARGET_LENGTH) -> list[np.ndarray]:
    """Coupe une branche longue en morceaux de longueur proche de ``target``."""

    n = branch.shape[0]
    if n <= target:
        return [branch]
    pieces = max(1, int(round(n / float(target))))
    bounds = np.linspace(0, n, pieces + 1).astype(int)
    out = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        # Un pixel de recouvrement garde les corridors jointifs.
        piece = branch[start : min(stop + 1, n)]
        if piece.shape[0] >= MIN_LENGTH:
            out.append(piece)
    return out or [branch]


def propose_fragments(
    evidence: Evidence,
    thresholds: Mapping[str, float],
    sources: Sequence[str] = SUPPORT_SOURCES,
    target_length: int = TARGET_LENGTH,
) -> list[Fragment]:
    """Chaîne complète support → squelette → branches → fragments.

    Retourne une liste éventuellement **vide** : c'est une sortie légale qui
    fait renvoyer ``z0`` exactement.
    """

    support, per_source = build_support(evidence, thresholds, sources)
    if not support.any():
        return []

    skeleton = skeletonize(support)
    if not skeleton.any():
        return []

    scale_plane = evidence.plane("scale")
    fragments: list[Fragment] = []
    for branch in _walk_branches(skeleton):
        for piece in _split_branch(branch, target_length):
            if piece.shape[0] < MIN_LENGTH:
                continue
            rows, cols = piece[:, 0], piece[:, 1]
            # Le rayon du corridor vient de l'échelle locale mesurée, jamais du
            # GT. C'est ce qui rend l'oracle de source honnête : il choisit une
            # action, il ne redimensionne pas le candidat.
            radius = float(np.median(scale_plane[rows, cols])) + 1.0
            origin = tuple(
                name for name, mask in per_source.items() if mask[rows, cols].any()
            )
            fragments.append(
                Fragment(
                    pixels=piece.astype(np.int32),
                    radius=float(np.clip(radius, MIN_RADIUS, MAX_RADIUS)),
                    sources=origin,
                )
            )
    return fragments


def corridor_union(
    fragments: Sequence[Fragment],
    shape: tuple[int, int],
    selected: Sequence[int] | None = None,
) -> np.ndarray:
    """Union ``B`` des corridors retenus. Vide si aucun n'est retenu."""

    union = np.zeros(shape, dtype=bool)
    indices = range(len(fragments)) if selected is None else selected
    for index in indices:
        union |= fragments[index].corridor(shape)
    return union
