"""Datasets and deterministic preprocessing for CrackSAM 2 experiments."""

from __future__ import annotations

import os
import random
import tempfile
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset


NoiseMode = Literal["original", "noisy1", "noisy2"]
PromptGenerator = Callable[[np.ndarray], np.ndarray | torch.Tensor]

_IMAGE_DIR_NAMES = ("images", "image", "imgs", "Images", "JPEGImages")
_MASK_DIR_NAMES = (
    "masks",
    "mask",
    "labels",
    "label",
    "Masks",
    "ground_truth",
    "gt",
)
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
PROMPT_CACHE_MANIFEST = ".cracksam2-frangi.json"
PROMPT_CACHE_VERSION = 1


def normalize_noise_mode(mode: NoiseMode | str | None) -> NoiseMode:
    """Normalize the clean/noise aliases used by the CLIs and datasets."""
    normalized = "original" if mode is None else str(mode).lower().replace("_", "")
    aliases = {
        "original": "original",
        "clean": "original",
        "none": "original",
        "noisy1": "noisy1",
        "noise1": "noisy1",
        "noisy2": "noisy2",
        "noise2": "noisy2",
    }
    try:
        return aliases[normalized]  # type: ignore[return-value]
    except KeyError as exc:
        raise ValueError(f"Unknown noise mode: {mode!r}") from exc


def sample_names_sha256(sample_names: Sequence[str]) -> str:
    """Hash an ordered, normalized split independently of its text encoding."""
    digest = hashlib.sha256()
    for name in sample_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_prompt_cache(
    cache_dir: str | os.PathLike[str],
    sample_names: Sequence[str],
    *,
    image_size: int | Sequence[int] = 448,
    prompt_size: int | Sequence[int] = 256,
    noise_mode: NoiseMode | str | None = None,
) -> dict[str, object]:
    """Validate a completed Frangi cache against the consuming dataset."""
    cache_path = Path(cache_dir).expanduser()
    manifest_path = cache_path / PROMPT_CACHE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Frangi cache manifest not found: {manifest_path}. "
            "Run precompute_frangi_prompts.py to completion."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid Frangi cache manifest: {manifest_path}") from exc
    expected = {
        "format_version": PROMPT_CACHE_VERSION,
        "status": "complete",
        "samples": len(sample_names),
        "sample_names_sha256": sample_names_sha256(sample_names),
        "image_size": list(_size_hw(image_size)),
        "prompt_size": list(_size_hw(prompt_size)),
        "noise": normalize_noise_mode(noise_mode),
    }
    mismatches = {
        key: {"observed": manifest.get(key), "expected": value}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    frangi = manifest.get("frangi")
    required_frangi = {
        "scales": [1.0, 3.0, 5.0, 9.0, 15.0],
        "R": 3,
        "K": 1,
        "eps": 1e-5,
    }
    if frangi != required_frangi:
        mismatches["frangi"] = {"observed": frangi, "expected": required_frangi}
    if mismatches:
        raise ValueError(
            f"Frangi cache {cache_path} is incompatible with this dataset: {mismatches}"
        )
    return manifest


def _size_hw(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("image_size must be positive")
        return size, size
    if len(size) != 2 or int(size[0]) <= 0 or int(size[1]) <= 0:
        raise ValueError("image_size must be an int or a positive (height, width) pair")
    return int(size[0]), int(size[1])


def read_sample_list(list_file: str | os.PathLike[str]) -> list[str]:
    """Read a CrackSAM split list, ignoring blank lines and comments."""
    path = Path(list_file).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Split list not found: {path}")
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not names:
        raise ValueError(f"Split list is empty: {path}")
    return names


def resolve_list_file(
    *,
    list_file: str | os.PathLike[str] | None = None,
    list_dir: str | os.PathLike[str] | None = None,
    split: str | None = None,
) -> Path:
    """Resolve either an explicit list file or ``list_dir/<split>.txt``."""
    if list_file is not None:
        path = Path(list_file).expanduser()
    else:
        if list_dir is None or split is None:
            raise ValueError("Provide list_file, or both list_dir and split")
        path = Path(list_dir).expanduser() / f"{split}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Split list not found: {path}")
    return path


def _candidate_dataset_roots(root: Path, split: str | None) -> Iterable[Path]:
    yield root
    if split is None:
        return
    split_lower = split.lower()
    names: list[str] = [split, split_lower]
    if split_lower.startswith("train"):
        names.extend(("trainingset", "training", "train"))
    elif split_lower.startswith("val"):
        names.extend(("validationset", "validation", "val", "valid"))
    elif split_lower.startswith("test"):
        names.extend(("testset", "testing", "test"))
    seen: set[str] = set()
    for name in names:
        if name not in seen:
            seen.add(name)
            yield root / name


def _resolve_data_dirs(
    root_dir: str | os.PathLike[str],
    split: str | None,
    image_dir: str | os.PathLike[str] | None,
    mask_dir: str | os.PathLike[str] | None,
) -> tuple[Path, Path]:
    root = Path(root_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    if image_dir is not None or mask_dir is not None:
        if image_dir is None or mask_dir is None:
            raise ValueError("image_dir and mask_dir must be provided together")
        image_path = Path(image_dir).expanduser()
        mask_path = Path(mask_dir).expanduser()
        image_path = image_path if image_path.is_absolute() else root / image_path
        mask_path = mask_path if mask_path.is_absolute() else root / mask_path
        if not image_path.is_dir() or not mask_path.is_dir():
            raise FileNotFoundError(
                f"Image/mask directories not found: {image_path}, {mask_path}"
            )
        return image_path, mask_path

    for base in _candidate_dataset_roots(root, split):
        if not base.is_dir():
            continue
        for image_name in _IMAGE_DIR_NAMES:
            candidate_images = base / image_name
            if not candidate_images.is_dir():
                continue
            for mask_name in _MASK_DIR_NAMES:
                candidate_masks = base / mask_name
                if candidate_masks.is_dir():
                    return candidate_images, candidate_masks

    raise FileNotFoundError(
        f"Could not find image and mask directories below {root}. "
        "Pass image_dir and mask_dir explicitly."
    )


def _relative_sample_name(sample_name: str) -> Path:
    sample = Path(sample_name.replace("\\", "/"))
    parts = [part for part in sample.parts if part not in ("", ".", "..", "/")]
    if parts and parts[0].lower() in {name.lower() for name in _IMAGE_DIR_NAMES}:
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Invalid sample name: {sample_name!r}")
    return Path(*parts)


def _case_insensitive_file(path: Path) -> Path | None:
    if path.is_file():
        return path
    parent = path.parent
    if not parent.is_dir():
        return None
    target = path.name.casefold()
    return next((child for child in parent.iterdir() if child.name.casefold() == target), None)


def _find_image(image_dir: Path, sample: Path) -> Path | None:
    direct = _case_insensitive_file(image_dir / sample)
    if direct is not None:
        return direct
    if not sample.suffix:
        for suffix in _IMAGE_SUFFIXES:
            candidate = _case_insensitive_file(image_dir / sample.with_suffix(suffix))
            if candidate is not None:
                return candidate
    return next(
        (path for path in image_dir.rglob(sample.name) if path.is_file()),
        None,
    )


def _find_mask(mask_dir: Path, sample: Path) -> Path | None:
    stem = sample.stem
    relative_parent = sample.parent
    names = [sample.name]
    names.extend(f"{stem}{suffix}" for suffix in _IMAGE_SUFFIXES)
    names.extend(
        f"{stem}{tag}{suffix}"
        for tag in ("_mask", "_label", "_gt")
        for suffix in _IMAGE_SUFFIXES
    )
    seen: set[str] = set()
    for name in names:
        if name.casefold() in seen:
            continue
        seen.add(name.casefold())
        candidate = _case_insensitive_file(mask_dir / relative_parent / name)
        if candidate is not None:
            return candidate
    for name in names:
        candidate = next((path for path in mask_dir.rglob(name) if path.is_file()), None)
        if candidate is not None:
            return candidate
    return None


def resolve_sample_paths(
    root_dir: str | os.PathLike[str],
    sample_name: str,
    *,
    split: str | None = None,
    image_dir: str | os.PathLike[str] | None = None,
    mask_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Resolve an image/mask pair across the layouts used by CrackSAM datasets."""
    images, masks = _resolve_data_dirs(root_dir, split, image_dir, mask_dir)
    sample = _relative_sample_name(sample_name)
    image_path = _find_image(images, sample)
    mask_path = _find_mask(masks, sample)
    if image_path is None or mask_path is None:
        missing = []
        if image_path is None:
            missing.append(f"image below {images}")
        if mask_path is None:
            missing.append(f"mask below {masks}")
        raise FileNotFoundError(f"Missing {' and '.join(missing)} for sample {sample_name!r}")
    return image_path, mask_path


def apply_noise_perturbation(
    image_rgb: np.ndarray,
    mode: NoiseMode | str | None,
    *,
    output_size: int | Sequence[int] = 448,
) -> np.ndarray:
    """Apply the two robustness perturbations defined by the CrackSAM paper."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.size and image.max() <= 1.0:
            image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.copy()

    normalized_mode = normalize_noise_mode(mode)
    if normalized_mode == "original":
        return image
    if normalized_mode == "noisy1":
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hsv[..., 2] = np.maximum(hsv[..., 2].astype(np.int16) - 50, 0).astype(np.uint8)
        darkened = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return cv2.GaussianBlur(darkened, (9, 9), sigmaX=0)
    if normalized_mode == "noisy2":
        blurred = cv2.GaussianBlur(image, (21, 21), sigmaX=0)
        height, width = blurred.shape[:2]
        half = cv2.resize(
            blurred,
            (max(1, width // 2), max(1, height // 2)),
            interpolation=cv2.INTER_CUBIC,
        )
        out_h, out_w = _size_hw(output_size)
        return cv2.resize(half, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    raise AssertionError(f"Unhandled normalized noise mode: {normalized_mode}")


def apply_geometric_transform(
    image: np.ndarray,
    mask: np.ndarray,
    prompt: np.ndarray | None = None,
    *,
    rotation_k: int = 0,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply exactly the same right-angle rotation and flips to all arrays."""
    k = int(rotation_k) % 4

    def transform(array: np.ndarray) -> np.ndarray:
        result = np.rot90(array, k=k, axes=(0, 1))
        if horizontal_flip:
            result = np.flip(result, axis=1)
        if vertical_flip:
            result = np.flip(result, axis=0)
        return np.ascontiguousarray(result)

    return transform(image), transform(mask), None if prompt is None else transform(prompt)


class SynchronizedRandomTransform:
    """Original CrackSAM augmentation shared by image, mask and prompt.

    The branch probabilities and angle range reproduce ``RandomGenerator`` in
    the SAM 1 code: 50% right-angle rotation plus one axis flip, otherwise
    25% arbitrary rotation in ``[-20, 20)``, otherwise no transform.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
    ) -> None:
        self._rng = random if seed is None else random.Random(seed)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        prompt: np.ndarray | None = None,
        *,
        rng: random.Random | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        active_rng = self._rng if rng is None else rng
        if active_rng.random() > 0.5:
            rotation_k = active_rng.randrange(4)
            axis = active_rng.randrange(2)
            return apply_geometric_transform(
                image,
                mask,
                prompt,
                rotation_k=rotation_k,
                horizontal_flip=axis == 1,
                vertical_flip=axis == 0,
            )
        if active_rng.random() > 0.5:
            angle = active_rng.randrange(-20, 20)

            def rotate(array: np.ndarray) -> np.ndarray:
                return np.ascontiguousarray(
                    ndimage.rotate(array, angle, order=0, reshape=False)
                )

            return rotate(image), rotate(mask), None if prompt is None else rotate(prompt)
        return image, mask, prompt


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _load_binary_mask(path: Path, threshold: float, invert: bool) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 4 and np.unique(array[..., 3]).size > 1:
            gray = array[..., 3]
        else:
            gray = np.asarray(image.convert("L"))
    mask = gray.astype(np.float32) / 255.0 > threshold
    return np.logical_not(mask) if invert else mask


def _prompt_array(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    prompt = np.asarray(value, dtype=np.float32)
    while prompt.ndim > 2 and prompt.shape[0] == 1:
        prompt = prompt[0]
    if prompt.ndim != 2:
        raise ValueError(f"A Frangi prompt must be two-dimensional, got {prompt.shape}")
    if not np.isfinite(prompt).all():
        raise ValueError("A Frangi prompt cannot contain NaN or infinity")
    return prompt


def _cache_path(cache_dir: Path, sample_name: str) -> Path:
    relative = _relative_sample_name(sample_name)
    return cache_dir / relative.parent / f"{relative.name}.npy"


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npy", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary:
            temporary_path = temporary.name
            np.save(temporary, array, allow_pickle=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)


class CrackSegmentationDataset(Dataset):
    """Load CrackSAM split lists with optional cached Frangi prompts.

    Samples contain ``image`` (float32 ``3xHxW``), ``mask`` (float32
    ``1xHxW``), ``case_name`` and, when enabled, ``prompt`` (float32
    ``1x256x256`` by default). Images are scaled to ``[0, 1]`` and masks are
    binary.
    """

    def __init__(
        self,
        root_dir: str | os.PathLike[str],
        *,
        list_file: str | os.PathLike[str] | None = None,
        list_dir: str | os.PathLike[str] | None = None,
        split: str | None = None,
        image_dir: str | os.PathLike[str] | None = None,
        mask_dir: str | os.PathLike[str] | None = None,
        image_size: int | Sequence[int] = 448,
        prompt_size: int | Sequence[int] = 256,
        augment: bool | SynchronizedRandomTransform = False,
        noise_mode: NoiseMode | str | None = None,
        mask_threshold: float = 0.5,
        invert_mask: bool = False,
        prompt_cache_dir: str | os.PathLike[str] | None = None,
        prompt_generator: PromptGenerator | None = None,
        augmentation_seed: int | None = None,
    ) -> None:
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must lie in [0, 1]")
        self.root_dir = Path(root_dir).expanduser()
        self.split = split
        self.list_file = resolve_list_file(
            list_file=list_file, list_dir=list_dir, split=split
        )
        self.sample_names = read_sample_list(self.list_file)
        self.image_dir, self.mask_dir = _resolve_data_dirs(
            self.root_dir, split, image_dir, mask_dir
        )
        self.image_size = _size_hw(image_size)
        self.prompt_size = _size_hw(prompt_size)
        self.noise_mode = normalize_noise_mode(noise_mode)
        self.mask_threshold = mask_threshold
        self.invert_mask = invert_mask
        self.prompt_cache_dir = (
            None if prompt_cache_dir is None else Path(prompt_cache_dir).expanduser()
        )
        self.prompt_generator = prompt_generator
        self.augmentation_seed = 0 if augmentation_seed is None else int(augmentation_seed)
        self.epoch = 0
        self.prompt_cache_manifest: dict[str, object] | None = None
        if self.prompt_cache_dir is not None and self.prompt_generator is None:
            self.prompt_cache_manifest = validate_prompt_cache(
                self.prompt_cache_dir,
                self.sample_names,
                image_size=self.image_size,
                prompt_size=self.prompt_size,
                noise_mode=self.noise_mode,
            )
        if isinstance(augment, SynchronizedRandomTransform):
            self.transform = augment
        elif augment:
            self.transform = SynchronizedRandomTransform(seed=augmentation_seed)
        else:
            self.transform = None

    def set_epoch(self, epoch: int) -> None:
        """Select the deterministic augmentation stream for one epoch."""
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)

    def _augmentation_rng(self, index: int) -> random.Random:
        # A stable 64-bit mix avoids copied worker RNG state and is resume-safe.
        value = (
            self.augmentation_seed * 6364136223846793005
            + self.epoch * 1442695040888963407
            + int(index) * 2862933555777941757
        ) & ((1 << 64) - 1)
        return random.Random(value)

    def __len__(self) -> int:
        return len(self.sample_names)

    def _paths(self, sample_name: str) -> tuple[Path, Path]:
        sample = _relative_sample_name(sample_name)
        image_path = _find_image(self.image_dir, sample)
        mask_path = _find_mask(self.mask_dir, sample)
        if image_path is None or mask_path is None:
            raise FileNotFoundError(
                f"Could not resolve image/mask pair for {sample_name!r} below "
                f"{self.image_dir.parent}"
            )
        return image_path, mask_path

    def _load_prompt(self, sample_name: str, image: np.ndarray) -> np.ndarray | None:
        cache_path = (
            None
            if self.prompt_cache_dir is None
            else _cache_path(self.prompt_cache_dir, sample_name)
        )
        if cache_path is not None and cache_path.is_file():
            prompt = _prompt_array(np.load(cache_path, allow_pickle=False))
            if prompt.shape != self.prompt_size:
                raise ValueError(
                    f"Cached Frangi prompt has shape {prompt.shape}, expected "
                    f"{self.prompt_size}: {cache_path}"
                )
            return prompt
        if self.prompt_generator is None:
            if cache_path is not None:
                raise FileNotFoundError(f"Cached Frangi prompt not found: {cache_path}")
            return None
        prompt = _prompt_array(self.prompt_generator(image.copy()))
        if prompt.shape != self.prompt_size:
            prompt = cv2.resize(
                prompt,
                (self.prompt_size[1], self.prompt_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        if cache_path is not None:
            _atomic_save_npy(cache_path, prompt)
        return prompt

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample_name = self.sample_names[index]
        image_path, mask_path = self._paths(sample_name)
        image = _load_rgb(image_path)
        mask = _load_binary_mask(mask_path, self.mask_threshold, self.invert_mask)
        image = apply_noise_perturbation(
            image, self.noise_mode, output_size=self.image_size
        )
        prompt = self._load_prompt(sample_name, image)

        out_h, out_w = self.image_size
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if mask.shape != self.image_size:
            mask = cv2.resize(
                mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        if prompt is not None and prompt.shape != self.prompt_size:
            prompt_h, prompt_w = self.prompt_size
            prompt = cv2.resize(
                prompt, (prompt_w, prompt_h), interpolation=cv2.INTER_LINEAR
            )

        if self.transform is not None:
            image, mask, prompt = self.transform(
                image, mask, prompt, rng=self._augmentation_rng(index)
            )

        image_tensor = torch.from_numpy(
            np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)
        ).div_(255.0)
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(mask[None, ...], dtype=np.float32)
        )
        sample: dict[str, torch.Tensor | str] = {
            "image": image_tensor,
            "mask": mask_tensor,
            "case_name": sample_name,
        }
        if prompt is not None:
            sample["prompt"] = torch.from_numpy(
                np.ascontiguousarray(prompt[None, ...], dtype=np.float32)
            )
        return sample


# Short aliases for callers migrating from the original Khanhha dataset class.
CrackDataset = CrackSegmentationDataset
CrackSAM2Dataset = CrackSegmentationDataset


__all__ = [
    "CrackDataset",
    "CrackSAM2Dataset",
    "CrackSegmentationDataset",
    "PROMPT_CACHE_MANIFEST",
    "PROMPT_CACHE_VERSION",
    "SynchronizedRandomTransform",
    "apply_geometric_transform",
    "apply_noise_perturbation",
    "normalize_noise_mode",
    "read_sample_list",
    "resolve_list_file",
    "resolve_sample_paths",
    "sample_names_sha256",
    "validate_prompt_cache",
]
