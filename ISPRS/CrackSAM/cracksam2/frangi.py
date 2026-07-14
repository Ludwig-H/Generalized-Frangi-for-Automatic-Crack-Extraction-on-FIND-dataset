"""Generalized Frangi-Graph prompts for CrackSAM 2.

This module exposes the GPU implementation originally developed in
``test_k2_clean.py`` without importing that notebook export or executing any
top-level experiment. The maintained implementation lives in ``ISPRS.src``;
the wrapper below fixes the CrackSAM 2 defaults and adds the image/prompt
conversion boundary required by SAM 2.

``extract_frangi_graph_gpu`` preserves the five-value return contract:

1. multi-scale Frangi response, ``float32[H, W]``;
2. node similarity probability map, ``float32[H, W]`` in ``[0, 1]``;
3. graph betweenness centrality, ``float32[H, W]``;
4. timing dictionary;
5. diagnostic masks (``tau_mask`` and ``comp_mask``).

The SAM 2 prompt is derived from item 2. It is clipped, converted to logits,
then bilinearly resized to ``(1, 256, 256)``. Stacking one cached ``.npy`` per
sample therefore produces the expected ``(B, 1, 256, 256)`` mask input.
Frangi-Graph is deliberately evaluated under ``torch.no_grad()``.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# Support both ``ISPRS.CrackSAM.cracksam2`` (repository root on sys.path) and
# ``cracksam2`` (ISPRS/CrackSAM on sys.path), which is convenient for scripts
# launched from the CrackSAM directory.
try:
    from ...src.frangi_hessian import FrangiHessianGPU
    from ...src.graph_extraction import (
        extract_frangi_graph_gpu as _extract_frangi_graph_gpu,
    )
except ImportError:
    _REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
    if str(_REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPOSITORY_ROOT))
    from ISPRS.src.frangi_hessian import FrangiHessianGPU
    from ISPRS.src.graph_extraction import (
        extract_frangi_graph_gpu as _extract_frangi_graph_gpu,
    )


DEFAULT_FRANGI_SCALES: tuple[float, ...] = (1.0, 3.0, 5.0, 9.0, 15.0)
DEFAULT_PROMPT_SIZE: tuple[int, int] = (256, 256)


def _as_float_image(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert an image to a detached float32 tensor in ``[0, 1]``."""
    tensor = image.detach() if isinstance(image, torch.Tensor) else torch.as_tensor(image)

    if tensor.dtype == torch.bool:
        tensor = tensor.to(torch.float32)
    elif not tensor.is_floating_point():
        dtype_max = torch.iinfo(tensor.dtype).max
        tensor = tensor.to(torch.float32) / float(dtype_max)
    else:
        tensor = tensor.to(torch.float32)
        if tensor.numel() and float(tensor.detach().max().cpu()) > 1.0:
            tensor = tensor / 255.0

    return tensor.clamp(0.0, 1.0)


def rgb_to_grayscale(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Return one ``float32[H, W]`` channel from an RGB image.

    Accepted layouts are ``H x W``, ``H x W x C`` and ``C x H x W`` with
    ``C`` equal to 1, 3 or 4. Alpha is ignored. Three-channel inputs are
    interpreted as RGB (not OpenCV BGR). Integer inputs and float images in
    ``[0, 255]`` are normalized to ``[0, 1]``.
    """
    tensor = _as_float_image(image)
    if tensor.ndim == 2:
        return tensor.contiguous()
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or a 3D single image, got shape {tuple(tensor.shape)}"
        )

    if tensor.shape[-1] in (1, 3, 4):
        channels_last = tensor
    elif tensor.shape[0] in (1, 3, 4):
        channels_last = tensor.permute(1, 2, 0)
    else:
        raise ValueError(
            "Cannot identify the channel dimension; expected 1, 3 or 4 channels, "
            f"got shape {tuple(tensor.shape)}"
        )

    if channels_last.shape[-1] == 1:
        return channels_last[..., 0].contiguous()

    coefficients = channels_last.new_tensor((0.2989, 0.5870, 0.1140))
    return (channels_last[..., :3] * coefficients).sum(dim=-1).contiguous()


def probability_to_logits(
    probability: np.ndarray | torch.Tensor,
    eps: float = 1e-5,
) -> np.ndarray | torch.Tensor:
    """Convert probabilities to finite logits after clipping to ``[eps, 1-eps]``.

    The result is float32 and has the same container type as the input.
    """
    if not 0.0 < eps < 0.5:
        raise ValueError(f"eps must be between 0 and 0.5, got {eps}")

    if isinstance(probability, torch.Tensor):
        clipped = probability.to(torch.float32).clamp(eps, 1.0 - eps)
        return torch.log(clipped) - torch.log1p(-clipped)

    clipped_np = np.clip(np.asarray(probability, dtype=np.float32), eps, 1.0 - eps)
    return (np.log(clipped_np) - np.log1p(-clipped_np)).astype(np.float32, copy=False)


def extract_frangi_graph_gpu(
    imgs_dict: Mapping[str, np.ndarray | torch.Tensor],
    weights: Mapping[str, float],
    scales: Sequence[float] = DEFAULT_FRANGI_SCALES,
    R: int = 3,
    ss: float = 1.0,
    si: float = 0.25,
    sa: float = 0.3,
    tau: float = 0.18,
    min_rel_size: float = 120.0,
    K: int = 1,
    device: str | torch.device | None = None,
    compute_centrality: bool = True,
):
    """Compute the generalized Frangi graph for one image.

    Images may be grayscale or RGB, NumPy arrays or tensors. Every active
    modality is converted to a 2D float32 tensor before delegating to the
    reference implementation. The function processes one image at a time;
    batching belongs in an offline prompt-generation loop.

    Args:
        imgs_dict: Modality name to single image.
        weights: Modality weights. Keys with a positive weight must exist in
            ``imgs_dict``.
        scales: Gaussian derivative scales in pixels. CrackSAM 2 uses
            ``(1, 3, 5, 9, 15)``.
        R: Euclidean graph-neighborhood radius in pixels.
        ss: Elongation-similarity bandwidth.
        si: Hessian-strength-similarity bandwidth.
        sa: Orientation-similarity bandwidth.
        tau: Fraction of strongest edges/nodes retained by graph pruning.
        min_rel_size: Relative component-size divisor used by centrality.
        K: ``1`` for the MST graph or ``2`` for the triangle dual graph.
        device: Torch device. Defaults to CUDA when available, otherwise CPU.
        compute_centrality: Whether to compute the MST and betweenness output.
            The similarity map is identical when this is false.

    Returns:
        ``(frangi_response, similarity_img, centrality, timings, diagnostics)``.
        The similarity map, not centrality, is the SAM 2 geometric prompt.
    """
    scale_tuple = tuple(float(scale) for scale in scales)
    if not scale_tuple or any(scale <= 0 for scale in scale_tuple):
        raise ValueError(f"scales must contain positive values, got {scale_tuple}")
    if int(R) != R or R < 1:
        raise ValueError(f"R must be a positive integer, got {R}")
    if K not in (1, 2):
        raise ValueError(f"K must be 1 or 2, got {K}")
    if not 0.0 < tau <= 1.0:
        raise ValueError(f"tau must be in (0, 1], got {tau}")

    active = {name for name, weight in weights.items() if float(weight) > 0.0}
    missing = active.difference(imgs_dict)
    if missing:
        raise KeyError(f"Missing active modalities: {sorted(missing)}")
    if not active:
        raise ValueError("At least one modality must have a positive weight")

    prepared = {name: rgb_to_grayscale(image) for name, image in imgs_dict.items()}
    active_shapes = {tuple(prepared[name].shape) for name in active}
    if len(active_shapes) != 1:
        raise ValueError(f"Active modalities must share one image shape, got {active_shapes}")

    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    with torch.no_grad():
        result = _extract_frangi_graph_gpu(
            prepared,
            {name: float(weight) for name, weight in weights.items()},
            Σ=list(scale_tuple),
            R=int(R),
            ss=float(ss),
            si=float(si),
            sa=float(sa),
            τ=float(tau),
            min_rel_size=float(min_rel_size),
            K=int(K),
            device=str(resolved_device),
            compute_centrality=bool(compute_centrality),
        )

    # The source implementation already returns float32 in normal cases. Cast
    # all image outputs here as an explicit cache/API guarantee, including its
    # early all-zero returns (which NumPy otherwise creates as float64).
    frangi_response, similarity_img, centrality, timings, diagnostics = result
    diagnostics = {
        name: np.asarray(mask, dtype=np.float32) for name, mask in diagnostics.items()
    }
    return (
        np.asarray(frangi_response, dtype=np.float32),
        np.asarray(similarity_img, dtype=np.float32),
        np.asarray(centrality, dtype=np.float32),
        timings,
        diagnostics,
    )


def save_prompt_atomic(path: str | os.PathLike[str], prompt: np.ndarray | torch.Tensor) -> Path:
    """Atomically save a float32 SAM 2 prompt as an unpickled ``.npy`` file."""
    destination = Path(path)
    if destination.suffix.lower() != ".npy":
        raise ValueError(f"Prompt cache path must end in .npy: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(prompt, torch.Tensor):
        array = prompt.detach().cpu().numpy()
    else:
        array = np.asarray(prompt)
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Refusing to cache a prompt containing NaN or infinity")

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            np.save(temporary_file, array, allow_pickle=False)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return destination


def generate_frangi_prompt(
    image: np.ndarray | torch.Tensor,
    output_path: str | os.PathLike[str] | None = None,
    *,
    prompt_size: tuple[int, int] = DEFAULT_PROMPT_SIZE,
    eps: float = 1e-5,
    scales: Sequence[float] = DEFAULT_FRANGI_SCALES,
    R: int = 3,
    K: int = 1,
    device: str | torch.device | None = None,
    **graph_parameters: float,
) -> np.ndarray:
    """Generate one static Frangi-Graph mask prompt, optionally caching it.

    The returned float32 array has shape ``(1, prompt_height, prompt_width)``.
    It contains resized pseudo-logits and is ready to be stacked by a PyTorch
    dataloader. ``graph_parameters`` forwards optional ``ss``, ``si``, ``sa``,
    ``tau`` and ``min_rel_size`` overrides to the graph extractor.
    """
    if len(prompt_size) != 2 or any(int(size) <= 0 for size in prompt_size):
        raise ValueError(f"prompt_size must contain two positive integers, got {prompt_size}")

    grayscale = rgb_to_grayscale(image)
    _, similarity_img, _, _, _ = extract_frangi_graph_gpu(
        {"visible": grayscale},
        {"visible": 1.0},
        scales=scales,
        R=R,
        K=K,
        device=device,
        compute_centrality=False,
        **graph_parameters,
    )
    logits = probability_to_logits(similarity_img, eps=eps)
    logits_tensor = torch.from_numpy(logits).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        logits_tensor,
        size=(int(prompt_size[0]), int(prompt_size[1])),
        mode="bilinear",
        align_corners=False,
    )
    prompt = resized.squeeze(0).numpy().astype(np.float32, copy=False)

    if output_path is not None:
        save_prompt_atomic(output_path, prompt)
    return prompt
