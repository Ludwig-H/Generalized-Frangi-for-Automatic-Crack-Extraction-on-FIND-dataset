from .frangi_hessian import FrangiHessianGPU
from .graph_extraction import extract_frangi_graph_gpu


def __getattr__(name):
    """Load optional visualization/metric dependencies only when requested."""
    if name in {"VTGraFDataset", "decode_jet_to_grayscale"}:
        from .dataloaders import VTGraFDataset, decode_jet_to_grayscale

        return {
            "VTGraFDataset": VTGraFDataset,
            "decode_jet_to_grayscale": decode_jet_to_grayscale,
        }[name]
    if name in {
        "skeletonize_lee",
        "thicken",
        "compute_metrics",
        "wasserstein_distance_skeletons",
    }:
        from .metrics import (
            compute_metrics,
            skeletonize_lee,
            thicken,
            wasserstein_distance_skeletons,
        )

        return {
            "skeletonize_lee": skeletonize_lee,
            "thicken": thicken,
            "compute_metrics": compute_metrics,
            "wasserstein_distance_skeletons": wasserstein_distance_skeletons,
        }[name]
    raise AttributeError(name)


__all__ = [
    "FrangiHessianGPU",
    "extract_frangi_graph_gpu",
    "VTGraFDataset",
    "decode_jet_to_grayscale",
    "skeletonize_lee",
    "thicken",
    "compute_metrics",
    "wasserstein_distance_skeletons",
]
