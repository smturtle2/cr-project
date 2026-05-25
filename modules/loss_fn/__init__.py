from .CAFM_loss import CloudAdaptiveLoss, SimpleMSELoss
from .fdt_loss import (
    FDTDecompositionLoss,
    FeatureUncorrelationLoss,
    PatchSlicedWassersteinLoss,
)
from .lcr_loss import LCRLoss

__all__ = [
    "CloudAdaptiveLoss",
    "FDTDecompositionLoss",
    "FeatureUncorrelationLoss",
    "LCRLoss",
    "PatchSlicedWassersteinLoss",
    "SimpleMSELoss",
]
