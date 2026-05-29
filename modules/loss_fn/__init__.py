from .CAFM_loss import CloudAdaptiveLoss, SimpleMSELoss
from .fdt_loss import (
    FDTCCALoss,
    FDTDecompositionLoss,
    FeatureUncorrelationLoss,
    PatchSlicedWassersteinLoss,
)
from .lcr_loss import LCRLoss

__all__ = [
    "CloudAdaptiveLoss",
    "FDTCCALoss",
    "FDTDecompositionLoss",
    "FeatureUncorrelationLoss",
    "LCRLoss",
    "PatchSlicedWassersteinLoss",
    "SimpleMSELoss",
]
