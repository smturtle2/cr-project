from .CAFM_loss import CloudAdaptiveLoss, SimpleMSELoss
from .fdt_loss import FDTCCALoss
from .lcr_loss import LCRLoss

__all__ = [
    "CloudAdaptiveLoss",
    "FDTCCALoss",
    "LCRLoss",
    "SimpleMSELoss",
]
