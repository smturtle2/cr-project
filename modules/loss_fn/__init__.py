from .CAFM_loss import CloudAdaptiveLoss, SimpleMSELoss
from .fdt_loss import FDTCCALoss, make_fdt_cca_loss_fn
from .focal_frequency_loss import FocalFrequencyLoss
from .lcr_loss import LCRLoss

__all__ = [
    "CloudAdaptiveLoss",
    "FDTCCALoss",
    "FocalFrequencyLoss",
    "LCRLoss",
    "make_fdt_cca_loss_fn",
    "SimpleMSELoss",
]
