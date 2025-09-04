"""
HAMBAE: Hierarchical Adaptive Multi-modal Biological Age Estimation

A comprehensive PyTorch implementation of the HAMBAE algorithm system
for biological age estimation using multi-modal data from UK Biobank.
"""

__version__ = "1.0.0"
__author__ = "HAMBAE Development Team"
__email__ = "hambae@example.com"

from .config import HAMBAEConfig, TierConfig
from .models.base_model import BaseBiologicalAgeModel
from .models.tier1_cban import ClinicalBiomarkerAgingNetwork
from .models.tier2_mnai import MetabolicNetworkAgingIntegrator
from .models.tier3_mmbat import MultiModalBiologicalAgeTransformer

__all__ = [
    "HAMBAEConfig",
    "TierConfig", 
    "BaseBiologicalAgeModel",
    "ClinicalBiomarkerAgingNetwork",
    "MetabolicNetworkAgingIntegrator",
    "MultiModalBiologicalAgeTransformer",
]