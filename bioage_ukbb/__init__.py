"""
UK Biobank Biological Age Algorithms Implementation
====================================================

Production-ready PyTorch implementations of three biological age algorithms:
- HENAW: Hierarchical Ensemble Network for Aging Waves
- MODAL: Multi-Organ Deep Aging Learner  
- METAGE: Metabolomic Trajectory Aging Estimator

Author: UK Biobank Analysis Team
Date: 2025
"""

__version__ = "1.0.0"
__author__ = "UK Biobank Analysis Team"

from .models.henaw import HENAW
from .models.modal import MODAL
from .models.metage import METAGE

__all__ = [
    "HENAW",
    "MODAL", 
    "METAGE"
]