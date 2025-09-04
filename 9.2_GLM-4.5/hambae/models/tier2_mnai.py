"""
Tier 2: Metabolic Network Aging Integrator (MNAI) for HAMBAE algorithm system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from dataclasses import dataclass

from .base_model import BaseBiologicalAgeModel, ModelOutput
from ..config import Tier2Config

logger = logging.getLogger(__name__)


class MetabolicNetworkAgingIntegrator(BaseBiologicalAgeModel):
    """
    Tier 2: Metabolic Network Aging Integrator (MNAI).
    
    This model implements the second tier of the HAMBAE algorithm system,
    integrating metabolomics data with blood biomarkers using graph neural
    networks and pathway analysis.
    """
    
    def __init__(self, config: Tier2Config, **kwargs):
        """Initialize Metabolic Network Aging Integrator."""
        self.config = config
        
        super().__init__(
            input_dim=config.blood_biomarker_count + config.metabolomic_count,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
            activation=config.activation,
            use_batch_norm=config.use_batch_norm,
            use_layer_norm=config.use_layer_norm,
            enable_uncertainty=config.enable_uncertainty,
            mc_dropout=config.mc_dropout,
            heteroscedastic=config.heteroscedastic,
            **kwargs
        )
        
        # Split input dimensions
        self.blood_dim = config.blood_biomarker_count
        self.metabolomic_dim = config.metabolomic_count
        
        # Build graph neural network components
        self._build_gnn_components()
        
        # Build pathway analysis components
        if config.enable_pathway_analysis:
            self._build_pathway_components()
        
        # Build cross-modal attention
        if config.cross_modal_attention:
            self._build_cross_modal_attention()
        
        logger.info(f"Initialized MNAI with {self.blood_dim} blood biomarkers and {self.metabolomic_dim} metabolomics features")
    
    def _build_model(self) -> None:
        """Build the main model architecture."""
        # Separate encoders for blood and metabolomics
        self.blood_encoder = nn.Sequential(
            nn.Linear(self.blood_dim, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
        )
        
        self.metabolomics_encoder = nn.Sequential(
            nn.Linear(self.metabolomic_dim, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
        )
        
        # Combined feature processing
        self.combined_processor = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
        )
        
        # Prediction head
        self.prediction_head = nn.Linear(self.hidden_dim, 1)
        
        # Uncertainty head
        if self.enable_uncertainty and self.heteroscedastic:
            self.uncertainty_head = nn.Linear(self.hidden_dim, 1)
    
    def _build_gnn_components(self) -> None:
        """Build graph neural network components."""
        # Graph attention layers
        self.gnn_layers = nn.ModuleList()
        
        for i in range(self.config.gnn_layers):
            gnn_layer = nn.Sequential(
                nn.Linear(self.hidden_dim, self.config.gnn_hidden_dim),
                self.activation_fn,
                nn.Dropout(self.dropout_rate),
                nn.MultiheadAttention(
                    embed_dim=self.config.gnn_hidden_dim,
                    num_heads=self.config.gnn_attention_heads,
                    dropout=self.dropout_rate,
                    batch_first=True,
                )[0],  # Get the MultiheadAttention module
                nn.Linear(self.config.gnn_hidden_dim, self.hidden_dim),
                self.activation_fn,
            )
            self.gnn_layers.append(gnn_layer)
    
    def _build_pathway_components(self) -> None:
        """Build pathway analysis components."""
        self.pathway_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.config.pathway_hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.config.pathway_hidden_dim, self.config.pathway_hidden_dim),
            self.activation_fn,
        )
        
        self.pathway_attention = nn.MultiheadAttention(
            embed_dim=self.config.pathway_hidden_dim,
            num_heads=4,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        
        self.pathway_output = nn.Linear(self.config.pathway_hidden_dim, 1)
    
    def _build_cross_modal_attention(self) -> None:
        """Build cross-modal attention components."""
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.config.attention_heads,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        
        self.modal_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.config.attention_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.config.attention_dim, self.hidden_dim),
        )
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features."""
        # Split input into blood and metabolomics
        blood_features = x[:, :self.blood_dim]
        metabolomics_features = x[:, self.blood_dim:]
        
        # Encode blood biomarkers
        blood_encoded = self.blood_encoder(blood_features)
        
        # Encode metabolomics
        metabolomics_encoded = self.metabolomics_encoder(metabolomics_features)
        
        # Apply graph neural network to metabolomics
        metabolomics_gnn = self._apply_gnn(metabolomics_encoded)
        
        # Cross-modal attention
        if self.config.cross_modal_attention:
            fused_features = self._apply_cross_modal_attention(
                blood_encoded, metabolomics_gnn
            )
        else:
            # Simple concatenation
            fused_features = torch.cat([blood_encoded, metabolomics_gnn], dim=1)
            fused_features = self.combined_processor(fused_features)
        
        # Apply pathway analysis
        if self.config.enable_pathway_analysis:
            pathway_features = self._apply_pathway_analysis(fused_features)
            fused_features = fused_features + pathway_features  # Residual connection
        
        return fused_features
    
    def _apply_gnn(self, metabolomics_features: torch.Tensor) -> torch.Tensor:
        """Apply graph neural network to metabolomics features."""
        # Reshape for attention (add sequence dimension)
        seq_features = metabolomics_features.unsqueeze(1)
        
        # Apply GNN layers
        gnn_features = seq_features
        for gnn_layer in self.gnn_layers:
            # Apply linear layers
            linear_out = gnn_layer[0](gnn_features)
            linear_out = gnn_layer[1](linear_out)
            linear_out = gnn_layer[2](linear_out)
            
            # Apply attention
            attended, _ = gnn_layer[3](linear_out, linear_out, linear_out)
            
            # Apply final linear layer
            gnn_features = gnn_layer[4](attended)
            gnn_features = gnn_layer[5](gnn_features)
        
        # Remove sequence dimension
        return gnn_features.squeeze(1)
    
    def _apply_cross_modal_attention(
        self, 
        blood_features: torch.Tensor, 
        metabolomics_features: torch.Tensor
    ) -> torch.Tensor:
        """Apply cross-modal attention."""
        # Reshape for attention
        blood_seq = blood_features.unsqueeze(1)
        metabolomics_seq = metabolomics_features.unsqueeze(1)
        
        # Cross attention: blood -> metabolomics
        metabolomics_attended, _ = self.cross_modal_attention(
            metabolomics_seq, blood_seq, blood_seq
        )
        
        # Cross attention: metabolomics -> blood
        blood_attended, _ = self.cross_modal_attention(
            blood_seq, metabolomics_seq, metabolomics_seq
        )
        
        # Fuse attended features
        blood_fused = blood_attended.squeeze(1)
        metabolomics_fused = metabolomics_attended.squeeze(1)
        
        # Combine features
        combined = torch.cat([blood_fused, metabolomics_fused], dim=1)
        fused = self.modal_fusion(combined)
        
        return fused
    
    def _apply_pathway_analysis(self, features: torch.Tensor) -> torch.Tensor:
        """Apply pathway analysis."""
        # Encode features for pathway analysis
        pathway_encoded = self.pathway_encoder(features)
        
        # Reshape for attention
        pathway_seq = pathway_encoded.unsqueeze(1)
        
        # Apply pathway attention
        pathway_attended, _ = self.pathway_attention(
            pathway_seq, pathway_seq, pathway_seq
        )
        
        # Generate pathway output
        pathway_output = self.pathway_output(pathway_attended.squeeze(1))
        
        return pathway_output
    
    def get_pathway_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Get pathway-level importance scores."""
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            features = self._forward_features(x)
            
            if self.config.enable_pathway_analysis:
                pathway_features = self._apply_pathway_analysis(features)
                return torch.abs(pathway_features)
            else:
                return torch.abs(features)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        
        mnai_info = {
            'model_tier': 2,
            'input_modalities': ['blood_biomarkers', 'metabolomics'],
            'blood_biomarker_count': self.blood_dim,
            'metabolomic_count': self.metabolomic_dim,
            'enable_pathway_analysis': self.config.enable_pathway_analysis,
            'enable_cross_modal_attention': self.config.cross_modal_attention,
            'gnn_layers': self.config.gnn_layers,
            'gnn_attention_heads': self.config.gnn_attention_heads,
            'target_mae': self.config.target_mae,
            'target_r2': self.config.target_r2,
        }
        
        base_info.update(mnai_info)
        return base_info


def create_mnai_model(config: Tier2Config) -> MetabolicNetworkAgingIntegrator:
    """Create a Metabolic Network Aging Integrator model."""
    return MetabolicNetworkAgingIntegrator(config)