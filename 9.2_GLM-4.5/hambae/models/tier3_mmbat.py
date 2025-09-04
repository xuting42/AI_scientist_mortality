"""
Tier 3: Multi-Modal Biological Age Transformer (MM-BAT) for HAMBAE algorithm system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from dataclasses import dataclass

from .base_model import BaseBiologicalAgeModel, ModelOutput
from ..config import Tier3Config

logger = logging.getLogger(__name__)


class MultiModalBiologicalAgeTransformer(BaseBiologicalAgeModel):
    """
    Tier 3: Multi-Modal Biological Age Transformer (MM-BAT).
    
    This model implements the third tier of the HAMBAE algorithm system,
    integrating blood biomarkers, metabolomics, retinal imaging, and genetic
    data using transformer-based multi-modal fusion.
    """
    
    def __init__(self, config: Tier3Config, **kwargs):
        """Initialize Multi-Modal Biological Age Transformer."""
        self.config = config
        
        # Calculate total input dimension
        total_input_dim = (
            config.blood_biomarker_count + 
            config.metabolomic_count + 
            config.retinal_feature_dim + 
            config.genetic_feature_dim
        )
        
        super().__init__(
            input_dim=total_input_dim,
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
        
        # Modality dimensions
        self.blood_dim = config.blood_biomarker_count
        self.metabolomic_dim = config.metabolomic_count
        self.retinal_dim = config.retinal_feature_dim
        self.genetic_dim = config.genetic_feature_dim
        
        # Build modality-specific encoders
        self._build_modality_encoders()
        
        # Build transformer components
        self._build_transformer()
        
        # Build multi-modal fusion
        self._build_multi_modal_fusion()
        
        # Build organ-specific components
        if config.organ_specific:
            self._build_organ_specific_components()
        
        # Build longitudinal components
        if config.enable_longitudinal:
            self._build_longitudinal_components()
        
        logger.info(f"Initialized MM-BAT with 4 modalities: blood ({self.blood_dim}), metabolomics ({self.metabolomic_dim}), retinal ({self.retinal_dim}), genetic ({self.genetic_dim})")
    
    def _build_model(self) -> None:
        """Build the main model architecture."""
        # This is handled by _build_modality_encoders and _build_transformer
        pass
    
    def _build_modality_encoders(self) -> None:
        """Build modality-specific encoders."""
        # Blood biomarker encoder
        self.blood_encoder = nn.Sequential(
            nn.Linear(self.blood_dim, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Metabolomics encoder
        self.metabolomics_encoder = nn.Sequential(
            nn.Linear(self.metabolomic_dim, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Retinal encoder (assuming pre-extracted features)
        self.retinal_encoder = nn.Sequential(
            nn.Linear(self.retinal_dim, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Genetic encoder
        self.genetic_encoder = nn.Sequential(
            nn.Linear(self.genetic_dim, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Modality projections
        self.modality_projections = nn.ModuleDict({
            'blood': nn.Linear(self.hidden_dim, self.config.transformer_dim),
            'metabolomics': nn.Linear(self.hidden_dim, self.config.transformer_dim),
            'retinal': nn.Linear(self.hidden_dim, self.config.transformer_dim),
            'genetic': nn.Linear(self.hidden_dim, self.config.transformer_dim),
        })
    
    def _build_transformer(self) -> None:
        """Build transformer components."""
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.transformer_dim,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.transformer_dim * 4,
            dropout=self.config.transformer_dropout,
            activation=self.activation,
            batch_first=True,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers,
        )
        
        # Modality type embeddings
        self.modality_embeddings = nn.Embedding(4, self.config.transformer_dim)
        
        # Positional embeddings
        self.positional_embeddings = nn.Embedding(4, self.config.transformer_dim)
    
    def _build_multi_modal_fusion(self) -> None:
        """Build multi-modal fusion components."""
        # Cross-modal attention
        if self.config.modal_attention == "cross":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.config.transformer_dim,
                num_heads=self.config.transformer_heads,
                dropout=self.config.transformer_dropout,
                batch_first=True,
            )
        
        # Fusion layer
        if self.config.modal_fusion == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=self.config.transformer_dim,
                num_heads=self.config.transformer_heads,
                dropout=self.config.transformer_dropout,
                batch_first=True,
            )
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.config.transformer_dim, self.hidden_dim),
                self.activation_fn,
                nn.Dropout(self.dropout_rate),
            )
        elif self.config.modal_fusion == "concatenation":
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.config.transformer_dim * 4, self.hidden_dim),
                self.activation_fn,
                nn.Dropout(self.dropout_rate),
            )
        elif self.config.modal_fusion == "gating":
            self.gating_network = nn.Sequential(
                nn.Linear(self.config.transformer_dim * 4, self.config.transformer_dim),
                self.activation_fn,
                nn.Linear(self.config.transformer_dim, 4),
                nn.Softmax(dim=-1),
            )
            
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.config.transformer_dim, self.hidden_dim),
                self.activation_fn,
                nn.Dropout(self.dropout_rate),
            )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        
        # Uncertainty head
        if self.enable_uncertainty and self.heteroscedastic:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                self.activation_fn,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, 1),
            )
    
    def _build_organ_specific_components(self) -> None:
        """Build organ-specific aging components."""
        self.organ_heads = nn.ModuleDict()
        
        for organ in self.config.organs:
            self.organ_heads[organ] = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                self.activation_fn,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim // 2, 1),
            )
    
    def _build_longitudinal_components(self) -> None:
        """Build longitudinal modeling components."""
        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
        )
        
        # Aging velocity predictor
        self.velocity_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1),
        )
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features."""
        # Split input into modalities
        blood_features = x[:, :self.blood_dim]
        metabolomics_features = x[:, self.blood_dim:self.blood_dim + self.metabolomic_dim]
        retinal_features = x[:, self.blood_dim + self.metabolomic_dim:self.blood_dim + self.metabolomic_dim + self.retinal_dim]
        genetic_features = x[:, self.blood_dim + self.metabolomic_dim + self.retinal_dim:]
        
        # Encode each modality
        blood_encoded = self.blood_encoder(blood_features)
        metabolomics_encoded = self.metabolomics_encoder(metabolomics_features)
        retinal_encoded = self.retinal_encoder(retinal_features)
        genetic_encoded = self.genetic_encoder(genetic_features)
        
        # Project to transformer dimension
        modality_features = {
            'blood': self.modality_projections['blood'](blood_encoded),
            'metabolomics': self.modality_projections['metabolomics'](metabolomics_encoded),
            'retinal': self.modality_projections['retinal'](retinal_encoded),
            'genetic': self.modality_projections['genetic'](genetic_encoded),
        }
        
        # Apply transformer
        transformer_features = self._apply_transformer(modality_features)
        
        # Apply multi-modal fusion
        fused_features = self._apply_multi_modal_fusion(transformer_features)
        
        return fused_features
    
    def _apply_transformer(self, modality_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply transformer to modality features."""
        # Create input sequence
        modality_names = ['blood', 'metabolomics', 'retinal', 'genetic']
        sequence = []
        
        for i, modality in enumerate(modality_names):
            # Get modality features
            features = modality_features[modality]
            
            # Add modality embedding
            modality_emb = self.modality_embeddings(torch.tensor(i, device=features.device))
            features = features + modality_emb
            
            # Add positional embedding
            pos_emb = self.positional_embeddings(torch.tensor(i, device=features.device))
            features = features + pos_emb
            
            sequence.append(features.unsqueeze(1))  # Add sequence dimension
        
        # Concatenate sequence
        sequence = torch.cat(sequence, dim=1)
        
        # Apply transformer encoder
        transformer_output = self.transformer_encoder(sequence)
        
        return transformer_output
    
    def _apply_multi_modal_fusion(self, transformer_features: torch.Tensor) -> torch.Tensor:
        """Apply multi-modal fusion."""
        if self.config.modal_fusion == "attention":
            # Self-attention fusion
            fused, _ = self.fusion_attention(
                transformer_features, transformer_features, transformer_features
            )
            
            # Global average pooling
            fused = torch.mean(fused, dim=1)
            
            # Apply fusion layer
            fused = self.fusion_layer(fused)
            
        elif self.config.modal_fusion == "concatenation":
            # Flatten and concatenate
            batch_size = transformer_features.size(0)
            fused = transformer_features.view(batch_size, -1)
            
            # Apply fusion layer
            fused = self.fusion_layer(fused)
            
        elif self.config.modal_fusion == "gating":
            # Global average pooling for each modality
            modality_pooled = []
            for i in range(4):
                modality_features = transformer_features[:, i, :]
                modality_pooled.append(modality_features)
            
            # Compute gating weights
            concatenated = torch.cat(modality_pooled, dim=1)
            gate_weights = self.gating_network(concatenated)
            
            # Apply gating
            gated_features = sum(w * feat for w, feat in zip(gate_weights.unbind(dim=1), modality_pooled))
            
            # Apply fusion layer
            fused = self.fusion_layer(gated_features)
        
        return fused
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
        return_uncertainty: bool = False,
        return_attention: bool = False,
        return_organ: bool = False,
        return_velocity: bool = False,
        **kwargs
    ) -> ModelOutput:
        """Forward pass through MM-BAT model."""
        # Extract main features
        features = self._forward_features(x)
        
        # Get predictions
        predicted_age = self.prediction_head(features)
        
        # Get uncertainty estimates
        uncertainty = None
        if return_uncertainty and self.enable_uncertainty:
            uncertainty = self.uncertainty_head(features)
        
        # Get organ-specific predictions
        organ_predictions = None
        if return_organ and self.config.organ_specific:
            organ_predictions = {}
            for organ, head in self.organ_heads.items():
                organ_predictions[organ] = head(features)
        
        # Get aging velocity
        velocity = None
        if return_velocity and self.config.enable_longitudinal:
            velocity = self.velocity_predictor(features)
        
        # Create auxiliary outputs
        auxiliary_outputs = {}
        if organ_predictions:
            auxiliary_outputs.update(organ_predictions)
        if velocity is not None:
            auxiliary_outputs['velocity'] = velocity
        
        # Create output object
        output = ModelOutput(
            predicted_age=predicted_age,
            uncertainty=uncertainty,
            features=features if return_features else None,
            auxiliary_outputs=auxiliary_outputs if auxiliary_outputs else None,
            metadata={
                'model_tier': 3,
                'input_modalities': ['blood_biomarkers', 'metabolomics', 'retinal', 'genetic'],
                'organ_specific': self.config.organ_specific,
                'longitudinal': self.config.enable_longitudinal,
            },
        )
        
        return output
    
    def get_modality_importance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get modality-level importance scores."""
        self.eval()
        
        with torch.no_grad():
            # Split input into modalities
            blood_features = x[:, :self.blood_dim]
            metabolomics_features = x[:, self.blood_dim:self.blood_dim + self.metabolomic_dim]
            retinal_features = x[:, self.blood_dim + self.metabolomic_dim:self.blood_dim + self.metabolomic_dim + self.retinal_dim]
            genetic_features = x[:, self.blood_dim + self.metabolomics_dim + self.retinal_dim:]
            
            # Get baseline prediction
            baseline_output = self.forward(x)
            baseline_prediction = baseline_output.predicted_age
            
            # Ablate each modality
            modalities = ['blood', 'metabolomics', 'retinal', 'genetic']
            importance_scores = {}
            
            # Blood ablation
            ablated_blood = torch.cat([
                torch.zeros_like(blood_features),
                metabolomics_features,
                retinal_features,
                genetic_features,
            ], dim=1)
            ablated_output = self.forward(ablated_blood)
            importance_scores['blood'] = torch.mean(torch.abs(ablated_output.predicted_age - baseline_prediction))
            
            # Metabolomics ablation
            ablated_metabolomics = torch.cat([
                blood_features,
                torch.zeros_like(metabolomics_features),
                retinal_features,
                genetic_features,
            ], dim=1)
            ablated_output = self.forward(ablated_metabolomics)
            importance_scores['metabolomics'] = torch.mean(torch.abs(ablated_output.predicted_age - baseline_prediction))
            
            # Retinal ablation
            ablated_retinal = torch.cat([
                blood_features,
                metabolomics_features,
                torch.zeros_like(retinal_features),
                genetic_features,
            ], dim=1)
            ablated_output = self.forward(ablated_retinal)
            importance_scores['retinal'] = torch.mean(torch.abs(ablated_output.predicted_age - baseline_prediction))
            
            # Genetic ablation
            ablated_genetic = torch.cat([
                blood_features,
                metabolomics_features,
                retinal_features,
                torch.zeros_like(genetic_features),
            ], dim=1)
            ablated_output = self.forward(ablated_genetic)
            importance_scores['genetic'] = torch.mean(torch.abs(ablated_output.predicted_age - baseline_prediction))
            
            return importance_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        
        mmbat_info = {
            'model_tier': 3,
            'input_modalities': ['blood_biomarkers', 'metabolomics', 'retinal', 'genetic'],
            'blood_biomarker_count': self.blood_dim,
            'metabolomic_count': self.metabolomic_dim,
            'retinal_feature_dim': self.retinal_dim,
            'genetic_feature_dim': self.genetic_dim,
            'transformer_layers': self.config.transformer_layers,
            'transformer_heads': self.config.transformer_heads,
            'transformer_dim': self.config.transformer_dim,
            'modal_attention': self.config.modal_attention,
            'modal_fusion': self.config.modal_fusion,
            'organ_specific': self.config.organ_specific,
            'organs': self.config.organs,
            'enable_longitudinal': self.config.enable_longitudinal,
            'target_mae': self.config.target_mae,
            'target_r2': self.config.target_r2,
        }
        
        base_info.update(mmbat_info)
        return base_info


def create_mmbat_model(config: Tier3Config) -> MultiModalBiologicalAgeTransformer:
    """Create a Multi-Modal Biological Age Transformer model."""
    return MultiModalBiologicalAgeTransformer(config)