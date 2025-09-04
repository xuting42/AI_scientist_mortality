"""
Tier 1: Clinical Biomarker Aging Network (CBAN) for HAMBAE algorithm system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from dataclasses import dataclass

from .base_model import BaseBiologicalAgeModel, ModelOutput
from ..config import Tier1Config

logger = logging.getLogger(__name__)


@dataclass
class EpigeneticProxyOutput:
    """Output from epigenetic proxy model."""
    epigenetic_age: torch.Tensor
    age_acceleration: torch.Tensor
    aging_rate: torch.Tensor
    biological_age: torch.Tensor


class ClinicalBiomarkerAgingNetwork(BaseBiologicalAgeModel):
    """
    Tier 1: Clinical Biomarker Aging Network (CBAN).
    
    This model implements the first tier of the HAMBAE algorithm system,
    focusing on blood biomarker-based biological age estimation with
    epigenetic proxy development and uncertainty quantification.
    """
    
    def __init__(
        self,
        config: Tier1Config,
        **kwargs
    ):
        """
        Initialize Clinical Biomarker Aging Network.
        
        Args:
            config: Tier 1 configuration
            **kwargs: Additional arguments
        """
        self.config = config
        
        super().__init__(
            input_dim=config.blood_biomarker_count,
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
        
        # Epigenetic proxy components
        if config.enable_epigenetic_proxy:
            self._build_epigenetic_proxy()
        
        # Multi-task learning components
        if config.multi_task:
            self._build_multi_task_components()
        
        # Explainability components
        if config.enable_explainability:
            self._build_explainability_components()
        
        # Biological constraint components
        if config.enforce_monotonicity:
            self._build_monotonicity_constraints()
        
        logger.info(f"Initialized CBAN with {config.blood_biomarker_count} blood biomarkers")
    
    def _build_model(self) -> None:
        """Build the main model architecture."""
        # Input projection layer
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.hidden_layers.append(
                self._create_layer(self.hidden_dim, self.hidden_dim, layer_idx=i + 1)
            )
        
        # Prediction head
        self.prediction_head = nn.Linear(self.hidden_dim, 1)
        
        # Uncertainty head
        if self.enable_uncertainty and self.heteroscedastic:
            self.uncertainty_head = nn.Linear(self.hidden_dim, 1)
    
    def _build_epigenetic_proxy(self) -> None:
        """Build epigenetic proxy model."""
        # Epigenetic age estimation from blood biomarkers
        self.epigenetic_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.config.epigenetic_hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.config.epigenetic_hidden_dim, self.config.epigenetic_hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
        )
        
        self.epigenetic_age_head = nn.Linear(self.config.epigenetic_hidden_dim, 1)
        self.aging_rate_head = nn.Linear(self.config.epigenetic_hidden_dim, 1)
        
        # Epigenetic aging signature extraction
        self.epigenetic_signature = nn.Sequential(
            nn.Linear(self.config.epigenetic_hidden_dim, 64),
            self.activation_fn,
            nn.Linear(64, 32),
            self.activation_fn,
        )
    
    def _build_multi_task_components(self) -> None:
        """Build multi-task learning components."""
        # Mortality prediction head
        self.mortality_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Healthspan prediction head
        self.healthspan_head = nn.Linear(self.hidden_dim, 1)
    
    def _build_explainability_components(self) -> None:
        """Build explainability components."""
        # Feature importance weights
        self.feature_weights = nn.Parameter(torch.ones(self.input_dim))
        
        # Attention mechanism for interpretability
        self.attention_layer = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        
        # Explainable feature extraction
        self.explainable_features = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim // 2),
            self.activation_fn,
            nn.Linear(self.hidden_dim // 2, self.input_dim),
        )
    
    def _build_monotonicity_constraints(self) -> None:
        """Build monotonicity constraint components."""
        # Monotonicity constraints for age-related biomarkers
        self.monotonicity_weights = nn.Parameter(torch.ones(self.input_dim))
        
        # Constraint regularization parameters
        self.monotonicity_penalty = self.config.monotonicity_penalty
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features."""
        # Input projection
        features = self.input_projection(x)
        
        # Apply hidden layers
        for i, layer in enumerate(self.hidden_layers):
            features = layer(features)
        
        return features
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
        return_uncertainty: bool = False,
        return_attention: bool = False,
        return_epigenetic: bool = False,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through CBAN model.
        
        Args:
            x: Input tensor (blood biomarkers)
            return_features: Whether to return intermediate features
            return_uncertainty: Whether to return uncertainty estimates
            return_attention: Whether to return attention weights
            return_epigenetic: Whether to return epigenetic proxy outputs
            
        Returns:
            ModelOutput object containing predictions and metadata
        """
        # Extract main features
        features = self._forward_features(x)
        
        # Get predictions
        predicted_age = self.prediction_head(features)
        
        # Get uncertainty estimates
        uncertainty = None
        if return_uncertainty and self.enable_uncertainty:
            uncertainty = self._get_uncertainty(features, x)
        
        # Get attention weights
        attention_weights = None
        if return_attention and self.config.enable_explainability:
            attention_weights = self._get_attention_weights(features)
        
        # Get epigenetic proxy outputs
        epigenetic_output = None
        if return_epigenetic and self.config.enable_epigenetic_proxy:
            epigenetic_output = self._get_epigenetic_proxy(x)
        
        # Get auxiliary outputs
        auxiliary_outputs = {}
        if self.config.multi_task:
            auxiliary_outputs['mortality'] = self.mortality_head(features)
            auxiliary_outputs['healthspan'] = self.healthspan_head(features)
        
        if self.config.enable_explainability:
            auxiliary_outputs['feature_importance'] = self.explainable_features(x)
        
        # Create output object
        output = ModelOutput(
            predicted_age=predicted_age,
            uncertainty=uncertainty,
            features=features if return_features else None,
            attention_weights=attention_weights,
            auxiliary_outputs=auxiliary_outputs if auxiliary_outputs else None,
            metadata={
                'epigenetic_proxy': epigenetic_output,
                'model_tier': 1,
                'input_modality': 'blood_biomarkers',
            },
        )
        
        return output
    
    def _get_epigenetic_proxy(self, x: torch.Tensor) -> EpigeneticProxyOutput:
        """Compute epigenetic proxy from blood biomarkers."""
        # Encode blood biomarkers for epigenetic age estimation
        epigenetic_features = self.epigenetic_encoder(x)
        
        # Estimate epigenetic age
        epigenetic_age = self.epigenetic_age_head(epigenetic_features)
        
        # Estimate aging rate
        aging_rate = self.aging_rate_head(epigenetic_features)
        
        # Compute age acceleration
        # This would require chronological age as input
        # For now, use a simplified approach
        age_acceleration = epigenetic_age - self.prediction_head(epigenetic_features)
        
        # Extract epigenetic signature
        epigenetic_signature = self.epigenetic_signature(epigenetic_features)
        
        # Compute biological age combining epigenetic and biomarker information
        biological_age = self._compute_biological_age(
            epigenetic_age, aging_rate, epigenetic_signature
        )
        
        return EpigeneticProxyOutput(
            epigenetic_age=epigenetic_age,
            age_acceleration=age_acceleration,
            aging_rate=aging_rate,
            biological_age=biological_age,
        )
    
    def _compute_biological_age(
        self, 
        epigenetic_age: torch.Tensor, 
        aging_rate: torch.Tensor,
        epigenetic_signature: torch.Tensor
    ) -> torch.Tensor:
        """Compute biological age from epigenetic components."""
        # Combine epigenetic age with aging rate
        biological_age = epigenetic_age + 0.1 * aging_rate
        
        # Apply epigenetic signature modulation
        signature_effect = torch.mean(epigenetic_signature, dim=1, keepdim=True)
        biological_age = biological_age + 0.05 * signature_effect
        
        return biological_age
    
    def _get_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Compute attention weights for explainability."""
        # Reshape features for attention
        seq_features = features.unsqueeze(1)  # Add sequence dimension
        
        # Self-attention
        attended_features, attention_weights = self.attention_layer(
            seq_features, seq_features, seq_features
        )
        
        return attention_weights.squeeze(1)  # Remove sequence dimension
    
    def compute_epigenetic_proxy_loss(
        self, 
        epigenetic_output: EpigeneticProxyOutput,
        chronological_age: torch.Tensor,
        mortality_status: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute loss for epigenetic proxy model.
        
        Args:
            epigenetic_output: Epigenetic proxy output
            chronological_age: Chronological age
            mortality_status: Optional mortality status
            
        Returns:
            Epigenetic proxy loss
        """
        # Epigenetic age prediction loss
        epigenetic_loss = F.mse_loss(
            epigenetic_output.epigenetic_age, chronological_age
        )
        
        # Age acceleration loss (should be small for healthy individuals)
        acceleration_loss = F.mse_loss(epigenetic_output.age_acceleration, torch.zeros_like(epigenetic_output.age_acceleration))
        
        # Biological age prediction loss
        biological_loss = F.mse_loss(
            epigenetic_output.biological_age, chronological_age
        )
        
        # Mortality prediction loss if available
        mortality_loss = 0.0
        if mortality_status is not None:
            # Predict mortality from epigenetic features
            mortality_pred = torch.sigmoid(epigenetic_output.epigenetic_age - chronological_age)
            mortality_loss = F.binary_cross_entropy(mortality_pred, mortality_status.float())
        
        # Combine losses
        total_loss = (
            0.4 * epigenetic_loss +
            0.2 * acceleration_loss +
            0.3 * biological_loss +
            0.1 * mortality_loss
        )
        
        return total_loss
    
    def compute_monotonicity_loss(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Compute monotonicity constraint loss.
        
        Args:
            x: Input tensor
            features: Feature tensor
            
        Returns:
            Monotonicity loss
        """
        if not self.config.enforce_monotonicity:
            return torch.tensor(0.0, device=x.device)
        
        # Compute gradients of output with respect to input
        gradients = torch.autograd.grad(
            outputs=features.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Apply monotonicity constraints
        # For age-related biomarkers, gradients should be positive
        monotonicity_loss = 0.0
        for i in range(self.input_dim):
            if self.monotonicity_weights[i] > 0:  # Should be monotonically increasing
                monotonicity_loss += torch.mean(torch.relu(-gradients[:, i]))
            else:  # Should be monotonically decreasing
                monotonicity_loss += torch.mean(torch.relu(gradients[:, i]))
        
        return self.monotonicity_penalty * monotonicity_loss
    
    def compute_feature_importance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature importance scores.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature importance scores
        """
        if self.config.enable_explainability:
            # Use explainable features
            explainable_output = self.explainable_features(x)
            importance = torch.abs(explainable_output).mean(dim=0)
            
            # Apply feature weights
            importance = importance * torch.abs(self.feature_weights)
            
            return importance
        else:
            # Use gradient-based importance
            return self._gradient_importance(x)
    
    def get_biomarker_interpretation(
        self, 
        x: torch.Tensor,
        biomarker_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get interpretation of biomarker contributions.
        
        Args:
            x: Input tensor
            biomarker_names: Names of biomarkers
            
        Returns:
            Dictionary with interpretation results
        """
        # Compute feature importance
        importance = self.compute_feature_importance(x)
        
        # Get predictions
        with torch.no_grad():
            output = self.forward(x, return_epigenetic=True)
            predicted_age = output.predicted_age
            
            if output.metadata['epigenetic_proxy'] is not None:
                epigenetic_output = output.metadata['epigenetic_proxy']
                epigenetic_age = epigenetic_output.epigenetic_age
                age_acceleration = epigenetic_output.age_acceleration
            else:
                epigenetic_age = None
                age_acceleration = None
        
        # Create interpretation dictionary
        interpretation = {
            'predicted_age': predicted_age,
            'feature_importance': importance,
            'epigenetic_age': epigenetic_age,
            'age_acceleration': age_acceleration,
        }
        
        # Add biomarker names if provided
        if biomarker_names and len(biomarker_names) == self.input_dim:
            interpretation['biomarker_contributions'] = {
                name: float(imp) for name, imp in zip(biomarker_names, importance)
            }
        
        return interpretation
    
    def predict_biological_age(
        self, 
        x: torch.Tensor,
        include_epigenetic: bool = True,
        return_uncertainty: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Predict biological age with optional epigenetic proxy.
        
        Args:
            x: Input tensor
            include_epigenetic: Whether to include epigenetic proxy
            return_uncertainty: Whether to return uncertainty
            
        Returns:
            Biological age prediction and optionally uncertainty
        """
        self.eval()
        
        with torch.no_grad():
            output = self.forward(
                x, 
                return_epigenetic=include_epigenetic,
                return_uncertainty=return_uncertainty
            )
            
            if include_epigenetic and output.metadata['epigenetic_proxy'] is not None:
                biological_age = output.metadata['epigenetic_proxy'].biological_age
            else:
                biological_age = output.predicted_age
            
            if return_uncertainty:
                return biological_age, output.uncertainty
            else:
                return biological_age
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        base_info = super().get_model_info()
        
        cban_info = {
            'model_tier': 1,
            'input_modality': 'blood_biomarkers',
            'biomarker_count': self.config.blood_biomarker_count,
            'enable_epigenetic_proxy': self.config.enable_epigenetic_proxy,
            'enable_multi_task': self.config.multi_task,
            'enable_explainability': self.config.enable_explainability,
            'enforce_monotonicity': self.config.enforce_monotonicity,
            'target_mae': self.config.target_mae,
            'target_r2': self.config.target_r2,
        }
        
        base_info.update(cban_info)
        return base_info


def create_cban_model(config: Tier1Config) -> ClinicalBiomarkerAgingNetwork:
    """
    Create a Clinical Biomarker Aging Network model.
    
    Args:
        config: Tier 1 configuration
        
    Returns:
        Configured CBAN model
    """
    return ClinicalBiomarkerAgingNetwork(config)