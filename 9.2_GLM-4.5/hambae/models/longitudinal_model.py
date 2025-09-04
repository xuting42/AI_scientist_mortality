"""
Longitudinal aging model for HAMBAE algorithm system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from dataclasses import dataclass

from .base_model import BaseBiologicalAgeModel, ModelOutput

logger = logging.getLogger(__name__)


@dataclass
class LongitudinalOutput:
    """Output from longitudinal aging model."""
    aging_trajectory: torch.Tensor
    aging_velocity: torch.Tensor
    intervention_effects: Optional[torch.Tensor] = None
    trajectory_uncertainty: Optional[torch.Tensor] = None


class LongitudinalAgingModel(BaseBiologicalAgeModel):
    """
    Longitudinal aging model for trajectory analysis and velocity estimation.
    
    This model implements Gaussian process-based longitudinal aging trajectory
    analysis and aging velocity estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        temporal_dim: int = 32,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        longitudinal_window: int = 5,
        **kwargs
    ):
        """Initialize longitudinal aging model."""
        self.temporal_dim = temporal_dim
        self.longitudinal_window = longitudinal_window
        
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            **kwargs
        )
        
        # Build longitudinal-specific components
        self._build_longitudinal_components()
        
        logger.info(f"Initialized Longitudinal Aging Model with {input_dim} features")
    
    def _build_model(self) -> None:
        """Build the main model architecture."""
        # Input encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    self.activation_fn,
                    nn.Dropout(self.dropout_rate),
                    nn.LayerNorm(self.hidden_dim),
                )
            )
        
        # Current age prediction head
        self.age_head = nn.Linear(self.hidden_dim, 1)
        
        # Uncertainty head
        if self.enable_uncertainty and self.heteroscedastic:
            self.uncertainty_head = nn.Linear(self.hidden_dim, 1)
    
    def _build_longitudinal_components(self) -> None:
        """Build longitudinal-specific components."""
        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, self.temporal_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
        )
        
        # Longitudinal feature encoder
        self.longitudinal_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim + self.temporal_dim, self.hidden_dim),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            self.activation_fn,
        )
        
        # Aging velocity predictor
        self.velocity_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 4, 1),
        )
        
        # Trajectory predictor
        self.trajectory_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 4, self.longitudinal_window),
        )
        
        # Intervention effect predictor
        self.intervention_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            self.activation_fn,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim // 4, 1),
        )
    
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features."""
        # Encode input
        features = self.input_encoder(x)
        
        # Apply hidden layers
        for layer in self.hidden_layers:
            features = layer(features)
        
        return features
    
    def forward(
        self, 
        x: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        return_features: bool = False,
        return_uncertainty: bool = False,
        return_longitudinal: bool = True,
        **kwargs
    ) -> ModelOutput:
        """Forward pass through longitudinal aging model."""
        # Extract main features
        features = self._forward_features(x)
        
        # Get age predictions
        predicted_age = self.age_head(features)
        
        # Get uncertainty estimates
        uncertainty = None
        if return_uncertainty and self.enable_uncertainty:
            uncertainty = self.uncertainty_head(features)
        
        # Get longitudinal outputs
        longitudinal_outputs = None
        if return_longitudinal:
            longitudinal_outputs = self._compute_longitudinal_outputs(features, timestamps)
        
        # Create auxiliary outputs
        auxiliary_outputs = {}
        if longitudinal_outputs is not None:
            auxiliary_outputs.update(longitudinal_outputs)
        
        # Create output object
        output = ModelOutput(
            predicted_age=predicted_age,
            uncertainty=uncertainty,
            features=features if return_features else None,
            auxiliary_outputs=auxiliary_outputs if auxiliary_outputs else None,
            metadata={
                'model_type': 'longitudinal',
                'longitudinal_outputs': longitudinal_outputs,
            },
        )
        
        return output
    
    def _compute_longitudinal_outputs(
        self, 
        features: torch.Tensor, 
        timestamps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute longitudinal-specific outputs."""
        longitudinal_outputs = {}
        
        # Encode temporal information
        if timestamps is not None:
            # Normalize timestamps
            normalized_timestamps = (timestamps - timestamps.mean()) / (timestamps.std() + 1e-8)
            temporal_features = self.temporal_encoder(normalized_timestamps.unsqueeze(1))
        else:
            # Use zero temporal features if no timestamps provided
            temporal_features = torch.zeros(features.size(0), self.temporal_dim, device=features.device)
        
        # Combine features with temporal information
        combined_features = torch.cat([features, temporal_features], dim=1)
        
        # Encode longitudinal features
        longitudinal_features = self.longitudinal_encoder(combined_features)
        
        # Compute aging velocity
        aging_velocity = self.velocity_predictor(longitudinal_features)
        longitudinal_outputs['aging_velocity'] = aging_velocity
        
        # Compute aging trajectory
        aging_trajectory = self.trajectory_predictor(longitudinal_features)
        longitudinal_outputs['aging_trajectory'] = aging_trajectory
        
        # Compute intervention effects
        intervention_effects = self.intervention_predictor(longitudinal_features)
        longitudinal_outputs['intervention_effects'] = intervention_effects
        
        return longitudinal_outputs
    
    def predict_trajectory(
        self, 
        x: torch.Tensor,
        future_timestamps: torch.Tensor,
        current_timestamp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict future aging trajectory.
        
        Args:
            x: Input features
            future_timestamps: Future time points
            current_timestamp: Current timestamp
            
        Returns:
            Predicted aging trajectory
        """
        self.eval()
        
        with torch.no_grad():
            # Get current features
            features = self._forward_features(x)
            
            # Encode temporal information
            if current_timestamp is not None:
                normalized_current = (current_timestamp - current_timestamp.mean()) / (current_timestamp.std() + 1e-8)
                temporal_features = self.temporal_encoder(normalized_current.unsqueeze(1))
            else:
                temporal_features = torch.zeros(features.size(0), self.temporal_dim, device=features.device)
            
            # Combine features
            combined_features = torch.cat([features, temporal_features], dim=1)
            longitudinal_features = self.longitudinal_encoder(combined_features)
            
            # Get current velocity
            current_velocity = self.velocity_predictor(longitudinal_features)
            
            # Predict future trajectory
            if current_timestamp is not None:
                time_deltas = future_timestamps - current_timestamp
                predicted_trajectory = features + current_velocity * time_deltas.unsqueeze(1)
            else:
                # Simple linear extrapolation
                predicted_trajectory = features + current_velocity * future_timestamps.unsqueeze(1)
            
            return predicted_trajectory
    
    def compute_velocity_loss(
        self,
        outputs: ModelOutput,
        true_velocities: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute velocity-specific loss.
        
        Args:
            outputs: Model outputs
            true_velocities: True aging velocities
            
        Returns:
            Velocity loss
        """
        if outputs.auxiliary_outputs is None or 'aging_velocity' not in outputs.auxiliary_outputs:
            return torch.tensor(0.0, device=outputs.predicted_age.device)
        
        predicted_velocities = outputs.auxiliary_outputs['aging_velocity']
        
        if true_velocities is not None:
            # Supervised velocity loss
            velocity_loss = F.mse_loss(predicted_velocities, true_velocities)
        else:
            # Unsupervised velocity regularization (encourage smooth trajectories)
            velocity_loss = torch.mean(torch.abs(predicted_velocities)) * 0.1
        
        return velocity_loss
    
    def compute_trajectory_loss(
        self,
        outputs: ModelOutput,
        true_trajectories: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute trajectory-specific loss.
        
        Args:
            outputs: Model outputs
            true_trajectories: True aging trajectories
            
        Returns:
            Trajectory loss
        """
        if outputs.auxiliary_outputs is None or 'aging_trajectory' not in outputs.auxiliary_outputs:
            return torch.tensor(0.0, device=outputs.predicted_age.device)
        
        predicted_trajectories = outputs.auxiliary_outputs['aging_trajectory']
        
        if true_trajectories is not None:
            # Supervised trajectory loss
            trajectory_loss = F.mse_loss(predicted_trajectories, true_trajectories)
        else:
            # Unsupervised trajectory regularization (encourage smooth trajectories)
            # Use second derivative penalty
            if predicted_trajectories.size(1) >= 3:
                second_derivative = predicted_trajectories[:, 2:] - 2 * predicted_trajectories[:, 1:-1] + predicted_trajectories[:, :-2]
                trajectory_loss = torch.mean(torch.square(second_derivative)) * 0.1
            else:
                trajectory_loss = torch.tensor(0.0, device=outputs.predicted_age.device)
        
        return trajectory_loss


def create_longitudinal_model(
    input_dim: int,
    hidden_dim: int = 256,
    temporal_dim: int = 32,
    num_layers: int = 3,
    dropout_rate: float = 0.1,
    longitudinal_window: int = 5,
    **kwargs
) -> LongitudinalAgingModel:
    """Create a longitudinal aging model."""
    return LongitudinalAgingModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        temporal_dim=temporal_dim,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        longitudinal_window=longitudinal_window,
        **kwargs
    )