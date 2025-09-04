"""
Base model class for HAMBAE algorithm system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelOutput:
    """Base model output structure."""
    predicted_age: torch.Tensor
    uncertainty: Optional[torch.Tensor] = None
    features: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    auxiliary_outputs: Optional[Dict[str, torch.Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseBiologicalAgeModel(nn.Module, ABC):
    """
    Base class for all HAMBAE biological age models.
    
    This abstract base class defines the interface that all biological age
    models must implement, providing common functionality and ensuring
    consistency across different model architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        enable_uncertainty: bool = True,
        mc_dropout: bool = True,
        heteroscedastic: bool = True,
        **kwargs
    ):
        """
        Initialize base biological age model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
            dropout_rate: Dropout rate
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            enable_uncertainty: Whether to enable uncertainty quantification
            mc_dropout: Whether to use Monte Carlo dropout
            heteroscedastic: Whether to use heteroscedastic uncertainty
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.enable_uncertainty = enable_uncertainty
        self.mc_dropout = mc_dropout
        self.heteroscedastic = heteroscedastic
        
        # Initialize activation function
        self.activation_fn = self._get_activation_function(activation)
        
        # Build model components
        self._build_model()
        
        # Initialize weights
        self._initialize_weights()
        
        # Set training mode
        self.training_mode = True
        
        logger.info(f"Initialized {self.__class__.__name__} with input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activation_map = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unknown activation function: {activation}")
        
        return activation_map[activation]
    
    @abstractmethod
    def _build_model(self) -> None:
        """Build model architecture - to be implemented by subclasses."""
        pass
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # Linear layer weights
                    nn.init.xavier_uniform_(param)
                else:
                    # Bias or 1D weights
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _create_layer(
        self, 
        in_dim: int, 
        out_dim: int, 
        layer_idx: int = 0
    ) -> nn.Module:
        """Create a single layer with optional normalization and dropout."""
        layers = []
        
        # Linear layer
        linear = nn.Linear(in_dim, out_dim)
        layers.append(linear)
        
        # Batch normalization
        if self.use_batch_norm and layer_idx > 0:
            batch_norm = nn.BatchNorm1d(out_dim)
            layers.append(batch_norm)
        
        # Layer normalization
        if self.use_layer_norm and layer_idx > 0:
            layer_norm = nn.LayerNorm(out_dim)
            layers.append(layer_norm)
        
        # Activation
        if layer_idx < self.num_layers - 1:  # No activation on final layer
            layers.append(self.activation_fn)
        
        # Dropout
        if self.dropout_rate > 0 and layer_idx > 0:
            if self.mc_dropout and self.enable_uncertainty:
                dropout = nn.Dropout(self.dropout_rate)
            else:
                dropout = nn.Dropout(self.dropout_rate)
            layers.append(dropout)
        
        return nn.Sequential(*layers)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
        return_uncertainty: bool = False,
        return_attention: bool = False,
        **kwargs
    ) -> ModelOutput:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            return_features: Whether to return intermediate features
            return_uncertainty: Whether to return uncertainty estimates
            return_attention: Whether to return attention weights
            **kwargs: Additional model-specific arguments
            
        Returns:
            ModelOutput object containing predictions and optional metadata
        """
        # Handle multi-modal input
        if isinstance(x, dict):
            x = self._process_multi_modal_input(x)
        
        # Ensure input is proper shape
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Forward pass through main model
        features = self._forward_features(x)
        
        # Get predictions
        predicted_age = self._get_predictions(features)
        
        # Get uncertainty estimates
        uncertainty = None
        if return_uncertainty and self.enable_uncertainty:
            uncertainty = self._get_uncertainty(features, x)
        
        # Get attention weights
        attention_weights = None
        if return_attention:
            attention_weights = self._get_attention_weights(features)
        
        # Get auxiliary outputs
        auxiliary_outputs = self._get_auxiliary_outputs(features, x)
        
        # Create output object
        output = ModelOutput(
            predicted_age=predicted_age,
            uncertainty=uncertainty,
            features=features if return_features else None,
            attention_weights=attention_weights,
            auxiliary_outputs=auxiliary_outputs,
            metadata=self._get_metadata(),
        )
        
        return output
    
    @abstractmethod
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to extract features - to be implemented by subclasses."""
        pass
    
    def _get_predictions(self, features: torch.Tensor) -> torch.Tensor:
        """Get age predictions from features."""
        if hasattr(self, 'prediction_head'):
            return self.prediction_head(features)
        else:
            return features
    
    def _get_uncertainty(
        self, 
        features: torch.Tensor, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Get uncertainty estimates."""
        if self.heteroscedastic and hasattr(self, 'uncertainty_head'):
            return self.uncertainty_head(features)
        elif self.mc_dropout:
            return self._mc_dropout_uncertainty(x)
        else:
            return None
    
    def _get_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """Get attention weights."""
        if hasattr(self, 'attention_weights'):
            return self.attention_weights
        else:
            return None
    
    def _get_auxiliary_outputs(
        self, 
        features: torch.Tensor, 
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Get auxiliary outputs."""
        if hasattr(self, 'auxiliary_heads'):
            auxiliary_outputs = {}
            for name, head in self.auxiliary_heads.items():
                auxiliary_outputs[name] = head(features)
            return auxiliary_outputs
        else:
            return {}
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            'model_class': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'enable_uncertainty': self.enable_uncertainty,
            'parameter_count': sum(p.numel() for p in self.parameters()),
        }
    
    def _process_multi_modal_input(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process multi-modal input into single tensor."""
        # Default implementation: concatenate all modalities
        tensors = []
        for key, value in x.items():
            if value is not None:
                tensors.append(value)
        
        if tensors:
            return torch.cat(tensors, dim=-1)
        else:
            raise ValueError("No valid input tensors found")
    
    def _mc_dropout_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        """Compute uncertainty using Monte Carlo dropout."""
        if not self.training:
            self.eval()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                # Enable dropout during inference
                for module in self.modules():
                    if isinstance(module, nn.Dropout):
                        module.train()
                
                output = self.forward(x, return_uncertainty=False)
                predictions.append(output.predicted_age)
        
        # Compute standard deviation as uncertainty
        predictions = torch.stack(predictions)
        uncertainty = torch.std(predictions, dim=0)
        
        return uncertainty
    
    def enable_uncertainty_estimation(self) -> None:
        """Enable uncertainty estimation."""
        self.enable_uncertainty = True
        if self.mc_dropout:
            # Enable dropout during inference
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
    
    def disable_uncertainty_estimation(self) -> None:
        """Disable uncertainty estimation."""
        self.enable_uncertainty = False
        if self.mc_dropout:
            # Disable dropout during inference
            self.eval()
    
    def get_feature_importance(self, x: torch.Tensor, method: str = "gradient") -> torch.Tensor:
        """
        Get feature importance scores.
        
        Args:
            x: Input tensor
            method: Importance computation method
            
        Returns:
            Feature importance scores
        """
        if method == "gradient":
            return self._gradient_importance(x)
        elif method == "permutation":
            return self._permutation_importance(x)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _gradient_importance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute feature importance using gradients."""
        x.requires_grad_(True)
        
        output = self.forward(x)
        loss = output.predicted_age.sum()
        
        # Compute gradients
        gradients = torch.autograd.grad(loss, x)[0]
        
        # Average absolute gradients
        importance = torch.abs(gradients).mean(dim=0)
        
        return importance
    
    def _permutation_importance(self, x: torch.Tensor, n_permutations: int = 10) -> torch.Tensor:
        """Compute feature importance using permutation."""
        baseline_output = self.forward(x).predicted_age
        
        importance_scores = torch.zeros(x.shape[1])
        
        for i in range(x.shape[1]):
            permuted_scores = []
            
            for _ in range(n_permutations):
                # Permute feature i
                x_permuted = x.clone()
                x_permuted[:, i] = x_permuted[torch.randperm(x.shape[0]), i]
                
                # Get prediction
                permuted_output = self.forward(x_permuted).predicted_age
                
                # Compute importance as prediction change
                importance = torch.mean(torch.abs(permuted_output - baseline_output))
                permuted_scores.append(importance)
            
            importance_scores[i] = torch.mean(torch.stack(permuted_scores))
        
        return importance_scores
    
    def compute_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
        auxiliary_outputs: Optional[Dict[str, torch.Tensor]] = None,
        auxiliary_targets: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss for model predictions.
        
        Args:
            predictions: Model predictions
            targets: Target values
            uncertainty: Uncertainty estimates
            auxiliary_outputs: Auxiliary model outputs
            auxiliary_targets: Auxiliary target values
            
        Returns:
            Computed loss
        """
        # Main age prediction loss
        if uncertainty is not None and self.heteroscedastic:
            # Heteroscedastic loss
            loss = self._heteroscedastic_loss(predictions, targets, uncertainty)
        else:
            # Standard MSE loss
            loss = F.mse_loss(predictions, targets)
        
        # Auxiliary losses
        if auxiliary_outputs and auxiliary_targets:
            auxiliary_loss = 0.0
            for name, output in auxiliary_outputs.items():
                if name in auxiliary_targets:
                    target = auxiliary_targets[name]
                    if output.shape == target.shape:
                        auxiliary_loss += F.mse_loss(output, target)
            
            # Weight auxiliary loss
            loss += 0.1 * auxiliary_loss
        
        return loss
    
    def _heteroscedastic_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Compute heteroscedastic loss."""
        # Predictive variance
        variance = torch.exp(uncertainty)
        
        # Gaussian negative log-likelihood
        loss = 0.5 * torch.exp(-uncertainty) * (predictions - targets) ** 2 + 0.5 * uncertainty
        
        return torch.mean(loss)
    
    def predict(
        self, 
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        return_uncertainty: bool = False,
        n_samples: int = 50
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor or dictionary of tensors
            return_uncertainty: Whether to return uncertainty estimates
            n_samples: Number of samples for MC dropout
            
        Returns:
            Predictions and optionally uncertainty
        """
        self.eval()
        
        with torch.no_grad():
            if return_uncertainty and self.enable_uncertainty:
                if self.mc_dropout:
                    # Monte Carlo dropout
                    predictions = []
                    for _ in range(n_samples):
                        output = self.forward(x, return_uncertainty=False)
                        predictions.append(output.predicted_age)
                    
                    predictions = torch.stack(predictions)
                    mean_prediction = torch.mean(predictions, dim=0)
                    uncertainty = torch.std(predictions, dim=0)
                    
                    return mean_prediction, uncertainty
                else:
                    # Standard uncertainty estimation
                    output = self.forward(x, return_uncertainty=True)
                    return output.predicted_age, output.uncertainty
            else:
                output = self.forward(x)
                return output.predicted_age
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self._get_config(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from file."""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")
    
    def _get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'enable_uncertainty': self.enable_uncertainty,
            'mc_dropout': self.mc_dropout,
            'heteroscedastic': self.heteroscedastic,
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_class': self.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'enable_uncertainty': self.enable_uncertainty,
            'mc_dropout': self.mc_dropout,
            'heteroscedastic': self.heteroscedastic,
            'device': next(self.parameters()).device,
            'dtype': next(self.parameters()).dtype,
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_model_info()
        return f"{self.__class__.__name__}({info})"