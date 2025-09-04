"""
HENAW (Hierarchical Ensemble Network for Aging Waves) implementation.

Multi-scale temporal aging pattern detection with hierarchical attention
and ensemble learning for biological age estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings


class HierarchicalAttention(nn.Module):
    """Hierarchical attention mechanism for multi-scale temporal features."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize hierarchical attention.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Multi-head attention for each temporal scale
        self.rapid_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.intermediate_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.slow_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-scale attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim * 3,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection layers
        self.rapid_proj = nn.Linear(input_dim, hidden_dim)
        self.intermediate_proj = nn.Linear(input_dim, hidden_dim)
        self.slow_proj = nn.Linear(input_dim, hidden_dim)
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        self.layer_norm_cross = nn.LayerNorm(input_dim * 3)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        rapid_features: torch.Tensor,
        intermediate_features: torch.Tensor,
        slow_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchical attention.
        
        Args:
            rapid_features: Rapid temporal features (batch, features)
            intermediate_features: Intermediate temporal features
            slow_features: Slow temporal features
        
        Returns:
            Tuple of (attended features, attention weights)
        """
        batch_size = rapid_features.size(0)
        
        # Add sequence dimension for attention
        rapid_seq = rapid_features.unsqueeze(1)  # (batch, 1, features)
        intermediate_seq = intermediate_features.unsqueeze(1)
        slow_seq = slow_features.unsqueeze(1)
        
        # Self-attention for each scale
        rapid_att, rapid_weights = self.rapid_attention(
            rapid_seq, rapid_seq, rapid_seq
        )
        rapid_att = self.layer_norm1(rapid_att + rapid_seq)
        
        intermediate_att, intermediate_weights = self.intermediate_attention(
            intermediate_seq, intermediate_seq, intermediate_seq
        )
        intermediate_att = self.layer_norm2(intermediate_att + intermediate_seq)
        
        slow_att, slow_weights = self.slow_attention(
            slow_seq, slow_seq, slow_seq
        )
        slow_att = self.layer_norm3(slow_att + slow_seq)
        
        # Remove sequence dimension
        rapid_att = rapid_att.squeeze(1)
        intermediate_att = intermediate_att.squeeze(1)
        slow_att = slow_att.squeeze(1)
        
        # Concatenate for cross-scale attention
        combined = torch.cat([rapid_att, intermediate_att, slow_att], dim=-1)
        combined_seq = combined.unsqueeze(1)
        
        # Cross-scale attention
        cross_att, cross_weights = self.cross_attention(
            combined_seq, combined_seq, combined_seq
        )
        cross_att = self.layer_norm_cross(cross_att + combined_seq)
        cross_att = cross_att.squeeze(1)
        
        # Project to hidden dimension
        rapid_hidden = self.rapid_proj(rapid_att)
        intermediate_hidden = self.intermediate_proj(intermediate_att)
        slow_hidden = self.slow_proj(slow_att)
        
        # Combine with learnable weights
        combined_hidden = torch.cat([rapid_hidden, intermediate_hidden, slow_hidden], dim=-1)
        
        attention_weights = {
            'rapid': rapid_weights,
            'intermediate': intermediate_weights,
            'slow': slow_weights,
            'cross': cross_weights
        }
        
        return combined_hidden, attention_weights


class EnsemblePredictor(nn.Module):
    """Ensemble predictor combining Ridge, Random Forest, and XGBoost."""
    
    def __init__(self, config: Any):
        """
        Initialize ensemble predictor.
        
        Args:
            config: HENAW configuration
        """
        super().__init__()
        self.config = config
        
        # Initialize base models
        self.ridge = None
        self.rf = None
        self.xgb = None
        
        if config.use_ridge:
            self.ridge = Ridge(
                alpha=config.ridge_alpha,
                max_iter=config.ridge_max_iter
            )
        
        if config.use_random_forest:
            self.rf = RandomForestRegressor(
                n_estimators=config.rf_n_estimators,
                max_depth=config.rf_max_depth,
                min_samples_split=config.rf_min_samples_split,
                min_samples_leaf=config.rf_min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )
        
        if config.use_xgboost:
            self.xgb = xgb.XGBRegressor(
                n_estimators=config.xgb_n_estimators,
                max_depth=config.xgb_max_depth,
                learning_rate=config.xgb_learning_rate,
                subsample=config.xgb_subsample,
                colsample_bytree=config.xgb_colsample_bytree,
                random_state=42,
                n_jobs=-1
            )
        
        # Ensemble weights (learnable)
        n_models = sum([config.use_ridge, config.use_random_forest, config.use_xgboost])
        self.ensemble_weights = nn.Parameter(torch.ones(n_models) / n_models)
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit ensemble models.
        
        Args:
            X: Training features
            y: Training targets
        """
        if self.ridge is not None:
            self.ridge.fit(X, y)
        
        if self.rf is not None:
            self.rf.fit(X, y)
        
        if self.xgb is not None:
            self.xgb.fit(X, y)
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble predictions.
        
        Args:
            X: Input features
        
        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            warnings.warn("Ensemble models not fitted, returning zeros")
            return np.zeros(len(X))
        
        predictions = []
        
        if self.ridge is not None:
            predictions.append(self.ridge.predict(X))
        
        if self.rf is not None:
            predictions.append(self.rf.predict(X))
        
        if self.xgb is not None:
            predictions.append(self.xgb.predict(X))
        
        if not predictions:
            return np.zeros(len(X))
        
        # Weighted average
        predictions = np.stack(predictions, axis=1)
        weights = F.softmax(self.ensemble_weights, dim=0).detach().cpu().numpy()
        
        return np.sum(predictions * weights, axis=1)


class HENAW(nn.Module):
    """
    Hierarchical Ensemble Network for Aging Waves.
    
    Combines multi-scale temporal feature extraction with hierarchical attention
    and ensemble learning for robust biological age prediction.
    """
    
    def __init__(self, config: Any):
        """
        Initialize HENAW model.
        
        Args:
            config: HENAW configuration object
        """
        super().__init__()
        self.config = config
        
        # Calculate input dimensions
        n_features = len(config.input_features)
        
        # Feature extraction layers for each temporal scale
        self.rapid_encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.intermediate_encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        self.slow_encoder = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Hierarchical attention
        self.attention = HierarchicalAttention(
            input_dim=128,
            hidden_dim=config.attention_dim,
            n_heads=config.attention_heads,
            dropout=config.attention_dropout
        )
        
        # Deep feature fusion
        fusion_input_dim = config.attention_dim * 3
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(config.mc_dropout_rate if config.use_monte_carlo_dropout else 0.0),
            nn.Linear(64, 1)
        )
        
        # Ensemble predictor
        self.ensemble = EnsemblePredictor(config)
        
        # Monte Carlo dropout for uncertainty
        self.mc_dropout = nn.Dropout(config.mc_dropout_rate) if config.use_monte_carlo_dropout else None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def extract_temporal_features(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features at different temporal scales.
        
        Args:
            inputs: Dictionary containing temporal features
        
        Returns:
            Tuple of (rapid, intermediate, slow) features
        """
        rapid_features = self.rapid_encoder(inputs['rapid'])
        intermediate_features = self.intermediate_encoder(inputs['intermediate'])
        slow_features = self.slow_encoder(inputs['slow'])
        
        return rapid_features, intermediate_features, slow_features
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_attention: bool = False,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HENAW.
        
        Args:
            inputs: Dictionary containing:
                - 'rapid': Rapid temporal features
                - 'intermediate': Intermediate temporal features
                - 'slow': Slow temporal features
            return_attention: Whether to return attention weights
            return_uncertainty: Whether to compute uncertainty estimates
        
        Returns:
            Dictionary containing predictions and optional outputs
        """
        # Extract temporal features
        rapid_feat, intermediate_feat, slow_feat = self.extract_temporal_features(inputs)
        
        # Apply hierarchical attention
        attended_features, attention_weights = self.attention(
            rapid_feat, intermediate_feat, slow_feat
        )
        
        # Feature fusion
        fused_features = self.fusion_network(attended_features)
        
        # Neural network prediction
        nn_prediction = self.prediction_head(fused_features)
        
        outputs = {
            'prediction': nn_prediction,
            'features': fused_features
        }
        
        # Add attention weights if requested
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        # Compute uncertainty if requested
        if return_uncertainty and self.config.use_monte_carlo_dropout:
            uncertainty = self._compute_uncertainty(inputs)
            outputs['uncertainty'] = uncertainty
        
        return outputs
    
    def _compute_uncertainty(
        self,
        inputs: Dict[str, torch.Tensor],
        n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute prediction uncertainty using Monte Carlo dropout.
        
        Args:
            inputs: Model inputs
            n_samples: Number of MC samples
        
        Returns:
            Uncertainty estimates
        """
        n_samples = n_samples or self.config.mc_n_samples
        
        # Enable dropout for MC sampling
        self.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                outputs = self.forward(inputs, return_uncertainty=False)
                predictions.append(outputs['prediction'])
        
        predictions = torch.stack(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return uncertainty
    
    def fit_ensemble(self, features: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit the ensemble models.
        
        Args:
            features: Training features
            targets: Training targets
        """
        self.ensemble.fit(features, targets)
    
    def predict_with_ensemble(
        self,
        inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Make predictions using both neural network and ensemble.
        
        Args:
            inputs: Model inputs
        
        Returns:
            Combined predictions
        """
        # Get neural network predictions
        with torch.no_grad():
            nn_outputs = self.forward(inputs, return_uncertainty=True)
            features = nn_outputs['features'].cpu().numpy()
        
        # Get ensemble predictions
        ensemble_preds = self.ensemble.predict(features)
        
        # Combine predictions (weighted average)
        nn_preds = nn_outputs['prediction'].squeeze().cpu().numpy()
        combined_preds = 0.6 * nn_preds + 0.4 * ensemble_preds
        
        return {
            'prediction': combined_preds,
            'nn_prediction': nn_preds,
            'ensemble_prediction': ensemble_preds,
            'uncertainty': nn_outputs.get('uncertainty', None)
        }
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from ensemble models.
        
        Returns:
            Dictionary of feature importance scores
        """
        importance = {}
        
        if self.ensemble.rf is not None and self.ensemble.is_fitted:
            importance['random_forest'] = self.ensemble.rf.feature_importances_
        
        if self.ensemble.xgb is not None and self.ensemble.is_fitted:
            importance['xgboost'] = self.ensemble.xgb.feature_importances_
        
        if self.ensemble.ridge is not None and self.ensemble.is_fitted:
            importance['ridge'] = np.abs(self.ensemble.ridge.coef_)
        
        return importance