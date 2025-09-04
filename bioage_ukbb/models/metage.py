"""
METAGE (Metabolomic Trajectory Aging Estimator) implementation.

Dynamic metabolic aging trajectory modeling using LSTM networks
with personalized aging rate estimation and intervention response prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for temporal sequences."""
    
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pos_embedding


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence modeling."""
    
    def __init__(
        self,
        hidden_dim: int,
        n_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequences (batch, seq_len, hidden_dim)
            mask: Attention mask
        
        Returns:
            Tuple of (attended sequences, attention weights)
        """
        attended, weights = self.attention(x, x, x, attn_mask=mask)
        x = self.norm(x + self.dropout(attended))
        
        return x, weights


class MetabolicLSTM(nn.Module):
    """LSTM network for metabolomic sequence modeling."""
    
    def __init__(self, config: Any):
        super().__init__()
        
        self.config = config
        input_dim = config.n_metabolomic_features
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, config.lstm_hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.lstm_hidden_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            batch_first=True,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            bidirectional=config.lstm_bidirectional
        )
        
        # Output dimension adjustment for bidirectional
        lstm_output_dim = config.lstm_hidden_dim * (2 if config.lstm_bidirectional else 1)
        
        # Temporal attention
        if config.use_attention:
            self.attention = TemporalAttention(
                lstm_output_dim,
                config.attention_heads
            )
        else:
            self.attention = None
        
        # Output projection
        self.output_projection = nn.Linear(lstm_output_dim, config.lstm_hidden_dim)
        
        self.dropout = nn.Dropout(config.lstm_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequences (batch, seq_len, n_features)
            lengths: Actual lengths of sequences
        
        Returns:
            Tuple of (sequence outputs, final hidden state)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # Pack sequences if lengths provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        
        # Apply attention if configured
        if self.attention is not None:
            lstm_out, attn_weights = self.attention(lstm_out)
        else:
            attn_weights = None
        
        # Project output
        output = self.output_projection(lstm_out)
        
        # Get final hidden state
        if self.config.lstm_bidirectional:
            # Concatenate forward and backward final states
            h_n = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            h_n = h_n[-1]
        
        return output, h_n


class AgingRateEstimator(nn.Module):
    """Estimates personalized aging rate from metabolomic trajectories."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        min_rate: float = 0.5,
        max_rate: float = 2.0
    ):
        super().__init__()
        
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Aging rates scaled to [min_rate, max_rate]
        """
        # Estimate rate in [0, 1]
        rate_01 = self.estimator(x)
        
        # Scale to [min_rate, max_rate]
        aging_rate = self.min_rate + (self.max_rate - self.min_rate) * rate_01
        
        return aging_rate


class InterventionPredictor(nn.Module):
    """Predicts response to different interventions."""
    
    def __init__(
        self,
        input_dim: int,
        intervention_types: List[str],
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.intervention_types = intervention_types
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Intervention-specific heads
        self.intervention_heads = nn.ModuleDict({
            intervention: nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for intervention in intervention_types
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Dictionary of intervention response predictions
        """
        features = self.feature_extractor(x)
        
        responses = {}
        for intervention, head in self.intervention_heads.items():
            responses[intervention] = head(features)
        
        return responses


class VariationalEncoder(nn.Module):
    """Variational encoder for metabolomic sequences."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input features (batch, input_dim)
        
        Returns:
            Tuple of (latent code, mu, log_variance)
        """
        encoded = self.encoder(x)
        
        mu = self.mu_head(encoded)
        logvar = self.logvar_head(encoded)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        return z, mu, logvar


class TrajectoryProjector(nn.Module):
    """Projects metabolomic trajectories into future timepoints."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_future_steps: int = 3
    ):
        super().__init__()
        
        self.n_future_steps = n_future_steps
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Generate future steps
        self.future_heads = nn.ModuleList([
            nn.Linear(hidden_dim, input_dim)
            for _ in range(n_future_steps)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Current state (batch, input_dim)
        
        Returns:
            List of future state predictions
        """
        features = self.projector(x)
        
        future_states = []
        for head in self.future_heads:
            future_states.append(head(features))
        
        return future_states


class METAGE(nn.Module):
    """
    Metabolomic Trajectory Aging Estimator.
    
    Models dynamic metabolic aging trajectories using LSTM networks
    with personalized aging rate estimation and intervention prediction.
    """
    
    def __init__(self, config: Any):
        """
        Initialize METAGE model.
        
        Args:
            config: METAGE configuration object
        """
        super().__init__()
        self.config = config
        
        # Positional encoding
        if config.time_encoding == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                config.n_metabolomic_features
            )
        elif config.time_encoding == "learnable":
            self.positional_encoding = LearnablePositionalEncoding(
                config.sequence_length,
                config.n_metabolomic_features
            )
        else:
            self.positional_encoding = None
        
        # Metabolomic feature groups
        self.feature_groups = self._create_feature_groups()
        
        # LSTM for sequence modeling
        self.metabolic_lstm = MetabolicLSTM(config)
        
        # Variational encoding if configured
        if config.use_variational:
            self.variational_encoder = VariationalEncoder(
                config.lstm_hidden_dim,
                latent_dim=64
            )
        else:
            self.variational_encoder = None
        
        # Trajectory projector
        self.trajectory_projector = TrajectoryProjector(
            config.lstm_hidden_dim,
            hidden_dim=256,
            n_future_steps=3
        )
        
        # Aging rate estimator
        if config.estimate_aging_rate:
            self.aging_rate_estimator = AgingRateEstimator(
                config.lstm_hidden_dim,
                min_rate=config.aging_rate_min,
                max_rate=config.aging_rate_max
            )
        else:
            self.aging_rate_estimator = None
        
        # Intervention predictor
        if config.predict_intervention_response:
            self.intervention_predictor = InterventionPredictor(
                config.lstm_hidden_dim,
                config.intervention_types
            )
        else:
            self.intervention_predictor = None
        
        # Age prediction network
        self.age_predictor = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def _create_feature_groups(self) -> Dict[str, List[int]]:
        """Create metabolomic feature groups."""
        # Placeholder for feature grouping
        # In practice, this would map actual metabolite indices to groups
        groups = {}
        features_per_group = self.config.n_metabolomic_features // len(self.config.metabolomic_groups)
        
        start_idx = 0
        for group in self.config.metabolomic_groups:
            end_idx = start_idx + features_per_group
            groups[group] = list(range(start_idx, min(end_idx, self.config.n_metabolomic_features)))
            start_idx = end_idx
        
        return groups
    
    def extract_group_features(
        self,
        x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features for each metabolomic group.
        
        Args:
            x: Input sequences (batch, seq_len, n_features)
        
        Returns:
            Dictionary of group features
        """
        group_features = {}
        
        for group_name, indices in self.feature_groups.items():
            group_features[group_name] = x[:, :, indices].mean(dim=-1, keepdim=True)
        
        return group_features
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_trajectories: bool = False
    ) -> Dict[str, Any]:
        """
        Forward pass through METAGE.
        
        Args:
            inputs: Dictionary containing:
                - 'sequence': Metabolomic sequences (batch, seq_len, n_features)
                - 'time_encoding': Optional time encodings
                - 'lengths': Optional sequence lengths
            return_trajectories: Whether to return future trajectory predictions
        
        Returns:
            Dictionary containing predictions and optional outputs
        """
        sequence = inputs['sequence']
        batch_size, seq_len, n_features = sequence.shape
        
        # Add positional encoding if configured
        if self.positional_encoding is not None:
            sequence = self.positional_encoding(sequence)
        elif 'time_encoding' in inputs:
            # Add provided time encoding
            sequence = sequence + inputs['time_encoding']
        
        # Extract group features for interpretability
        group_features = self.extract_group_features(sequence)
        
        # Process through LSTM
        lstm_output, final_hidden = self.metabolic_lstm(
            sequence,
            inputs.get('lengths', None)
        )
        
        # Use final hidden state for predictions
        features = final_hidden
        
        # Variational encoding if configured
        if self.variational_encoder is not None:
            z, mu, logvar = self.variational_encoder(features)
            features = z
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        else:
            kl_loss = None
        
        # Age prediction
        age_prediction = self.age_predictor(features)
        
        outputs = {
            'prediction': age_prediction,
            'features': features,
            'sequence_output': lstm_output,
            'group_features': group_features
        }
        
        # Add KL loss if using variational
        if kl_loss is not None:
            outputs['kl_loss'] = kl_loss
        
        # Estimate aging rate
        if self.aging_rate_estimator is not None:
            aging_rate = self.aging_rate_estimator(features)
            outputs['aging_rate'] = aging_rate
            
            # Adjust prediction by aging rate
            outputs['adjusted_prediction'] = age_prediction * aging_rate
        
        # Predict intervention responses
        if self.intervention_predictor is not None:
            intervention_responses = self.intervention_predictor(features)
            outputs['intervention_responses'] = intervention_responses
        
        # Project future trajectories
        if return_trajectories:
            future_states = self.trajectory_projector(features)
            outputs['future_trajectories'] = future_states
        
        return outputs
    
    def compute_trajectory_loss(
        self,
        predicted_trajectories: List[torch.Tensor],
        true_future_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for trajectory predictions.
        
        Args:
            predicted_trajectories: List of predicted future states
            true_future_sequences: True future sequences (batch, n_future, n_features)
        
        Returns:
            Trajectory prediction loss
        """
        losses = []
        
        for i, pred in enumerate(predicted_trajectories):
            if i < true_future_sequences.size(1):
                true_state = true_future_sequences[:, i, :]
                loss = F.mse_loss(pred, true_state)
                losses.append(loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)
    
    def interpret_metabolic_profile(
        self,
        group_features: Dict[str, torch.Tensor]
    ) -> Dict[str, str]:
        """
        Interpret metabolomic group features.
        
        Args:
            group_features: Dictionary of group features
        
        Returns:
            Dictionary of interpretations
        """
        interpretations = {}
        
        for group_name, features in group_features.items():
            # Get mean value across sequence
            mean_value = features.mean().item()
            
            # Simple interpretation based on deviation from normal
            if mean_value > 1.5:
                status = "elevated"
            elif mean_value < -1.5:
                status = "reduced"
            else:
                status = "normal"
            
            interpretations[group_name] = f"{group_name}: {status} (mean: {mean_value:.2f})"
        
        return interpretations
    
    def predict_optimal_intervention(
        self,
        intervention_responses: Dict[str, torch.Tensor]
    ) -> Tuple[str, float]:
        """
        Determine optimal intervention based on predicted responses.
        
        Args:
            intervention_responses: Dictionary of intervention predictions
        
        Returns:
            Tuple of (best intervention, expected improvement)
        """
        best_intervention = None
        best_improvement = float('-inf')
        
        for intervention, response in intervention_responses.items():
            improvement = -response.mean().item()  # Negative for age reduction
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_intervention = intervention
        
        return best_intervention, best_improvement