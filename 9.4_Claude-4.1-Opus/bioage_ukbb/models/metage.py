"""
METAGE (Metabolomic Trajectory Aging Estimator) implementation.

Dynamic metabolic aging trajectory modeling using NMR metabolomics data
with personalized aging rate estimation and intervention response prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import math
from torchdiffeq import odeint_adjoint as odeint


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:, :x.size(1)]


class MetabolomicEncoder(nn.Module):
    """Encoder for NMR metabolomic features."""
    
    def __init__(
        self,
        n_metabolites: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.n_metabolites = n_metabolites
        self.hidden_dim = hidden_dim
        
        # Metabolite group embeddings
        self.metabolite_groups = {
            'lipids': list(range(0, 40)),
            'lipoproteins': list(range(40, 112)),
            'fatty_acids': list(range(112, 140)),
            'amino_acids': list(range(140, 149)),
            'glycolysis': list(range(149, 155)),
            'ketone_bodies': list(range(155, 158)),
            'inflammation': list(range(158, 168))
        }
        
        # Group-specific encoders
        self.group_encoders = nn.ModuleDict({
            group: nn.Sequential(
                nn.Linear(len(indices), hidden_dim // 4),
                nn.LayerNorm(hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for group, indices in self.metabolite_groups.items()
        })
        
        # Global encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(n_metabolites, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2)
        )
        
        # Fusion layer
        n_groups = len(self.metabolite_groups)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + (hidden_dim // 4) * n_groups, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode metabolomic features.
        
        Args:
            x: Metabolomic features (batch, n_metabolites) or (batch, seq_len, n_metabolites)
        
        Returns:
            Encoded features
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.reshape(-1, self.n_metabolites)
            encoded = self._encode_features(x_flat)
            return encoded.reshape(batch_size, seq_len, -1)
        else:
            return self._encode_features(x)
    
    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Encode features with group-specific and global processing."""
        # Global encoding
        global_features = self.global_encoder(x)
        
        # Group-specific encoding
        group_features = []
        for group, indices in self.metabolite_groups.items():
            if max(indices) < x.shape[1]:
                group_data = x[:, indices]
                group_encoded = self.group_encoders[group](group_data)
                group_features.append(group_encoded)
        
        # Concatenate all features
        if group_features:
            all_features = torch.cat([global_features] + group_features, dim=-1)
            return self.fusion(all_features)
        else:
            return global_features


class TrajectoryLSTM(nn.Module):
    """LSTM for modeling metabolic trajectories."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process temporal sequences.
        
        Args:
            x: Input sequences (batch, seq_len, input_dim)
            lengths: Actual sequence lengths for packing
        
        Returns:
            Tuple of (output sequences, final hidden state)
        """
        if lengths is not None:
            # Pack padded sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (hidden, cell) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True
            )
        else:
            output, (hidden, cell) = self.lstm(x)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        
        return output, hidden


class NeuralODE(nn.Module):
    """Neural ODE for continuous trajectory modeling."""
    
    def __init__(self, hidden_dim: int, n_layers: int = 3):
        super().__init__()
        
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(hidden_dim + 1, hidden_dim * 2),  # +1 for time
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim * 2)
                ])
            elif i == n_layers - 1:
                layers.append(nn.Linear(hidden_dim * 2, hidden_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim * 2, hidden_dim * 2),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim * 2)
                ])
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Compute derivative of state with respect to time.
        
        Args:
            t: Time point
            state: Current state
        
        Returns:
            Derivative of state
        """
        # Expand time to match batch size
        t_vec = t.expand(state.shape[0], 1)
        
        # Concatenate time with state
        x = torch.cat([state, t_vec], dim=-1)
        
        # Compute derivative
        return self.net(x)


class AgingRatePredictor(nn.Module):
    """Predict personalized aging rate from metabolic features."""
    
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
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict aging rate.
        
        Args:
            x: Input features
        
        Returns:
            Aging rate in [min_rate, max_rate]
        """
        # Get rate in [0, 1]
        normalized_rate = self.predictor(x)
        
        # Scale to [min_rate, max_rate]
        aging_rate = self.min_rate + (self.max_rate - self.min_rate) * normalized_rate
        
        return aging_rate


class InterventionResponsePredictor(nn.Module):
    """Predict response to different interventions."""
    
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
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 2)  # Response magnitude and confidence
            )
            for intervention in intervention_types
        })
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict intervention responses.
        
        Args:
            x: Input features
        
        Returns:
            Dictionary of intervention responses
        """
        # Extract shared features
        features = self.feature_extractor(x)
        
        # Predict response for each intervention
        responses = {}
        for intervention, head in self.intervention_heads.items():
            response = head(features)
            responses[intervention] = {
                'magnitude': torch.tanh(response[:, 0:1]),  # [-1, 1]
                'confidence': torch.sigmoid(response[:, 1:2])  # [0, 1]
            }
        
        return responses


class METAGE(nn.Module):
    """
    Metabolomic Trajectory Aging Estimator.
    
    Models dynamic metabolic aging trajectories with personalized aging rates
    and intervention response prediction using NMR metabolomics data.
    """
    
    def __init__(self, config: Any):
        """
        Initialize METAGE model.
        
        Args:
            config: METAGE configuration object
        """
        super().__init__()
        self.config = config
        
        # Metabolomic encoder
        self.metabolomic_encoder = MetabolomicEncoder(
            n_metabolites=config.n_metabolomic_features,
            hidden_dim=config.lstm_hidden_dim,
            dropout=config.lstm_dropout
        )
        
        # Temporal modeling
        if config.sequence_length > 1:
            # Positional encoding for sequences
            if config.time_encoding == "sinusoidal":
                self.positional_encoding = PositionalEncoding(
                    config.lstm_hidden_dim,
                    max_len=config.sequence_length
                )
            else:
                self.positional_encoding = None
            
            # LSTM for trajectory modeling
            self.trajectory_lstm = TrajectoryLSTM(
                input_dim=config.lstm_hidden_dim,
                hidden_dim=config.lstm_hidden_dim,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout,
                bidirectional=config.lstm_bidirectional
            )
            
            lstm_output_dim = self.trajectory_lstm.output_dim
        else:
            self.positional_encoding = None
            self.trajectory_lstm = None
            lstm_output_dim = config.lstm_hidden_dim
        
        # Neural ODE for continuous trajectories
        if config.use_ode_solver:
            self.neural_ode = NeuralODE(lstm_output_dim)
            self.ode_solver = config.ode_solver
            self.ode_rtol = config.ode_rtol
            self.ode_atol = config.ode_atol
        else:
            self.neural_ode = None
        
        # Attention mechanism
        if config.use_attention:
            if config.attention_type == "self":
                self.attention = nn.MultiheadAttention(
                    lstm_output_dim,
                    config.attention_heads,
                    dropout=0.1,
                    batch_first=True
                )
            else:
                self.attention = None
        else:
            self.attention = None
        
        # Aging rate predictor
        if config.estimate_aging_rate:
            self.aging_rate_predictor = AgingRatePredictor(
                input_dim=lstm_output_dim,
                min_rate=config.aging_rate_min,
                max_rate=config.aging_rate_max
            )
        else:
            self.aging_rate_predictor = None
        
        # Intervention response predictor
        if config.predict_intervention_response:
            self.intervention_predictor = InterventionResponsePredictor(
                input_dim=lstm_output_dim,
                intervention_types=config.intervention_types
            )
        else:
            self.intervention_predictor = None
        
        # Variational components for uncertainty
        if config.use_variational:
            self.variational_encoder = nn.Sequential(
                nn.Linear(lstm_output_dim, lstm_output_dim // 2),
                nn.ReLU()
            )
            self.mu_head = nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4)
            self.logvar_head = nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4)
        else:
            self.variational_encoder = None
        
        # Final prediction network
        final_input_dim = lstm_output_dim
        if config.use_variational:
            final_input_dim = lstm_output_dim + lstm_output_dim // 4
        
        self.prediction_network = nn.Sequential(
            nn.Linear(final_input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Age prediction head
        self.age_head = nn.Linear(64, 1)
        
        # Trajectory projection head
        self.trajectory_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for variational inference."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def compute_trajectory(
        self,
        initial_state: torch.Tensor,
        time_points: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute continuous trajectory using Neural ODE.
        
        Args:
            initial_state: Initial metabolic state
            time_points: Time points to evaluate
        
        Returns:
            Trajectory values at time points
        """
        if self.neural_ode is None:
            return initial_state.unsqueeze(1).expand(-1, len(time_points), -1)
        
        # Solve ODE
        trajectory = odeint(
            self.neural_ode,
            initial_state,
            time_points,
            method=self.ode_solver,
            rtol=self.ode_rtol,
            atol=self.ode_atol
        )
        
        # Reshape: (time, batch, features) -> (batch, time, features)
        return trajectory.permute(1, 0, 2)
    
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        return_trajectories: bool = False,
        return_rates: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through METAGE.
        
        Args:
            inputs: Dictionary containing:
                - 'metabolomics': NMR metabolomic features (batch, seq_len, n_features)
                  or (batch, n_features) for single timepoint
                - 'time_points': Optional time points for each sequence element
                - 'lengths': Optional actual sequence lengths
            return_trajectories: Whether to return trajectory predictions
            return_rates: Whether to return aging rate predictions
        
        Returns:
            Dictionary containing predictions and optional outputs
        """
        metabolomics = inputs['metabolomics']
        
        # Encode metabolomic features
        encoded = self.metabolomic_encoder(metabolomics)
        
        # Handle temporal sequences
        if metabolomics.dim() == 3 and self.trajectory_lstm is not None:
            batch_size, seq_len, _ = encoded.shape
            
            # Add positional encoding if available
            if self.positional_encoding is not None:
                encoded = self.positional_encoding(encoded)
            
            # Process with LSTM
            lengths = inputs.get('lengths', None)
            lstm_output, final_hidden = self.trajectory_lstm(encoded, lengths)
            
            # Apply attention if configured
            if self.attention is not None:
                attended, _ = self.attention(lstm_output, lstm_output, lstm_output)
                # Use attended final state
                features = attended[:, -1, :]
            else:
                # Use final hidden state
                features = final_hidden
        else:
            # Single timepoint - use encoded features directly
            features = encoded.squeeze(1) if encoded.dim() == 3 else encoded
        
        # Variational encoding for uncertainty
        kl_loss = None
        if self.variational_encoder is not None:
            var_features = self.variational_encoder(features)
            mu = self.mu_head(var_features)
            logvar = self.logvar_head(var_features)
            z = self.reparameterize(mu, logvar)
            
            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
            
            # Concatenate with original features
            features = torch.cat([features, z], dim=-1)
        
        # Pass through prediction network
        pred_features = self.prediction_network(features)
        
        # Main age prediction
        age_prediction = self.age_head(pred_features)
        
        outputs = {
            'prediction': age_prediction,
            'features': pred_features
        }
        
        # Add KL loss if computed
        if kl_loss is not None:
            outputs['kl_loss'] = kl_loss
        
        # Predict aging rate
        if return_rates and self.aging_rate_predictor is not None:
            aging_rate = self.aging_rate_predictor(features)
            outputs['aging_rate'] = aging_rate
            
            # Adjust prediction based on aging rate
            adjusted_prediction = age_prediction * aging_rate
            outputs['adjusted_prediction'] = adjusted_prediction
        
        # Compute trajectories
        if return_trajectories:
            # Project future trajectory
            trajectory_projection = self.trajectory_head(pred_features)
            outputs['trajectory_projection'] = trajectory_projection
            
            # If using Neural ODE, compute continuous trajectory
            if self.neural_ode is not None and 'time_points' in inputs:
                time_points = inputs['time_points']
                trajectory = self.compute_trajectory(features, time_points)
                outputs['continuous_trajectory'] = trajectory
        
        # Predict intervention responses
        if self.intervention_predictor is not None:
            intervention_responses = self.intervention_predictor(features)
            outputs['intervention_responses'] = intervention_responses
        
        return outputs
    
    def predict_future_age(
        self,
        current_features: torch.Tensor,
        years_ahead: float,
        aging_rate: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict biological age at a future time point.
        
        Args:
            current_features: Current metabolomic features
            years_ahead: Number of years to project
            aging_rate: Optional personalized aging rate
        
        Returns:
            Predicted biological age
        """
        # Encode features
        encoded = self.metabolomic_encoder(current_features)
        
        # Get aging rate if not provided
        if aging_rate is None and self.aging_rate_predictor is not None:
            aging_rate = self.aging_rate_predictor(encoded)
        elif aging_rate is None:
            aging_rate = torch.ones(current_features.shape[0], 1, device=current_features.device)
        
        # Pass through prediction network
        pred_features = self.prediction_network(encoded)
        
        # Get current age
        current_age = self.age_head(pred_features)
        
        # Project future age based on aging rate
        future_age = current_age + years_ahead * aging_rate
        
        return future_age
    
    def recommend_interventions(
        self,
        features: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, List[str]]:
        """
        Recommend interventions based on predicted responses.
        
        Args:
            features: Metabolomic features
            threshold: Confidence threshold for recommendations
        
        Returns:
            Dictionary of recommendations per sample
        """
        if self.intervention_predictor is None:
            return {}
        
        # Encode features
        encoded = self.metabolomic_encoder(features)
        
        # Get intervention responses
        responses = self.intervention_predictor(encoded)
        
        # Generate recommendations
        batch_size = features.shape[0]
        recommendations = {i: [] for i in range(batch_size)}
        
        for intervention, response_dict in responses.items():
            magnitude = response_dict['magnitude']
            confidence = response_dict['confidence']
            
            for i in range(batch_size):
                # Recommend if high confidence and negative magnitude (age reduction)
                if confidence[i] > threshold and magnitude[i] < 0:
                    recommendations[i].append(intervention)
        
        return recommendations