"""
Advanced Missing Data Imputation Strategies for HAMNet

This module implements advanced imputation methods including variational autoencoders,
probabilistic matrix factorization, temporal imputation, and multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from torch.distributions import Normal, Bernoulli, Categorical


@dataclass
class AdvancedImputationConfig:
    """Configuration for advanced imputation strategies"""
    # Model architecture
    latent_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    
    # VAE parameters
    beta_vae: float = 1.0
    annealing_steps: int = 1000
    
    # Matrix factorization parameters
    rank: int = 50
    regularization: float = 0.01
    
    # Temporal parameters
    temporal_window: int = 5
    temporal_stride: int = 1
    
    # Multi-task parameters
    task_weights: Dict[str, float] = None
    
    # Data dimensions
    clinical_dim: int = 100
    imaging_dim: int = 512
    genetic_dim: int = 1000
    lifestyle_dim: int = 50
    
    # Modality settings
    modalities: List[str] = None
    modality_dims: Dict[str, int] = None
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.modalities is None:
            self.modalities = ["clinical", "imaging", "genetic", "lifestyle"]
        if self.modality_dims is None:
            self.modality_dims = {
                "clinical": self.clinical_dim,
                "imaging": self.imaging_dim,
                "genetic": self.genetic_dim,
                "lifestyle": self.lifestyle_dim
            }
        if self.task_weights is None:
            self.task_weights = {
                "reconstruction": 1.0,
                "prediction": 0.5,
                "consistency": 0.3
            }


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for uncertainty-aware imputation"""
    
    def __init__(self, config: AdvancedImputationConfig):
        super().__init__()
        self.config = config
        
        # Calculate total input dimension
        self.total_dim = sum(config.modality_dims.values())
        
        # Encoder
        encoder_layers = []
        in_features = self.total_dim
        
        for i in range(config.num_layers):
            out_features = config.hidden_dim // (2 ** i)
            encoder_layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_features = out_features
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Variational parameters
        self.mu_layer = nn.Linear(in_features, config.latent_dim)
        self.logvar_layer = nn.Linear(in_features, config.latent_dim)
        
        # Decoder
        decoder_layers = []
        in_features = config.latent_dim
        
        for i in range(config.num_layers - 1):
            out_features = config.hidden_dim // (2 ** (config.num_layers - 2 - i))
            decoder_layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            in_features = out_features
        
        decoder_layers.append(nn.Linear(in_features, self.total_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Modality-specific decoders
        self.modality_decoders = nn.ModuleDict()
        split_indices = self._get_split_indices()
        
        for i, modality in enumerate(config.modalities):
            start_idx, end_idx = split_indices[i]
            self.modality_decoders[modality] = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, end_idx - start_idx)
            )
        
        # Uncertainty estimation
        self.uncertainty_heads = nn.ModuleDict()
        for modality in config.modalities:
            self.uncertainty_heads[modality] = nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 4, config.modality_dims[modality]),
                nn.Softplus()  # Ensure positive variance
            )
    
    def _get_split_indices(self) -> List[Tuple[int, int]]:
        """Get split indices for different modalities"""
        split_indices = []
        current_idx = 0
        for modality in self.config.modalities:
            dim = self.config.modality_dims[modality]
            split_indices.append((current_idx, current_idx + dim))
            current_idx += dim
        return split_indices
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input data to latent distribution"""
        if mask is not None:
            x = x * mask
        
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode latent samples to modality-specific outputs"""
        outputs = {}
        
        # Modality-specific decoding
        for modality in self.config.modalities:
            outputs[modality] = self.modality_decoders[modality](z)
        
        # Estimate uncertainties
        uncertainties = {}
        for modality in self.config.modalities:
            uncertainties[modality] = self.uncertainty_heads[modality](z)
        
        return outputs, uncertainties
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                num_samples: int = 1) -> Dict[str, torch.Tensor]:
        """Forward pass with optional sampling"""
        mu, logvar = self.encode(x, mask)
        
        if num_samples > 1:
            # Multiple samples for uncertainty estimation
            samples = []
            uncertainties = []
            
            for _ in range(num_samples):
                z = self.reparameterize(mu, logvar)
                outputs, unc = self.decode(z)
                samples.append(outputs)
                uncertainties.append(unc)
            
            # Average predictions
            avg_outputs = {}
            for modality in self.config.modalities:
                mod_samples = [s[modality] for s in samples]
                avg_outputs[modality] = torch.stack(mod_samples).mean(dim=0)
                
                # Average uncertainties
                mod_unc = [u[modality] for u in uncertainties]
                avg_outputs[f"{modality}_uncertainty"] = torch.stack(mod_unc).mean(dim=0)
            
            avg_outputs["mu"] = mu
            avg_outputs["logvar"] = logvar
            
            return avg_outputs
        else:
            z = self.reparameterize(mu, logvar)
            outputs, uncertainties = self.decode(z)
            
            outputs["mu"] = mu
            outputs["logvar"] = logvar
            outputs.update({f"{k}_uncertainty": v for k, v in uncertainties.items()})
            
            return outputs
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class ProbabilisticMatrixFactorization(nn.Module):
    """Probabilistic Matrix Factorization for sparse biomarker data"""
    
    def __init__(self, config: AdvancedImputationConfig):
        super().__init__()
        self.config = config
        
        # User and item embeddings
        self.user_embeddings = nn.Embedding(config.num_patients, config.rank)
        self.item_embeddings = nn.Embedding(config.num_features, config.rank)
        
        # Bias terms
        self.user_bias = nn.Embedding(config.num_patients, 1)
        self.item_bias = nn.Embedding(config.num_features, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Uncertainty parameters
        self.log_precision = nn.Parameter(torch.zeros(1))
        
        # Initialize embeddings
        nn.init.normal_(self.user_embeddings.weight, 0, 0.1)
        nn.init.normal_(self.item_embeddings.weight, 0, 0.1)
        nn.init.normal_(self.user_bias.weight, 0, 0.1)
        nn.init.normal_(self.item_bias.weight, 0, 0.1)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for matrix factorization"""
        # Get embeddings
        user_emb = self.user_embeddings(user_ids)
        item_emb = self.item_embeddings(item_ids)
        
        # Get biases
        user_b = self.user_bias(user_ids)
        item_b = self.item_bias(item_ids)
        
        # Compute prediction
        prediction = (user_emb * item_emb).sum(dim=-1, keepdim=True)
        prediction += user_b + item_b + self.global_bias
        
        # Compute uncertainty
        uncertainty = 1.0 / torch.exp(self.log_precision)
        
        return {
            'prediction': prediction,
            'uncertainty': uncertainty,
            'user_embedding': user_emb,
            'item_embedding': item_emb
        }
    
    def compute_loss(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                    values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute loss for matrix factorization"""
        # Forward pass
        outputs = self.forward(user_ids, item_ids)
        prediction = outputs['prediction']
        
        # Reconstruction loss (only for observed values)
        reconstruction_loss = F.mse_loss(prediction * mask, values * mask, reduction='sum')
        
        # Regularization
        reg_loss = self.config.regularization * (
            self.user_embeddings.weight.norm() + 
            self.item_embeddings.weight.norm() +
            self.user_bias.weight.norm() + 
            self.item_bias.weight.norm()
        )
        
        return reconstruction_loss + reg_loss


class TemporalImputer(nn.Module):
    """Temporal imputation for longitudinal missing data"""
    
    def __init__(self, config: AdvancedImputationConfig):
        super().__init__()
        self.config = config
        
        # Temporal encoder
        self.temporal_encoder = nn.LSTM(
            input_size=sum(config.modality_dims.values()),
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Temporal decoder
        self.temporal_decoder = nn.LSTM(
            input_size=config.latent_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            config.hidden_dim,
            sum(config.modality_dims.values())
        )
        
        # Temporal dynamics modeling
        self.dynamics_model = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                time_steps: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for temporal imputation
        
        Args:
            x: Input tensor (batch_size, time_steps, features)
            mask: Missing data mask
            time_steps: Time step indices
            
        Returns:
            Dictionary with imputed values and temporal dynamics
        """
        batch_size, seq_len, features_dim = x.shape
        
        # Encode temporal sequence
        encoded, (hidden, cell) = self.temporal_encoder(x)
        
        # Apply temporal attention
        attended, _ = self.temporal_attention(encoded, encoded, encoded)
        
        # Model temporal dynamics
        dynamics = self.dynamics_model(attended)
        
        # Decode to impute missing values
        # Use last hidden state as latent representation
        latent = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        decoded, _ = self.temporal_decoder(latent)
        
        # Project to output space
        output = self.output_projection(decoded)
        
        # Apply mask to preserve observed values
        output = output * (1 - mask) + x * mask
        
        return {
            'imputed': output,
            'temporal_dynamics': dynamics,
            'encoded': encoded,
            'attended': attended
        }
    
    def compute_temporal_loss(self, x: torch.Tensor, imputed: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """Compute temporal consistency loss"""
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(imputed * mask, x * mask)
        
        # Temporal smoothness loss
        if imputed.size(1) > 1:
            temporal_diff = torch.diff(imputed, dim=1)
            smoothness_loss = torch.mean(temporal_diff ** 2)
        else:
            smoothness_loss = torch.tensor(0.0, device=x.device)
        
        return reconstruction_loss + 0.1 * smoothness_loss


class MultiTaskImputer(nn.Module):
    """Multi-task learning for correlated missing patterns"""
    
    def __init__(self, config: AdvancedImputationConfig):
        super().__init__()
        self.config = config
        
        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(sum(config.modality_dims.values()), config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        
        # Imputation task
        self.task_heads['imputation'] = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, sum(config.modality_dims.values()))
        )
        
        # Missing pattern prediction task
        self.task_heads['pattern'] = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, len(config.modalities)),
            nn.Sigmoid()
        )
        
        # Quality assessment task
        self.task_heads['quality'] = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, sum(config.modality_dims.values())),
            nn.Softplus()
        )
        
        # Attention mechanism for task weighting
        self.task_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim // 2,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for multi-task imputer"""
        # Shared encoding
        shared_features = self.shared_encoder(x)
        
        # Task-specific predictions
        imputation = self.task_heads['imputation'](shared_features)
        pattern_prediction = self.task_heads['pattern'](shared_features)
        quality_assessment = self.task_heads['quality'](shared_features)
        uncertainty = self.uncertainty_head(shared_features)
        
        # Apply mask to imputation
        imputation = imputation * (1 - mask) + x * mask
        
        # Task attention for adaptive weighting
        task_features = torch.stack([
            shared_features,  # Shared representation
            self.task_heads['imputation'][:-2](shared_features),  # Imputation features
            self.task_heads['pattern'][:-2](shared_features),  # Pattern features
            self.task_heads['quality'][:-2](shared_features)  # Quality features
        ], dim=1)
        
        attended_task, _ = self.task_attention(
            task_features, task_features, task_features
        )
        
        return {
            'imputed': imputation,
            'pattern_prediction': pattern_prediction,
            'quality_assessment': quality_assessment,
            'uncertainty': uncertainty,
            'shared_features': shared_features,
            'task_features': attended_task
        }
    
    def compute_multi_task_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                              mask: torch.Tensor, true_pattern: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        losses = {}
        
        # Imputation loss
        imputation_loss = F.mse_loss(outputs['imputed'] * mask, x * mask)
        losses['imputation'] = self.config.task_weights['reconstruction'] * imputation_loss
        
        # Pattern prediction loss
        pattern_loss = F.binary_cross_entropy(
            outputs['pattern_prediction'], true_pattern
        )
        losses['pattern'] = self.config.task_weights['prediction'] * pattern_loss
        
        # Quality assessment loss
        # Quality is based on reconstruction error
        reconstruction_error = F.mse_loss(outputs['imputed'], x, reduction='none')
        quality_target = 1.0 / (1.0 + reconstruction_error.mean(dim=-1, keepdim=True))
        quality_loss = F.mse_loss(outputs['quality_assessment'], quality_target)
        losses['quality'] = self.config.task_weights['consistency'] * quality_loss
        
        # Uncertainty calibration loss
        uncertainty_loss = self._compute_uncertainty_loss(
            x, outputs['imputed'], outputs['uncertainty'], mask
        )
        losses['uncertainty'] = 0.1 * uncertainty_loss
        
        return losses
    
    def _compute_uncertainty_loss(self, x: torch.Tensor, imputed: torch.Tensor,
                                uncertainty: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute uncertainty calibration loss"""
        # Negative log likelihood with learned uncertainty
        error = (x - imputed) ** 2
        nll = 0.5 * (error / uncertainty + torch.log(uncertainty))
        
        # Only compute for missing values
        nll = nll * mask
        
        return nll.mean()


class AdvancedImputer:
    """Main advanced imputation class"""
    
    def __init__(self, config: AdvancedImputationConfig, method: str = "vae"):
        self.config = config
        self.method = method
        self.device = torch.device(config.device)
        
        # Initialize model based on method
        if method == "vae":
            self.model = VariationalAutoencoder(config).to(self.device)
        elif method == "matrix_factorization":
            self.model = ProbabilisticMatrixFactorization(config).to(self.device)
        elif method == "temporal":
            self.model = TemporalImputer(config).to(self.device)
        elif method == "multi_task":
            self.model = MultiTaskImputer(config).to(self.device)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training history
        self.training_history = {}
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        # Move to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        self.optimizer.zero_grad()
        
        if self.method == "vae":
            losses = self._train_step_vae(batch)
        elif self.method == "matrix_factorization":
            losses = self._train_step_matrix_factorization(batch)
        elif self.method == "temporal":
            losses = self._train_step_temporal(batch)
        elif self.method == "multi_task":
            losses = self._train_step_multi_task(batch)
        
        # Backpropagate
        total_loss = sum(losses.values())
        total_loss.backward()
        self.optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def _train_step_vae(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step for VAE"""
        x = batch['data']
        mask = batch['mask']
        
        outputs = self.model(x, mask)
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(outputs['imputed'] * mask, x * mask)
        
        # KL divergence
        kl_loss = self.model.compute_kl_loss(outputs['mu'], outputs['logvar'])
        
        # Total loss with annealing
        if hasattr(self, 'annealing_step'):
            beta = min(1.0, self.annealing_step / self.config.annealing_steps)
            self.annealing_step += 1
        else:
            beta = self.config.beta_vae
            self.annealing_step = 1
        
        total_loss = reconstruction_loss + beta * kl_loss
        
        return {
            'reconstruction': reconstruction_loss,
            'kl_divergence': kl_loss,
            'total': total_loss
        }
    
    def _train_step_matrix_factorization(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step for matrix factorization"""
        user_ids = batch['user_ids']
        item_ids = batch['item_ids']
        values = batch['values']
        mask = batch['mask']
        
        loss = self.model.compute_loss(user_ids, item_ids, values, mask)
        
        return {'total': loss}
    
    def _train_step_temporal(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step for temporal imputation"""
        x = batch['data']
        mask = batch['mask']
        time_steps = batch['time_steps']
        
        outputs = self.model(x, mask, time_steps)
        loss = self.model.compute_temporal_loss(x, outputs['imputed'], mask)
        
        return {'total': loss}
    
    def _train_step_multi_task(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step for multi-task imputer"""
        x = batch['data']
        mask = batch['mask']
        true_pattern = batch['true_pattern']
        
        outputs = self.model(x, mask)
        losses = self.model.compute_multi_task_loss(x, outputs, mask, true_pattern)
        
        return losses
    
    def impute(self, data: torch.Tensor, mask: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Impute missing data"""
        self.model.eval()
        
        with torch.no_grad():
            if self.method == "vae":
                return self.model(data, mask, **kwargs)
            elif self.method == "matrix_factorization":
                return self.model(kwargs['user_ids'], kwargs['item_ids'])
            elif self.method == "temporal":
                return self.model(data, mask, kwargs['time_steps'])
            elif self.method == "multi_task":
                return self.model(data, mask)
    
    def train(self, dataloader, num_epochs: int = None):
        """Train the advanced imputer"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = {}
            
            for batch_idx, batch in enumerate(dataloader):
                losses = self.train_step(batch)
                
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = []
                    epoch_losses[key].append(value.item())
            
            # Log epoch averages
            epoch_avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            
            for key, value in epoch_avg.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                for key, value in epoch_avg.items():
                    print(f"  {key}: {value:.4f}")
    
    def save(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'method': self.method,
            'training_history': self.training_history
        }, path)
    
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.method = checkpoint['method']
        self.training_history = checkpoint['training_history']