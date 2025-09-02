"""
HAMNet (Hierarchical Attention-based Multimodal Network) for Biological Age Prediction

This module implements the core HAMNet architecture for biological age prediction
using multimodal data from UK Biobank. The architecture includes:
- Modality-specific encoders
- Cross-modal attention fusion
- Temporal integration layers
- Uncertainty quantification
- Missing data handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class HAMNetConfig:
    """Configuration for HAMNet model"""
    # Model architecture
    model_tier: str = "standard"  # base, standard, comprehensive
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    # Modality configurations
    clinical_dim: int = 100
    imaging_dim: int = 512
    genetic_dim: int = 1000
    lifestyle_dim: int = 50
    
    # Temporal configuration
    temporal_window: int = 5
    temporal_stride: int = 1
    
    # Uncertainty quantification
    enable_uncertainty: bool = True
    num_monte_carlo: int = 20
    
    # Missing data handling
    missing_data_strategy: str = "attention"  # attention, zero, mean
    
    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True


class ModalityEncoder(nn.Module):
    """Base class for modality-specific encoders"""
    
    def __init__(self, input_dim: int, output_dim: int, config: HAMNetConfig):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config
        
        # Common encoder architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, output_dim)
        )
        
        # Modality-specific projection
        self.projection = nn.Linear(output_dim, config.embedding_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for modality encoder
        
        Args:
            x: Input tensor of shape (batch_size, ..., input_dim)
            mask: Optional mask for missing values
            
        Returns:
            Encoded tensor of shape (batch_size, ..., embedding_dim)
        """
        # Handle missing data
        if mask is not None:
            x = x * mask.unsqueeze(-1)
            
        # Flatten spatial/temporal dimensions if present
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(-1, self.input_dim)
            
        # Encode
        encoded = self.encoder(x)
        projected = self.projection(encoded)
        
        # Restore original shape (excluding input dimension)
        if len(original_shape) > 2:
            projected = projected.view(*original_shape[:-1], self.config.embedding_dim)
            
        return projected


class ClinicalEncoder(ModalityEncoder):
    """Encoder for clinical data (biomarkers, lab results, etc.)"""
    
    def __init__(self, config: HAMNetConfig):
        super().__init__(config.clinical_dim, config.embedding_dim, config)
        
        # Clinical-specific normalization layers
        self.batch_norm = nn.BatchNorm1d(config.clinical_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Apply clinical-specific preprocessing
        x = self.batch_norm(x)
        return super().forward(x, mask)


class ImagingEncoder(ModalityEncoder):
    """Encoder for imaging data (MRI, DXA, retinal features)"""
    
    def __init__(self, config: HAMNetConfig):
        super().__init__(config.imaging_dim, config.embedding_dim, config)
        
        # Imaging-specific convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(config.imaging_dim // 4)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Add channel dimension for conv layers
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(batch_size, -1)
        
        return super().forward(x, mask)


class GeneticEncoder(ModalityEncoder):
    """Encoder for genetic data (SNPs, polygenic risk scores)"""
    
    def __init__(self, config: HAMNetConfig):
        super().__init__(config.genetic_dim, config.embedding_dim, config)
        
        # Genetic-specific attention mechanism
        self.genetic_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads // 2,
            dropout=config.dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Encode genetic features
        encoded = super().forward(x, mask)
        
        # Apply genetic-specific attention
        if len(encoded.shape) == 2:
            encoded = encoded.unsqueeze(1)  # Add sequence dimension
            
        attended, _ = self.genetic_attention(encoded, encoded, encoded)
        
        return attended.squeeze(1) if len(encoded.shape) == 3 else attended


class LifestyleEncoder(ModalityEncoder):
    """Encoder for lifestyle data (diet, exercise, smoking, etc.)"""
    
    def __init__(self, config: HAMNetConfig):
        super().__init__(config.lifestyle_dim, config.embedding_dim, config)
        
        # Lifestyle-specific embedding for categorical variables
        self.categorical_embed = nn.Embedding(10, config.embedding_dim // 4)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Handle categorical lifestyle variables
        if x.dtype == torch.long:
            embedded = self.categorical_embed(x)
            embedded = embedded.view(x.shape[0], -1)
            x = torch.cat([x.float(), embedded], dim=-1)
            
        return super().forward(x, mask)


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for information exchange"""
    
    def __init__(self, config: HAMNetConfig):
        super().__init__()
        self.config = config
        
        # Multi-head attention for cross-modal interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Gated fusion mechanism
        self.gate = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(config.embedding_dim, config.embedding_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                query_mask: Optional[torch.Tensor] = None,
                key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Cross-modal attention forward pass
        
        Args:
            query: Query tensor (batch_size, seq_len, embedding_dim)
            key: Key tensor (batch_size, seq_len, embedding_dim)
            value: Value tensor (batch_size, seq_len, embedding_dim)
            query_mask: Optional mask for query
            key_mask: Optional mask for key
            
        Returns:
            Attended tensor
        """
        # Apply attention
        attended, attention_weights = self.attention(
            query, key, value,
            key_padding_mask=key_mask
        )
        
        # Gated fusion
        gate_input = torch.cat([query, attended], dim=-1)
        gate = self.gate(gate_input)
        
        # Apply gate and project
        output = gate * attended + (1 - gate) * query
        output = self.output_proj(output)
        
        return output, attention_weights


class TemporalIntegrationLayer(nn.Module):
    """Temporal integration layer for longitudinal data"""
    
    def __init__(self, config: HAMNetConfig):
        super().__init__()
        self.config = config
        
        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Temporal convolution
        self.temporal_conv = nn.Conv1d(
            config.embedding_dim, config.embedding_dim,
            kernel_size=3, padding=1
        )
        
        # Temporal gate
        self.temporal_gate = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, temporal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Temporal integration forward pass
        
        Args:
            x: Input tensor (batch_size, time_steps, embedding_dim)
            temporal_mask: Optional mask for temporal positions
            
        Returns:
            Temporally integrated tensor
        """
        # Apply temporal attention
        attended, _ = self.temporal_attention(x, x, x)
        
        # Apply temporal convolution
        conv_input = x.transpose(1, 2)  # (batch_size, embedding_dim, time_steps)
        conv_output = self.temporal_conv(conv_input)
        conv_output = conv_output.transpose(1, 2)  # Back to (batch_size, time_steps, embedding_dim)
        
        # Gated fusion
        gate_input = torch.cat([attended, conv_output], dim=-1)
        gate = self.temporal_gate(gate_input)
        
        # Apply gate
        output = gate * attended + (1 - gate) * conv_output
        
        return output


class UncertaintyQuantification(nn.Module):
    """Bayesian neural network for uncertainty quantification"""
    
    def __init__(self, config: HAMNetConfig):
        super().__init__()
        self.config = config
        
        # Bayesian layers with dropout
        self.bayesian_layers = nn.ModuleList([
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.Linear(config.hidden_dim // 4, 2)  # Mean and variance
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Monte Carlo dropout
        
        Args:
            x: Input tensor
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with mean and variance predictions
        """
        if num_samples is None:
            num_samples = self.config.num_monte_carlo
            
        predictions = []
        
        # Monte Carlo sampling
        for _ in range(num_samples):
            h = x
            for layer in self.bayesian_layers[:-1]:
                h = self.dropout(F.relu(layer(h)))
            pred = self.bayesian_layers[-1](h)
            predictions.append(pred)
            
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate mean and variance
        mean = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        
        return {
            'mean': mean,
            'variance': variance,
            'uncertainty': torch.sqrt(variance + 1e-8)
        }


class HAMNet(nn.Module):
    """Main HAMNet model for biological age prediction"""
    
    def __init__(self, config: HAMNetConfig):
        super().__init__()
        self.config = config
        
        # Modality encoders
        self.clinical_encoder = ClinicalEncoder(config)
        self.imaging_encoder = ImagingEncoder(config)
        self.genetic_encoder = GeneticEncoder(config)
        self.lifestyle_encoder = LifestyleEncoder(config)
        
        # Cross-modal attention modules
        self.cross_modal_attention = CrossModalAttention(config)
        
        # Temporal integration
        self.temporal_integration = TemporalIntegrationLayer(config)
        
        # Uncertainty quantification
        if config.enable_uncertainty:
            self.uncertainty_module = UncertaintyQuantification(config)
        
        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 4, 1)
        )
        
        # Modality availability weights
        self.modality_weights = nn.Parameter(torch.ones(4) / 4)
        
    def forward(self, inputs: Dict[str, torch.Tensor], 
                masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for HAMNet
        
        Args:
            inputs: Dictionary of modality inputs
            masks: Optional dictionary of modality masks
            
        Returns:
            Dictionary with predictions and uncertainty
        """
        if masks is None:
            masks = {}
            
        # Encode each modality
        modality_embeddings = {}
        
        # Clinical data
        if 'clinical' in inputs:
            clinical_mask = masks.get('clinical', None)
            modality_embeddings['clinical'] = self.clinical_encoder(
                inputs['clinical'], clinical_mask
            )
            
        # Imaging data
        if 'imaging' in inputs:
            imaging_mask = masks.get('imaging', None)
            modality_embeddings['imaging'] = self.imaging_encoder(
                inputs['imaging'], imaging_mask
            )
            
        # Genetic data
        if 'genetic' in inputs:
            genetic_mask = masks.get('genetic', None)
            modality_embeddings['genetic'] = self.genetic_encoder(
                inputs['genetic'], genetic_mask
            )
            
        # Lifestyle data
        if 'lifestyle' in inputs:
            lifestyle_mask = masks.get('lifestyle', None)
            modality_embeddings['lifestyle'] = self.lifestyle_encoder(
                inputs['lifestyle'], lifestyle_mask
            )
            
        # Cross-modal fusion
        fused_embedding = self._cross_modal_fusion(modality_embeddings)
        
        # Temporal integration (if applicable)
        if len(fused_embedding.shape) > 2:  # Has temporal dimension
            fused_embedding = self.temporal_integration(fused_embedding)
            
        # Generate predictions
        predictions = self.final_layers(fused_embedding)
        
        # Uncertainty quantification
        output = {'predictions': predictions}
        
        if self.config.enable_uncertainty:
            uncertainty_results = self.uncertainty_module(fused_embedding)
            output.update(uncertainty_results)
            
        return output
        
    def _cross_modal_fusion(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse modality embeddings using cross-modal attention"""
        if not modality_embeddings:
            return torch.zeros(1, self.config.embedding_dim).to(next(self.parameters()).device)
            
        # Get available modalities
        modalities = list(modality_embeddings.keys())
        
        if len(modalities) == 1:
            return modality_embeddings[modalities[0]]
            
        # Stack embeddings for cross-modal attention
        embeddings = []
        for modality in modalities:
            emb = modality_embeddings[modality]
            if len(emb.shape) == 2:
                emb = emb.unsqueeze(1)  # Add sequence dimension
            embeddings.append(emb)
            
        stacked_embeddings = torch.cat(embeddings, dim=1)
        
        # Apply cross-modal attention
        fused_embeddings = []
        for i, modality in enumerate(modalities):
            query = stacked_embeddings[:, i:i+1, :]
            key_value = stacked_embeddings
            
            fused, _ = self.cross_modal_attention(query, key_value, key_value)
            fused_embeddings.append(fused.squeeze(1))
            
        # Weighted combination
        weights = F.softmax(self.modality_weights[:len(modalities)], dim=0)
        weights = weights.view(1, -1, 1)
        
        fused_embeddings = torch.stack(fused_embeddings, dim=1)
        final_fusion = (fused_embeddings * weights).sum(dim=1)
        
        return final_fusion
        
    def get_attention_weights(self, inputs: Dict[str, torch.Tensor],
                            masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Extract attention weights for interpretability"""
        # This would be implemented to return attention weights
        # for model interpretability and analysis
        pass
        
    def predict_with_uncertainty(self, inputs: Dict[str, torch.Tensor],
                               masks: Optional[Dict[str, torch.Tensor]] = None,
                               num_samples: int = None) -> Dict[str, torch.Tensor]:
        """Predict with uncertainty quantification"""
        if num_samples is None:
            num_samples = self.config.num_monte_carlo
            
        predictions = []
        
        # Monte Carlo dropout sampling
        for _ in range(num_samples):
            pred = self.forward(inputs, masks)
            predictions.append(pred['predictions'])
            
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        uncertainty = torch.sqrt(variance + 1e-8)
        
        return {
            'mean_prediction': mean_pred,
            'variance': variance,
            'uncertainty': uncertainty,
            'samples': predictions
        }


def create_hamnet_model(config: HAMNetConfig) -> HAMNet:
    """Factory function to create HAMNet model"""
    return HAMNet(config)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_shape: Dict[str, Tuple[int, ...]]) -> str:
    """Generate a summary of the model architecture"""
    summary = []
    summary.append("HAMNet Model Summary")
    summary.append("=" * 50)
    summary.append(f"Total parameters: {count_parameters(model):,}")
    summary.append(f"Trainable parameters: {count_parameters(model):,}")
    summary.append("")
    
    # Add layer-wise summary
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf module
            summary.append(f"{name}: {module.__class__.__name__}")
            
    return "\n".join(summary)