"""
GAN-based Missing Data Imputation for HAMNet

This module implements advanced GAN-based imputation methods for handling missing data
in multimodal biological age prediction. It includes conditional GANs, multi-modal GANs,
and cycle-consistency constraints for robust imputation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
import numpy as np
from torch.autograd import Variable


@dataclass
class GANImputationConfig:
    """Configuration for GAN-based imputation"""
    # Model architecture
    latent_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    
    # GAN-specific
    generator_lr: float = 1e-4
    discriminator_lr: float = 1e-4
    lambda_cycle: float = 10.0
    lambda_gp: float = 10.0
    n_critic: int = 5
    
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


class ResidualBlock(nn.Module):
    """Residual block for GAN generators"""
    
    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.1):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, in_features),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConditionalGenerator(nn.Module):
    """Conditional generator for modality-specific imputation"""
    
    def __init__(self, config: GANImputationConfig, modality: str):
        super().__init__()
        self.config = config
        self.modality = modality
        self.output_dim = config.modality_dims[modality]
        
        # Encoder for observed data
        self.encoder = nn.Sequential(
            nn.Linear(self.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU()
        )
        
        # Noise injection
        self.noise_projection = nn.Linear(config.latent_dim, config.hidden_dim // 2)
        
        # Conditional information processing
        self.cond_encoder = nn.Sequential(
            nn.Linear(sum(config.modality_dims.values()), config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        )
        
        # Decoder with residual blocks
        decoder_layers = []
        in_features = config.hidden_dim
        
        for _ in range(config.num_layers):
            decoder_layers.append(ResidualBlock(in_features, config.hidden_dim, config.dropout))
            
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            *decoder_layers,
            nn.Linear(config.hidden_dim, self.output_dim)
        )
        
        # Output normalization
        self.output_norm = nn.BatchNorm1d(self.output_dim)
        
    def forward(self, observed_data: torch.Tensor, 
                condition: torch.Tensor,
                noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate imputed data
        
        Args:
            observed_data: Partially observed data for the target modality
            condition: Conditional information from other modalities
            noise: Optional noise vector
            
        Returns:
            Generated complete data
        """
        batch_size = observed_data.shape[0]
        
        # Encode observed data
        encoded_obs = self.encoder(observed_data)
        
        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.config.latent_dim, 
                              device=observed_data.device)
        noise_proj = self.noise_projection(noise)
        
        # Encode conditional information
        cond_encoded = self.cond_encoder(condition)
        
        # Combine features
        combined = torch.cat([encoded_obs, noise_proj, cond_encoded], dim=-1)
        
        # Decode to generate complete data
        generated = self.decoder(combined)
        generated = self.output_norm(generated)
        
        return generated


class MultimodalDiscriminator(nn.Module):
    """Discriminator for multi-modal GAN"""
    
    def __init__(self, config: GANImputationConfig):
        super().__init__()
        self.config = config
        
        # Modality-specific feature extractors
        self.modality_discriminators = nn.ModuleDict()
        
        for modality in config.modalities:
            input_dim = config.modality_dims[modality]
            self.modality_discriminators[modality] = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.LeakyReLU(0.2)
            )
        
        # Multi-modal fusion discriminator
        total_features = len(config.modalities) * (config.hidden_dim // 2)
        self.fusion_discriminator = nn.Sequential(
            nn.Linear(total_features, config.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Discriminate between real and generated multi-modal data
        
        Args:
            modalities: Dictionary of modality tensors
            
        Returns:
            Discriminator output (real/fake probability)
        """
        features = []
        
        for modality in self.config.modalities:
            if modality in modalities:
                feat = self.modality_discriminators[modality](modalities[modality])
                features.append(feat)
            else:
                # Zero features for missing modalities
                batch_size = next(iter(modalities.values())).shape[0]
                feat = torch.zeros(batch_size, self.config.hidden_dim // 2,
                                 device=next(iter(modalities.values())).device)
                features.append(feat)
        
        # Concatenate all features
        combined_features = torch.cat(features, dim=-1)
        
        # Final discrimination
        output = self.fusion_discriminator(combined_features)
        
        return output


class CycleConsistentGAN(nn.Module):
    """Cycle-consistent GAN for multi-modal imputation"""
    
    def __init__(self, config: GANImputationConfig):
        super().__init__()
        self.config = config
        
        # Generators for each modality
        self.generators = nn.ModuleDict()
        for modality in config.modalities:
            self.generators[modality] = ConditionalGenerator(config, modality)
        
        # Discriminators
        self.discriminators = nn.ModuleDict()
        for modality in config.modalities:
            self.discriminators[modality] = MultimodalDiscriminator(config)
        
        # Multi-modal discriminator
        self.multimodal_discriminator = MultimodalDiscriminator(config)
        
    def forward(self, partial_data: Dict[str, torch.Tensor],
                masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for cycle-consistent GAN
        
        Args:
            partial_data: Dictionary of partial modality data
            masks: Dictionary of binary masks indicating observed values
            
        Returns:
            Dictionary of imputed data for all modalities
        """
        imputed_data = {}
        
        # Generate complete data for each modality
        for modality in self.config.modalities:
            if modality in partial_data:
                # Prepare conditional information from other modalities
                condition = self._prepare_condition(partial_data, modality)
                
                # Generate noise
                batch_size = partial_data[modality].shape[0]
                noise = torch.randn(batch_size, self.config.latent_dim,
                                  device=partial_data[modality].device)
                
                # Generate complete data
                generated = self.generators[modality](
                    partial_data[modality], condition, noise
                )
                
                # Apply mask to keep observed values
                if modality in masks:
                    mask = masks[modality].unsqueeze(-1)
                    generated = generated * mask + partial_data[modality] * (1 - mask)
                
                imputed_data[modality] = generated
            else:
                # If modality is completely missing, generate from scratch
                condition = self._prepare_condition(partial_data, modality)
                batch_size = next(iter(partial_data.values())).shape[0]
                noise = torch.randn(batch_size, self.config.latent_dim,
                                  device=next(iter(partial_data.values())).device)
                
                # Create zero observed data
                zero_observed = torch.zeros(batch_size, self.config.modality_dims[modality],
                                          device=next(iter(partial_data.values())).device)
                
                generated = self.generators[modality](zero_observed, condition, noise)
                imputed_data[modality] = generated
        
        return imputed_data
    
    def _prepare_condition(self, data: Dict[str, torch.Tensor], target_modality: str) -> torch.Tensor:
        """Prepare conditional information from other modalities"""
        condition_features = []
        
        for modality in self.config.modalities:
            if modality != target_modality and modality in data:
                condition_features.append(data[modality])
            elif modality != target_modality:
                # Zero features for missing modalities
                batch_size = next(iter(data.values())).shape[0]
                zero_feat = torch.zeros(batch_size, self.config.modality_dims[modality],
                                       device=next(iter(data.values())).device)
                condition_features.append(zero_feat)
        
        return torch.cat(condition_features, dim=-1)
    
    def compute_cycle_consistency(self, partial_data: Dict[str, torch.Tensor],
                                 masks: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cycle consistency loss"""
        # Forward cycle: partial -> complete -> partial
        complete_data = self.forward(partial_data, masks)
        
        # Backward cycle: complete -> partial -> complete
        # This is a simplified version - in practice, you'd need to simulate missingness
        cycle_loss = 0.0
        
        for modality in self.config.modalities:
            if modality in partial_data:
                # Compare original partial data with reconstructed partial data
                original = partial_data[modality]
                reconstructed = complete_data[modality]
                
                # Only compare observed values
                if modality in masks:
                    mask = masks[modality].unsqueeze(-1)
                    cycle_loss += F.mse_loss(original * mask, reconstructed * mask)
        
        return cycle_loss / len([m for m in self.config.modalities if m in partial_data])


class WassersteinGAN(nn.Module):
    """Wasserstein GAN with gradient penalty for stable training"""
    
    def __init__(self, config: GANImputationConfig):
        super().__init__()
        self.config = config
        
        # Generator and critic
        self.generator = ConditionalGenerator(config, "multimodal")
        self.critic = MultimodalDiscriminator(config)
        
    def forward(self, partial_data: Dict[str, torch.Tensor],
                masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for Wasserstein GAN"""
        # This is a simplified version - would need adaptation for multi-modal case
        return self.generator(partial_data, masks)
    
    def compute_gradient_penalty(self, real_data: Dict[str, torch.Tensor],
                                fake_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute gradient penalty for Wasserstein GAN"""
        batch_size = next(iter(real_data.values())).shape[0]
        device = next(iter(real_data.values())).device
        
        # Interpolate between real and fake data
        alpha = torch.rand(batch_size, 1, device=device)
        
        interpolated = {}
        for modality in real_data:
            if modality in fake_data:
                interpolated[modality] = alpha * real_data[modality] + (1 - alpha) * fake_data[modality]
        
        # Compute critic output for interpolated data
        interpolated_output = self.critic(interpolated)
        
        # Compute gradient penalty
        gradients = torch.autograd.grad(
            outputs=interpolated_output,
            inputs=list(interpolated.values()),
            grad_outputs=torch.ones_like(interpolated_output),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


class GANImputer:
    """Main GAN-based imputation class"""
    
    def __init__(self, config: GANImputationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize models
        if config.missing_data_strategy == "cycle_gan":
            self.model = CycleConsistentGAN(config).to(self.device)
        elif config.missing_data_strategy == "wgan":
            self.model = WassersteinGAN(config).to(self.device)
        else:
            self.model = CycleConsistentGAN(config).to(self.device)
        
        # Optimizers
        self.g_optimizer = torch.optim.Adam(
            self.model.generators.parameters(),
            lr=config.generator_lr,
            weight_decay=config.weight_decay
        )
        
        self.d_optimizer = torch.optim.Adam(
            self.model.discriminators.parameters(),
            lr=config.discriminator_lr,
            weight_decay=config.weight_decay
        )
        
        # Training history
        self.training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'cycle_loss': [],
            'gradient_penalty': []
        }
        
    def train_step(self, real_data: Dict[str, torch.Tensor],
                   masks: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        batch_size = next(iter(real_data.values())).shape[0]
        
        # Move to device
        real_data = {k: v.to(self.device) for k, v in real_data.items()}
        masks = {k: v.to(self.device) for k, v in masks.items()}
        
        # Train discriminator
        self.d_optimizer.zero_grad()
        
        # Generate fake data
        with torch.no_grad():
            fake_data = self.model(real_data, masks)
        
        # Discriminator loss
        real_output = self.model.multimodal_discriminator(real_data)
        fake_output = self.model.multimodal_discriminator(fake_data)
        
        d_loss = -torch.mean(real_output) + torch.mean(fake_output)
        
        # Add gradient penalty for Wasserstein GAN
        if isinstance(self.model, WassersteinGAN):
            gp = self.model.compute_gradient_penalty(real_data, fake_data)
            d_loss += self.config.lambda_gp * gp
        
        d_loss.backward()
        self.d_optimizer.step()
        
        # Train generator
        self.g_optimizer.zero_grad()
        
        # Generate fake data
        fake_data = self.model(real_data, masks)
        
        # Generator loss
        fake_output = self.model.multimodal_discriminator(fake_data)
        g_loss = -torch.mean(fake_output)
        
        # Add cycle consistency loss
        if isinstance(self.model, CycleConsistentGAN):
            cycle_loss = self.model.compute_cycle_consistency(real_data, masks)
            g_loss += self.config.lambda_cycle * cycle_loss
        else:
            cycle_loss = torch.tensor(0.0, device=self.device)
        
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'generator_loss': g_loss.item(),
            'discriminator_loss': d_loss.item(),
            'cycle_loss': cycle_loss.item() if isinstance(cycle_loss, torch.Tensor) else 0.0,
            'gradient_penalty': gp.item() if isinstance(self.model, WassersteinGAN) else 0.0
        }
    
    def impute(self, partial_data: Dict[str, torch.Tensor],
               masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Impute missing data"""
        self.model.eval()
        
        with torch.no_grad():
            imputed_data = self.model(partial_data, masks)
        
        return imputed_data
    
    def train(self, dataloader, num_epochs: int = None):
        """Train the GAN imputer"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'generator_loss': [],
                'discriminator_loss': [],
                'cycle_loss': [],
                'gradient_penalty': []
            }
            
            for batch_idx, (real_data, masks) in enumerate(dataloader):
                losses = self.train_step(real_data, masks)
                
                for key, value in losses.items():
                    epoch_losses[key].append(value)
            
            # Log epoch averages
            epoch_avg = {k: np.mean(v) for k, v in epoch_losses.items()}
            self.training_history['generator_loss'].append(epoch_avg['generator_loss'])
            self.training_history['discriminator_loss'].append(epoch_avg['discriminator_loss'])
            self.training_history['cycle_loss'].append(epoch_avg['cycle_loss'])
            self.training_history['gradient_penalty'].append(epoch_avg['gradient_penalty'])
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}")
                print(f"  Generator Loss: {epoch_avg['generator_loss']:.4f}")
                print(f"  Discriminator Loss: {epoch_avg['discriminator_loss']:.4f}")
                print(f"  Cycle Loss: {epoch_avg['cycle_loss']:.4f}")
                print(f"  Gradient Penalty: {epoch_avg['gradient_penalty']:.4f}")
    
    def save(self, path: str):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
    
    def load(self, path: str):
        """Load the model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        self.training_history = checkpoint['training_history']