"""
Comprehensive Training Framework for HAMNet Biological Age Prediction

This module provides a complete training framework for the HAMNet model with:
- Multi-objective loss functions
- Mixed-precision training
- Distributed training support
- Advanced optimization strategies
- Comprehensive logging and monitoring
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau
)
from torch.optim import AdamW, SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Import HAMNet components
from ..models.hamnet import HAMNet, HAMNetConfig
from ..models.uncertainty_quantification import UncertaintyQuantification
from ..models.xai_module import XAIModule
from ..utils.utils import setup_logging, save_checkpoint, load_checkpoint


@dataclass
class TrainingConfig:
    """Configuration for training framework"""
    # Model configuration
    model_config: HAMNetConfig = field(default_factory=HAMNetConfig)
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_epochs: int = 100
    patience: int = 15
    
    # Optimization
    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-7
    
    # Mixed precision
    use_amp: bool = True
    grad_clip: float = 1.0
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Loss weights
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'mae': 1.0,
        'mse': 0.5,
        'uncertainty': 0.3,
        'consistency': 0.2,
        'attention': 0.1
    })
    
    # Logging
    log_dir: str = './logs'
    checkpoint_dir: str = './checkpoints'
    save_every: int = 10
    eval_every: int = 5
    
    # Advanced
    use_curriculum: bool = True
    curriculum_stages: int = 3
    use_ensemble: bool = False
    ensemble_size: int = 5


class MultiObjectiveLoss(nn.Module):
    """Multi-objective loss function for biological age prediction"""
    
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {
            'mae': 1.0,
            'mse': 0.5,
            'uncertainty': 0.3,
            'consistency': 0.2,
            'attention': 0.1
        }
        
        # Individual loss components
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.uncertainty_loss = HeteroscedasticLoss()
        self.consistency_loss = ConsistencyLoss()
        self.attention_loss = AttentionLoss()
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: torch.Tensor, 
                model_outputs: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-objective loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            model_outputs: Additional model outputs for consistency and attention losses
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # MAE loss
        if 'bio_age' in predictions:
            losses['mae'] = self.mae_loss(predictions['bio_age'], targets)
        
        # MSE loss
        if 'bio_age' in predictions:
            losses['mse'] = self.mse_loss(predictions['bio_age'], targets)
        
        # Uncertainty loss
        if 'uncertainty' in predictions and 'bio_age' in predictions:
            losses['uncertainty'] = self.uncertainty_loss(
                predictions['bio_age'], targets, predictions['uncertainty']
            )
        
        # Consistency loss
        if model_outputs and 'attention_weights' in model_outputs:
            losses['consistency'] = self.consistency_loss(model_outputs['attention_weights'])
        
        # Attention regularization loss
        if model_outputs and 'attention_weights' in model_outputs:
            losses['attention'] = self.attention_loss(model_outputs['attention_weights'])
        
        # Calculate total loss
        total_loss = 0.0
        for loss_name, loss_value in losses.items():
            if loss_name in self.weights:
                total_loss += self.weights[loss_name] * loss_value
        
        losses['total'] = total_loss
        
        return losses


class HeteroscedasticLoss(nn.Module):
    """Heteroscedastic loss for uncertainty estimation"""
    
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, 
                targets: torch.Tensor, 
                uncertainties: torch.Tensor) -> torch.Tensor:
        """
        Calculate heteroscedastic loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            uncertainties: Predicted uncertainties
            
        Returns:
            Heteroscedastic loss
        """
        # Clamp uncertainties to avoid numerical issues
        uncertainties = torch.clamp(uncertainties, min=self.epsilon)
        
        # Calculate squared error
        squared_error = (predictions - targets) ** 2
        
        # Calculate loss
        loss = 0.5 * (squared_error / uncertainties**2 + torch.log(uncertainties**2))
        
        return torch.mean(loss)


class ConsistencyLoss(nn.Module):
    """Consistency loss for attention weights"""
    
    def __init__(self, lambda_temp: float = 0.1):
        super().__init__()
        self.lambda_temp = lambda_temp
    
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate consistency loss for attention weights
        
        Args:
            attention_weights: Multi-head attention weights
            
        Returns:
            Consistency loss
        """
        # Calculate entropy of attention weights
        attention_probs = torch.softmax(attention_weights, dim=-1)
        log_attention_probs = torch.log(attention_probs + 1e-8)
        entropy = -torch.sum(attention_probs * log_attention_probs, dim=-1)
        
        # Minimize entropy (encourage focused attention)
        loss = torch.mean(entropy)
        
        return loss


class AttentionLoss(nn.Module):
    """Regularization loss for attention mechanisms"""
    
    def __init__(self, lambda_sparse: float = 0.01):
        super().__init__()
        self.lambda_sparse = lambda_sparse
    
    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate attention regularization loss
        
        Args:
            attention_weights: Multi-head attention weights
            
        Returns:
            Attention regularization loss
        """
        # L1 regularization on attention weights
        l1_loss = torch.mean(torch.abs(attention_weights))
        
        return self.lambda_sparse * l1_loss


class CurriculumLearning:
    """Curriculum learning strategy for progressive training"""
    
    def __init__(self, stages: int = 3, initial_ratio: float = 0.3):
        self.stages = stages
        self.initial_ratio = initial_ratio
        self.current_stage = 0
        self.stage_epochs = []
        
    def get_current_ratio(self, epoch: int, total_epochs: int) -> float:
        """
        Get current data ratio for curriculum learning
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            
        Returns:
            Data ratio for current stage
        """
        # Calculate epochs per stage
        epochs_per_stage = total_epochs // self.stages
        
        # Determine current stage
        current_stage = min(epoch // epochs_per_stage, self.stages - 1)
        
        # Calculate ratio
        ratio = self.initial_ratio + (1.0 - self.initial_ratio) * (current_stage / (self.stages - 1))
        
        return min(ratio, 1.0)
    
    def should_advance_stage(self, epoch: int, total_epochs: int) -> bool:
        """Check if should advance to next stage"""
        epochs_per_stage = total_epochs // self.stages
        return epoch % epochs_per_stage == 0 and epoch > 0


class AdvancedTrainer:
    """Advanced trainer for HAMNet with multiple optimization strategies"""
    
    def __init__(self, config: TrainingConfig, model: HAMNet):
        self.config = config
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = setup_logging('HAMNet_Trainer')
        
        # Initialize components
        self.criterion = MultiObjectiveLoss(config.loss_weights)
        self.curriculum = CurriculumLearning(config.curriculum_stages) if config.use_curriculum else None
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.use_amp else None
        
        # Setup logging
        self.writer = SummaryWriter(config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup distributed training if enabled
        if config.distributed:
            self._setup_distributed()
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on configuration"""
        if self.config.optimizer.lower() == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.config.optimizer.lower() == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        if self.config.scheduler.lower() == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler.lower() == 'cosine_warm_restarts':
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler.lower() == 'onecycle':
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.max_epochs,
                steps_per_epoch=100  # Will be updated during training
            )
        elif self.config.scheduler.lower() == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _setup_distributed(self):
        """Setup distributed training"""
        if self.config.distributed:
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            # Wrap model with DDP
            self.model = DDP(self.model, device_ids=[self.config.rank])
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {key: [] for key in self.config.loss_weights.keys()}
        epoch_losses['total'] = []
        
        # Set curriculum ratio if enabled
        curriculum_ratio = 1.0
        if self.curriculum:
            curriculum_ratio = self.curriculum.get_current_ratio(
                self.current_epoch, self.config.max_epochs
            )
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                predictions, model_outputs = self.model(batch)
                targets = batch.get('chronological_age', batch.get('age'))
                
                # Calculate losses
                losses = self.criterion(predictions, targets, model_outputs)
            
            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(losses['total']).backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total'].backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.grad_clip
                    )
                
                self.optimizer.step()
            
            # Update learning rate for OneCycleLR
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()
            
            # Record losses
            for loss_name, loss_value in losses.items():
                epoch_losses[loss_name].append(loss_value.item())
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}, "
                    f"Loss: {losses['total'].item():.4f}"
                )
        
        # Step scheduler if not OneCycleLR
        if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
            self.scheduler.step()
        
        # Calculate average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        val_losses = {key: [] for key in self.config.loss_weights.keys()}
        val_losses['total'] = []
        
        predictions_list = []
        targets_list = []
        uncertainties_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_to_device(batch)
                
                # Forward pass
                with autocast(enabled=self.config.use_amp):
                    predictions, model_outputs = self.model(batch)
                    targets = batch.get('chronological_age', batch.get('age'))
                    
                    # Calculate losses
                    losses = self.criterion(predictions, targets, model_outputs)
                
                # Record losses
                for loss_name, loss_value in losses.items():
                    val_losses[loss_name].append(loss_value.item())
                
                # Store predictions for metrics calculation
                predictions_list.append(predictions['bio_age'].cpu().numpy())
                targets_list.append(targets.cpu().numpy())
                if 'uncertainty' in predictions:
                    uncertainties_list.append(predictions['uncertainty'].cpu().numpy())
        
        # Calculate average losses
        avg_losses = {key: np.mean(values) for key, values in val_losses.items()}
        
        # Calculate additional metrics
        predictions_all = np.concatenate(predictions_list)
        targets_all = np.concatenate(targets_list)
        
        # MAE
        mae = np.mean(np.abs(predictions_all - targets_all))
        avg_losses['mae_metric'] = mae
        
        # RMSE
        rmse = np.sqrt(np.mean((predictions_all - targets_all) ** 2))
        avg_losses['rmse'] = rmse
        
        # R²
        r2 = 1 - np.sum((predictions_all - targets_all) ** 2) / np.sum((targets_all - np.mean(targets_all)) ** 2)
        avg_losses['r2'] = r2
        
        # Uncertainty calibration if available
        if uncertainties_list:
            uncertainties_all = np.concatenate(uncertainties_list)
            calibration_error = self._calculate_calibration_error(
                predictions_all, targets_all, uncertainties_all
            )
            avg_losses['calibration_error'] = calibration_error
        
        return avg_losses
    
    def _calculate_calibration_error(self, predictions: np.ndarray, 
                                   targets: np.ndarray, 
                                   uncertainties: np.ndarray) -> float:
        """Calculate expected calibration error"""
        # Calculate prediction intervals
        lower_bound = predictions - 1.96 * uncertainties
        upper_bound = predictions + 1.96 * uncertainties
        
        # Calculate coverage
        coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))
        
        # Expected calibration error
        ece = np.abs(coverage - 0.95)
        
        return ece
    
    def _move_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            if val_loader is not None:
                val_losses = self.validate(val_loader)
                
                # Log validation losses
                for loss_name, loss_value in val_losses.items():
                    self.writer.add_scalar(f'val/{loss_name}', loss_value, epoch)
                
                # Early stopping
                if val_losses['total'] < self.best_loss:
                    self.best_loss = val_losses['total']
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        os.path.join(self.config.checkpoint_dir, 'best_model.pth'),
                        is_best=True
                    )
                else:
                    self.early_stopping_counter += 1
                    
                    if self.early_stopping_counter >= self.config.patience:
                        self.logger.info("Early stopping triggered")
                        break
            
            # Log training losses
            for loss_name, loss_value in train_losses.items():
                self.writer.add_scalar(f'train/{loss_name}', loss_value, epoch)
            
            # Log learning rate
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            if epoch % self.config.save_every == 0:
                self.save_checkpoint(
                    os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
                )
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch}/{self.config.max_epochs} - "
                f"Train Loss: {train_losses['total']:.4f}, "
                f"Val Loss: {val_losses['total']:.4f if val_loader else 'N/A'}"
            )
        
        self.logger.info("Training completed!")
        self.writer.close()
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            # Create symlink to best model
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            if os.path.exists(best_path):
                os.remove(best_path)
            os.symlink(filepath, best_path)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"Checkpoint loaded from {filepath}")


class EnsembleTrainer:
    """Ensemble trainer for multiple HAMNet models"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = []
        self.trainers = []
        
        # Create ensemble of models
        for i in range(config.ensemble_size):
            # Create model with different random seed
            torch.manual_seed(config.rank * config.ensemble_size + i)
            np.random.seed(config.rank * config.ensemble_size + i)
            
            model = HAMNet(config.model_config)
            trainer = AdvancedTrainer(config, model)
            
            self.models.append(model)
            self.trainers.append(trainer)
    
    def train_all(self, train_loader: DataLoader, val_loader: DataLoader = None):
        """Train all models in ensemble"""
        for i, trainer in enumerate(self.trainers):
            self.logger.info(f"Training model {i+1}/{len(self.trainers)}")
            trainer.fit(train_loader, val_loader)
    
    def predict_ensemble(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """Make ensemble predictions"""
        all_predictions = []
        all_uncertainties = []
        
        for trainer in self.trainers:
            trainer.model.eval()
            model_predictions = []
            model_uncertainties = []
            
            with torch.no_grad():
                for batch in data_loader:
                    batch = trainer._move_to_device(batch)
                    predictions, _ = trainer.model(batch)
                    
                    model_predictions.append(predictions['bio_age'].cpu().numpy())
                    if 'uncertainty' in predictions:
                        model_uncertainties.append(predictions['uncertainty'].cpu().numpy())
            
            all_predictions.append(np.concatenate(model_predictions))
            if model_uncertainties:
                all_uncertainties.append(np.concatenate(model_uncertainties))
        
        # Calculate ensemble predictions
        ensemble_predictions = np.mean(all_predictions, axis=0)
        ensemble_uncertainties = np.std(all_predictions, axis=0)
        
        return {
            'predictions': ensemble_predictions,
            'uncertainties': ensemble_uncertainties,
            'individual_predictions': all_predictions
        }


def create_training_config(config_path: str = None) -> TrainingConfig:
    """Create training configuration from file or defaults"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return TrainingConfig(**config_dict)
    else:
        return TrainingConfig()


def main():
    """Main training script"""
    # Create configuration
    config = create_training_config()
    
    # Create model
    model = HAMNet(config.model_config)
    
    # Create trainer
    trainer = AdvancedTrainer(config, model)
    
    # Create data loaders (placeholder - should be implemented based on data)
    train_loader = None  # TODO: Implement data loading
    val_loader = None    # TODO: Implement data loading
    
    # Train model
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    main()