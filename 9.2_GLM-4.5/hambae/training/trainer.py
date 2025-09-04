"""
Main trainer class for HAMBAE algorithm system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import os
from pathlib import Path
from dataclasses import dataclass
import json

from ..config import HAMBAEConfig, TrainingConfig
from ..models.base_model import BaseBiologicalAgeModel, ModelOutput
from ..uncertainty.bayesian_layers import compute_bayesian_kl_divergence
from .optimization import OptimizerManager, LearningRateScheduler
from .validation import ValidationManager
from .callbacks import TrainingCallback

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    epoch: int
    train_loss: float
    val_loss: float
    train_mae: float
    val_mae: float
    train_r2: float
    val_r2: float
    learning_rate: float
    epoch_time: float
    best_val_loss: float
    best_val_mae: float
    best_val_r2: float
    convergence_reached: bool = False


class HAMBATrainer:
    """
    Main trainer class for HAMBAE algorithm system.
    
    This class handles the complete training pipeline including
    optimization, validation, checkpointing, and monitoring.
    """
    
    def __init__(
        self,
        model: BaseBiologicalAgeModel,
        config: HAMBAEConfig,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        experiment_dir: Optional[str] = None,
    ):
        """
        Initialize HAMBAE trainer.
        
        Args:
            model: Model to train
            config: HAMBAE configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            callbacks: List of training callbacks
            experiment_dir: Experiment directory
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.callbacks = callbacks or []
        
        # Set up experiment directory
        self.experiment_dir = Path(experiment_dir or f"experiments/{config.experiment_name}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training components
        self.optimizer_manager = OptimizerManager(model, config)
        self.lr_scheduler = LearningRateScheduler(self.optimizer_manager.optimizer, config)
        self.validation_manager = ValidationManager(config)
        
        # Set up device
        self.device = torch.device(config.training.device if config.training.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize metrics tracking
        self.metrics_history = []
        self.best_val_loss = float('inf')
        self.best_val_mae = float('inf')
        self.best_val_r2 = -float('inf')
        self.patience_counter = 0
        self.convergence_reached = False
        
        # Set up logging
        self._setup_logging()
        
        # Set up tensorboard
        self.writer = SummaryWriter(log_dir=str(self.experiment_dir / "logs"))
        
        logger.info(f"Initialized HAMBATrainer with experiment directory: {self.experiment_dir}")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_file = self.experiment_dir / "training.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    def train(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            max_epochs: Maximum number of epochs
            
        Returns:
            Training results dictionary
        """
        max_epochs = max_epochs or self.config.training.max_epochs
        
        logger.info(f"Starting training for {max_epochs} epochs")
        
        # Call on_train_begin callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        for epoch in range(max_epochs):
            if self.convergence_reached:
                logger.info(f"Convergence reached at epoch {epoch}")
                break
            
            # Call on_epoch_begin callbacks
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self)
            
            # Train epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validate epoch
            val_metrics = self._validate_epoch(epoch)
            
            # Update learning rate
            self.lr_scheduler.step(val_metrics['val_loss'])
            
            # Compute metrics
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics['loss'],
                val_loss=val_metrics['loss'],
                train_mae=train_metrics['mae'],
                val_mae=val_metrics['mae'],
                train_r2=train_metrics['r2'],
                val_r2=val_metrics['r2'],
                learning_rate=self.lr_scheduler.get_last_lr()[0],
                epoch_time=train_metrics['epoch_time'],
                best_val_loss=self.best_val_loss,
                best_val_mae=self.best_val_mae,
                best_val_r2=self.best_val_r2,
                convergence_reached=self.convergence_reached,
            )
            
            # Log metrics
            self._log_metrics(metrics)
            
            # Update best metrics
            self._update_best_metrics(val_metrics)
            
            # Save checkpoint
            if val_metrics['loss'] < self.best_val_loss:
                self._save_checkpoint(epoch, is_best=True)
            
            # Check for early stopping
            if self._check_early_stopping(val_metrics['loss']):
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Call on_epoch_end callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics, self)
            
            # Save regular checkpoint
            if epoch % self.config.training.save_freq == 0:
                self._save_checkpoint(epoch)
        
        # Final evaluation
        if self.test_loader is not None:
            test_metrics = self._evaluate_test_set()
            logger.info(f"Test set evaluation: {test_metrics}")
        
        # Call on_train_end callbacks
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        # Close tensorboard writer
        self.writer.close()
        
        # Return training results
        results = {
            'metrics_history': self.metrics_history,
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'best_val_r2': self.best_val_r2,
            'convergence_epoch': epoch if self.convergence_reached else max_epochs,
            'total_epochs': epoch + 1,
        }
        
        if self.test_loader is not None:
            results['test_metrics'] = test_metrics
        
        return results
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_mae = 0.0
        total_r2 = 0.0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Compute loss
            loss = self._compute_loss(outputs, batch)
            
            # Add KL divergence for Bayesian models
            kl_divergence = compute_bayesian_kl_divergence(self.model)
            loss = loss + kl_divergence
            
            # Backward pass
            self.optimizer_manager.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.gradient_clip_norm
                )
            
            # Optimization step
            self.optimizer_manager.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_mae += self._compute_mae(outputs.predicted_age, batch['age']).item()
            total_r2 += self._compute_r2(outputs.predicted_age, batch['age']).item()
            num_batches += 1
            
            # Log batch metrics
            if batch_idx % self.config.training.log_freq == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, MAE: {total_mae / num_batches:.4f}"
                )
        
        epoch_time = time.time() - epoch_start_time
        
        # Compute average metrics
        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_r2 = total_r2 / num_batches
        
        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'r2': avg_r2,
            'epoch_time': epoch_time,
        }
    
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        return self.validation_manager.validate(self.model, self.val_loader, self.device)
    
    def _evaluate_test_set(self) -> Dict[str, float]:
        """Evaluate on test set."""
        if self.test_loader is None:
            return {}
        
        return self.validation_manager.validate(self.model, self.test_loader, self.device)
    
    def _compute_loss(self, outputs: ModelOutput, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for model outputs."""
        # Main age prediction loss
        age_loss = F.mse_loss(outputs.predicted_age, batch['age'])
        
        # Add uncertainty loss if available
        if outputs.uncertainty is not None and self.config.uncertainty.heteroscedastic_loss:
            uncertainty_loss = self._compute_uncertainty_loss(outputs.predicted_age, batch['age'], outputs.uncertainty)
            age_loss += self.config.uncertainty.uncertainty_weight * uncertainty_loss
        
        # Add auxiliary losses if available
        auxiliary_loss = 0.0
        if outputs.auxiliary_outputs is not None:
            auxiliary_loss = self._compute_auxiliary_losses(outputs.auxiliary_outputs, batch)
        
        total_loss = age_loss + auxiliary_loss
        
        return total_loss
    
    def _compute_uncertainty_loss(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Compute heteroscedastic uncertainty loss."""
        # Gaussian negative log-likelihood
        variance = torch.exp(uncertainty)
        loss = 0.5 * torch.exp(-uncertainty) * (predictions - targets) ** 2 + 0.5 * uncertainty
        return torch.mean(loss)
    
    def _compute_auxiliary_losses(
        self, 
        auxiliary_outputs: Dict[str, torch.Tensor], 
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute auxiliary losses."""
        total_loss = 0.0
        
        # Mortality prediction loss
        if 'mortality' in auxiliary_outputs and 'mortality_status' in batch:
            mortality_loss = F.binary_cross_entropy(
                auxiliary_outputs['mortality'], 
                batch['mortality_status'].float()
            )
            total_loss += 0.1 * mortality_loss
        
        # Organ-specific losses
        for key, value in auxiliary_outputs.items():
            if key.startswith('organ_') and key.endswith('_age'):
                # Assume organ age targets are available
                organ_name = key.replace('organ_', '').replace('_age', '')
                if f'{organ_name}_age' in batch:
                    organ_loss = F.mse_loss(value, batch[f'{organ_name}_age'])
                    total_loss += 0.05 * organ_loss
        
        return total_loss
    
    def _compute_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Mean Absolute Error."""
        return torch.mean(torch.abs(predictions - targets))
    
    def _compute_r2(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute R-squared score."""
        target_mean = torch.mean(targets)
        ss_tot = torch.sum((targets - target_mean) ** 2)
        ss_res = torch.sum((targets - predictions) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _update_best_metrics(self, val_metrics: Dict[str, float]) -> None:
        """Update best metrics."""
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
        
        if val_metrics['mae'] < self.best_val_mae:
            self.best_val_mae = val_metrics['mae']
        
        if val_metrics['r2'] > self.best_val_r2:
            self.best_val_r2 = val_metrics['r2']
    
    def _check_early_stopping(self, val_loss: float) -> bool:
        """Check for early stopping."""
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.training.early_stopping_patience:
            self.convergence_reached = True
            return True
        
        return False
    
    def _log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log metrics to tensorboard and console."""
        # Add to history
        self.metrics_history.append(metrics)
        
        # Log to tensorboard
        self.writer.add_scalar('Loss/train', metrics.train_loss, metrics.epoch)
        self.writer.add_scalar('Loss/val', metrics.val_loss, metrics.epoch)
        self.writer.add_scalar('MAE/train', metrics.train_mae, metrics.epoch)
        self.writer.add_scalar('MAE/val', metrics.val_mae, metrics.epoch)
        self.writer.add_scalar('R2/train', metrics.train_r2, metrics.epoch)
        self.writer.add_scalar('R2/val', metrics.val_r2, metrics.epoch)
        self.writer.add_scalar('Learning_Rate', metrics.learning_rate, metrics.epoch)
        self.writer.add_scalar('Epoch_Time', metrics.epoch_time, metrics.epoch)
        
        # Log to console
        logger.info(
            f"Epoch {metrics.epoch}: "
            f"Train Loss: {metrics.train_loss:.4f}, Val Loss: {metrics.val_loss:.4f}, "
            f"Train MAE: {metrics.train_mae:.4f}, Val MAE: {metrics.val_mae:.4f}, "
            f"Train R2: {metrics.train_r2:.4f}, Val R2: {metrics.val_r2:.4f}, "
            f"LR: {metrics.learning_rate:.6f}, Time: {metrics.epoch_time:.2f}s"
        )
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer_manager.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_mae': self.best_val_mae,
            'best_val_r2': self.best_val_r2,
            'config': self.config,
            'metrics_history': self.metrics_history,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.experiment_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_manager.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_mae = checkpoint['best_val_mae']
        self.best_val_r2 = checkpoint['best_val_r2']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def save_training_results(self, results: Dict[str, Any]) -> None:
        """Save training results to file."""
        results_path = self.experiment_dir / "training_results.json"
        
        # Convert tensors to numpy arrays for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return str(obj)
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved training results to {results_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.model.get_model_info()
    
    def __repr__(self) -> str:
        """String representation of the trainer."""
        return f"HAMBATrainer(model={self.model.__class__.__name__}, device={self.device})"