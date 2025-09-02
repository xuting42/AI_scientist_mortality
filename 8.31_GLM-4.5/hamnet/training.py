"""
Training and evaluation scripts for HAMNet

This module provides comprehensive training and evaluation functionality:
- Training loops with mixed precision support
- Evaluation with multiple metrics
- Cross-validation
- Hyperparameter optimization
- Model ensemble methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
import json
import time
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .models.hamnet import HAMNet, HAMNetConfig
from .utils.utils import (
    TrainingConfig, CheckpointManager, EarlyStopping, MetricsTracker,
    create_optimizer, create_scheduler, compute_loss, setup_logging,
    save_config, save_results, profile_model, get_model_size
)


class HAMNetTrainer:
    """Main trainer class for HAMNet"""
    
    def __init__(self, model: HAMNet, config: TrainingConfig,
                 experiment_dir: str = None):
        self.model = model
        self.config = config
        self.experiment_dir = Path(experiment_dir) if experiment_dir else None
        
        # Setup device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Setup optimizer and scheduler
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        
        # Setup checkpointing and early stopping
        if self.experiment_dir:
            self.checkpoint_manager = CheckpointManager(
                self.experiment_dir / "checkpoints", "hamnet"
            )
            self.early_stopping = EarlyStopping(config.patience, config.min_delta)
        
        # Setup logging
        self.logger = setup_logging(
            self.experiment_dir / "logs" if self.experiment_dir else "logs",
            f"hamnet_trainer_{int(time.time())}"
        )
        
        # Metrics tracking
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_r2': [], 'val_r2': [],
            'lr': []
        }
        
        self.logger.info("HAMNet Trainer initialized")
        self.logger.info(f"Model device: {self.device}")
        self.logger.info(f"Mixed precision: {config.mixed_precision}")
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(batch)
                predictions = outputs['predictions'].squeeze()
                targets = batch['targets']
                
                # Get uncertainties if available
                uncertainties = outputs.get('uncertainty', None)
                
                # Compute loss
                loss = compute_loss(predictions, targets, uncertainties)
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clipping
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clipping
                    )
                
                self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(predictions, targets, loss.item(), uncertainties)
            epoch_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log batch metrics
            if batch_idx % self.config.log_every == 0:
                batch_metrics = self.train_metrics.compute_metrics()
                self.logger.info(
                    f"Batch {batch_idx}/{num_batches} - "
                    f"Loss: {loss.item():.4f}, "
                    f"MAE: {batch_metrics.get('mae', 0):.4f}"
                )
        
        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute_metrics()
        epoch_metrics['loss'] = epoch_loss / num_batches
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        epoch_loss = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", leave=False)
            
            for batch in progress_bar:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(batch)
                    predictions = outputs['predictions'].squeeze()
                    targets = batch['targets']
                    
                    # Get uncertainties if available
                    uncertainties = outputs.get('uncertainty', None)
                    
                    # Compute loss
                    loss = compute_loss(predictions, targets, uncertainties)
                
                # Update metrics
                self.val_metrics.update(predictions, targets, loss.item(), uncertainties)
                epoch_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
        
        # Compute epoch metrics
        epoch_metrics = self.val_metrics.compute_metrics()
        epoch_metrics['loss'] = epoch_loss / num_batches
        
        return epoch_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              start_epoch: int = 0) -> Dict[str, Any]:
        """Main training loop"""
        self.logger.info("Starting training...")
        
        best_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(start_epoch, self.config.epochs):
            self.logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Update learning rate
            if self.config.scheduler == "reduce_on_plateau":
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
            
            # Update history
            self._update_history(train_metrics, val_metrics)
            
            # Log metrics
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)
            
            # Check for best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                is_best = True
            else:
                is_best = False
            
            # Save checkpoint
            if self.experiment_dir and (epoch % self.config.save_every == 0 or is_best):
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics['loss'],
                    asdict(self.config), is_best
                )
                self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Early stopping
            if self.experiment_dir and self.early_stopping(val_metrics['loss']):
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Final evaluation
        final_results = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'final_train_metrics': train_metrics,
            'final_val_metrics': val_metrics,
            'history': self.history,
            'model_size': get_model_size(self.model)
        }
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
        
        return final_results
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test set"""
        self.logger.info("Starting evaluation...")
        
        self.model.eval()
        test_metrics = MetricsTracker()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(batch)
                predictions = outputs['predictions'].squeeze()
                targets = batch['targets']
                
                # Get uncertainties if available
                uncertainties = outputs.get('uncertainty', None)
                
                # Store predictions
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                if uncertainties is not None:
                    all_uncertainties.extend(uncertainties.cpu().numpy())
                
                # Update metrics
                loss = compute_loss(predictions, targets, uncertainties)
                test_metrics.update(predictions, targets, loss.item(), uncertainties)
        
        # Compute metrics
        metrics = test_metrics.compute_metrics()
        
        # Additional analysis
        analysis_results = self._analyze_predictions(
            np.array(all_predictions), np.array(all_targets),
            np.array(all_uncertainties) if all_uncertainties else None
        )
        
        results = {
            'metrics': metrics,
            'analysis': analysis_results,
            'predictions': all_predictions,
            'targets': all_targets,
            'uncertainties': all_uncertainties if all_uncertainties else None
        }
        
        self.logger.info("Evaluation completed!")
        self.logger.info(f"Test MSE: {metrics['mse']:.4f}")
        self.logger.info(f"Test MAE: {metrics['mae']:.4f}")
        self.logger.info(f"Test R²: {metrics['r2']:.4f}")
        
        return results
    
    def predict_with_uncertainty(self, data_loader: DataLoader,
                                num_samples: int = 20) -> Dict[str, Any]:
        """Predict with uncertainty quantification"""
        self.model.eval()
        
        all_predictions = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                batch = self._move_batch_to_device(batch)
                
                # Monte Carlo dropout sampling
                predictions = []
                uncertainties = []
                
                for _ in range(num_samples):
                    outputs = self.model(batch)
                    predictions.append(outputs['predictions'].cpu().numpy())
                    if 'uncertainty' in outputs:
                        uncertainties.append(outputs['uncertainty'].cpu().numpy())
                
                # Compute statistics
                predictions = np.array(predictions)
                mean_pred = predictions.mean(axis=0)
                std_pred = predictions.std(axis=0)
                
                all_predictions.extend(mean_pred.flatten())
                all_uncertainties.extend(std_pred.flatten())
        
        return {
            'predictions': all_predictions,
            'uncertainties': all_uncertainties
        }
    
    def _move_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _update_history(self, train_metrics: Dict[str, float],
                       val_metrics: Dict[str, float]):
        """Update training history"""
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        
        if 'mae' in train_metrics:
            self.history['train_mae'].append(train_metrics['mae'])
        if 'mae' in val_metrics:
            self.history['val_mae'].append(val_metrics['mae'])
        
        if 'r2' in train_metrics:
            self.history['train_r2'].append(train_metrics['r2'])
        if 'r2' in val_metrics:
            self.history['val_r2'].append(val_metrics['r2'])
        
        self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
    
    def _log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float]):
        """Log epoch metrics"""
        self.logger.info(
            f"Epoch {epoch + 1} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}"
        )
        
        if 'mae' in train_metrics and 'mae' in val_metrics:
            self.logger.info(
                f"Train MAE: {train_metrics['mae']:.4f}, "
                f"Val MAE: {val_metrics['mae']:.4f}"
            )
        
        if 'r2' in train_metrics and 'r2' in val_metrics:
            self.logger.info(
                f"Train R²: {train_metrics['r2']:.4f}, "
                f"Val R²: {val_metrics['r2']:.4f}"
            )
    
    def _analyze_predictions(self, predictions: np.ndarray, targets: np.ndarray,
                           uncertainties: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze predictions and compute additional metrics"""
        analysis = {}
        
        # Basic statistics
        analysis['prediction_mean'] = np.mean(predictions)
        analysis['prediction_std'] = np.std(predictions)
        analysis['target_mean'] = np.mean(targets)
        analysis['target_std'] = np.std(targets)
        
        # Error analysis
        errors = predictions - targets
        analysis['error_mean'] = np.mean(errors)
        analysis['error_std'] = np.std(errors)
        analysis['error_median'] = np.median(errors)
        analysis['error_mad'] = np.median(np.abs(errors - np.median(errors)))
        
        # Uncertainty analysis
        if uncertainties is not None:
            analysis['uncertainty_mean'] = np.mean(uncertainties)
            analysis['uncertainty_std'] = np.std(uncertainties)
            
            # Uncertainty calibration
            abs_errors = np.abs(errors)
            analysis['uncertainty_correlation'] = np.corrcoef(abs_errors, uncertainties)[0, 1]
            
            # Expected calibration error
            n_bins = 10
            bin_indices = np.digitize(uncertainties, np.linspace(uncertainties.min(), uncertainties.max(), n_bins))
            calibration_errors = []
            
            for i in range(1, n_bins + 1):
                mask = bin_indices == i
                if np.sum(mask) > 0:
                    observed_error = np.mean(abs_errors[mask])
                    predicted_uncertainty = np.mean(uncertainties[mask])
                    calibration_errors.append(np.abs(observed_error - predicted_uncertainty))
            
            analysis['expected_calibration_error'] = np.mean(calibration_errors)
        
        return analysis


class CrossValidator:
    """Cross-validation for HAMNet"""
    
    def __init__(self, model_config: HAMNetConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
    def cross_validate(self, dataset, n_folds: int = 5,
                      stratified: bool = False) -> Dict[str, Any]:
        """Perform cross-validation"""
        
        if stratified:
            # Use age bins for stratification
            age_bins = np.digitize(dataset.targets, np.percentile(dataset.targets, [20, 40, 60, 80]))
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_generator = kf.split(dataset.targets, age_bins)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_generator = kf.split(dataset.targets)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(fold_generator):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            # Create fold datasets
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=self.training_config.batch_size,
                shuffle=True, num_workers=self.training_config.num_workers
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.training_config.batch_size,
                shuffle=False, num_workers=self.training_config.num_workers
            )
            
            # Create model and trainer
            model = HAMNet(self.model_config)
            trainer = HAMNetTrainer(model, self.training_config)
            
            # Train
            fold_results.append(trainer.train(train_loader, val_loader))
        
        # Aggregate results
        aggregated_results = self._aggregate_fold_results(fold_results)
        
        return aggregated_results
    
    def _aggregate_fold_results(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from all folds"""
        aggregated = {}
        
        # Extract metrics
        val_losses = [result['best_val_loss'] for result in fold_results]
        val_maes = [result['final_val_metrics']['mae'] for result in fold_results]
        val_r2s = [result['final_val_metrics']['r2'] for result in fold_results]
        
        aggregated['mean_val_loss'] = np.mean(val_losses)
        aggregated['std_val_loss'] = np.std(val_losses)
        aggregated['mean_val_mae'] = np.mean(val_maes)
        aggregated['std_val_mae'] = np.std(val_maes)
        aggregated['mean_val_r2'] = np.mean(val_r2s)
        aggregated['std_val_r2'] = np.std(val_r2s)
        
        aggregated['fold_results'] = fold_results
        
        return aggregated


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, model_config: HAMNetConfig, training_config: TrainingConfig,
                 dataset, n_trials: int = 50):
        self.model_config = model_config
        self.training_config = training_config
        self.dataset = dataset
        self.n_trials = n_trials
        
    def optimize(self) -> optuna.Study:
        """Run hyperparameter optimization"""
        
        def objective(trial):
            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            embedding_dim = trial.suggest_categorical('embedding_dim', [128, 256, 512])
            hidden_dim = trial.suggest_categorical('hidden_dim', [256, 512, 1024])
            num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
            
            # Update configs
            self.training_config.learning_rate = learning_rate
            self.training_config.dropout = dropout
            self.model_config.embedding_dim = embedding_dim
            self.model_config.hidden_dim = hidden_dim
            self.model_config.dropout = dropout
            self.model_config.num_heads = num_heads
            
            # Create data loaders
            train_loader = DataLoader(
                self.dataset, batch_size=self.training_config.batch_size,
                shuffle=True, num_workers=self.training_config.num_workers
            )
            
            # Quick validation (small subset)
            val_size = min(1000, len(self.dataset) // 5)
            val_indices = np.random.choice(len(self.dataset), val_size, replace=False)
            val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
            val_loader = DataLoader(
                val_dataset, batch_size=self.training_config.batch_size,
                shuffle=False, num_workers=self.training_config.num_workers
            )
            
            # Create model and trainer
            model = HAMNet(self.model_config)
            trainer = HAMNetTrainer(model, self.training_config)
            
            # Train for few epochs
            results = trainer.train(train_loader, val_loader, start_epoch=0)
            
            return results['best_val_loss']
        
        # Create study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        return study


class ModelEnsemble:
    """Model ensemble for improved predictions"""
    
    def __init__(self, models: List[HAMNet]):
        self.models = models
        
    def predict(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Ensemble prediction"""
        all_predictions = []
        
        for model in self.models:
            model.eval()
            model_predictions = []
            
            with torch.no_grad():
                for batch in data_loader:
                    # Move batch to device
                    device_batch = {}
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            device_batch[key] = value.to(next(model.parameters()).device)
                        else:
                            device_batch[key] = value
                    
                    outputs = model(device_batch)
                    predictions = outputs['predictions'].squeeze()
                    model_predictions.extend(predictions.cpu().numpy())
            
            all_predictions.append(model_predictions)
        
        # Average predictions
        all_predictions = np.array(all_predictions)
        ensemble_predictions = np.mean(all_predictions, axis=0)
        ensemble_std = np.std(all_predictions, axis=0)
        
        return {
            'predictions': ensemble_predictions,
            'uncertainties': ensemble_std
        }
    
    def save(self, save_path: str):
        """Save ensemble models"""
        save_dict = {
            'model_configs': [],
            'model_states': []
        }
        
        for model in self.models:
            save_dict['model_configs'].append(model.config.__dict__)
            save_dict['model_states'].append(model.state_dict())
        
        torch.save(save_dict, save_path)
    
    @classmethod
    def load(cls, load_path: str, device: str = 'cuda'):
        """Load ensemble models"""
        save_dict = torch.load(load_path, map_location=device)
        
        models = []
        for config_dict, state_dict in zip(
            save_dict['model_configs'], save_dict['model_states']
        ):
            config = HAMNetConfig(**config_dict)
            model = HAMNet(config)
            model.load_state_dict(state_dict)
            model.to(device)
            models.append(model)
        
        return cls(models)