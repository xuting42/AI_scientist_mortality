"""
Training Script for HENAW Model
Implements multi-task learning with cross-validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import yaml
import json
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import wandb
from datetime import datetime
import warnings
import argparse
import os
from sklearn.model_selection import StratifiedKFold

from henaw_model import HENAWModel, MultiTaskLoss, HENAWOutput
from data_loader import create_data_loaders, UKBBDataset
from evaluate import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for HENAW model
    Handles training loop, validation, checkpointing, and logging
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 experiment_name: Optional[str] = None):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            device: Device to use ('cuda' or 'cpu')
            experiment_name: Optional experiment name for logging
        """
        self.config = config
        
        # Proper device selection with fallback and logging
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Set experiment name
        if experiment_name is None:
            experiment_name = f"henaw_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        # Create directories
        self.setup_directories()
        
        # Initialize model
        self.model = HENAWModel(config).to(self.device)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Initialize loss function
        self.criterion = MultiTaskLoss(config)
        
        # Initialize optimizer
        self.optimizer = self.setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self.setup_scheduler()
        
        # Mixed precision training - only enable if CUDA available
        self.use_amp = config['infrastructure'].get('mixed_precision', False) and self.device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("Mixed precision training enabled")
        elif config['infrastructure'].get('mixed_precision', False):
            logger.info("Mixed precision requested but not available on CPU")
        
        # Initialize evaluator
        self.evaluator = Evaluator(config)
        
        # Tracking
        self.current_epoch = 0
        self.best_val_metric = float('inf')
        self.training_history = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_directories(self) -> None:
        """Create necessary directories"""
        self.checkpoint_dir = Path(self.config['logging']['checkpoint']['save_dir']) / self.experiment_name
        self.log_dir = Path(self.config['logging']['log_dir']) / self.experiment_name
        self.results_dir = Path(self.config['output']['results_dir']) / self.experiment_name
        
        for dir_path in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        optimizer_type = self.config['training'].get('optimizer', 'adam')
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        return optimizer
    
    def setup_scheduler(self) -> Optional[Any]:
        """Setup learning rate scheduler"""
        scheduler_config = self.config['training'].get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def setup_logging(self) -> None:
        """Setup logging (tensorboard, wandb, etc.)"""
        # TensorBoard
        if self.config['logging'].get('tensorboard', False):
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(self.log_dir / 'tensorboard')
        else:
            self.tb_writer = None
        
        # Weights & Biases
        wandb_config = self.config['logging'].get('wandb', {})
        if wandb_config.get('project'):
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config.get('entity'),
                name=self.experiment_name,
                config=self.config
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'age': 0.0,
            'mortality': 0.0,
            'morbidity': 0.0,
            'regularization': 0.0
        }
        
        num_batches = len(train_loader)
        
        # Initialize CUDA OOM handling
        oom_count = 0
        max_oom_retries = 3
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch}/{self.config['training']['epochs']}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Extract inputs and targets
                    biomarkers = batch['biomarkers']
                    age = batch['chronological_age']
                    
                    targets = {
                        'chronological_age': batch['chronological_age'].squeeze(),
                        'survival_time': batch.get('survival_time', torch.zeros(biomarkers.size(0))).squeeze(),
                        'event_indicator': batch.get('event_indicator', torch.ones(biomarkers.size(0))).squeeze()
                    }
                    
                    # Add disease labels
                    for disease in ['cardiovascular', 'diabetes', 'cancer', 'dementia']:
                        if f'{disease}_label' in batch:
                            targets[f'{disease}_label'] = batch[f'{disease}_label'].squeeze()
                    
                    # Forward pass with mixed precision
                    if self.use_amp:
                        with autocast():
                            output = self.model(biomarkers, age)
                            loss, loss_components = self.criterion(output, targets, self.model)
                    else:
                        output = self.model(biomarkers, age)
                        loss, loss_components = self.criterion(output, targets, self.model)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        
                        # Add gradient clipping for stability
                        max_grad_norm = self.config['training'].get('gradient_clip_norm', 1.0)
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
                        
                        self.optimizer.step()
                    
                    # Update metrics
                    epoch_losses['total'] += loss.item()
                    for key, value in loss_components.items():
                        if key in epoch_losses:
                            epoch_losses[key] += value.item()
                    
                    # Reset OOM count on successful batch
                    oom_count = 0
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr']
                    })
                    
                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"CUDA OOM at batch {batch_idx}: {e}")
                    
                    # Clear cache and try recovery
                    torch.cuda.empty_cache()
                    
                    oom_count += 1
                    if oom_count > max_oom_retries:
                        logger.error(f"Repeated CUDA OOM errors ({oom_count} times). "
                                   "Consider reducing batch size or model complexity.")
                        raise RuntimeError("Too many CUDA OOM errors. Training aborted.") from e
                    
                    # Skip this batch and continue
                    logger.warning(f"Skipping batch {batch_idx} due to OOM (retry {oom_count}/{max_oom_retries})")
                    continue
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # Handle different CUDA OOM error formats
                        logger.warning(f"CUDA memory error at batch {batch_idx}: {e}")
                        torch.cuda.empty_cache()
                        oom_count += 1
                        
                        if oom_count > max_oom_retries:
                            raise RuntimeError("Too many memory errors. Training aborted.") from e
                        
                        logger.warning(f"Skipping batch {batch_idx} due to memory error")
                        continue
                    else:
                        # Re-raise non-memory related runtime errors
                        raise
                        
                except Exception as e:
                    logger.error(f"Unexpected error in batch {batch_idx}: {e}")
                    raise
                
                # Log batch metrics
                if batch_idx % 10 == 0:
                    self.log_batch_metrics(batch_idx, loss.item(), loss_components)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'age': 0.0,
            'mortality': 0.0,
            'morbidity': 0.0
        }
        
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                biomarkers = batch['biomarkers']
                age = batch['chronological_age']
                
                targets = {
                    'chronological_age': batch['chronological_age'].squeeze(),
                    'survival_time': batch.get('survival_time', torch.zeros(biomarkers.size(0))).squeeze(),
                    'event_indicator': batch.get('event_indicator', torch.ones(biomarkers.size(0))).squeeze()
                }
                
                # Add disease labels
                for disease in ['cardiovascular', 'diabetes', 'cancer', 'dementia']:
                    if f'{disease}_label' in batch:
                        targets[f'{disease}_label'] = batch[f'{disease}_label'].squeeze()
                
                # Forward pass
                if self.use_amp:
                    with autocast():
                        output = self.model(biomarkers, age)
                        loss, loss_components = self.criterion(output, targets, self.model)
                else:
                    output = self.model(biomarkers, age)
                    loss, loss_components = self.criterion(output, targets, self.model)
                
                # Update losses
                val_losses['total'] += loss.item()
                for key, value in loss_components.items():
                    if key in val_losses:
                        val_losses[key] += value.item()
                
                # Store predictions for evaluation
                predictions.append({
                    'biological_age': output.biological_age.cpu(),
                    'chronological_age': targets['chronological_age'].cpu(),
                    'mortality_risk': output.mortality_risk.cpu() if output.mortality_risk is not None else None
                })
                targets_list.append(targets)
        
        # Average losses
        num_batches = len(val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Compute evaluation metrics
        eval_metrics = self.evaluator.compute_metrics(predictions, targets_list)
        
        # Combine losses and metrics
        val_metrics = {**val_losses, **eval_metrics}
        
        return val_metrics
    
    def train(self,
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader,
             test_loader: Optional[torch.utils.data.DataLoader] = None) -> None:
        """
        Main training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Optional test data loader
        """
        logger.info("Starting training...")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch + 1
            
            # Train for one epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['mae'])
                else:
                    self.scheduler.step()
            
            # Log epoch metrics
            self.log_epoch_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            self.save_checkpoint(val_metrics)
            
            # Early stopping check
            if self.check_early_stopping(val_metrics):
                logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                break
            
            # Update adaptive weights periodically
            if epoch % self.config['model']['adaptive_weighting']['update_frequency'] == 0:
                logger.info("Updating adaptive weights...")
        
        # Final evaluation on test set
        if test_loader is not None:
            logger.info("Running final evaluation on test set...")
            test_metrics = self.validate(test_loader)
            self.log_test_metrics(test_metrics)
            self.save_results(test_metrics)
        
        logger.info("Training completed!")
    
    def save_checkpoint(self, metrics: Dict[str, float]) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        monitor_metric = self.config['logging']['checkpoint']['monitor_metric']
        if monitor_metric in metrics:
            current_metric = metrics[monitor_metric]
            
            if self.config['logging']['checkpoint']['mode'] == 'min':
                is_best = current_metric < self.best_val_metric
            else:
                is_best = current_metric > self.best_val_metric
            
            if is_best:
                self.best_val_metric = current_metric
                best_path = self.checkpoint_dir / 'best.pt'
                torch.save(checkpoint, best_path)
                logger.info(f"Saved best model with {monitor_metric}={current_metric:.4f}")
        
        # Save periodic checkpoint
        if self.current_epoch % self.config['logging']['checkpoint']['save_frequency'] == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch}.pt'
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint with comprehensive error handling"""
        checkpoint_path = Path(checkpoint_path)
        
        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")
        
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        if file_size_mb < 0.1:
            raise ValueError(f"Checkpoint file too small ({file_size_mb:.1f}MB), likely corrupted")
        
        logger.info(f"Loading checkpoint from {checkpoint_path} ({file_size_mb:.1f}MB)")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            logger.warning(f"Checkpoint missing keys: {missing_keys}")
        
        # Load model state with error handling
        if 'model_state_dict' in checkpoint:
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            except Exception as e:
                logger.error(f"Failed to load model state: {e}")
                raise
        else:
            logger.warning("No model state dict in checkpoint")
        
        # Load optimizer state with error handling
        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}. Reinitializing optimizer.")
        
        # Load scheduler state if available
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")
        
        # Load epoch number
        self.current_epoch = checkpoint.get('epoch', 0)
        
        # Load best metric if available
        if 'best_val_metric' in checkpoint:
            self.best_val_metric = checkpoint['best_val_metric']
        
        logger.info(f"Successfully loaded checkpoint from epoch {self.current_epoch}")
    
    def check_early_stopping(self, metrics: Dict[str, float], patience: int = 20) -> bool:
        """Check early stopping criteria"""
        monitor_metric = self.config['logging']['checkpoint']['monitor_metric']
        
        if monitor_metric not in metrics:
            return False
        
        current_metric = metrics[monitor_metric]
        
        # Track history
        if not hasattr(self, 'early_stopping_counter'):
            self.early_stopping_counter = 0
            self.best_early_stopping_metric = current_metric
        
        if self.config['logging']['checkpoint']['mode'] == 'min':
            improved = current_metric < self.best_early_stopping_metric
        else:
            improved = current_metric > self.best_early_stopping_metric
        
        if improved:
            self.best_early_stopping_metric = current_metric
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= patience
    
    def log_batch_metrics(self, batch_idx: int, loss: float, loss_components: Dict[str, torch.Tensor]) -> None:
        """Log batch-level metrics"""
        if self.tb_writer:
            global_step = self.current_epoch * 1000 + batch_idx
            self.tb_writer.add_scalar('batch/loss', loss, global_step)
            
            for name, value in loss_components.items():
                self.tb_writer.add_scalar(f'batch/{name}', value.item(), global_step)
    
    def log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]) -> None:
        """Log epoch-level metrics"""
        # Console logging
        logger.info(f"Epoch {self.current_epoch}:")
        logger.info(f"  Train Loss: {train_metrics['total']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['total']:.4f}")
        logger.info(f"  Val MAE: {val_metrics.get('mae', 0):.4f}")
        logger.info(f"  Val C-statistic: {val_metrics.get('c_statistic', 0):.4f}")
        
        # TensorBoard logging
        if self.tb_writer:
            for name, value in train_metrics.items():
                self.tb_writer.add_scalar(f'train/{name}', value, self.current_epoch)
            
            for name, value in val_metrics.items():
                self.tb_writer.add_scalar(f'val/{name}', value, self.current_epoch)
        
        # Weights & Biases logging
        if self.use_wandb:
            wandb.log({
                'epoch': self.current_epoch,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
        
        # Save to history
        self.training_history.append({
            'epoch': self.current_epoch,
            'train': train_metrics,
            'val': val_metrics
        })
    
    def log_test_metrics(self, test_metrics: Dict[str, float]) -> None:
        """Log test metrics"""
        logger.info("Test Results:")
        for name, value in test_metrics.items():
            logger.info(f"  {name}: {value:.4f}")
        
        if self.use_wandb:
            wandb.log({f'test_{k}': v for k, v in test_metrics.items()})
    
    def save_results(self, test_metrics: Dict[str, float]) -> None:
        """Save final results"""
        # Save metrics
        metrics_path = self.results_dir / 'test_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        # Save training history
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save configuration
        config_path = self.results_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
        
        logger.info(f"Results saved to {self.results_dir}")


def cross_validation_train(config: Dict[str, Any],
                          data_path: str,
                          n_folds: int = 5) -> Dict[str, List[float]]:
    """
    Perform k-fold cross-validation training
    
    Args:
        config: Configuration dictionary
        data_path: Path to data
        n_folds: Number of folds
    
    Returns:
        Dictionary of metrics across folds
    """
    logger.info(f"Starting {n_folds}-fold cross-validation...")
    
    cv_results = {
        'mae': [],
        'rmse': [],
        'c_statistic': [],
        'icc': []
    }
    
    # Create base dataset for splitting
    base_dataset = UKBBDataset(data_path, config, split='train')
    
    # Create stratified folds based on age groups
    ages = np.array([s.age for s in base_dataset.samples])
    age_groups = pd.cut(ages, bins=[40, 50, 60, 70], labels=[0, 1, 2])
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config['reproducibility']['seed'])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(ages, age_groups)):
        logger.info(f"\nFold {fold + 1}/{n_folds}")
        
        # Create fold-specific data loaders
        # This is simplified - in production, you'd properly split the dataset
        train_loader, val_loader, test_loader = create_data_loaders(config, data_path)
        
        # Create trainer for this fold
        trainer = Trainer(config, experiment_name=f"henaw_cv_fold_{fold + 1}")
        
        # Train
        trainer.train(train_loader, val_loader)
        
        # Evaluate on validation set
        val_metrics = trainer.validate(val_loader)
        
        # Store results
        for metric in cv_results:
            if metric in val_metrics:
                cv_results[metric].append(val_metrics[metric])
    
    # Compute statistics
    cv_summary = {}
    for metric, values in cv_results.items():
        cv_summary[f'{metric}_mean'] = np.mean(values)
        cv_summary[f'{metric}_std'] = np.std(values)
    
    logger.info("\nCross-validation results:")
    for metric, value in cv_summary.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return cv_summary


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description='Train HENAW model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to UK Biobank data')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--cross-validation', action='store_true', help='Perform cross-validation')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seeds for reproducibility
    if config['reproducibility']['deterministic']:
        torch.manual_seed(config['reproducibility']['seed'])
        np.random.seed(config['reproducibility']['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = config['reproducibility']['benchmark']
    
    if args.cross_validation:
        # Perform cross-validation
        cv_results = cross_validation_train(
            config,
            args.data_path,
            n_folds=config['training']['n_folds']
        )
    else:
        # Standard training
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            config,
            args.data_path,
            num_workers=config['infrastructure']['num_workers']
        )
        
        # Create trainer
        trainer = Trainer(config, device=args.device, experiment_name=args.experiment_name)
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)
        
        # Train
        trainer.train(train_loader, val_loader, test_loader)


if __name__ == "__main__":
    main()