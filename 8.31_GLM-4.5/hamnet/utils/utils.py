"""
Utility functions and helper modules for HAMNet

This module provides various utilities for:
- Data preprocessing and validation
- Model initialization and checkpointing
- Training and evaluation helpers
- Visualization and interpretability
- Performance monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import json
import os
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Configuration for training HAMNet"""
    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 10
    min_delta: float = 1e-4
    
    # Data parameters
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Optimization
    optimizer: str = "adam"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    
    # Regularization
    dropout: float = 0.1
    l1_lambda: float = 0.0
    l2_lambda: float = 1e-5
    
    # Mixed precision
    mixed_precision: bool = True
    gradient_clipping: float = 1.0
    
    # Checkpointing
    save_every: int = 10
    save_best: bool = True
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    log_every: int = 10
    eval_every: int = 5
    tensorboard_dir: str = "runs"
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True


class EarlyStopping:
    """Early stopping implementation"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class CheckpointManager:
    """Checkpoint management for HAMNet"""
    
    def __init__(self, checkpoint_dir: str, model_name: str = "hamnet"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_name = model_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       epoch: int, loss: float, config: Dict[str, Any],
                       is_best: bool = False) -> str:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'config': config
        }
        
        filename = f"{self.model_name}_epoch_{epoch}.pt"
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            
        return str(checkpoint_path)
        
    def load_checkpoint(self, checkpoint_path: str, model: nn.Module,
                       optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        return checkpoint
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob(f"{self.model_name}_epoch_*.pt"))
        if not checkpoints:
            return None
            
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        return str(latest)
        
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the best checkpoint"""
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        return str(best_path) if best_path.exists() else None


class MetricsTracker:
    """Track and compute training metrics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset metrics"""
        self.predictions = []
        self.targets = []
        self.losses = []
        self.uncertainties = []
        
    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               loss: float, uncertainties: Optional[torch.Tensor] = None):
        """Update metrics with new batch"""
        self.predictions.extend(predictions.cpu().numpy().flatten())
        self.targets.extend(targets.cpu().numpy().flatten())
        self.losses.append(loss)
        
        if uncertainties is not None:
            self.uncertainties.extend(uncertainties.cpu().numpy().flatten())
            
    def compute_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        if not self.predictions:
            return {}
            
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {
            'mse': mean_squared_error(targets, predictions),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mean_loss': np.mean(self.losses)
        }
        
        # Add uncertainty metrics if available
        if self.uncertainties:
            uncertainties = np.array(self.uncertainties)
            metrics['mean_uncertainty'] = np.mean(uncertainties)
            metrics['uncertainty_calibration'] = np.corrcoef(
                np.abs(predictions - targets), uncertainties
            )[0, 1]
            
        return metrics


class HAMNetDataset(Dataset):
    """Dataset class for HAMNet"""
    
    def __init__(self, data: Dict[str, np.ndarray], targets: np.ndarray,
                 masks: Optional[Dict[str, np.ndarray]] = None,
                 transform=None):
        self.data = data
        self.targets = targets
        self.masks = masks or {}
        self.transform = transform
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Validate data consistency"""
        sample_size = len(self.targets)
        
        for modality, data in self.data.items():
            if len(data) != sample_size:
                raise ValueError(f"Modality {modality} has {len(data)} samples, "
                               f"but targets have {sample_size} samples")
                               
        for modality, mask in self.masks.items():
            if len(mask) != sample_size:
                raise ValueError(f"Mask for {modality} has {len(mask)} samples, "
                               f"but targets have {sample_size} samples")
                               
    def __len__(self) -> int:
        return len(self.targets)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        sample = {
            'targets': self.targets[idx],
            'index': idx
        }
        
        # Add modality data
        for modality, data in self.data.items():
            sample[modality] = torch.FloatTensor(data[idx])
            
        # Add masks
        for modality, mask in self.masks.items():
            sample[f'{modality}_mask'] = torch.FloatTensor(mask[idx])
            
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
            
        return sample


def create_data_loaders(dataset: HAMNetDataset, batch_size: int = 32,
                       train_split: float = 0.8, val_split: float = 0.1,
                       num_workers: int = 4, pin_memory: bool = True) -> Dict[str, DataLoader]:
    """Create train/validation/test data loaders"""
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    data_loaders = {
        'train': DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        'val': DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        'test': DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
    }
    
    return data_loaders


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Create optimizer for HAMNet"""
    
    if config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")
        
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler"""
    
    if config.scheduler.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-6
        )
    elif config.scheduler.lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.epochs // 3, gamma=0.1
        )
    elif config.scheduler.lower() == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
    else:
        raise ValueError(f"Unknown scheduler: {config.scheduler}")
        
    return scheduler


def compute_loss(predictions: torch.Tensor, targets: torch.Tensor,
                uncertainties: Optional[torch.Tensor] = None,
                loss_type: str = "mse") -> torch.Tensor:
    """Compute loss with optional uncertainty weighting"""
    
    if loss_type == "mse":
        base_loss = F.mse_loss(predictions, targets, reduction='none')
    elif loss_type == "mae":
        base_loss = F.l1_loss(predictions, targets, reduction='none')
    elif loss_type == "huber":
        base_loss = F.huber_loss(predictions, targets, reduction='none')
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
        
    # Apply uncertainty weighting if available
    if uncertainties is not None:
        # Negative log likelihood with uncertainty
        loss = 0.5 * (base_loss / (uncertainties ** 2 + 1e-8) + torch.log(uncertainties ** 2 + 1e-8))
    else:
        loss = base_loss
        
    return loss.mean()


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for HAMNet data"""
    
    collated = {}
    
    # Get all keys from the first batch
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'targets':
            collated[key] = torch.stack([item[key] for item in batch])
        elif key == 'index':
            collated[key] = torch.tensor([item[key] for item in batch])
        elif key.endswith('_mask'):
            collated[key] = torch.stack([item[key] for item in batch])
        else:  # Modality data
            # Handle variable length sequences
            if len(batch[0][key].shape) > 1:
                # Pad sequences to same length
                max_len = max(len(item[key]) for item in batch)
                padded = []
                for item in batch:
                    if len(item[key]) < max_len:
                        padding = torch.zeros(max_len - len(item[key]), *item[key].shape[1:])
                        padded_item = torch.cat([item[key], padding], dim=0)
                    else:
                        padded_item = item[key]
                    padded.append(padded_item)
                collated[key] = torch.stack(padded)
            else:
                collated[key] = torch.stack([item[key] for item in batch])
                
    return collated


def setup_logging(log_dir: str = "logs", experiment_name: str = None) -> logging.Logger:
    """Setup logging for HAMNet"""
    
    if experiment_name is None:
        experiment_name = f"hamnet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    log_file = log_dir / f"{experiment_name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_config(config: Union[TrainingConfig, HAMNetConfig], config_path: str):
    """Save configuration to file"""
    
    config_dict = asdict(config)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file"""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    return config


def visualize_attention_weights(attention_weights: torch.Tensor,
                              modality_names: List[str],
                              save_path: str = None):
    """Visualize attention weights"""
    
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy
    weights = attention_weights.cpu().numpy()
    
    # Average over batch and heads
    if len(weights.shape) > 2:
        weights = weights.mean(axis=(0, 1))
    
    # Create heatmap
    sns.heatmap(weights, annot=True, cmap='viridis', 
                xticklabels=modality_names,
                yticklabels=modality_names)
    
    plt.title('Cross-Modal Attention Weights')
    plt.xlabel('Key Modality')
    plt.ylabel('Query Modality')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # Metrics plots
    if 'train_mae' in history:
        axes[0, 1].plot(history['train_mae'], label='Train MAE')
        axes[0, 1].plot(history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('MAE')
        axes[0, 1].legend()
    
    if 'train_r2' in history:
        axes[1, 0].plot(history['train_r2'], label='Train R²')
        axes[1, 0].plot(history['val_r2'], label='Val R²')
        axes[1, 0].set_title('R²')
        axes[1, 0].legend()
    
    # Learning rate plot
    if 'lr' in history:
        axes[1, 1].plot(history['lr'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def get_model_size(model: nn.Module) -> Dict[str, int]:
    """Get model size information"""
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        'total_parameters': param_count,
        'trainable_parameters': trainable_count,
        'model_size_mb': model_size_mb
    }


def profile_model(model: nn.Module, input_shape: Dict[str, Tuple[int, ...]],
                  device: str = 'cuda') -> Dict[str, float]:
    """Profile model performance"""
    
    model.eval()
    model.to(device)
    
    # Create dummy inputs
    dummy_inputs = {}
    for modality, shape in input_shape.items():
        dummy_inputs[modality] = torch.randn(shape).to(device)
    
    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_inputs)
    
    # Profile
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if torch.cuda.is_available():
        start_time.record()
    
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_inputs)
    
    if torch.cuda.is_available():
        end_time.record()
        torch.cuda.synchronize()
        avg_time = start_time.elapsed_time(end_time) / 100
    else:
        import time
        start = time.time()
        for _ in range(100):
            _ = model(dummy_inputs)
        avg_time = (time.time() - start) / 100
    
    return {
        'avg_inference_time_ms': avg_time,
        'fps': 1000 / avg_time if avg_time > 0 else 0
    }


class DataValidator:
    """Validate input data for HAMNet"""
    
    @staticmethod
    def validate_input_data(data: Dict[str, np.ndarray], 
                          targets: np.ndarray) -> List[str]:
        """Validate input data consistency"""
        errors = []
        
        # Check if targets are valid
        if len(targets) == 0:
            errors.append("Targets array is empty")
        
        # Check if data dictionary is not empty
        if not data:
            errors.append("Data dictionary is empty")
        
        # Check sample consistency
        sample_size = len(targets)
        for modality, modality_data in data.items():
            if len(modality_data) != sample_size:
                errors.append(f"Modality {modality} has {len(modality_data)} samples, "
                            f"expected {sample_size}")
        
        # Check for NaN values
        for modality, modality_data in data.items():
            if np.isnan(modality_data).any():
                errors.append(f"Modality {modality} contains NaN values")
        
        if np.isnan(targets).any():
            errors.append("Targets contain NaN values")
        
        return errors
    
    @staticmethod
    def validate_config(config: HAMNetConfig) -> List[str]:
        """Validate model configuration"""
        errors = []
        
        # Check model tier
        if config.model_tier not in ["base", "standard", "comprehensive"]:
            errors.append(f"Invalid model tier: {config.model_tier}")
        
        # Check dimensions
        if config.embedding_dim <= 0:
            errors.append("Embedding dimension must be positive")
        
        if config.hidden_dim <= 0:
            errors.append("Hidden dimension must be positive")
        
        # Check dropout
        if not 0 <= config.dropout <= 1:
            errors.append("Dropout must be between 0 and 1")
        
        return errors


def create_experiment_dir(base_dir: str = "experiments") -> str:
    """Create experiment directory with timestamp"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(base_dir) / f"experiment_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "configs").mkdir(exist_ok=True)
    
    return str(experiment_dir)


def save_results(results: Dict[str, Any], save_path: str):
    """Save experiment results"""
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64)):
            results_serializable[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            results_serializable[key] = int(value)
        else:
            results_serializable[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)