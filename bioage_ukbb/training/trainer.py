"""
Main training loop for biological age algorithms.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path
import warnings
from tqdm import tqdm
import wandb


class BioAgeTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for biological age models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        loss_fn: Optional[nn.Module] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: The biological age model
            config: Configuration object
            loss_fn: Custom loss function
        """
        super().__init__()
        
        self.model = model
        self.config = config
        self.loss_fn = loss_fn or nn.MSELoss()
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.test_results = {}
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Select optimizer
        if self.config.training_config.optimizer == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                weight_decay=self.config.training_config.weight_decay
            )
        elif self.config.training_config.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                weight_decay=self.config.training_config.weight_decay
            )
        elif self.config.training_config.optimizer == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training_config.weight_decay
            )
        else:
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                weight_decay=self.config.training_config.weight_decay
            )
        
        # Select scheduler
        if self.config.training_config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training_config.max_epochs
            )
        elif self.config.training_config.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif self.config.training_config.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        elif self.config.training_config.scheduler == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config.training_config.learning_rate * 10,
                total_steps=self.config.training_config.max_epochs * 1000  # Approximate
            )
        else:
            scheduler = None
        
        if scheduler:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss' if self.config.training_config.scheduler == "plateau" else None
                }
            }
        else:
            return optimizer
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        return self.model(inputs)
    
    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        inputs, targets = batch
        
        # Forward pass
        outputs = self.model(inputs)
        predictions = outputs['prediction'].squeeze()
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets)
        
        # Add additional losses if present
        if 'contrastive_loss' in outputs:
            loss += outputs['contrastive_loss'] * 0.1
        
        if 'kl_loss' in outputs:
            loss += outputs['kl_loss'] * self.config.metage_config.kl_weight
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mae', torch.abs(predictions - targets).mean(), on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        inputs, targets = batch
        
        # Forward pass
        outputs = self.model(inputs)
        predictions = outputs['prediction'].squeeze()
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets)
        
        # Calculate metrics
        mae = torch.abs(predictions - targets).mean()
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mae', mae, on_epoch=True, prog_bar=True)
        self.log('val_rmse', rmse, on_epoch=True)
        
        return {
            'val_loss': loss,
            'predictions': predictions,
            'targets': targets
        }
    
    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        inputs, targets = batch
        
        # Forward pass with uncertainty if available
        outputs = self.model(inputs, return_uncertainty=True)
        predictions = outputs['prediction'].squeeze()
        
        # Calculate metrics
        mae = torch.abs(predictions - targets).mean()
        rmse = torch.sqrt(torch.mean((predictions - targets) ** 2))
        
        # Calculate correlation
        if len(predictions) > 1:
            pearson = torch.corrcoef(
                torch.stack([predictions, targets])
            )[0, 1]
        else:
            pearson = torch.tensor(0.0)
        
        # Log metrics
        self.log('test_mae', mae, on_epoch=True)
        self.log('test_rmse', rmse, on_epoch=True)
        self.log('test_pearson', pearson, on_epoch=True)
        
        return {
            'predictions': predictions,
            'targets': targets,
            'uncertainty': outputs.get('uncertainty')
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Store epoch losses
        if self.trainer.callback_metrics:
            train_loss = self.trainer.callback_metrics.get('train_loss_epoch')
            if train_loss:
                self.train_losses.append(train_loss.item())
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        # Store validation losses
        if self.trainer.callback_metrics:
            val_loss = self.trainer.callback_metrics.get('val_loss')
            if val_loss:
                self.val_losses.append(val_loss.item())


class StandardTrainer:
    """Standard PyTorch trainer for biological age models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: str = 'cuda'
    ):
        """
        Initialize trainer.
        
        Args:
            model: The biological age model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration object
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.training_config.use_amp else None
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        if self.config.training_config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                weight_decay=self.config.training_config.weight_decay
            )
        elif self.config.training_config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                weight_decay=self.config.training_config.weight_decay
            )
        else:
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.training_config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training_config.weight_decay
            )
    
    def _create_scheduler(self) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.config.training_config.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training_config.max_epochs
            )
        elif self.config.training_config.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5
            )
        return None
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # Move to device
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(inputs)
                    predictions = outputs['prediction'].squeeze()
                    loss = self.criterion(predictions, targets)
                    
                    # Add auxiliary losses
                    if 'contrastive_loss' in outputs:
                        loss += outputs['contrastive_loss'] * 0.1
                    if 'kl_loss' in outputs:
                        loss += outputs['kl_loss'] * 0.001
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if hasattr(self.config.training_config, 'gradient_clip_val'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training_config.gradient_clip_val
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward-backward pass
                outputs = self.model(inputs)
                predictions = outputs['prediction'].squeeze()
                loss = self.criterion(predictions, targets)
                
                # Add auxiliary losses
                if 'contrastive_loss' in outputs:
                    loss += outputs['contrastive_loss'] * 0.1
                if 'kl_loss' in outputs:
                    loss += outputs['kl_loss'] * 0.001
                
                loss.backward()
                
                # Gradient clipping
                if hasattr(self.config.training_config, 'gradient_clip_val'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training_config.gradient_clip_val
                    )
                
                self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            n_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> Tuple[float, float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                if isinstance(inputs, dict):
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                else:
                    inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                predictions = outputs['prediction'].squeeze()
                
                # Calculate loss
                loss = self.criterion(predictions, targets)
                
                # Store results
                total_loss += loss.item()
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Aggregate metrics
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)
        
        mae = torch.abs(all_predictions - all_targets).mean().item()
        rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()
        
        self.val_losses.append(avg_loss)
        
        return avg_loss, mae, rmse
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        print(f"Starting training for {self.config.training_config.max_epochs} epochs")
        
        for epoch in range(self.config.training_config.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.training_config.max_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_mae, val_rmse = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.training_config.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': len(self.train_losses),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        save_path = Path(self.config.training_config.checkpoint_dir) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {filepath}")


def create_trainer(
    model: nn.Module,
    config: Any,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    use_lightning: bool = True
) -> Union[pl.Trainer, StandardTrainer]:
    """
    Create trainer instance.
    
    Args:
        model: Model to train
        config: Configuration object
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Optional test data loader
        use_lightning: Whether to use PyTorch Lightning
    
    Returns:
        Trainer instance
    """
    if use_lightning:
        # Create Lightning module
        lightning_model = BioAgeTrainer(model, config)
        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=config.training_config.checkpoint_dir,
                filename='best-{epoch:02d}-{val_loss:.4f}',
                monitor='val_loss',
                mode='min',
                save_top_k=config.training_config.save_top_k
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=config.training_config.early_stopping_patience,
                min_delta=config.training_config.early_stopping_min_delta
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # Loggers
        loggers = []
        if config.training_config.use_wandb:
            loggers.append(WandbLogger(
                project=config.training_config.wandb_project,
                name=f"{model.__class__.__name__}_training"
            ))
        if config.training_config.use_tensorboard:
            loggers.append(TensorBoardLogger(
                config.training_config.tensorboard_dir,
                name=model.__class__.__name__
            ))
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=config.training_config.max_epochs,
            callbacks=callbacks,
            logger=loggers,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=16 if config.training_config.use_amp else 32,
            gradient_clip_val=getattr(config.training_config, 'gradient_clip_val', None),
            accumulate_grad_batches=config.training_config.gradient_accumulation_steps,
            deterministic=config.training_config.deterministic,
            benchmark=config.training_config.benchmark,
            num_sanity_val_steps=config.training_config.num_sanity_val_steps,
            limit_train_batches=config.training_config.limit_train_batches,
            limit_val_batches=config.training_config.limit_val_batches,
            log_every_n_steps=config.training_config.log_every_n_steps
        )
        
        return trainer
    else:
        # Create standard trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return StandardTrainer(model, train_loader, val_loader, config, device)