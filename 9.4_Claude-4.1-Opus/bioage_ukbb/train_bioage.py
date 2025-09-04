#!/usr/bin/env python
"""
Main training script for biological age estimation models.

Supports HENAW, MODAL, and METAGE algorithms with comprehensive
error handling, logging, and checkpoint management.
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os
import logging
import traceback
from datetime import datetime
import json
import wandb
import numpy as np
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from bioage_ukbb.models.henaw import HENAW
from bioage_ukbb.models.modal import MODAL
from bioage_ukbb.models.metage import METAGE
from bioage_ukbb.data.ukbb_loader import create_dataloaders
from bioage_ukbb.training.trainer import StandardTrainer, create_trainer
from bioage_ukbb.evaluation.metrics import BioAgeEvaluator
from bioage_ukbb.utils.config import ConfigManager


# Set up logging
def setup_logging(log_dir: Path, model_name: str) -> logging.Logger:
    """Set up comprehensive logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('bioage_training')
    logger.setLevel(logging.DEBUG)
    
    # File handler for all logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{model_name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for important logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Error file handler
    error_file = log_dir / f'{model_name}_{timestamp}_errors.log'
    error_handler = logging.FileHandler(error_file)
    error_handler.setLevel(logging.ERROR)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)
    
    return logger


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_gpu_availability(logger: logging.Logger) -> str:
    """Check GPU availability and return device."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.2f} GB)")
        device = 'cuda'
    else:
        logger.warning("No GPU available, using CPU")
        device = 'cpu'
    
    return device


def create_model(
    model_type: str,
    config: ConfigManager,
    logger: logging.Logger
) -> nn.Module:
    """
    Create model instance based on type.
    
    Args:
        model_type: Type of model to create
        config: Configuration manager
        logger: Logger instance
    
    Returns:
        Model instance
    """
    model_type = model_type.lower()
    
    try:
        if model_type == 'henaw':
            model_config = config.henaw_config
            model = HENAW(model_config)
            logger.info(f"Created HENAW model with {sum(p.numel() for p in model.parameters())} parameters")
            
        elif model_type == 'modal':
            model_config = config.modal_config
            model = MODAL(model_config)
            logger.info(f"Created MODAL model with {sum(p.numel() for p in model.parameters())} parameters")
            
        elif model_type == 'metage':
            model_config = config.metage_config
            model = METAGE(model_config)
            logger.info(f"Created METAGE model with {sum(p.numel() for p in model.parameters())} parameters")
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model
    
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        logger.error(traceback.format_exc())
        raise


def validate_config(config: ConfigManager, logger: logging.Logger) -> bool:
    """
    Validate configuration settings.
    
    Args:
        config: Configuration manager
        logger: Logger instance
    
    Returns:
        True if valid, False otherwise
    """
    try:
        # Check data paths
        ukbb_path = Path(config.data_config.ukbb_data_path)
        if not ukbb_path.exists():
            logger.warning(f"UK Biobank data path does not exist: {ukbb_path}")
            logger.warning("Creating mock data for demonstration")
            # Continue with mock data for demonstration
        
        # Check output directories
        output_dir = Path(config.data_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Check cache directory
        cache_dir = Path(config.data_config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {cache_dir}")
        
        # Validate training parameters
        if config.training_config.max_epochs < 1:
            logger.error("Invalid max_epochs: must be >= 1")
            return False
        
        if config.data_config.batch_size < 1:
            logger.error("Invalid batch_size: must be >= 1")
            return False
        
        # Check for incompatible settings
        if config.training_config.use_ddp and not torch.cuda.is_available():
            logger.warning("DDP requested but no GPU available, disabling DDP")
            config.training_config.use_ddp = False
        
        return True
    
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        logger.error(traceback.format_exc())
        return False


def train_model(
    model: nn.Module,
    config: ConfigManager,
    model_type: str,
    device: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Train the model with error handling.
    
    Args:
        model: Model to train
        config: Configuration manager
        model_type: Type of model
        device: Device to train on
        logger: Logger instance
    
    Returns:
        Training results dictionary
    """
    try:
        # Move model to device
        model = model.to(device)
        
        # Create data loaders
        logger.info("Creating data loaders...")
        dataloaders = create_dataloaders(
            config.data_config,
            config.get_model_config(model_type),
            model_type
        )
        
        # Initialize wandb if configured
        if config.training_config.use_wandb:
            logger.info("Initializing Weights & Biases...")
            wandb.init(
                project=config.training_config.wandb_project,
                name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    'model_type': model_type,
                    'batch_size': config.data_config.batch_size,
                    'learning_rate': config.training_config.learning_rate,
                    'max_epochs': config.training_config.max_epochs
                }
            )
        
        # Create trainer
        logger.info("Initializing trainer...")
        if config.training_config.use_lightning:
            trainer = create_trainer(
                model,
                config,
                dataloaders['train'],
                dataloaders['val'],
                dataloaders['test'],
                use_lightning=True
            )
            
            # Train with PyTorch Lightning
            logger.info("Starting training with PyTorch Lightning...")
            trainer.fit(model, dataloaders['train'], dataloaders['val'])
            
            # Test
            logger.info("Running test evaluation...")
            test_results = trainer.test(model, dataloaders['test'])
            
            results = {
                'final_train_loss': trainer.callback_metrics.get('train_loss_epoch', 0),
                'final_val_loss': trainer.callback_metrics.get('val_loss', 0),
                'test_results': test_results
            }
            
        else:
            # Use standard trainer
            trainer = StandardTrainer(
                model,
                dataloaders['train'],
                dataloaders['val'],
                config,
                device
            )
            
            # Train
            logger.info("Starting training...")
            train_results = trainer.train()
            
            # Evaluate on test set
            logger.info("Running test evaluation...")
            test_loss, test_mae, test_rmse = trainer.validate()
            
            results = {
                'train_losses': train_results['train_losses'],
                'val_losses': train_results['val_losses'],
                'best_val_loss': train_results['best_val_loss'],
                'test_loss': test_loss,
                'test_mae': test_mae,
                'test_rmse': test_rmse
            }
        
        logger.info("Training completed successfully!")
        
        # Save final model
        checkpoint_dir = Path(config.training_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path = checkpoint_dir / f'{model_type}_final.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'results': results
        }, final_path)
        logger.info(f"Final model saved to {final_path}")
        
        # Close wandb
        if config.training_config.use_wandb:
            wandb.finish()
        
        return results
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        if config.training_config.use_wandb:
            wandb.finish()
        raise
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        if config.training_config.use_wandb:
            wandb.finish()
        raise


def evaluate_model(
    model: nn.Module,
    config: ConfigManager,
    model_type: str,
    device: str,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Trained model
        config: Configuration manager
        model_type: Type of model
        device: Device for evaluation
        logger: Logger instance
    
    Returns:
        Evaluation results
    """
    try:
        logger.info("Starting comprehensive evaluation...")
        
        # Create evaluator
        evaluator = BioAgeEvaluator(model, config, device)
        
        # Create test dataloader
        dataloaders = create_dataloaders(
            config.data_config,
            config.get_model_config(model_type),
            model_type
        )
        
        # Run evaluation
        results = evaluator.evaluate(dataloaders['test'])
        
        # Log results
        logger.info("Evaluation Results:")
        logger.info(f"  MAE: {results['mae']:.3f} years")
        logger.info(f"  RMSE: {results['rmse']:.3f} years")
        logger.info(f"  Pearson correlation: {results['pearson']:.3f}")
        logger.info(f"  Spearman correlation: {results['spearman']:.3f}")
        logger.info(f"  R²: {results['r2']:.3f}")
        
        if 'c_index' in results:
            logger.info(f"  C-index (mortality): {results['c_index']:.3f}")
        
        return results
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main training function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train biological age estimation models')
    parser.add_argument('--model', type=str, default='henaw',
                       choices=['henaw', 'modal', 'metage'],
                       help='Model type to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate existing model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for evaluation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Set up logging
    log_dir = Path('logs')
    logger = setup_logging(log_dir, args.model)
    
    logger.info("="*60)
    logger.info(f"Biological Age Training - Model: {args.model.upper()}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    try:
        # Set seed for reproducibility
        set_seed(args.seed)
        logger.info(f"Random seed: {args.seed}")
        
        # Load configuration
        logger.info("Loading configuration...")
        config = ConfigManager(args.config)
        
        # Override parameters if specified
        if args.epochs is not None:
            config.training_config.max_epochs = args.epochs
        if args.batch_size is not None:
            config.data_config.batch_size = args.batch_size
        if args.lr is not None:
            model_config = config.get_model_config(args.model)
            model_config.learning_rate = args.lr
        
        # Enable debug mode if requested
        if args.debug:
            config.training_config.limit_train_batches = 0.01
            config.training_config.limit_val_batches = 0.01
            config.training_config.max_epochs = 2
            logger.info("Debug mode enabled - using limited data")
        
        # Validate configuration
        if not validate_config(config, logger):
            logger.error("Configuration validation failed")
            sys.exit(1)
        
        # Check GPU availability
        device = check_gpu_availability(logger)
        
        # Create model
        model = create_model(args.model, config, logger)
        
        if args.evaluate_only:
            # Load checkpoint
            if args.checkpoint:
                logger.info(f"Loading checkpoint from {args.checkpoint}")
                checkpoint = torch.load(args.checkpoint, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.error("No checkpoint specified for evaluation")
                sys.exit(1)
            
            # Evaluate model
            eval_results = evaluate_model(model, config, args.model, device, logger)
            
            # Save evaluation results
            results_file = log_dir / f'{args.model}_evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(eval_results, f, indent=2)
            logger.info(f"Evaluation results saved to {results_file}")
            
        else:
            # Train model
            train_results = train_model(model, config, args.model, device, logger)
            
            # Evaluate after training
            eval_results = evaluate_model(model, config, args.model, device, logger)
            
            # Combine results
            all_results = {
                'training': train_results,
                'evaluation': eval_results,
                'config': {
                    'model_type': args.model,
                    'epochs': config.training_config.max_epochs,
                    'batch_size': config.data_config.batch_size,
                    'learning_rate': config.get_model_config(args.model).learning_rate
                }
            }
            
            # Save all results
            results_file = log_dir / f'{args.model}_all_results.json'
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            logger.info(f"All results saved to {results_file}")
        
        logger.info("="*60)
        logger.info("Process completed successfully!")
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()