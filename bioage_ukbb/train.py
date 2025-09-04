#!/usr/bin/env python
"""
Main training script for UK Biobank biological age algorithms.

Usage:
    python train.py --model henaw --config configs/henaw_config.yaml
    python train.py --model modal --config configs/modal_config.yaml
    python train.py --model metage --config configs/metage_config.yaml
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import random
from datetime import datetime
import yaml
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.henaw import HENAW
from models.modal import MODAL
from models.metage import METAGE
from data.loaders import (
    HENAWDataset, MODALDataset, METAGEDataset,
    create_data_loaders
)
from data.preprocessing import create_preprocessor
from training.trainer import create_trainer, StandardTrainer
from evaluation.metrics import BioAgeMetrics
from utils.config import ConfigManager
import pytorch_lightning as pl


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(model_name: str, config: ConfigManager) -> torch.nn.Module:
    """Create model based on name."""
    model_map = {
        'henaw': (HENAW, config.henaw_config),
        'modal': (MODAL, config.modal_config),
        'metage': (METAGE, config.metage_config)
    }
    
    if model_name.lower() not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class, model_config = model_map[model_name.lower()]
    return model_class(model_config)


def create_dataset(model_name: str, config: ConfigManager):
    """Create dataset based on model type."""
    if model_name.lower() == 'henaw':
        dataset_class = HENAWDataset
        dataset_kwargs = {
            'temporal_windows': [
                config.henaw_config.rapid_window,
                config.henaw_config.intermediate_window,
                config.henaw_config.slow_window
            ],
            'biomarkers': config.henaw_config.input_features
        }
    elif model_name.lower() == 'modal':
        dataset_class = MODALDataset
        dataset_kwargs = {
            'oct_image_dir': config.data_config.retinal_img_path,
            'biomarker_features': config.modal_config.biomarker_features,
            'image_size': config.modal_config.oct_image_size,
            'augment': config.modal_config.oct_augmentation
        }
    elif model_name.lower() == 'metage':
        dataset_class = METAGEDataset
        dataset_kwargs = {
            'metabolomics_file': os.path.join(
                config.data_config.ukbb_data_path,
                'metabolomics.h5'
            ),
            'sequence_length': config.metage_config.sequence_length
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return dataset_class, dataset_kwargs


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train biological age models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['henaw', 'modal', 'metage'],
                       help='Model to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default='/mnt/data1/UKBB',
                       help='Path to UK Biobank data')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_lightning', action='store_true',
                       help='Use PyTorch Lightning trainer')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs to use')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode with reduced data')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = ConfigManager(args.config)
    
    # Override with command line arguments
    if args.data_path:
        config.data_config.ukbb_data_path = args.data_path
    if args.output_dir:
        config.data_config.output_dir = args.output_dir
    
    # Create output directory
    output_dir = Path(config.data_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"{args.model}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config.save_config(run_dir / 'config.yaml')
    
    print(f"\n{'='*60}")
    print(f"Training {args.model.upper()} Model")
    print(f"{'='*60}")
    print(f"Output directory: {run_dir}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"Seed: {args.seed}")
    
    # Create model
    print("\n📦 Creating model...")
    model = create_model(args.model, config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataset and data loaders
    print("\n📊 Loading data...")
    dataset_class, dataset_kwargs = create_dataset(args.model, config)
    
    # Debug mode - use subset of data
    if args.debug:
        config.data_config.batch_size = 4
        config.training_config.max_epochs = 2
        config.training_config.limit_train_batches = 10
        config.training_config.limit_val_batches = 5
    
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset_class,
        config.data_config,
        config.training_config,
        config.get_model_config(args.model),
        **dataset_kwargs
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create trainer
    print("\n🏋️ Starting training...")
    
    if args.use_lightning:
        # PyTorch Lightning trainer
        trainer = create_trainer(
            model,
            config,
            train_loader,
            val_loader,
            test_loader,
            use_lightning=True
        )
        
        # Create Lightning module
        from training.trainer import BioAgeTrainer
        lightning_model = BioAgeTrainer(model, config)
        
        # Train
        trainer.fit(lightning_model, train_loader, val_loader)
        
        # Test
        if test_loader:
            print("\n🧪 Running test evaluation...")
            test_results = trainer.test(lightning_model, test_loader)
            
            # Save test results
            with open(run_dir / 'test_results.json', 'w') as f:
                json.dump(test_results, f, indent=2)
    else:
        # Standard PyTorch trainer
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        trainer = StandardTrainer(
            model,
            train_loader,
            val_loader,
            config,
            device
        )
        
        # Load checkpoint if provided
        if args.checkpoint:
            print(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)
        
        # Train
        train_results = trainer.train()
        
        # Save training history
        with open(run_dir / 'training_history.json', 'w') as f:
            json.dump({
                'train_losses': train_results['train_losses'],
                'val_losses': train_results['val_losses']
            }, f, indent=2)
        
        # Evaluate on test set
        if test_loader:
            print("\n🧪 Running test evaluation...")
            model.eval()
            
            all_predictions = []
            all_targets = []
            all_uncertainties = []
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    if isinstance(inputs, dict):
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                    else:
                        inputs = inputs.to(device)
                    
                    outputs = model(inputs, return_uncertainty=True)
                    predictions = outputs['prediction'].squeeze().cpu().numpy()
                    
                    all_predictions.append(predictions)
                    all_targets.append(targets.numpy())
                    
                    if 'uncertainty' in outputs:
                        all_uncertainties.append(outputs['uncertainty'].cpu().numpy())
            
            # Aggregate predictions
            all_predictions = np.concatenate(all_predictions)
            all_targets = np.concatenate(all_targets)
            
            if all_uncertainties:
                all_uncertainties = np.concatenate(all_uncertainties)
            else:
                all_uncertainties = None
            
            # Calculate metrics
            metrics_calculator = BioAgeMetrics()
            test_results = metrics_calculator.comprehensive_evaluation(
                all_predictions,
                all_targets,
                uncertainties=all_uncertainties
            )
            
            # Print summary
            metrics_calculator.print_summary()
            
            # Save test results
            with open(run_dir / 'test_metrics.json', 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                def convert_to_serializable(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, (np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_to_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    else:
                        return obj
                
                serializable_results = convert_to_serializable(test_results)
                json.dump(serializable_results, f, indent=2)
    
    print(f"\n✅ Training complete! Results saved to {run_dir}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"Total epochs: {config.training_config.max_epochs}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Output directory: {run_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()