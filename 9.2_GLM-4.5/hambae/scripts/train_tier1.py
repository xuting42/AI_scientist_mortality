"""
Training script for Tier 1: Clinical Biomarker Aging Network (CBAN).
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
import yaml
import json

from hambae.config import HAMBAEConfig, load_config
from hambae.models.tier1_cban import ClinicalBiomarkerAgingNetwork, create_cban_model
from hambae.data.ukbb_loader import create_data_loaders
from hambae.training.trainer import HAMBATrainer
from hambae.utils.metrics import compute_metrics, format_metrics_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Tier 1: Clinical Biomarker Aging Network')
    
    parser.add_argument('--config', type=str, default='hambae/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_root', type=str, default='data/ukbb',
                       help='Path to UK Biobank data directory')
    parser.add_argument('--experiment_dir', type=str, default='experiments/tier1',
                       help='Experiment directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update configuration with command line arguments
    config.data.data_root = args.data_root
    config.training.device = args.device
    config.training.num_workers = args.num_workers
    config.data.batch_size = args.batch_size
    config.training.max_epochs = args.max_epochs
    config.training.seed = args.seed
    
    # Set up experiment directory
    experiment_dir = Path(args.experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_path = experiment_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
    
    logger.info(f"Starting Tier 1 training with experiment directory: {experiment_dir}")
    logger.info(f"Configuration: {config}")
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data_root=args.data_root,
        config=config.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Train: {len(data_loaders['train'].dataset)} samples")
    logger.info(f"  Val: {len(data_loaders['val'].dataset)} samples")
    logger.info(f"  Test: {len(data_loaders['test'].dataset)} samples")
    
    # Create model
    model = create_cban_model(config.tier1)
    
    # Log model information
    model_info = model.get_model_info()
    logger.info(f"Model created: {model_info}")
    
    # Create trainer
    trainer = HAMBATrainer(
        model=model,
        config=config,
        train_loader=data_loaders['train'],
        val_loader=data_loaders['val'],
        test_loader=data_loaders['test'],
        experiment_dir=str(experiment_dir),
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train model
    training_results = trainer.train(max_epochs=args.max_epochs)
    
    # Save training results
    trainer.save_training_results(training_results)
    
    # Final evaluation
    logger.info("Performing final evaluation...")
    
    # Evaluate on test set
    model.eval()
    test_predictions = []
    test_targets = []
    test_uncertainties = []
    
    with torch.no_grad():
        for batch in data_loaders['test']:
            # Move batch to device
            device_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    device_batch[key] = value.to(trainer.device)
                else:
                    device_batch[key] = value
            
            # Forward pass
            outputs = model(device_batch['blood_biomarkers'], 
                          return_uncertainty=True)
            
            test_predictions.append(outputs.predicted_age.cpu())
            test_targets.append(device_batch['age'].cpu())
            if outputs.uncertainty is not None:
                test_uncertainties.append(outputs.uncertainty.cpu())
    
    # Concatenate predictions
    test_predictions = torch.cat(test_predictions, dim=0).numpy()
    test_targets = torch.cat(test_targets, dim=0).numpy()
    test_uncertainties = torch.cat(test_uncertainties, dim=0).numpy() if test_uncertainties else None
    
    # Compute metrics
    test_metrics = compute_metrics(
        predictions=test_predictions,
        targets=test_targets,
        uncertainty=test_uncertainties,
    )
    
    # Log results
    logger.info("Training completed!")
    logger.info("Final Test Metrics:")
    logger.info(format_metrics_report(test_metrics))
    
    # Compare with targets
    target_mae = config.tier1.target_mae
    target_r2 = config.tier1.target_r2
    
    logger.info(f"Target Performance: MAE ≤ {target_mae}, R² ≥ {target_r2}")
    logger.info(f"Actual Performance: MAE = {test_metrics['mae']:.4f}, R² = {test_metrics['r2']:.4f}")
    
    if test_metrics['mae'] <= target_mae and test_metrics['r2'] >= target_r2:
        logger.info("✅ Performance targets achieved!")
    else:
        logger.info("⚠️  Performance targets not fully achieved")
    
    # Save final results
    final_results = {
        'training_results': training_results,
        'test_metrics': test_metrics,
        'model_info': model_info,
        'performance_targets': {
            'target_mae': target_mae,
            'target_r2': target_r2,
            'mae_achieved': test_metrics['mae'] <= target_mae,
            'r2_achieved': test_metrics['r2'] >= target_r2,
        },
    }
    
    results_path = experiment_dir / 'final_results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()