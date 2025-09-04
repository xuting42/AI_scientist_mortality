"""
Evaluation script for HAMBAE algorithm system.
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
import pandas as pd

from hambae.config import HAMBAEConfig, load_config
from hambae.models.tier1_cban import ClinicalBiomarkerAgingNetwork, create_cban_model
from hambae.models.tier2_mnai import MetabolicNetworkAgingIntegrator, create_mnai_model
from hambae.models.tier3_mmbat import MultiModalBiologicalAgeTransformer, create_mmbat_model
from hambae.data.ukbb_loader import create_data_loaders
from hambae.utils.metrics import compute_metrics, compute_fairness_metrics, format_metrics_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate HAMBAE models')
    
    parser.add_argument('--config', type=str, default='hambae/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_root', type=str, default='data/ukbb',
                       help='Path to UK Biobank data directory')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_tier', type=int, choices=[1, 2, 3], required=True,
                       help='Model tier to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--compute_fairness', action='store_true',
                       help='Compute fairness metrics')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions')
    
    return parser.parse_args()


def load_model(model_path: str, model_tier: int, config: HAMBAEConfig, device: torch.device):
    """Load trained model from checkpoint."""
    logger.info(f"Loading {model_tier} model from {model_path}")
    
    # Create model based on tier
    if model_tier == 1:
        model = create_cban_model(config.tier1)
    elif model_tier == 2:
        model = create_mnai_model(config.tier2)
    elif model_tier == 3:
        model = create_mmbat_model(config.tier3)
    else:
        raise ValueError(f"Invalid model tier: {model_tier}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    return model


def prepare_input_for_tier(batch: dict, model_tier: int):
    """Prepare input batch for specific model tier."""
    if model_tier == 1:
        # Tier 1: Blood biomarkers only
        return batch['blood_biomarkers']
    elif model_tier == 2:
        # Tier 2: Blood biomarkers + metabolomics
        blood = batch['blood_biomarkers']
        metabolomics = batch.get('metabolomics', torch.zeros(blood.size(0), 400, device=blood.device))
        return torch.cat([blood, metabolomics], dim=1)
    elif model_tier == 3:
        # Tier 3: All modalities
        blood = batch['blood_biomarkers']
        metabolomics = batch.get('metabolomics', torch.zeros(blood.size(0), 400, device=blood.device))
        retinal = batch.get('retinal_features', torch.zeros(blood.size(0), 768, device=blood.device))
        genetic = batch.get('genetic_features', torch.zeros(blood.size(0), 1000, device=blood.device))
        return torch.cat([blood, metabolomics, retinal, genetic], dim=1)
    else:
        raise ValueError(f"Invalid model tier: {model_tier}")


def evaluate_model(model, data_loader, device, model_tier, compute_uncertainty=True):
    """Evaluate model on data loader."""
    logger.info("Starting evaluation...")
    
    predictions = []
    targets = []
    uncertainties = []
    participant_ids = []
    ages = []
    sexes = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            device_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    device_batch[key] = value.to(device)
                else:
                    device_batch[key] = value
            
            # Prepare input for model tier
            model_input = prepare_input_for_tier(device_batch, model_tier)
            
            # Forward pass
            outputs = model(
                model_input,
                return_uncertainty=compute_uncertainty,
                return_features=False,
            )
            
            # Collect predictions
            predictions.append(outputs.predicted_age.cpu())
            targets.append(device_batch['age'].cpu())
            
            if outputs.uncertainty is not None:
                uncertainties.append(outputs.uncertainty.cpu())
            
            # Collect metadata
            if 'participant_id' in device_batch:
                participant_ids.extend(device_batch['participant_id'])
            ages.extend(device_batch['age'].cpu().tolist())
            if 'sex' in device_batch:
                sexes.extend(device_batch['sex'].cpu().tolist())
    
    # Concatenate results
    predictions = torch.cat(predictions, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    uncertainties = torch.cat(uncertainties, dim=0).numpy() if uncertainties else None
    
    return {
        'predictions': predictions,
        'targets': targets,
        'uncertainties': uncertainties,
        'participant_ids': participant_ids,
        'ages': ages,
        'sexes': sexes,
    }


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device(args.device if args.device != 'auto' else 'cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create data loaders
    data_loaders = create_data_loaders(
        data_root=args.data_root,
        config=config.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    logger.info(f"Data loaded:")
    logger.info(f"  Test: {len(data_loaders['test'].dataset)} samples")
    
    # Load model
    model = load_model(args.model_path, args.model_tier, config, device)
    
    # Evaluate model
    evaluation_results = evaluate_model(
        model, data_loaders['test'], device, args.model_tier
    )
    
    # Compute metrics
    logger.info("Computing evaluation metrics...")
    
    metrics = compute_metrics(
        predictions=evaluation_results['predictions'],
        targets=evaluation_results['targets'],
        uncertainty=evaluation_results['uncertainties'],
    )
    
    # Compute fairness metrics if requested
    if args.compute_fairness and evaluation_results['sexes']:
        logger.info("Computing fairness metrics...")
        
        sensitive_attributes = {
            'sex': np.array(evaluation_results['sexes'])
        }
        
        fairness_metrics = compute_fairness_metrics(
            predictions=evaluation_results['predictions'],
            targets=evaluation_results['targets'],
            sensitive_attributes=sensitive_attributes,
        )
        metrics.update(fairness_metrics)
    
    # Log results
    logger.info("Evaluation Results:")
    logger.info(format_metrics_report(metrics))
    
    # Compare with targets
    if args.model_tier == 1:
        target_mae = config.tier1.target_mae
        target_r2 = config.tier1.target_r2
    elif args.model_tier == 2:
        target_mae = config.tier2.target_mae
        target_r2 = config.tier2.target_r2
    elif args.model_tier == 3:
        target_mae = config.tier3.target_mae
        target_r2 = config.tier3.target_r2
    
    logger.info(f"Target Performance: MAE ≤ {target_mae}, R² ≥ {target_r2}")
    logger.info(f"Actual Performance: MAE = {metrics['mae']:.4f}, R² = {metrics['r2']:.4f}")
    
    if metrics['mae'] <= target_mae and metrics['r2'] >= target_r2:
        logger.info("✅ Performance targets achieved!")
    else:
        logger.info("⚠️  Performance targets not fully achieved")
    
    # Save results
    results = {
        'model_tier': args.model_tier,
        'model_path': args.model_path,
        'evaluation_metrics': metrics,
        'performance_targets': {
            'target_mae': target_mae,
            'target_r2': target_r2,
            'mae_achieved': metrics['mae'] <= target_mae,
            'r2_achieved': metrics['r2'] >= target_r2,
        },
        'evaluation_summary': {
            'num_samples': len(evaluation_results['predictions']),
            'mean_age': np.mean(evaluation_results['ages']),
            'std_age': np.std(evaluation_results['ages']),
            'age_range': [np.min(evaluation_results['ages']), np.max(evaluation_results['ages'])],
        },
    }
    
    # Save detailed results
    results_path = output_dir / f'tier{args.model_tier}_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'participant_id': evaluation_results['participant_ids'],
            'chronological_age': evaluation_results['ages'],
            'predicted_age': evaluation_results['predictions'].flatten(),
            'age_acceleration': evaluation_results['predictions'].flatten() - evaluation_results['targets'],
        })
        
        if evaluation_results['uncertainties'] is not None:
            predictions_df['uncertainty'] = evaluation_results['uncertainties'].flatten()
        
        if evaluation_results['sexes']:
            predictions_df['sex'] = evaluation_results['sexes']
        
        predictions_path = output_dir / f'tier{args.model_tier}_predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Predictions saved to {predictions_path}")
    
    # Generate summary report
    report = f"""
HAMBAE Tier {args.model_tier} Evaluation Report
{'='*50}

Model Information:
- Model Tier: {args.model_tier}
- Model Path: {args.model_path}
- Device: {device}
- Number of Samples: {len(evaluation_results['predictions'])}

Performance Metrics:
- Mean Absolute Error (MAE): {metrics['mae']:.4f}
- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}
- R-squared (R²): {metrics['r2']:.4f}
- Pearson Correlation: {metrics['pearson_correlation']:.4f}
- Spearman Correlation: {metrics['spearman_correlation']:.4f}

Target Performance:
- Target MAE: ≤ {target_mae}
- Target R²: ≥ {target_r2}
- MAE Achieved: {'✅' if metrics['mae'] <= target_mae else '❌'}
- R² Achieved: {'✅' if metrics['r2'] >= target_r2 else '❌'}

Age Acceleration:
- Mean Age Acceleration: {metrics['mean_age_acceleration']:.4f}
- Std Age Acceleration: {metrics['std_age_acceleration']:.4f}

"""
    
    if 'expected_calibration_error' in metrics:
        report += f"""
Uncertainty Quantification:
- Expected Calibration Error: {metrics['expected_calibration_error']:.4f}
- Negative Log Likelihood: {metrics['negative_log_likelihood']:.4f}
- Mean Uncertainty: {metrics['mean_uncertainty']:.4f}
"""
    
    if args.compute_fairness and 'sex_mae_disparity' in metrics:
        report += f"""
Fairness Metrics:
- Sex MAE Disparity: {metrics['sex_mae_disparity']:.4f}
- Sex R² Disparity: {metrics['sex_r2_disparity']:.4f}
"""
    
    report_path = output_dir / f'tier{args.model_tier}_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Evaluation results saved to {output_dir}")
    logger.info("Evaluation completed successfully!")


if __name__ == '__main__':
    main()