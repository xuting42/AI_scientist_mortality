"""
Example usage of HAMNet for biological age prediction

This script demonstrates:
- How to create and configure HAMNet models
- How to prepare data for training
- How to train and evaluate models
- How to use uncertainty quantification
- How to perform cross-validation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Import HAMNet components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hamnet.models.hamnet import HAMNet, HAMNetConfig
from hamnet.utils.utils import (
    TrainingConfig, HAMNetDataset, create_data_loaders,
    setup_logging, save_config, save_results, plot_training_history
)
from hamnet.training import HAMNetTrainer, CrossValidator, ModelEnsemble


def create_sample_data(num_samples: int = 1000) -> Dict[str, Any]:
    """Create sample multimodal data for demonstration"""
    
    np.random.seed(42)
    
    # Clinical data (biomarkers, lab results)
    clinical_data = np.random.randn(num_samples, 100)
    # Add some correlation with age
    clinical_data[:, 0] += np.random.normal(0, 0.1, num_samples)  # Cholesterol
    clinical_data[:, 1] += np.random.normal(0, 0.1, num_samples)  # Blood pressure
    
    # Imaging data (MRI features, DXA measurements)
    imaging_data = np.random.randn(num_samples, 512)
    # Add age-related changes
    imaging_data[:, 0] += np.random.normal(0, 0.05, num_samples)  # Brain volume
    imaging_data[:, 1] += np.random.normal(0, 0.05, num_samples)  # Bone density
    
    # Genetic data (SNPs, polygenic risk scores)
    genetic_data = np.random.randn(num_samples, 1000)
    # Add some genetic effects
    genetic_snp_effect = np.random.randn(1000) * 0.1
    genetic_data = genetic_data + genetic_snp_effect
    
    # Lifestyle data (diet, exercise, smoking)
    lifestyle_data = np.random.randint(0, 10, (num_samples, 50))
    # Convert to float for processing
    lifestyle_data = lifestyle_data.astype(float)
    
    # Generate target ages (20-80 years)
    ages = np.random.uniform(20, 80, num_samples)
    
    # Add multimodal effects to ages
    clinical_effect = np.mean(clinical_data[:, :5], axis=1) * 2
    imaging_effect = np.mean(imaging_data[:, :5], axis=1) * 1.5
    genetic_effect = np.mean(genetic_data[:, :10], axis=1) * 0.5
    lifestyle_effect = np.mean(lifestyle_data[:, :5], axis=1) * 0.3
    
    # Combine effects with noise
    biological_ages = (
        ages * 0.7 +  # Chronological age baseline
        clinical_effect * 0.15 +
        imaging_effect * 0.1 +
        genetic_effect * 0.03 +
        lifestyle_effect * 0.02 +
        np.random.normal(0, 2, num_samples)  # Noise
    )
    
    # Create masks for missing data simulation
    clinical_mask = np.ones((num_samples, 100))
    imaging_mask = np.ones((num_samples, 512))
    
    # Simulate missing data (10% missing)
    missing_idx = np.random.choice(num_samples, int(num_samples * 0.1), replace=False)
    clinical_mask[missing_idx, :10] = 0
    imaging_mask[missing_idx, :20] = 0
    
    return {
        'data': {
            'clinical': clinical_data,
            'imaging': imaging_data,
            'genetic': genetic_data,
            'lifestyle': lifestyle_data
        },
        'targets': biological_ages,
        'masks': {
            'clinical': clinical_mask,
            'imaging': imaging_mask
        },
        'chronological_ages': ages
    }


def example_basic_usage():
    """Basic usage example"""
    print("=== Basic HAMNet Usage Example ===")
    
    # Create sample data
    sample_data = create_sample_data(num_samples=1000)
    
    # Create model configuration
    model_config = HAMNetConfig(
        model_tier="standard",
        embedding_dim=256,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
        enable_uncertainty=True,
        num_monte_carlo=20
    )
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        patience=10,
        mixed_precision=True
    )
    
    # Create dataset
    dataset = HAMNetDataset(
        data=sample_data['data'],
        targets=sample_data['targets'],
        masks=sample_data['masks']
    )
    
    # Create data loaders
    data_loaders = create_data_loaders(
        dataset, 
        batch_size=training_config.batch_size,
        train_split=0.8,
        val_split=0.1,
        num_workers=4
    )
    
    # Create model
    model = HAMNet(model_config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create trainer
    trainer = HAMNetTrainer(model, training_config)
    
    # Train model
    print("Starting training...")
    results = trainer.train(
        data_loaders['train'],
        data_loaders['val']
    )
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = trainer.evaluate(data_loaders['test'])
    
    # Print results
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Test MSE: {eval_results['metrics']['mse']:.4f}")
    print(f"Test MAE: {eval_results['metrics']['mae']:.4f}")
    print(f"Test R²: {eval_results['metrics']['r2']:.4f}")
    
    return model, results, eval_results


def example_uncertainty_quantification():
    """Uncertainty quantification example"""
    print("\n=== Uncertainty Quantification Example ===")
    
    # Create sample data
    sample_data = create_sample_data(num_samples=500)
    
    # Create model with uncertainty
    model_config = HAMNetConfig(
        model_tier="standard",
        embedding_dim=256,
        hidden_dim=512,
        num_heads=8,
        dropout=0.2,  # Higher dropout for uncertainty
        enable_uncertainty=True,
        num_monte_carlo=50
    )
    
    # Create dataset
    dataset = HAMNetDataset(
        data=sample_data['data'],
        targets=sample_data['targets'],
        masks=sample_data['masks']
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False
    )
    
    # Create model
    model = HAMNet(model_config)
    
    # Quick training (for demonstration)
    training_config = TrainingConfig(epochs=10, batch_size=32)
    trainer = HAMNetTrainer(model, training_config)
    
    # Train briefly
    trainer.train(data_loader, data_loader)
    
    # Predict with uncertainty
    uncertainty_results = model.predict_with_uncertainty(
        sample_data['data'], sample_data['masks'], num_samples=30
    )
    
    print(f"Mean prediction: {uncertainty_results['mean_prediction'].mean():.2f}")
    print(f"Mean uncertainty: {uncertainty_results['uncertainty'].mean():.2f}")
    print(f"Max uncertainty: {uncertainty_results['uncertainty'].max():.2f}")
    
    # Plot uncertainty vs error
    predictions = uncertainty_results['mean_prediction']
    uncertainties = uncertainty_results['uncertainty']
    targets = sample_data['targets']
    
    errors = np.abs(predictions - targets)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(uncertainties, errors, alpha=0.6)
    plt.xlabel('Predicted Uncertainty')
    plt.ylabel('Absolute Error')
    plt.title('Uncertainty Calibration')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Calculate correlation
    correlation = np.corrcoef(uncertainties, errors)[0, 1]
    print(f"Uncertainty-error correlation: {correlation:.3f}")
    
    return uncertainty_results


def example_cross_validation():
    """Cross-validation example"""
    print("\n=== Cross-Validation Example ===")
    
    # Create sample data
    sample_data = create_sample_data(num_samples=500)
    
    # Create configurations
    model_config = HAMNetConfig(
        model_tier="base",  # Smaller model for faster CV
        embedding_dim=128,
        hidden_dim=256,
        num_heads=4,
        dropout=0.1
    )
    
    training_config = TrainingConfig(
        epochs=20,  # Shorter training for CV
        batch_size=32,
        learning_rate=1e-4
    )
    
    # Create dataset
    dataset = HAMNetDataset(
        data=sample_data['data'],
        targets=sample_data['targets'],
        masks=sample_data['masks']
    )
    
    # Create cross-validator
    cv = CrossValidator(model_config, training_config)
    
    # Run cross-validation
    print("Running 5-fold cross-validation...")
    cv_results = cv.cross_validate(dataset, n_folds=5)
    
    # Print results
    print(f"Mean validation loss: {cv_results['mean_val_loss']:.4f} ± {cv_results['std_val_loss']:.4f}")
    print(f"Mean validation MAE: {cv_results['mean_val_mae']:.4f} ± {cv_results['std_val_mae']:.4f}")
    print(f"Mean validation R²: {cv_results['mean_val_r2']:.4f} ± {cv_results['std_val_r2']:.4f}")
    
    # Plot fold results
    fold_losses = [result['best_val_loss'] for result in cv_results['fold_results']]
    fold_maes = [result['final_val_metrics']['mae'] for result in cv_results['fold_results']]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([fold_losses], labels=['Validation Loss'])
    plt.title('Cross-Validation Loss Distribution')
    
    plt.subplot(1, 2, 2)
    plt.boxplot([fold_maes], labels=['Validation MAE'])
    plt.title('Cross-Validation MAE Distribution')
    
    plt.tight_layout()
    plt.show()
    
    return cv_results


def example_model_ensemble():
    """Model ensemble example"""
    print("\n=== Model Ensemble Example ===")
    
    # Create sample data
    sample_data = create_sample_data(num_samples=500)
    
    # Create dataset
    dataset = HAMNetDataset(
        data=sample_data['data'],
        targets=sample_data['targets'],
        masks=sample_data['masks']
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False
    )
    
    # Create multiple models with different configurations
    models = []
    model_configs = [
        HAMNetConfig(embedding_dim=128, hidden_dim=256, num_heads=4),
        HAMNetConfig(embedding_dim=256, hidden_dim=512, num_heads=8),
        HAMNetConfig(embedding_dim=512, hidden_dim=1024, num_heads=16)
    ]
    
    print("Training ensemble models...")
    for i, config in enumerate(model_configs):
        print(f"Training model {i+1}/{len(model_configs)}")
        
        model = HAMNet(config)
        
        # Quick training
        training_config = TrainingConfig(epochs=15, batch_size=32)
        trainer = HAMNetTrainer(model, training_config)
        
        trainer.train(data_loader, data_loader)
        models.append(model)
    
    # Create ensemble
    ensemble = ModelEnsemble(models)
    
    # Make ensemble predictions
    ensemble_results = ensemble.predict(data_loader)
    
    print(f"Ensemble predictions shape: {ensemble_results['predictions'].shape}")
    print(f"Ensemble uncertainty shape: {ensemble_results['uncertainties'].shape}")
    
    # Compare with individual models
    individual_predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = []
            for batch in data_loader:
                inputs = {k: v for k, v in batch.items() if k in ['clinical', 'imaging', 'genetic', 'lifestyle']}
                outputs = model(inputs)
                preds.append(outputs['predictions'].cpu().numpy())
            individual_predictions.append(np.concatenate(preds))
    
    # Calculate individual model errors
    individual_errors = []
    for preds in individual_predictions:
        errors = np.abs(preds - sample_data['targets'])
        individual_errors.append(errors.mean())
    
    # Calculate ensemble error
    ensemble_errors = np.abs(ensemble_results['predictions'] - sample_data['targets'])
    ensemble_error = ensemble_errors.mean()
    
    print(f"Individual model errors: {[f'{e:.4f}' for e in individual_errors]}")
    print(f"Ensemble error: {ensemble_error:.4f}")
    print(f"Ensemble improvement: {(min(individual_errors) - ensemble_error):.4f}")
    
    return ensemble_results


def example_missing_data_handling():
    """Missing data handling example"""
    print("\n=== Missing Data Handling Example ===")
    
    # Create sample data with different levels of missingness
    sample_data = create_sample_data(num_samples=500)
    
    # Create different missing data scenarios
    missing_scenarios = {
        'no_missing': 0.0,
        'low_missing': 0.1,
        'medium_missing': 0.3,
        'high_missing': 0.5
    }
    
    results = {}
    
    for scenario_name, missing_rate in missing_scenarios.items():
        print(f"Testing {scenario_name} (missing rate: {missing_rate})")
        
        # Create masks with different missing rates
        masks = {
            'clinical': np.ones_like(sample_data['data']['clinical']),
            'imaging': np.ones_like(sample_data['data']['imaging'])
        }
        
        if missing_rate > 0:
            n_samples = len(sample_data['targets'])
            n_missing = int(n_samples * missing_rate)
            
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
            
            masks['clinical'][missing_indices, :20] = 0
            masks['imaging'][missing_indices, :50] = 0
        
        # Create dataset
        dataset = HAMNetDataset(
            data=sample_data['data'],
            targets=sample_data['targets'],
            masks=masks
        )
        
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False
        )
        
        # Create and train model
        model_config = HAMNetConfig(embedding_dim=128, hidden_dim=256)
        model = HAMNet(model_config)
        
        training_config = TrainingConfig(epochs=20, batch_size=32)
        trainer = HAMNetTrainer(model, training_config)
        
        trainer.train(data_loader, data_loader)
        
        # Evaluate
        eval_results = trainer.evaluate(data_loader)
        
        results[scenario_name] = eval_results['metrics']
        
        print(f"  MSE: {eval_results['metrics']['mse']:.4f}")
        print(f"  MAE: {eval_results['metrics']['mae']:.4f}")
        print(f"  R²: {eval_results['metrics']['r2']:.4f}")
    
    # Plot results
    scenarios = list(results.keys())
    mse_values = [results[s]['mse'] for s in scenarios]
    mae_values = [results[s]['mae'] for s in scenarios]
    r2_values = [results[s]['r2'] for s in scenarios]
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(scenarios, mse_values)
    plt.title('MSE vs Missing Data Rate')
    plt.ylabel('MSE')
    
    plt.subplot(1, 3, 2)
    plt.bar(scenarios, mae_values)
    plt.title('MAE vs Missing Data Rate')
    plt.ylabel('MAE')
    
    plt.subplot(1, 3, 3)
    plt.bar(scenarios, r2_values)
    plt.title('R² vs Missing Data Rate')
    plt.ylabel('R²')
    
    plt.tight_layout()
    plt.show()
    
    return results


def example_temporal_data():
    """Temporal data handling example"""
    print("\n=== Temporal Data Example ===")
    
    # Create temporal data (multiple time points)
    num_samples = 200
    time_points = 5
    
    # Create longitudinal data
    temporal_data = {
        'clinical': np.random.randn(num_samples, time_points, 100),
        'imaging': np.random.randn(num_samples, time_points, 512)
    }
    
    # Create targets (final age)
    ages = np.random.uniform(20, 80, num_samples)
    
    # Add temporal trends
    for t in range(time_points):
        temporal_data['clinical'][:, t, 0] += ages * 0.1 * (t + 1) / time_points
        temporal_data['imaging'][:, t, 0] += ages * 0.05 * (t + 1) / time_points
    
    # Create model configuration for temporal data
    model_config = HAMNetConfig(
        model_tier="standard",
        embedding_dim=256,
        hidden_dim=512,
        num_heads=8,
        temporal_window=time_points,
        enable_uncertainty=True
    )
    
    # Create model
    model = HAMNet(model_config)
    
    # Create dataset
    dataset = HAMNetDataset(
        data=temporal_data,
        targets=ages
    )
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False
    )
    
    # Train model
    training_config = TrainingConfig(epochs=30, batch_size=32)
    trainer = HAMNetTrainer(model, training_config)
    
    print("Training on temporal data...")
    trainer.train(data_loader, data_loader)
    
    # Evaluate
    eval_results = trainer.evaluate(data_loader)
    
    print(f"Temporal model results:")
    print(f"  MSE: {eval_results['metrics']['mse']:.4f}")
    print(f"  MAE: {eval_results['metrics']['mae']:.4f}")
    print(f"  R²: {eval_results['metrics']['r2']:.4f}")
    
    # Extract attention weights for analysis
    model.eval()
    with torch.no_grad():
        batch = next(iter(data_loader))
        inputs = {k: v for k, v in batch.items() if k in ['clinical', 'imaging']}
        outputs = model(inputs)
    
    return eval_results


def main():
    """Main function to run all examples"""
    
    # Setup logging
    logger = setup_logging("examples", "hamnet_examples")
    
    print("HAMNet Usage Examples")
    print("=" * 50)
    
    # Run examples
    try:
        # Basic usage
        model, results, eval_results = example_basic_usage()
        
        # Uncertainty quantification
        uncertainty_results = example_uncertainty_quantification()
        
        # Cross-validation
        cv_results = example_cross_validation()
        
        # Model ensemble
        ensemble_results = example_model_ensemble()
        
        # Missing data handling
        missing_data_results = example_missing_data_handling()
        
        # Temporal data
        temporal_results = example_temporal_data()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()