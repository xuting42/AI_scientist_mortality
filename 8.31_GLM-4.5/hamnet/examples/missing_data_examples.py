"""
Usage Examples for Missing Data Imputation Components

This module provides comprehensive examples demonstrating how to use the missing data
imputation components with HAMNet for biological age prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Import HAMNet components
from .hamnet import HAMNet, HAMNetConfig
from .gan_imputation import GANImputationConfig, GANImputer
from .graph_imputation import GraphImputationConfig, GraphImputer
from .advanced_imputation import AdvancedImputationConfig, AdvancedImputer
from .integrated_missing_data import IntegratedMissingDataConfig, IntegratedMissingDataHandler


def create_sample_data(batch_size: int = 64, missing_rate: float = 0.3) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Create sample multi-modal data with missing values for demonstration
    
    Args:
        batch_size: Number of samples
        missing_rate: Rate of missing data (0.0 to 1.0)
    
    Returns:
        Tuple of (data_dict, mask_dict)
    """
    # Define modality dimensions
    modalities = {
        'clinical': 100,      # Clinical measurements
        'imaging': 512,        # Imaging features
        'genetic': 1000,       # Genetic markers
        'lifestyle': 50        # Lifestyle factors
    }
    
    # Generate synthetic data
    data = {}
    masks = {}
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    for modality, dim in modalities.items():
        # Generate realistic data patterns
        if modality == 'clinical':
            # Clinical data: mix of normal and slightly skewed distributions
            data[modality] = torch.randn(batch_size, dim)
            # Add some correlations
            data[modality][:, :10] += 0.5 * data[modality][:, 10:20]
            
        elif modality == 'imaging':
            # Imaging data: more structured patterns
            base_patterns = torch.randn(batch_size, 50)
            data[modality] = torch.cat([
                base_patterns,
                base_patterns * 0.8 + torch.randn(batch_size, 50) * 0.2,
                torch.randn(batch_size, dim - 100)
            ], dim=1)
            
        elif modality == 'genetic':
            # Genetic data: sparse binary-like patterns
            data[modality] = torch.randn(batch_size, dim)
            # Make some features more discrete
            data[modality][:, :200] = (torch.sigmoid(data[modality][:, :200]) > 0.5).float()
            
        elif modality == 'lifestyle':
            # Lifestyle data: categorical and continuous mix
            data[modality] = torch.randn(batch_size, dim)
            # Add categorical patterns
            data[modality][:, :10] = torch.randint(0, 5, (batch_size, 10)).float()
        
        # Create missing data masks with realistic patterns
        if modality == 'genetic':
            # Genetic data often has more missing values
            mod_missing_rate = min(missing_rate * 1.5, 0.8)
        elif modality == 'imaging':
            # Imaging data usually has fewer missing values
            mod_missing_rate = missing_rate * 0.5
        else:
            mod_missing_rate = missing_rate
        
        # Create structured missingness (not completely random)
        mask = torch.ones(batch_size, dim)
        
        # Add random missingness
        random_missing = torch.rand(batch_size, dim) < mod_missing_rate
        
        # Add block missingness (some patients missing entire modality sections)
        block_missing = torch.zeros(batch_size, dim).bool()
        n_blocks = int(batch_size * mod_missing_rate * 0.3)  # 30% of missingness is block-based
        if n_blocks > 0:
            block_patients = torch.randperm(batch_size)[:n_blocks]
            block_size = dim // 4
            for patient in block_patients:
                start_idx = torch.randint(0, dim - block_size, (1,)).item()
                block_missing[patient, start_idx:start_idx + block_size] = True
        
        # Combine missing patterns
        combined_missing = random_missing | block_missing
        mask[combined_missing] = 0
        masks[modality] = mask.bool()
    
    return data, masks


def example_gan_imputation():
    """
    Example: Using GAN-based imputation for missing data
    """
    print("=== GAN-based Imputation Example ===")
    
    # Create configuration
    config = GANImputationConfig(
        latent_dim=128,
        hidden_dim=256,
        batch_size=32,
        num_epochs=50,
        generator_lr=1e-4,
        discriminator_lr=1e-4,
        lambda_cycle=10.0,
        clinical_dim=100,
        imaging_dim=512,
        genetic_dim=1000,
        lifestyle_dim=50
    )
    
    # Create sample data
    data, masks = create_sample_data(batch_size=128, missing_rate=0.4)
    
    # Initialize GAN imputer
    imputer = GANImputer(config)
    
    # Train the imputer
    print("Training GAN imputer...")
    # Note: In practice, you would use a proper DataLoader
    # This is a simplified example
    for epoch in range(10):  # Reduced for demonstration
        losses = imputer.train_step(data, masks)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: G_loss={losses['generator_loss']:.4f}, "
                  f"D_loss={losses['discriminator_loss']:.4f}")
    
    # Perform imputation
    imputed_data = imputer.impute(data, masks)
    
    # Evaluate imputation quality
    print("\nImputation Results:")
    for modality in data:
        mask = masks[modality]
        if mask.any():
            original = data[modality][mask]
            imputed = imputed_data[modality][mask]
            
            mse = torch.nn.functional.mse_loss(imputed, original).item()
            mae = torch.nn.functional.l1_loss(imputed, original).item()
            
            print(f"{modality}: MSE={mse:.4f}, MAE={mae:.4f}")
    
    return imputed_data


def example_graph_imputation():
    """
    Example: Using graph-based imputation with patient similarity
    """
    print("\n=== Graph-based Imputation Example ===")
    
    # Create configuration
    config = GraphImputationConfig(
        hidden_dim=256,
        num_layers=3,
        num_heads=4,
        similarity_threshold=0.6,
        k_neighbors=8,
        batch_size=32,
        clinical_dim=100,
        imaging_dim=512,
        genetic_dim=1000,
        lifestyle_dim=50
    )
    
    # Create sample data
    data, masks = create_sample_data(batch_size=100, missing_rate=0.35)
    
    # Initialize graph imputer
    imputer = GraphImputer(config)
    
    # Train the imputer
    print("Training graph imputer...")
    for epoch in range(10):  # Reduced for demonstration
        losses = imputer.train_step(data, masks)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Total_loss={losses['total_loss']:.4f}, "
                  f"Reconstruction={losses['reconstruction_loss']:.4f}")
    
    # Perform imputation
    imputed_data = imputer.impute(data, masks)
    
    # Analyze graph structure
    print(f"\nGraph Analysis:")
    print(f"Number of communities detected: {len(torch.unique(imputer.graph_constructor.communities))}")
    
    # Evaluate imputation quality
    print("\nImputation Results:")
    for modality in data:
        mask = masks[modality]
        if mask.any():
            original = data[modality][mask]
            imputed = imputed_data[modality][mask]
            
            mse = torch.nn.functional.mse_loss(imputed, original).item()
            mae = torch.nn.functional.l1_loss(imputed, original).item()
            
            print(f"{modality}: MSE={mse:.4f}, MAE={mae:.4f}")
    
    return imputed_data


def example_vae_imputation():
    """
    Example: Using VAE-based imputation with uncertainty estimation
    """
    print("\n=== VAE-based Imputation Example ===")
    
    # Create configuration
    config = AdvancedImputationConfig(
        latent_dim=64,
        hidden_dim=256,
        num_layers=3,
        beta_vae=1.0,
        annealing_steps=1000,
        clinical_dim=100,
        imaging_dim=512,
        genetic_dim=1000,
        lifestyle_dim=50
    )
    
    # Create sample data
    data, masks = create_sample_data(batch_size=128, missing_rate=0.3)
    
    # Initialize VAE imputer
    imputer = AdvancedImputer(config, method="vae")
    
    # Train the imputer
    print("Training VAE imputer...")
    for epoch in range(10):  # Reduced for demonstration
        batch = {
            'data': torch.cat([data[m] for m in data], dim=1),
            'mask': torch.cat([masks[m].float() for m in masks], dim=1)
        }
        losses = imputer.train_step(batch)
        if epoch % 2 == 0:
            print(f"Epoch {epoch}: Total={losses['total']:.4f}, "
                  f"Reconstruction={losses['reconstruction']:.4f}")
    
    # Perform imputation with uncertainty estimation
    combined_data = torch.cat([data[m] for m in data], dim=1)
    combined_masks = torch.cat([masks[m].float() for m in masks], dim=1)
    
    imputed_outputs = imputer.impute(combined_data, combined_masks, num_samples=10)
    
    # Split outputs back to modalities
    imputed_data = {}
    uncertainties = {}
    
    split_indices = [0, 100, 612, 1612, 1662]  # Cumulative dimensions
    modality_names = ['clinical', 'imaging', 'genetic', 'lifestyle']
    
    for i, modality in enumerate(modality_names):
        start_idx, end_idx = split_indices[i], split_indices[i + 1]
        imputed_data[modality] = imputed_outputs[modality]
        uncertainties[modality] = imputed_outputs[f"{modality}_uncertainty"]
    
    # Evaluate results
    print("\nImputation Results with Uncertainty:")
    for modality in data:
        mask = masks[modality]
        if mask.any():
            original = data[modality][mask]
            imputed = imputed_data[modality][mask]
            uncertainty = uncertainties[modality][mask]
            
            mse = torch.nn.functional.mse_loss(imputed, original).item()
            mae = torch.nn.functional.l1_loss(imputed, original).item()
            avg_uncertainty = uncertainty.mean().item()
            
            print(f"{modality}: MSE={mse:.4f}, MAE={mae:.4f}, "
                  f"Avg_Uncertainty={avg_uncertainty:.4f}")
    
    return imputed_data, uncertainties


def example_integrated_imputation():
    """
    Example: Using integrated missing data handling with HAMNet
    """
    print("\n=== Integrated Missing Data Handling Example ===")
    
    # Create integrated configuration
    config = IntegratedMissingDataConfig(
        primary_method="adaptive",
        batch_size=32,
        missing_threshold=0.3,
        ensemble_weights={
            "gan": 0.3,
            "graph": 0.3,
            "vae": 0.2,
            "multi_task": 0.2
        },
        gan_config=GANImputationConfig(
            latent_dim=64,
            hidden_dim=128,
            clinical_dim=100,
            imaging_dim=512,
            genetic_dim=1000,
            lifestyle_dim=50
        ),
        graph_config=GraphImputationConfig(
            hidden_dim=128,
            clinical_dim=100,
            imaging_dim=512,
            genetic_dim=1000,
            lifestyle_dim=50
        ),
        advanced_config=AdvancedImputationConfig(
            latent_dim=64,
            hidden_dim=128,
            clinical_dim=100,
            imaging_dim=512,
            genetic_dim=1000,
            lifestyle_dim=50
        )
    )
    
    # Create sample data
    data, masks = create_sample_data(batch_size=64, missing_rate=0.4)
    
    # Initialize integrated handler
    handler = IntegratedMissingDataHandler(config)
    
    # Analyze missing patterns
    analysis = handler.analyze_missing_data(masks)
    print(f"Missing Data Analysis:")
    print(f"Overall missing rate: {analysis['overall_missing_rate']:.2%}")
    print(f"Recommended method: {analysis['recommended_method']}")
    
    # Perform imputation and prediction
    outputs = handler.predict(data, masks)
    
    # Display results
    print(f"\nSelected imputation method: {outputs['selected_method']}")
    print(f"Prediction shape: {outputs['predictions'].shape}")
    
    # Evaluate imputation quality
    print("\nImputation Quality:")
    for modality in data:
        mask = masks[modality]
        if mask.any():
            original = data[modality][mask]
            imputed = outputs['imputed_data'][modality][mask]
            
            mse = torch.nn.functional.mse_loss(imputed, original).item()
            mae = torch.nn.functional.l1_loss(imputed, original).item()
            
            print(f"{modality}: MSE={mse:.4f}, MAE={mae:.4f}")
    
    return outputs


def example_comparison_study():
    """
    Example: Comparing different imputation methods
    """
    print("\n=== Imputation Method Comparison ===")
    
    # Create test data
    data, masks = create_sample_data(batch_size=200, missing_rate=0.4)
    
    # Test different methods
    methods = {
        'GAN': GANImputer(GANImputationConfig(
            latent_dim=64, hidden_dim=128, clinical_dim=100,
            imaging_dim=512, genetic_dim=1000, lifestyle_dim=50
        )),
        'Graph': GraphImputer(GraphImputationConfig(
            hidden_dim=128, clinical_dim=100, imaging_dim=512,
            genetic_dim=1000, lifestyle_dim=50
        )),
        'VAE': AdvancedImputer(AdvancedImputationConfig(
            latent_dim=64, hidden_dim=128, clinical_dim=100,
            imaging_dim=512, genetic_dim=1000, lifestyle_dim=50
        ), method="vae"),
        'Multi-task': AdvancedImputer(AdvancedImputationConfig(
            latent_dim=64, hidden_dim=128, clinical_dim=100,
            imaging_dim=512, genetic_dim=1000, lifestyle_dim=50
        ), method="multi_task")
    }
    
    results = {}
    
    for method_name, imputer in methods.items():
        print(f"\nTesting {method_name}...")
        
        # Brief training (for demonstration)
        if method_name == 'VAE' or method_name == 'Multi-task':
            batch = {
                'data': torch.cat([data[m] for m in data], dim=1),
                'mask': torch.cat([masks[m].float() for m in masks], dim=1)
            }
            for _ in range(5):
                imputer.train_step(batch)
        else:
            for _ in range(5):
                imputer.train_step(data, masks)
        
        # Impute
        if method_name in ['VAE', 'Multi-task']:
            combined_data = torch.cat([data[m] for m in data], dim=1)
            combined_masks = torch.cat([masks[m].float() for m in masks], dim=1)
            imputed_output = imputer.impute(combined_data, combined_masks)
            
            # Split back to modalities
            imputed_data = {}
            split_indices = [0, 100, 612, 1612, 1662]
            modality_names = ['clinical', 'imaging', 'genetic', 'lifestyle']
            
            for i, modality in enumerate(modality_names):
                start_idx, end_idx = split_indices[i], split_indices[i + 1]
                imputed_data[modality] = imputed_output[modality]
        else:
            imputed_data = imputer.impute(data, masks)
        
        # Evaluate
        method_results = {}
        for modality in data:
            mask = masks[modality]
            if mask.any():
                original = data[modality][mask]
                imputed = imputed_data[modality][mask]
                
                mse = torch.nn.functional.mse_loss(imputed, original).item()
                mae = torch.nn.functional.l1_loss(imputed, original).item()
                
                method_results[modality] = {'mse': mse, 'mae': mae}
        
        results[method_name] = method_results
    
    # Display comparison
    print("\n=== Method Comparison Results ===")
    print("{:<12} {:<15} {:<15} {:<15} {:<15}".format(
        "Method", "Clinical MSE", "Imaging MSE", "Genetic MSE", "Lifestyle MSE"
    ))
    
    for method_name, method_results in results.items():
        row = [method_name]
        for modality in ['clinical', 'imaging', 'genetic', 'lifestyle']:
            if modality in method_results:
                row.append(f"{method_results[modality]['mse']:.4f}")
            else:
                row.append("N/A")
        
        print("{:<12} {:<15} {:<15} {:<15} {:<15}".format(*row))
    
    return results


def visualize_imputation_results(data, masks, imputed_data, method_name="Imputation"):
    """
    Visualize imputation results
    
    Args:
        data: Original data dictionary
        masks: Missing data masks
        imputed_data: Imputed data dictionary
        method_name: Name of the imputation method
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    modality_names = list(data.keys())
    
    for i, modality in enumerate(modality_names):
        if i >= 4:
            break
            
        ax = axes[i]
        
        # Get data for this modality
        original = data[modality].cpu().numpy()
        mask = masks[modality].cpu().numpy()
        imputed = imputed_data[modality].cpu().numpy()
        
        # Flatten for visualization
        original_flat = original.flatten()
        mask_flat = mask.flatten()
        imputed_flat = imputed.flatten()
        
        # Create scatter plot
        observed_idx = mask_flat
        missing_idx = ~mask_flat
        
        if observed_idx.any():
            ax.scatter(original_flat[observed_idx], imputed_flat[observed_idx], 
                      alpha=0.6, label='Observed', c='blue')
        
        if missing_idx.any():
            ax.scatter(original_flat[missing_idx], imputed_flat[missing_idx], 
                      alpha=0.6, label='Imputed', c='red')
        
        # Add diagonal line
        min_val = min(original_flat.min(), imputed_flat.min())
        max_val = max(original_flat.max(), imputed_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        ax.set_xlabel('Original Values')
        ax.set_ylabel('Imputed Values')
        ax.set_title(f'{modality.capitalize()} - {method_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_pipeline_with_hamnet():
    """
    Example: Complete pipeline with HAMNet integration
    """
    print("\n=== Complete HAMNet Pipeline with Missing Data Handling ===")
    
    # Create configuration
    config = IntegratedMissingDataConfig(
        primary_method="adaptive",
        batch_size=32,
        hamnet_config=HAMNetConfig(
            embedding_dim=256,
            hidden_dim=512,
            num_heads=8,
            enable_uncertainty=True,
            clinical_dim=100,
            imaging_dim=512,
            genetic_dim=1000,
            lifestyle_dim=50
        )
    )
    
    # Create sample data with targets (for training)
    data, masks = create_sample_data(batch_size=128, missing_rate=0.35)
    
    # Add synthetic targets (biological age)
    # In practice, these would be real chronological ages
    targets = torch.randn(128, 1) * 10 + 50  # Mean age 50, std 10
    
    # Create integrated handler
    handler = IntegratedMissingDataHandler(config)
    
    # Train the system (simplified for demonstration)
    print("Training integrated HAMNet system...")
    
    # In practice, you would use proper DataLoaders and training loops
    # This is a simplified example
    dataloader = [{
        'data': data,
        'masks': masks,
        'targets': targets
    } for _ in range(10)]  # 10 batches for demo
    
    for epoch in range(5):  # 5 epochs for demo
        total_loss = 0.0
        
        for batch in dataloader:
            handler.optimizer.zero_grad()
            
            # Forward pass
            outputs = handler.module(
                batch['data'], batch['masks'], training=True
            )
            
            # Add targets to outputs for loss computation
            outputs['targets'] = batch['targets']
            
            # Compute loss
            loss = handler._compute_integrated_loss(outputs, batch)
            
            loss.backward()
            handler.optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/len(dataloader):.4f}")
    
    # Make predictions
    print("\nMaking predictions...")
    test_data, test_masks = create_sample_data(batch_size=32, missing_rate=0.4)
    predictions = handler.predict(test_data, test_masks)
    
    # Display results
    print(f"Prediction shape: {predictions['predictions'].shape}")
    print(f"Uncertainty shape: {predictions['uncertainty'].shape}")
    print(f"Used imputation method: {predictions['selected_method']}")
    
    # Analyze missing data handling
    analysis = predictions['missing_analysis']
    print(f"\nMissing data analysis:")
    print(f"Overall missing rate: {analysis['overall_missing_rate']:.2%}")
    
    return predictions


if __name__ == "__main__":
    """
    Run all examples
    """
    print("Running Missing Data Imputation Examples")
    print("=" * 50)
    
    # Run individual examples
    gan_results = example_gan_imputation()
    graph_results = example_graph_imputation()
    vae_results, vae_uncertainties = example_vae_imputation()
    integrated_results = example_integrated_imputation()
    comparison_results = example_comparison_study()
    hamnet_results = example_pipeline_with_hamnet()
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("Key insights:")
    print("1. GAN-based methods work well for generating complete modalities")
    print("2. Graph-based methods leverage patient similarity patterns")
    print("3. VAE methods provide uncertainty estimates")
    print("4. Integrated methods adapt to missing data patterns")
    print("5. All methods integrate seamlessly with HAMNet")