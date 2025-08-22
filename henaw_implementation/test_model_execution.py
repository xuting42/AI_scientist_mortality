#!/usr/bin/env python
"""
Simple test script to verify HENAW model execution
"""

import torch
import numpy as np
import yaml
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from henaw_model import HENAWModel

def test_model_inference():
    """Test basic model inference"""
    print("=" * 60)
    print("HENAW MODEL EXECUTION TEST")
    print("=" * 60)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device to CPU for testing
    device = torch.device('cpu')
    
    # Load model
    print("\n1. Loading model...")
    model = HENAWModel(config).to(device)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model.pt'
    if os.path.exists(checkpoint_path):
        print(f"   Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("   ✓ Checkpoint loaded successfully")
    else:
        print("   ! No checkpoint found, using random weights")
    
    model.eval()
    
    # Create test batch
    print("\n2. Creating test data...")
    batch_size = 5
    n_features = config['model']['input_dim']
    
    # Generate realistic biomarker values
    test_biomarkers = torch.randn(batch_size, n_features).to(device)
    test_ages = torch.tensor([45.0, 55.0, 65.0, 50.0, 60.0]).unsqueeze(1).to(device)
    
    print(f"   Test batch shape: {test_biomarkers.shape}")
    print(f"   Age values: {test_ages.squeeze().tolist()}")
    
    # Run inference
    print("\n3. Running inference...")
    with torch.no_grad():
        try:
            # Try basic forward pass without intermediates
            output = model(test_biomarkers, test_ages, return_intermediates=False)
            
            print("   ✓ Inference successful!")
            print(f"\n   Results:")
            print(f"   - Biological ages: {output.biological_age.squeeze().tolist()}")
            
            if hasattr(output, 'uncertainty') and output.uncertainty is not None:
                print(f"   - Uncertainties: {output.uncertainty.squeeze().tolist()}")
            
            if hasattr(output, 'mortality_risk') and output.mortality_risk is not None:
                print(f"   - Mortality risks: {output.mortality_risk.squeeze().tolist()}")
            
            # Calculate age gaps
            age_gaps = output.biological_age.squeeze() - test_ages.squeeze()
            print(f"   - Age gaps: {age_gaps.tolist()}")
            
            # Model statistics
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n4. Model Statistics:")
            print(f"   - Total parameters: {total_params:,}")
            print(f"   - Trainable parameters: {trainable_params:,}")
            print(f"   - Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
            
        except Exception as e:
            print(f"   ✗ Inference failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return True

def test_batch_inference():
    """Test batch inference performance"""
    print("\n" + "=" * 60)
    print("BATCH INFERENCE TEST")
    print("=" * 60)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Load model
    model = HENAWModel(config).to(device)
    checkpoint_path = 'checkpoints/best_model.pt'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Test different batch sizes
    batch_sizes = [1, 10, 100, 500]
    n_features = config['model']['input_dim']
    
    print("\nTesting different batch sizes:")
    for batch_size in batch_sizes:
        test_biomarkers = torch.randn(batch_size, n_features).to(device)
        test_ages = torch.randn(batch_size, 1).to(device) * 10 + 55  # Ages around 55±10
        
        import time
        start_time = time.time()
        
        with torch.no_grad():
            output = model(test_biomarkers, test_ages, return_intermediates=False)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        per_sample_time = inference_time / batch_size
        
        print(f"   Batch size {batch_size:4d}: {inference_time:7.2f}ms total, {per_sample_time:6.3f}ms per sample")
    
    print("\n✓ Batch inference test completed")
    return True

if __name__ == "__main__":
    # Run tests
    success = test_model_inference()
    if success:
        test_batch_inference()
    
    sys.exit(0 if success else 1)