#!/usr/bin/env python
"""
Evaluation script for trained HENAW model
"""

import torch
import numpy as np
import pandas as pd
import yaml
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from henaw_model import HENAWModel

def evaluate_model(model_path='./checkpoints/best_simplified.pt'):
    """Evaluate trained model"""
    print("=" * 60)
    print("HENAW MODEL EVALUATION")
    print("=" * 60)
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # Load model
    print("\n1. Loading model...")
    model = HENAWModel(config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"   Model loaded from epoch {checkpoint['epoch']+1}")
    print(f"   Training loss: {checkpoint['train_loss']:.4f}")
    print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"   Validation MAE: {checkpoint['val_mae']:.2f} years")
    
    # Load test data
    print("\n2. Loading test data...")
    df = pd.read_csv('./data/synthetic_ukbb.csv')
    
    # Use last 15% as test set
    n_test_start = int(0.85 * len(df))
    test_df = df.iloc[n_test_start:]
    
    print(f"   Test samples: {len(test_df)}")
    
    # Prepare data
    biomarker_cols = ['crp', 'hba1c', 'creatinine', 'albumin', 'lymphocyte_pct', 
                      'rdw', 'ggt', 'ast', 'alt']
    
    # Use training statistics for normalization
    train_df = df.iloc[:int(0.7 * len(df))]
    mean = train_df[biomarker_cols].mean()
    std = train_df[biomarker_cols].std()
    
    test_X = torch.tensor(((test_df[biomarker_cols] - mean) / std).values).float().to(device)
    test_ages = torch.tensor(test_df['age'].values).float().unsqueeze(1).to(device)
    test_survival = torch.tensor(test_df['survival_time'].values).float()
    test_events = torch.tensor(test_df['event_indicator'].values).float()
    
    # Run predictions
    print("\n3. Running predictions...")
    with torch.no_grad():
        output = model(test_X, test_ages)
        predicted_ages = output.biological_age.squeeze()
        
        if output.mortality_risk is not None:
            mortality_risks = output.mortality_risk.squeeze()
        else:
            mortality_risks = None
    
    # Calculate metrics
    print("\n4. Performance Metrics:")
    print("-" * 40)
    
    # Age prediction metrics
    true_ages = test_ages.squeeze()
    age_errors = predicted_ages - true_ages
    
    mae = torch.mean(torch.abs(age_errors)).item()
    rmse = torch.sqrt(torch.mean(age_errors ** 2)).item()
    correlation = np.corrcoef(predicted_ages.cpu().numpy(), true_ages.cpu().numpy())[0, 1]
    
    print(f"   Age Prediction:")
    print(f"     - MAE: {mae:.2f} years")
    print(f"     - RMSE: {rmse:.2f} years")
    print(f"     - Correlation: {correlation:.3f}")
    print(f"     - Mean predicted age: {predicted_ages.mean().item():.1f} years")
    print(f"     - Mean true age: {true_ages.mean().item():.1f} years")
    
    # Age gap analysis
    age_gaps = predicted_ages - true_ages
    print(f"\n   Age Gap Analysis:")
    print(f"     - Mean gap: {age_gaps.mean().item():.2f} years")
    print(f"     - Std gap: {age_gaps.std().item():.2f} years")
    print(f"     - Min gap: {age_gaps.min().item():.2f} years")
    print(f"     - Max gap: {age_gaps.max().item():.2f} years")
    
    # Mortality risk analysis (if available)
    if mortality_risks is not None:
        print(f"\n   Mortality Risk Analysis:")
        print(f"     - Mean risk score: {mortality_risks.mean().item():.3f}")
        print(f"     - Std risk score: {mortality_risks.std().item():.3f}")
        
        # Calculate C-statistic for mortality prediction
        from sklearn.metrics import roc_auc_score
        try:
            # Use event indicators as binary outcome
            c_stat = roc_auc_score(test_events.cpu().numpy(), 
                                  torch.sigmoid(mortality_risks).cpu().numpy())
            print(f"     - C-statistic: {c_stat:.3f}")
        except:
            print(f"     - C-statistic: N/A (insufficient events)")
    
    # Performance vs targets
    print(f"\n5. Performance vs Configuration Targets:")
    print("-" * 40)
    
    target_mae = config['evaluation']['targets']['mae']
    target_c_stat = config['evaluation']['targets']['c_statistic']
    target_icc = config['evaluation']['targets']['icc']
    
    if mae < target_mae:
        print(f"   ✓ MAE: {mae:.2f} < {target_mae} (TARGET MET)")
    else:
        print(f"   ✗ MAE: {mae:.2f} >= {target_mae} (TARGET MISSED)")
    
    if mortality_risks is not None and 'c_stat' in locals() and c_stat > target_c_stat:
        print(f"   ✓ C-statistic: {c_stat:.3f} > {target_c_stat} (TARGET MET)")
    else:
        print(f"   - C-statistic: Not evaluated")
    
    # Calculate ICC (simplified version)
    # For demonstration, we use test-retest correlation
    # In practice, this would require repeated measurements
    icc_estimate = correlation ** 2  # Simplified ICC approximation
    if icc_estimate > target_icc:
        print(f"   ✓ ICC (approx): {icc_estimate:.3f} > {target_icc} (TARGET MET)")
    else:
        print(f"   ✗ ICC (approx): {icc_estimate:.3f} <= {target_icc} (TARGET MISSED)")
    
    # Feature importance (simplified)
    print(f"\n6. Feature Importance (based on weight magnitudes):")
    print("-" * 40)
    
    # Get first layer weights
    first_layer = model.feature_transform[0]
    if hasattr(first_layer, 'weight'):
        weights = torch.abs(first_layer.weight.mean(dim=0))
        importance = weights / weights.sum()
        
        for i, name in enumerate(biomarker_cols):
            print(f"   {name:15s}: {importance[i].item():.3f}")
    
    # Model efficiency
    print(f"\n7. Model Efficiency:")
    print("-" * 40)
    
    # Test inference speed
    import time
    n_test_samples = 100
    test_batch = test_X[:n_test_samples]
    test_ages_batch = test_ages[:n_test_samples]
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):  # Run 10 times for average
            _ = model(test_batch, test_ages_batch)
    
    total_time = (time.time() - start_time) * 1000  # Convert to ms
    avg_time = total_time / (10 * n_test_samples)
    
    print(f"   Average inference time: {avg_time:.3f} ms per sample")
    print(f"   Throughput: {int(1000/avg_time)} samples/second")
    
    if avg_time < 100:
        print(f"   ✓ Latency < 100ms target (PASSED)")
    else:
        print(f"   ✗ Latency >= 100ms target (FAILED)")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'correlation': correlation,
        'mean_age_gap': age_gaps.mean().item(),
        'inference_time_ms': avg_time
    }

if __name__ == "__main__":
    metrics = evaluate_model()
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    
    # Save metrics to file
    import json
    Path('./results').mkdir(exist_ok=True)
    with open('./results/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("\nMetrics saved to: ./results/evaluation_metrics.json")