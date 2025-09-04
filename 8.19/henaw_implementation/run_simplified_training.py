#!/usr/bin/env python
"""
Simplified HENAW training script without tensorboard/wandb dependencies
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from henaw_model import HENAWModel, MultiTaskLoss
from data_loader import UKBBDataset
from torch.utils.data import DataLoader

def train_henaw(config, epochs=10):
    """Simplified training function"""
    print("=" * 60)
    print("SIMPLIFIED HENAW TRAINING")
    print("=" * 60)
    
    # Set device
    device = torch.device('cpu')  # Force CPU for reliability
    print(f"Using device: {device}")
    
    # Create data directory
    Path('./checkpoints').mkdir(parents=True, exist_ok=True)
    
    # Load synthetic data
    print("\n1. Loading synthetic data...")
    import pandas as pd
    df = pd.read_csv('./data/synthetic_ukbb.csv')
    
    # Split data
    n_samples = len(df)
    n_train = int(0.7 * n_samples)
    n_val = int(0.15 * n_samples)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train+n_val]
    test_df = df.iloc[n_train+n_val:]
    
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Prepare data tensors
    biomarker_cols = ['crp', 'hba1c', 'creatinine', 'albumin', 'lymphocyte_pct', 
                      'rdw', 'ggt', 'ast', 'alt']
    
    # Normalize data
    mean = train_df[biomarker_cols].mean()
    std = train_df[biomarker_cols].std()
    
    train_X = torch.tensor(((train_df[biomarker_cols] - mean) / std).values).float()
    train_ages = torch.tensor(train_df['age'].values).float().unsqueeze(1)
    train_survival = torch.tensor(train_df['survival_time'].values).float().unsqueeze(1)
    train_events = torch.tensor(train_df['event_indicator'].values).float().unsqueeze(1)
    
    val_X = torch.tensor(((val_df[biomarker_cols] - mean) / std).values).float()
    val_ages = torch.tensor(val_df['age'].values).float().unsqueeze(1)
    
    # Initialize model
    print("\n2. Initializing model...")
    model = HENAWModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Initialize loss and optimizer
    criterion = MultiTaskLoss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Training loop
    print(f"\n3. Training for {epochs} epochs...")
    batch_size = min(512, len(train_X))
    n_batches = len(train_X) // batch_size
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        # Shuffle data
        perm = torch.randperm(len(train_X))
        train_X = train_X[perm]
        train_ages = train_ages[perm]
        train_survival = train_survival[perm]
        train_events = train_events[perm]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(train_X))
            
            batch_X = train_X[start_idx:end_idx].to(device)
            batch_ages = train_ages[start_idx:end_idx].to(device)
            batch_survival = train_survival[start_idx:end_idx].to(device)
            batch_events = train_events[start_idx:end_idx].to(device)
            
            # Forward pass
            output = model(batch_X, batch_ages)
            
            # Compute loss
            targets = {
                'chronological_age': batch_ages.squeeze(),
                'survival_time': batch_survival.squeeze(),
                'event_indicator': batch_events.squeeze()
            }
            loss, loss_components = criterion(output, targets, model)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(val_X.to(device), val_ages.to(device))
            val_loss = nn.functional.mse_loss(val_output.biological_age, val_ages.to(device))
            val_losses.append(val_loss.item())
            
            # Calculate MAE
            mae = torch.mean(torch.abs(val_output.biological_age - val_ages.to(device))).item()
        
        # Print progress
        print(f"   Epoch {epoch+1:3d}/{epochs}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={val_loss.item():.4f}, Val MAE={mae:.2f} years")
        
        # Save best model
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss.item(),
                'val_mae': mae,
                'config': config
            }, './checkpoints/best_simplified.pt')
            print(f"      ✓ New best model saved (MAE={mae:.2f} years)")
    
    print("\n4. Training completed!")
    print(f"   Best validation MAE: {mae:.2f} years")
    print(f"   Model saved to: ./checkpoints/best_simplified.pt")
    
    # Test the model
    print("\n5. Testing final model...")
    test_X = torch.tensor(((test_df[biomarker_cols] - mean) / std).values).float()
    test_ages = torch.tensor(test_df['age'].values).float().unsqueeze(1)
    
    model.eval()
    with torch.no_grad():
        test_output = model(test_X.to(device), test_ages.to(device))
        test_mae = torch.mean(torch.abs(test_output.biological_age - test_ages.to(device))).item()
        
        # Calculate age gaps
        age_gaps = test_output.biological_age.squeeze() - test_ages.squeeze().to(device)
        
    print(f"   Test MAE: {test_mae:.2f} years")
    print(f"   Mean age gap: {age_gaps.mean().item():.2f} years")
    print(f"   Std age gap: {age_gaps.std().item():.2f} years")
    
    return model, train_losses, val_losses

if __name__ == "__main__":
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train model
    model, train_losses, val_losses = train_henaw(config, epochs=20)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)