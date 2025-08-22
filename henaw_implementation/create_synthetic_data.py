#!/usr/bin/env python
"""
Create synthetic dataset for HENAW model demonstration
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import h5py

def generate_synthetic_ukbb_data(n_samples=10000, output_dir='./data'):
    """
    Generate synthetic UK Biobank-like data for demonstration
    """
    print("=" * 60)
    print("GENERATING SYNTHETIC UK BIOBANK-LIKE DATA")
    print("=" * 60)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    np.random.seed(42)
    
    # Generate participant IDs
    participant_ids = np.arange(1000000, 1000000 + n_samples)
    
    # Generate demographics
    ages = np.random.normal(55, 10, n_samples)
    ages = np.clip(ages, 40, 70)
    sex = np.random.binomial(1, 0.5, n_samples)  # 0: Female, 1: Male
    
    # Generate biomarkers with realistic distributions
    biomarkers = {}
    
    # C-reactive protein (mg/L) - log-normal distribution
    biomarkers['crp'] = np.random.lognormal(0.7, 0.8, n_samples)
    biomarkers['crp'] = np.clip(biomarkers['crp'], 0.1, 10)
    
    # HbA1c (mmol/mol) - normal distribution
    biomarkers['hba1c'] = np.random.normal(36, 6, n_samples)
    biomarkers['hba1c'] = np.clip(biomarkers['hba1c'], 20, 120)
    
    # Creatinine (μmol/L) - normal distribution
    biomarkers['creatinine'] = np.random.normal(70, 15, n_samples)
    biomarkers['creatinine'] = np.clip(biomarkers['creatinine'], 40, 150)
    
    # Albumin (g/L) - normal distribution
    biomarkers['albumin'] = np.random.normal(45, 3, n_samples)
    biomarkers['albumin'] = np.clip(biomarkers['albumin'], 35, 50)
    
    # Lymphocyte percentage (%) - normal distribution
    biomarkers['lymphocyte_pct'] = np.random.normal(30, 7, n_samples)
    biomarkers['lymphocyte_pct'] = np.clip(biomarkers['lymphocyte_pct'], 15, 45)
    
    # Red cell distribution width (%) - normal distribution
    biomarkers['rdw'] = np.random.normal(13, 1, n_samples)
    biomarkers['rdw'] = np.clip(biomarkers['rdw'], 11, 16)
    
    # Gamma glutamyltransferase (U/L) - log-normal distribution
    biomarkers['ggt'] = np.random.lognormal(3.4, 0.7, n_samples)
    biomarkers['ggt'] = np.clip(biomarkers['ggt'], 10, 300)
    
    # Aspartate aminotransferase (U/L) - normal distribution
    biomarkers['ast'] = np.random.normal(25, 10, n_samples)
    biomarkers['ast'] = np.clip(biomarkers['ast'], 10, 100)
    
    # Alanine aminotransferase (U/L) - normal distribution
    biomarkers['alt'] = np.random.normal(30, 12, n_samples)
    biomarkers['alt'] = np.clip(biomarkers['alt'], 10, 100)
    
    # Add some correlation with age and sex
    for marker in biomarkers:
        # Add age effect (older = higher values for most markers)
        age_effect = (ages - 55) * np.random.normal(0.1, 0.05)
        biomarkers[marker] += age_effect
        
        # Add sex effect
        if marker in ['creatinine', 'alt', 'ast']:  # Higher in males
            biomarkers[marker] += sex * np.random.normal(5, 2)
        elif marker == 'albumin':  # Slightly lower in females
            biomarkers[marker] -= (1 - sex) * np.random.normal(1, 0.5)
    
    # Generate synthetic survival data
    # Hazard increases with age and certain biomarker levels
    base_hazard = 0.001
    hazard = base_hazard * np.exp(
        0.05 * (ages - 55) +  # Age effect
        0.1 * (biomarkers['crp'] > 3) +  # High CRP
        0.1 * (biomarkers['hba1c'] > 42) +  # High HbA1c
        0.1 * (biomarkers['creatinine'] > 90)  # High creatinine
    )
    
    # Generate survival times (exponential distribution)
    survival_times = np.random.exponential(1 / hazard)
    survival_times = np.clip(survival_times, 0.1, 10)  # Follow-up limited to 10 years
    
    # Generate event indicators (1 = event occurred, 0 = censored)
    event_indicators = np.random.binomial(1, 0.2, n_samples)  # 20% event rate
    
    # Create DataFrame
    df = pd.DataFrame({
        'participant_id': participant_ids,
        'age': ages,
        'sex': sex,
        **biomarkers,
        'survival_time': survival_times,
        'event_indicator': event_indicators
    })
    
    # Save as HDF5 (UK Biobank format)
    h5_path = os.path.join(output_dir, 'synthetic_ukbb.h5')
    with h5py.File(h5_path, 'w') as f:
        # Save each column as a dataset
        for col in df.columns:
            f.create_dataset(col, data=df[col].values)
        
        # Add metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['created'] = pd.Timestamp.now().isoformat()
        f.attrs['description'] = 'Synthetic UK Biobank-like data for HENAW model demonstration'
    
    # Also save as CSV for easy inspection
    csv_path = os.path.join(output_dir, 'synthetic_ukbb.csv')
    df.to_csv(csv_path, index=False)
    
    # Create train/val/test splits
    n_train = int(0.6 * n_samples)
    n_val = int(0.2 * n_samples)
    n_test = n_samples - n_train - n_val
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    # Save splits
    splits = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    
    np.savez(os.path.join(output_dir, 'splits.npz'), **splits)
    
    # Print summary statistics
    print(f"\nDataset created successfully!")
    print(f"  Total samples: {n_samples:,}")
    print(f"  Train samples: {n_train:,}")
    print(f"  Val samples: {n_val:,}")
    print(f"  Test samples: {n_test:,}")
    print(f"\nFiles created:")
    print(f"  - {h5_path}")
    print(f"  - {csv_path}")
    print(f"  - {os.path.join(output_dir, 'splits.npz')}")
    
    print(f"\nBiomarker summary statistics:")
    print("-" * 40)
    for marker in ['crp', 'hba1c', 'creatinine', 'albumin', 'lymphocyte_pct', 
                   'rdw', 'ggt', 'ast', 'alt']:
        values = df[marker].values
        print(f"  {marker:15s}: mean={np.mean(values):6.2f}, std={np.std(values):6.2f}, "
              f"range=[{np.min(values):6.2f}, {np.max(values):6.2f}]")
    
    print(f"\nDemographics:")
    print(f"  Age: mean={np.mean(ages):.1f}, std={np.std(ages):.1f}")
    print(f"  Sex: {np.sum(sex == 0)} females, {np.sum(sex == 1)} males")
    print(f"  Events: {np.sum(event_indicators)} ({100*np.mean(event_indicators):.1f}%)")
    
    return df

if __name__ == "__main__":
    # Generate synthetic data
    df = generate_synthetic_ukbb_data(n_samples=10000)
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)