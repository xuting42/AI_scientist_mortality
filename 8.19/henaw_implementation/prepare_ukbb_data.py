#!/usr/bin/env python3
"""
UK Biobank Data Preparation Script
Extracts required biomarker fields from UK Biobank datasets and prepares them for HENAW model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import yaml
import logging
from typing import Dict, List, Optional, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from ukbb_data_loader import UKBBRealDataLoader, UKBBDataConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UKBBDataPreparator:
    """
    Prepares UK Biobank data for HENAW model training
    """
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize data preparator
        
        Args:
            config_path: Path to HENAW configuration file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize data loader
        self.loader = UKBBRealDataLoader(self.config)
        
        # Field mappings
        self.biomarker_fields = self.config['ukbb_fields']['biomarkers']
        self.age_field = self.config['ukbb_fields'].get('age', 21022)
        self.sex_field = self.config['ukbb_fields'].get('sex', 31)
        
        logger.info(f"Initialized with {len(self.biomarker_fields)} biomarkers")
    
    def prepare_from_ukbb_csv(self,
                             input_path: str,
                             output_path: str,
                             max_samples: Optional[int] = None,
                             require_complete: bool = False,
                             split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> None:
        """
        Prepare UK Biobank data from raw CSV file
        
        Args:
            input_path: Path to raw UK Biobank CSV
            output_path: Path to save processed data
            max_samples: Maximum number of samples to process
            require_complete: Whether to require all biomarkers present
            split_ratio: Train/val/test split ratios
        """
        logger.info(f"Processing UK Biobank data from {input_path}")
        
        # Load raw data
        df = self.loader.load_ukbb_csv(
            csv_path=input_path,
            required_complete=require_complete,
            max_samples=max_samples
        )
        
        logger.info(f"Loaded {len(df)} samples")
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Split data
        train_df, val_df, test_df = self._split_data(df, split_ratio)
        
        # Save processed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.h5':
            # Save as HDF5
            train_df.to_hdf(output_path, key='train', mode='w')
            val_df.to_hdf(output_path, key='val', mode='a')
            test_df.to_hdf(output_path, key='test', mode='a')
            logger.info(f"Saved to HDF5: {output_path}")
        else:
            # Save as separate CSV files
            base_path = output_path.with_suffix('')
            train_df.to_csv(f"{base_path}_train.csv", index=False)
            val_df.to_csv(f"{base_path}_val.csv", index=False)
            test_df.to_csv(f"{base_path}_test.csv", index=False)
            logger.info(f"Saved CSV files: {base_path}_[train/val/test].csv")
        
        # Print statistics
        self._print_statistics(train_df, val_df, test_df)
    
    def prepare_from_multiple_files(self,
                                   phenotype_file: str,
                                   biomarker_file: Optional[str] = None,
                                   output_path: str = 'processed_ukbb_data.h5') -> None:
        """
        Prepare data from multiple UK Biobank files
        
        Args:
            phenotype_file: Path to phenotype data
            biomarker_file: Optional path to biomarker data
            output_path: Path to save processed data
        """
        logger.info("Loading data from multiple files")
        
        # Load phenotype data
        pheno_df = pd.read_csv(phenotype_file, nrows=None)
        logger.info(f"Loaded {len(pheno_df)} samples from phenotype file")
        
        # Load biomarker data if separate
        if biomarker_file:
            bio_df = pd.read_csv(biomarker_file, nrows=None)
            logger.info(f"Loaded biomarker data from {biomarker_file}")
            
            # Merge on participant ID
            if 'eid' in pheno_df.columns and 'eid' in bio_df.columns:
                df = pd.merge(pheno_df, bio_df, on='eid', how='inner')
                logger.info(f"Merged to {len(df)} samples")
            else:
                logger.error("Cannot merge files without 'eid' column")
                return
        else:
            df = pheno_df
        
        # Process with loader
        df = self.loader._process_ukbb_data(df, required_complete=False)
        
        # Add derived features
        df = self._add_derived_features(df)
        
        # Split and save
        train_df, val_df, test_df = self._split_data(df)
        
        # Save processed data
        output_path = Path(output_path)
        if output_path.suffix == '.h5':
            train_df.to_hdf(output_path, key='train', mode='w')
            val_df.to_hdf(output_path, key='val', mode='a')
            test_df.to_hdf(output_path, key='test', mode='a')
        else:
            base_path = output_path.with_suffix('')
            train_df.to_csv(f"{base_path}_train.csv", index=False)
            val_df.to_csv(f"{base_path}_val.csv", index=False)
            test_df.to_csv(f"{base_path}_test.csv", index=False)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features for HENAW model
        
        Args:
            df: Input dataframe
        
        Returns:
            Dataframe with derived features
        """
        # Add biological age proxy (for supervised learning)
        # This would be replaced with actual mortality/morbidity outcomes
        if 'biological_age' not in df.columns:
            # Simple proxy based on biomarkers
            df['biological_age'] = df['age'].copy()
            
            # Adjust based on biomarker values
            if 'crp' in df.columns:
                # Higher CRP indicates inflammation/aging
                df['biological_age'] += (df['crp'] - 2.0) * 0.5
            
            if 'hba1c' in df.columns:
                # Higher HbA1c indicates metabolic aging
                df['biological_age'] += (df['hba1c'] - 36) * 0.2
            
            if 'albumin' in df.columns:
                # Lower albumin indicates aging
                df['biological_age'] -= (df['albumin'] - 45) * 0.3
        
        # Add age groups for stratification
        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 45, 50, 55, 60, 65, 70, 100],
                                 labels=['<45', '45-50', '50-55', '55-60', '60-65', '65-70', '70+'])
        
        # Add biomarker composite scores
        inflammation_markers = ['crp']
        metabolic_markers = ['hba1c', 'ggt', 'alt', 'ast']
        kidney_markers = ['creatinine', 'albumin']
        
        # Inflammation score
        if all(m in df.columns for m in inflammation_markers):
            df['inflammation_score'] = df[inflammation_markers].mean(axis=1)
        
        # Metabolic score
        available_metabolic = [m for m in metabolic_markers if m in df.columns]
        if available_metabolic:
            df['metabolic_score'] = df[available_metabolic].mean(axis=1)
        
        # Kidney function score
        available_kidney = [m for m in kidney_markers if m in df.columns]
        if available_kidney:
            df['kidney_score'] = df[available_kidney].mean(axis=1)
        
        return df
    
    def _split_data(self, 
                   df: pd.DataFrame,
                   split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets
        
        Args:
            df: Input dataframe
            split_ratio: Train/val/test split ratios
        
        Returns:
            Train, validation, and test dataframes
        """
        # Stratify by age group if available
        if 'age_group' in df.columns:
            # Sort by age group for stratified split
            df = df.sort_values('age_group')
        
        n_samples = len(df)
        train_end = int(n_samples * split_ratio[0])
        val_end = int(n_samples * (split_ratio[0] + split_ratio[1]))
        
        # Shuffle within splits
        train_df = df.iloc[:train_end].sample(frac=1, random_state=42)
        val_df = df.iloc[train_end:val_end].sample(frac=1, random_state=42)
        test_df = df.iloc[val_end:].sample(frac=1, random_state=42)
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _print_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Print statistics about the processed data
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
        """
        print("\n" + "="*60)
        print("UK BIOBANK DATA PREPARATION SUMMARY")
        print("="*60)
        
        print(f"\nDataset Sizes:")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        print(f"  Total: {len(train_df) + len(val_df) + len(test_df):,} samples")
        
        print(f"\nAge Distribution (Training Set):")
        print(f"  Mean: {train_df['age'].mean():.1f} years")
        print(f"  Std:  {train_df['age'].std():.1f} years")
        print(f"  Range: [{train_df['age'].min():.1f}, {train_df['age'].max():.1f}]")
        
        print(f"\nSex Distribution (Training Set):")
        sex_counts = train_df['sex'].value_counts()
        print(f"  Female: {sex_counts.get(0, 0):,} ({sex_counts.get(0, 0)/len(train_df)*100:.1f}%)")
        print(f"  Male:   {sex_counts.get(1, 0):,} ({sex_counts.get(1, 0)/len(train_df)*100:.1f}%)")
        
        print(f"\nBiomarker Completeness:")
        for name in self.biomarker_fields.keys():
            if name in train_df.columns:
                completeness = (train_df[name].notna().sum() / len(train_df)) * 100
                print(f"  {name:15s}: {completeness:.1f}% complete")
        
        print(f"\nMissing Data Summary:")
        missing_any = train_df[list(self.biomarker_fields.keys())].isna().any(axis=1).sum()
        complete_cases = len(train_df) - missing_any
        print(f"  Complete cases: {complete_cases:,} ({complete_cases/len(train_df)*100:.1f}%)")
        print(f"  Cases with missing data: {missing_any:,} ({missing_any/len(train_df)*100:.1f}%)")
        
        print("="*60)


def main():
    """Main entry point for data preparation"""
    parser = argparse.ArgumentParser(description='Prepare UK Biobank data for HENAW model')
    parser.add_argument('--input', type=str, help='Input UK Biobank CSV file')
    parser.add_argument('--phenotype', type=str, help='Phenotype file (if separate)')
    parser.add_argument('--biomarker', type=str, help='Biomarker file (if separate)')
    parser.add_argument('--output', type=str, default='processed_ukbb_data.h5',
                       help='Output file path (.h5 or .csv)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='HENAW configuration file')
    parser.add_argument('--max-samples', type=int, help='Maximum samples to process')
    parser.add_argument('--require-complete', action='store_true',
                       help='Only keep samples with all biomarkers')
    parser.add_argument('--split-ratio', type=float, nargs=3, default=[0.7, 0.15, 0.15],
                       help='Train/val/test split ratios')
    parser.add_argument('--generate-synthetic', action='store_true',
                       help='Generate synthetic data for testing')
    
    args = parser.parse_args()
    
    # Initialize preparator
    preparator = UKBBDataPreparator(args.config)
    
    if args.generate_synthetic:
        # Generate synthetic data for testing
        logger.info("Generating synthetic UK Biobank-like data")
        loader = UKBBRealDataLoader(preparator.config)
        df = loader._generate_synthetic_data(args.max_samples or 10000)
        
        # Add derived features
        df = preparator._add_derived_features(df)
        
        # Split and save
        train_df, val_df, test_df = preparator._split_data(df, tuple(args.split_ratio))
        
        output_path = Path(args.output)
        if output_path.suffix == '.h5':
            train_df.to_hdf(output_path, key='train', mode='w')
            val_df.to_hdf(output_path, key='val', mode='a')
            test_df.to_hdf(output_path, key='test', mode='a')
        else:
            base_path = output_path.with_suffix('')
            train_df.to_csv(f"{base_path}_train.csv", index=False)
            val_df.to_csv(f"{base_path}_val.csv", index=False)
            test_df.to_csv(f"{base_path}_test.csv", index=False)
        
        logger.info(f"Generated synthetic data saved to {output_path}")
        preparator._print_statistics(train_df, val_df, test_df)
        
    elif args.input:
        # Process single file
        preparator.prepare_from_ukbb_csv(
            input_path=args.input,
            output_path=args.output,
            max_samples=args.max_samples,
            require_complete=args.require_complete,
            split_ratio=tuple(args.split_ratio)
        )
    elif args.phenotype:
        # Process multiple files
        preparator.prepare_from_multiple_files(
            phenotype_file=args.phenotype,
            biomarker_file=args.biomarker,
            output_path=args.output
        )
    else:
        # Try to find data automatically
        logger.info("No input specified, searching for UK Biobank data...")
        loader = UKBBRealDataLoader(preparator.config)
        df = loader.load_ukbb_csv(max_samples=args.max_samples)
        
        # Process and save
        df = preparator._add_derived_features(df)
        train_df, val_df, test_df = preparator._split_data(df, tuple(args.split_ratio))
        
        output_path = Path(args.output)
        if output_path.suffix == '.h5':
            train_df.to_hdf(output_path, key='train', mode='w')
            val_df.to_hdf(output_path, key='val', mode='a')
            test_df.to_hdf(output_path, key='test', mode='a')
        else:
            base_path = output_path.with_suffix('')
            train_df.to_csv(f"{base_path}_train.csv", index=False)
            val_df.to_csv(f"{base_path}_val.csv", index=False)
            test_df.to_csv(f"{base_path}_test.csv", index=False)
        
        logger.info(f"Data saved to {output_path}")
        preparator._print_statistics(train_df, val_df, test_df)


if __name__ == "__main__":
    main()