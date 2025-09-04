"""
Data preprocessing utilities for UK Biobank biological age algorithms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.decomposition import PCA
from scipy import stats
import warnings
from dataclasses import dataclass
import torch


@dataclass
class PreprocessingStats:
    """Statistics from preprocessing for reproducibility."""
    
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    median: Optional[np.ndarray] = None
    mad: Optional[np.ndarray] = None  # Median absolute deviation
    min: Optional[np.ndarray] = None
    max: Optional[np.ndarray] = None
    missing_rates: Optional[np.ndarray] = None
    outlier_counts: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None


class UKBiobankPreprocessor:
    """Comprehensive preprocessor for UK Biobank data."""
    
    def __init__(
        self,
        normalization: str = "standard",
        imputation: str = "median",
        handle_outliers: bool = True,
        outlier_threshold: float = 5.0,
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            normalization: Normalization method ("standard", "robust", "minmax", None)
            imputation: Imputation strategy ("mean", "median", "knn", "mice", None)
            handle_outliers: Whether to handle outliers
            outlier_threshold: Z-score threshold for outlier detection
            feature_selection: Feature selection method ("variance", "mutual_info", "lasso", None)
            n_features: Number of features to select
        """
        self.normalization = normalization
        self.imputation = imputation
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.feature_selection = feature_selection
        self.n_features = n_features
        
        # Initialize transformers
        self.imputer = None
        self.scaler = None
        self.feature_selector = None
        self.stats = PreprocessingStats()
        
        self._init_transformers()
    
    def _init_transformers(self) -> None:
        """Initialize preprocessing transformers."""
        # Imputation
        if self.imputation == "mean":
            self.imputer = SimpleImputer(strategy="mean")
        elif self.imputation == "median":
            self.imputer = SimpleImputer(strategy="median")
        elif self.imputation == "knn":
            self.imputer = KNNImputer(n_neighbors=5)
        
        # Normalization
        if self.normalization == "standard":
            self.scaler = StandardScaler()
        elif self.normalization == "robust":
            self.scaler = RobustScaler()
        elif self.normalization == "minmax":
            self.scaler = MinMaxScaler()
        
        # Feature selection
        if self.feature_selection == "mutual_info":
            self.feature_selector = SelectKBest(
                mutual_info_regression,
                k=self.n_features or "all"
            )
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, 
            feature_names: Optional[List[str]] = None) -> 'UKBiobankPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            X: Input features
            y: Target values (for supervised feature selection)
            feature_names: Names of features
        
        Returns:
            Self
        """
        self.stats.feature_names = feature_names
        
        # Calculate statistics before preprocessing
        self.stats.mean = np.nanmean(X, axis=0)
        self.stats.std = np.nanstd(X, axis=0)
        self.stats.median = np.nanmedian(X, axis=0)
        self.stats.mad = np.nanmedian(np.abs(X - self.stats.median), axis=0)
        self.stats.min = np.nanmin(X, axis=0)
        self.stats.max = np.nanmax(X, axis=0)
        self.stats.missing_rates = np.mean(np.isnan(X), axis=0)
        
        # Handle outliers
        if self.handle_outliers:
            X = self._handle_outliers(X)
        
        # Imputation
        if self.imputer:
            X = self.imputer.fit_transform(X)
        
        # Normalization
        if self.scaler:
            X = self.scaler.fit_transform(X)
        
        # Feature selection
        if self.feature_selector and y is not None:
            self.feature_selector.fit(X, y)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Input features
        
        Returns:
            Transformed features
        """
        # Handle outliers
        if self.handle_outliers:
            X = self._handle_outliers(X)
        
        # Imputation
        if self.imputer:
            X = self.imputer.transform(X)
        
        # Normalization
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Feature selection
        if self.feature_selector:
            X = self.feature_selector.transform(X)
        
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _handle_outliers(self, X: np.ndarray) -> np.ndarray:
        """Handle outliers using z-score method."""
        X_clean = X.copy()
        
        # Calculate z-scores
        z_scores = np.abs(stats.zscore(X, nan_policy='omit'))
        
        # Count outliers
        outlier_mask = z_scores > self.outlier_threshold
        self.stats.outlier_counts = np.sum(outlier_mask, axis=0)
        
        # Cap outliers at threshold
        X_clean[outlier_mask] = np.nan
        
        return X_clean
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform normalized data."""
        if self.scaler:
            return self.scaler.inverse_transform(X)
        return X


class BiomarkerProcessor:
    """Specialized processor for blood biomarkers."""
    
    # Reference ranges for common biomarkers (for validation)
    REFERENCE_RANGES = {
        'albumin': (35, 50),  # g/L
        'alkaline_phosphatase': (30, 130),  # U/L
        'creatinine': (60, 120),  # μmol/L
        'c_reactive_protein': (0, 10),  # mg/L
        'glucose': (3.9, 7.8),  # mmol/L
        'hemoglobin': (120, 180),  # g/L
        'white_blood_cell_count': (4, 11),  # 10^9/L
        'systolic_blood_pressure': (90, 140),  # mmHg
        'diastolic_blood_pressure': (60, 90),  # mmHg
        'bmi': (18.5, 30),  # kg/m²
    }
    
    def __init__(self, validate_ranges: bool = True):
        """
        Initialize biomarker processor.
        
        Args:
            validate_ranges: Whether to validate against reference ranges
        """
        self.validate_ranges = validate_ranges
        self.out_of_range_counts = {}
    
    def process(self, biomarkers: pd.DataFrame) -> pd.DataFrame:
        """
        Process biomarker data.
        
        Args:
            biomarkers: DataFrame with biomarker values
        
        Returns:
            Processed biomarkers
        """
        processed = biomarkers.copy()
        
        # Validate ranges
        if self.validate_ranges:
            for col in processed.columns:
                if col in self.REFERENCE_RANGES:
                    min_val, max_val = self.REFERENCE_RANGES[col]
                    out_of_range = (processed[col] < min_val) | (processed[col] > max_val)
                    self.out_of_range_counts[col] = out_of_range.sum()
                    
                    if out_of_range.sum() > len(processed) * 0.1:
                        warnings.warn(f"More than 10% of {col} values are out of reference range")
        
        # Log-transform skewed biomarkers
        skewed_biomarkers = ['c_reactive_protein', 'triglycerides', 'gamma_gt']
        for col in skewed_biomarkers:
            if col in processed.columns:
                processed[col] = np.log1p(processed[col])
        
        # Create derived features
        if 'systolic_blood_pressure' in processed.columns and 'diastolic_blood_pressure' in processed.columns:
            processed['pulse_pressure'] = processed['systolic_blood_pressure'] - processed['diastolic_blood_pressure']
            processed['mean_arterial_pressure'] = (processed['systolic_blood_pressure'] + 
                                                   2 * processed['diastolic_blood_pressure']) / 3
        
        return processed


class OCTImageProcessor:
    """Processor for OCT retinal images."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        extract_features: bool = False
    ):
        """
        Initialize OCT processor.
        
        Args:
            target_size: Target image size
            normalize: Whether to normalize images
            extract_features: Whether to extract handcrafted features
        """
        self.target_size = target_size
        self.normalize = normalize
        self.extract_features = extract_features
    
    def process_image(self, image: np.ndarray) -> Union[np.ndarray, Dict[str, float]]:
        """
        Process single OCT image.
        
        Args:
            image: Input OCT image
        
        Returns:
            Processed image or extracted features
        """
        if self.extract_features:
            return self._extract_features(image)
        
        # Resize if needed
        if image.shape[:2] != self.target_size:
            from skimage.transform import resize
            image = resize(image, self.target_size, preserve_range=True)
        
        # Normalize
        if self.normalize:
            image = (image - image.mean()) / (image.std() + 1e-8)
        
        return image
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract handcrafted features from OCT image."""
        features = {}
        
        # Basic statistics
        features['mean_intensity'] = np.mean(image)
        features['std_intensity'] = np.std(image)
        features['max_intensity'] = np.max(image)
        features['min_intensity'] = np.min(image)
        
        # Texture features (simplified)
        features['contrast'] = np.std(image) ** 2
        features['energy'] = np.sum(image ** 2)
        features['homogeneity'] = 1 / (1 + np.var(image))
        
        # Layer thickness measurements (placeholder)
        features['retinal_thickness'] = np.random.uniform(200, 300)
        features['rnfl_thickness'] = np.random.uniform(80, 120)
        
        return features


class MetabolomicsProcessor:
    """Processor for NMR metabolomics data."""
    
    # Metabolite groups for feature engineering
    METABOLITE_GROUPS = {
        'lipids': ['ldl', 'hdl', 'vldl', 'triglycerides', 'cholesterol'],
        'amino_acids': ['alanine', 'glutamine', 'glycine', 'histidine', 'isoleucine',
                       'leucine', 'valine', 'phenylalanine', 'tyrosine'],
        'glycolysis': ['glucose', 'lactate', 'pyruvate', 'citrate'],
        'ketones': ['acetate', 'acetoacetate', 'beta_hydroxybutyrate'],
        'inflammation': ['glycoprotein_acetyls', 'creatinine', 'albumin']
    }
    
    def __init__(self, create_ratios: bool = True, group_features: bool = True):
        """
        Initialize metabolomics processor.
        
        Args:
            create_ratios: Whether to create metabolite ratios
            group_features: Whether to create group-level features
        """
        self.create_ratios = create_ratios
        self.group_features = group_features
    
    def process(self, metabolomics: pd.DataFrame) -> pd.DataFrame:
        """
        Process metabolomics data.
        
        Args:
            metabolomics: DataFrame with metabolite concentrations
        
        Returns:
            Processed metabolomics with engineered features
        """
        processed = metabolomics.copy()
        
        # Create metabolite ratios
        if self.create_ratios:
            processed = self._create_ratios(processed)
        
        # Create group-level features
        if self.group_features:
            processed = self._create_group_features(processed)
        
        # Log-transform to handle skewness
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        processed[numeric_cols] = np.log1p(processed[numeric_cols])
        
        return processed
    
    def _create_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create biologically meaningful metabolite ratios."""
        # HDL/LDL ratio (cardiovascular risk)
        if 'hdl' in df.columns and 'ldl' in df.columns:
            df['hdl_ldl_ratio'] = df['hdl'] / (df['ldl'] + 1e-8)
        
        # Glucose/lactate ratio (metabolic efficiency)
        if 'glucose' in df.columns and 'lactate' in df.columns:
            df['glucose_lactate_ratio'] = df['glucose'] / (df['lactate'] + 1e-8)
        
        # Branched-chain amino acids ratio
        bcaa_cols = ['leucine', 'isoleucine', 'valine']
        if all(col in df.columns for col in bcaa_cols):
            df['bcaa_total'] = df[bcaa_cols].sum(axis=1)
        
        return df
    
    def _create_group_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features for metabolite groups."""
        for group_name, metabolites in self.METABOLITE_GROUPS.items():
            group_cols = [col for col in metabolites if col in df.columns]
            if group_cols:
                df[f'{group_name}_mean'] = df[group_cols].mean(axis=1)
                df[f'{group_name}_std'] = df[group_cols].std(axis=1)
                df[f'{group_name}_sum'] = df[group_cols].sum(axis=1)
        
        return df


class TemporalProcessor:
    """Processor for longitudinal/temporal data."""
    
    def __init__(
        self,
        method: str = "difference",
        window_size: int = 3,
        handle_missing: str = "interpolate"
    ):
        """
        Initialize temporal processor.
        
        Args:
            method: Processing method ("difference", "slope", "moving_average")
            window_size: Window size for moving operations
            handle_missing: How to handle missing timepoints
        """
        self.method = method
        self.window_size = window_size
        self.handle_missing = handle_missing
    
    def process_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """
        Process temporal sequences.
        
        Args:
            sequences: 3D array (batch, time, features)
        
        Returns:
            Processed sequences with temporal features
        """
        batch_size, seq_len, n_features = sequences.shape
        
        # Handle missing values
        if self.handle_missing == "interpolate":
            sequences = self._interpolate_missing(sequences)
        
        if self.method == "difference":
            # Calculate differences between consecutive timepoints
            diffs = np.diff(sequences, axis=1)
            # Pad to maintain sequence length
            diffs = np.concatenate([np.zeros((batch_size, 1, n_features)), diffs], axis=1)
            return np.concatenate([sequences, diffs], axis=-1)
        
        elif self.method == "slope":
            # Calculate slopes over time
            slopes = np.zeros_like(sequences)
            for i in range(1, seq_len):
                slopes[:, i, :] = (sequences[:, i, :] - sequences[:, 0, :]) / i
            return np.concatenate([sequences, slopes], axis=-1)
        
        elif self.method == "moving_average":
            # Calculate moving averages
            ma = np.zeros_like(sequences)
            for i in range(seq_len):
                start_idx = max(0, i - self.window_size + 1)
                ma[:, i, :] = np.mean(sequences[:, start_idx:i+1, :], axis=1)
            return np.concatenate([sequences, ma], axis=-1)
        
        return sequences
    
    def _interpolate_missing(self, sequences: np.ndarray) -> np.ndarray:
        """Interpolate missing values in sequences."""
        from scipy.interpolate import interp1d
        
        batch_size, seq_len, n_features = sequences.shape
        filled = sequences.copy()
        
        for b in range(batch_size):
            for f in range(n_features):
                series = sequences[b, :, f]
                if np.any(np.isnan(series)):
                    valid_idx = ~np.isnan(series)
                    if np.sum(valid_idx) >= 2:
                        # Interpolate
                        x = np.arange(seq_len)[valid_idx]
                        y = series[valid_idx]
                        interp = interp1d(x, y, kind='linear', fill_value='extrapolate')
                        filled[b, :, f] = interp(np.arange(seq_len))
        
        return filled


def create_preprocessor(config: Any) -> UKBiobankPreprocessor:
    """
    Create preprocessor from configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Configured preprocessor
    """
    return UKBiobankPreprocessor(
        normalization=config.normalize_features if hasattr(config, 'normalize_features') else "standard",
        imputation=config.imputation_strategy if hasattr(config, 'imputation_strategy') else "median",
        handle_outliers=config.handle_outliers if hasattr(config, 'handle_outliers') else True,
        outlier_threshold=config.outlier_threshold if hasattr(config, 'outlier_threshold') else 5.0,
        feature_selection=config.feature_selection_method if hasattr(config, 'feature_selection_method') else None,
        n_features=config.n_features_to_select if hasattr(config, 'n_features_to_select') else None
    )