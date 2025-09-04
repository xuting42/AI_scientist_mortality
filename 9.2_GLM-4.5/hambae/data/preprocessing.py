"""
Data preprocessing pipeline for HAMBAE algorithm system.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.spatial.distance import mahalanobis
import logging
from dataclasses import dataclass
from ..config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of data preprocessing."""
    processed_data: Dict[str, np.ndarray]
    scalers: Dict[str, Any]
    imputers: Dict[str, Any]
    outlier_masks: Dict[str, np.ndarray]
    quality_scores: Dict[str, np.ndarray]
    preprocessing_stats: Dict[str, Any]


class DataPreprocessor:
    """
    Data preprocessing pipeline for HAMBAE algorithm system.
    
    This class handles data normalization, missing value imputation,
    outlier detection, and quality control for multi-modal data.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize data preprocessor.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.scalers = {}
        self.imputers = {}
        self.outlier_detectors = {}
        self.quality_thresholds = {}
        
        # Initialize preprocessing components
        self._initialize_preprocessors()
    
    def _initialize_preprocessors(self) -> None:
        """Initialize preprocessing components."""
        # Initialize scalers
        if self.config.normalization_method == "standard":
            self.scalers['blood_biomarkers'] = StandardScaler()
            self.scalers['metabolomics'] = StandardScaler()
            self.scalers['retinal_features'] = StandardScaler()
            self.scalers['genetic_features'] = StandardScaler()
        elif self.config.normalization_method == "robust":
            self.scalers['blood_biomarkers'] = RobustScaler()
            self.scalers['metabolomics'] = RobustScaler()
            self.scalers['retinal_features'] = RobustScaler()
            self.scalers['genetic_features'] = RobustScaler()
        elif self.config.normalization_method == "minmax":
            self.scalers['blood_biomarkers'] = MinMaxScaler()
            self.scalers['metabolomics'] = MinMaxScaler()
            self.scalers['retinal_features'] = MinMaxScaler()
            self.scalers['genetic_features'] = MinMaxScaler()
        
        # Initialize imputers
        if self.config.handle_missing == "median":
            self.imputers['blood_biomarkers'] = SimpleImputer(strategy='median')
            self.imputers['metabolomics'] = SimpleImputer(strategy='median')
            self.imputers['retinal_features'] = SimpleImputer(strategy='median')
            self.imputers['genetic_features'] = SimpleImputer(strategy='median')
        elif self.config.handle_missing == "mean":
            self.imputers['blood_biomarkers'] = SimpleImputer(strategy='mean')
            self.imputers['metabolomics'] = SimpleImputer(strategy='mean')
            self.imputers['retinal_features'] = SimpleImputer(strategy='mean')
            self.imputers['genetic_features'] = SimpleImputer(strategy='mean')
        elif self.config.handle_missing == "knn":
            self.imputers['blood_biomarkers'] = KNNImputer(n_neighbors=5)
            self.imputers['metabolomics'] = KNNImputer(n_neighbors=5)
            self.imputers['retinal_features'] = KNNImputer(n_neighbors=5)
            self.imputers['genetic_features'] = KNNImputer(n_neighbors=5)
        
        # Initialize outlier detectors
        if self.config.outlier_method == "isolation_forest":
            self.outlier_detectors['blood_biomarkers'] = IsolationForest(
                contamination=0.1, random_state=42
            )
            self.outlier_detectors['metabolomics'] = IsolationForest(
                contamination=0.1, random_state=42
            )
            self.outlier_detectors['retinal_features'] = IsolationForest(
                contamination=0.1, random_state=42
            )
            self.outlier_detectors['genetic_features'] = IsolationForest(
                contamination=0.1, random_state=42
            )
    
    def fit(self, data: Dict[str, np.ndarray]) -> 'DataPreprocessor':
        """
        Fit preprocessing components on training data.
        
        Args:
            data: Dictionary of modality-specific data arrays
            
        Returns:
            Fitted preprocessor
        """
        logger.info("Fitting preprocessing components")
        
        preprocessing_stats = {}
        
        for modality, modality_data in data.items():
            if modality_data is None or len(modality_data) == 0:
                continue
            
            logger.info(f"Fitting preprocessor for {modality}")
            
            # Handle missing values
            if np.isnan(modality_data).any():
                logger.info(f"Imputing missing values for {modality}")
                self.imputers[modality].fit(modality_data)
            
            # Fit scaler
            if modality in self.scalers:
                logger.info(f"Fitting scaler for {modality}")
                self.scalers[modality].fit(modality_data)
            
            # Fit outlier detector
            if modality in self.outlier_detectors:
                logger.info(f"Fitting outlier detector for {modality}")
                self.outlier_detectors[modality].fit(modality_data)
            
            # Compute statistics
            preprocessing_stats[modality] = {
                'n_samples': len(modality_data),
                'n_features': modality_data.shape[1] if len(modality_data.shape) > 1 else 1,
                'missing_rate': np.isnan(modality_data).mean(),
                'mean': np.nanmean(modality_data),
                'std': np.nanstd(modality_data),
                'min': np.nanmin(modality_data),
                'max': np.nanmax(modality_data),
            }
        
        self.preprocessing_stats = preprocessing_stats
        logger.info("Preprocessing components fitted successfully")
        
        return self
    
    def transform(self, data: Dict[str, np.ndarray]) -> PreprocessingResult:
        """
        Transform data using fitted preprocessing components.
        
        Args:
            data: Dictionary of modality-specific data arrays
            
        Returns:
            Preprocessing result with processed data and metadata
        """
        logger.info("Transforming data")
        
        processed_data = {}
        outlier_masks = {}
        quality_scores = {}
        
        for modality, modality_data in data.items():
            if modality_data is None or len(modality_data) == 0:
                continue
            
            logger.info(f"Processing {modality}")
            
            # Handle missing values
            if np.isnan(modality_data).any():
                logger.info(f"Imputing missing values for {modality}")
                modality_data = self.imputers[modality].transform(modality_data)
            
            # Detect outliers
            if modality in self.outlier_detectors:
                logger.info(f"Detecting outliers for {modality}")
                outlier_labels = self.outlier_detectors[modality].predict(modality_data)
                outlier_masks[modality] = (outlier_labels == -1)
                
                # Additional outlier detection using z-score
                if self.config.outlier_method == "zscore":
                    z_scores = np.abs(stats.zscore(modality_data, nan_policy='omit'))
                    z_outliers = z_scores > self.config.outlier_threshold
                    outlier_masks[modality] = outlier_masks[modality] | z_outliers
                elif self.config.outlier_method == "iqr":
                    q1 = np.percentile(modality_data, 25, axis=0)
                    q3 = np.percentile(modality_data, 75, axis=0)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    iqr_outliers = (modality_data < lower_bound) | (modality_data > upper_bound)
                    outlier_masks[modality] = outlier_masks[modality] | iqr_outliers.any(axis=1)
            
            # Normalize data
            if modality in self.scalers:
                logger.info(f"Normalizing {modality}")
                modality_data = self.scalers[modality].transform(modality_data)
            
            # Compute quality scores
            quality_scores[modality] = self._compute_quality_scores(
                modality_data, outlier_masks.get(modality, None)
            )
            
            processed_data[modality] = modality_data
        
        # Apply batch effect correction if enabled
        if self.config.correct_batch_effects:
            processed_data = self._correct_batch_effects(processed_data)
        
        result = PreprocessingResult(
            processed_data=processed_data,
            scalers=self.scalers,
            imputers=self.imputers,
            outlier_masks=outlier_masks,
            quality_scores=quality_scores,
            preprocessing_stats=getattr(self, 'preprocessing_stats', {}),
        )
        
        logger.info("Data transformation completed")
        
        return result
    
    def fit_transform(self, data: Dict[str, np.ndarray]) -> PreprocessingResult:
        """
        Fit preprocessing components and transform data.
        
        Args:
            data: Dictionary of modality-specific data arrays
            
        Returns:
            Preprocessing result with processed data and metadata
        """
        return self.fit(data).transform(data)
    
    def _compute_quality_scores(
        self, 
        data: np.ndarray, 
        outlier_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute quality scores for each sample.
        
        Args:
            data: Data array
            outlier_mask: Optional outlier mask
            
        Returns:
            Quality scores for each sample
        """
        n_samples = data.shape[0]
        quality_scores = np.ones(n_samples)
        
        # Penalty for missing values (should be handled by imputation)
        missing_penalty = np.isnan(data).mean(axis=1)
        quality_scores -= 0.5 * missing_penalty
        
        # Penalty for outliers
        if outlier_mask is not None:
            quality_scores[outlier_mask] *= 0.5
        
        # Compute Mahalanobis distance quality
        try:
            # Compute covariance matrix
            cov_matrix = np.cov(data, rowvar=False)
            if np.linalg.det(cov_matrix) != 0:
                mean_vector = np.mean(data, axis=0)
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                
                # Compute Mahalanobis distances
                mahal_distances = []
                for i in range(n_samples):
                    distance = mahalanobis(
                        data[i], mean_vector, inv_cov_matrix
                    )
                    mahal_distances.append(distance)
                
                mahal_distances = np.array(mahal_distances)
                
                # Convert to quality scores (higher distance = lower quality)
                mahal_quality = 1 / (1 + mahal_distances / np.median(mahal_distances))
                quality_scores *= mahal_quality
        except Exception as e:
            logger.warning(f"Could not compute Mahalanobis distance: {e}")
        
        # Ensure quality scores are in [0, 1]
        quality_scores = np.clip(quality_scores, 0, 1)
        
        return quality_scores
    
    def _correct_batch_effects(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Correct batch effects in data.
        
        Args:
            data: Dictionary of modality-specific data arrays
            
        Returns:
            Batch-corrected data
        """
        # This is a simplified implementation
        # In practice, you would use more sophisticated batch correction methods
        # like ComBat, SVA, or RUV
        
        logger.info("Applying batch effect correction")
        
        corrected_data = {}
        for modality, modality_data in data.items():
            # Simple batch correction: remove batch mean
            # In practice, you would need batch information
            batch_means = np.mean(modality_data, axis=0)
            corrected_data[modality] = modality_data - batch_means
        
        return corrected_data
    
    def inverse_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Inverse transform normalized data back to original scale.
        
        Args:
            data: Dictionary of normalized data arrays
            
        Returns:
            Original scale data
        """
        original_data = {}
        
        for modality, modality_data in data.items():
            if modality in self.scalers:
                original_data[modality] = self.scalers[modality].inverse_transform(modality_data)
            else:
                original_data[modality] = modality_data
        
        return original_data
    
    def save_preprocessors(self, save_path: str) -> None:
        """
        Save fitted preprocessing components.
        
        Args:
            save_path: Path to save preprocessing components
        """
        import joblib
        
        preprocessors = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'outlier_detectors': self.outlier_detectors,
            'quality_thresholds': self.quality_thresholds,
            'config': self.config,
        }
        
        joblib.dump(preprocessors, save_path)
        logger.info(f"Preprocessing components saved to {save_path}")
    
    def load_preprocessors(self, load_path: str) -> None:
        """
        Load fitted preprocessing components.
        
        Args:
            load_path: Path to load preprocessing components
        """
        import joblib
        
        preprocessors = joblib.load(load_path)
        self.scalers = preprocessors['scalers']
        self.imputers = preprocessors['imputers']
        self.outlier_detectors = preprocessors['outlier_detectors']
        self.quality_thresholds = preprocessors['quality_thresholds']
        
        logger.info(f"Preprocessing components loaded from {load_path}")


def create_preprocessor(config: DataConfig) -> DataPreprocessor:
    """
    Create a data preprocessor with the given configuration.
    
    Args:
        config: Data configuration
        
    Returns:
        Configured data preprocessor
    """
    return DataPreprocessor(config)


def preprocess_ukbb_data(
    data: Dict[str, np.ndarray],
    config: DataConfig,
    fit: bool = True,
) -> PreprocessingResult:
    """
    Preprocess UK Biobank data.
    
    Args:
        data: Dictionary of modality-specific data arrays
        config: Data configuration
        fit: Whether to fit preprocessing components
        
    Returns:
        Preprocessing result
    """
    preprocessor = DataPreprocessor(config)
    
    if fit:
        result = preprocessor.fit_transform(data)
    else:
        result = preprocessor.transform(data)
    
    return result