"""
Quality control for HAMBAE algorithm system.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from scipy import stats
from scipy.spatial.distance import mahalanobis
import logging
from dataclasses import dataclass
from ..config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class QualityControlResult:
    """Result of quality control."""
    quality_scores: Dict[str, np.ndarray]
    outlier_flags: Dict[str, np.ndarray]
    quality_flags: Dict[str, np.ndarray]
    quality_stats: Dict[str, Any]
    recommendations: Dict[str, List[str]]


class QualityController:
    """
    Quality control for HAMBAE algorithm system.
    
    This class handles data quality assessment, outlier detection,
    and quality flagging for multi-modal data.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize quality controller.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.quality_thresholds = {}
        self.outlier_detectors = {}
        self.quality_stats = {}
        
        # Initialize quality control components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize quality control components."""
        # Set quality thresholds
        self.quality_thresholds = {
            'min_data_quality': self.config.min_data_quality,
            'max_missing_rate': self.config.max_missing_rate,
            'min_samples': 100,
            'min_features': 5,
            'max_outlier_rate': 0.1,
            'min_correlation': 0.1,
            'max_variance_inflation': 10.0,
        }
    
    def assess_quality(self, data: Dict[str, np.ndarray]) -> QualityControlResult:
        """
        Assess data quality for all modalities.
        
        Args:
            data: Dictionary of modality-specific data arrays
            
        Returns:
            Quality control result
        """
        logger.info("Assessing data quality")
        
        quality_scores = {}
        outlier_flags = {}
        quality_flags = {}
        quality_stats = {}
        recommendations = {}
        
        for modality, modality_data in data.items():
            if modality_data is None or len(modality_data) == 0:
                continue
            
            logger.info(f"Assessing quality for {modality}")
            
            # Compute quality scores
            modality_scores = self._compute_modality_quality_scores(modality_data)
            quality_scores[modality] = modality_scores
            
            # Detect outliers
            modality_outliers = self._detect_outliers(modality_data)
            outlier_flags[modality] = modality_outliers
            
            # Generate quality flags
            modality_flags = self._generate_quality_flags(
                modality_data, modality_scores, modality_outliers
            )
            quality_flags[modality] = modality_flags
            
            # Compute quality statistics
            modality_stats = self._compute_quality_statistics(
                modality_data, modality_scores, modality_outliers
            )
            quality_stats[modality] = modality_stats
            
            # Generate recommendations
            modality_recommendations = self._generate_recommendations(
                modality_data, modality_stats
            )
            recommendations[modality] = modality_recommendations
        
        # Generate overall quality assessment
        overall_stats = self._compute_overall_quality_stats(quality_stats)
        quality_stats['overall'] = overall_stats
        
        # Generate overall recommendations
        overall_recommendations = self._generate_overall_recommendations(quality_stats)
        recommendations['overall'] = overall_recommendations
        
        result = QualityControlResult(
            quality_scores=quality_scores,
            outlier_flags=outlier_flags,
            quality_flags=quality_flags,
            quality_stats=quality_stats,
            recommendations=recommendations,
        )
        
        logger.info("Quality assessment completed")
        
        return result
    
    def _compute_modality_quality_scores(self, data: np.ndarray) -> np.ndarray:
        """
        Compute quality scores for each sample in a modality.
        
        Args:
            data: Data array
            
        Returns:
            Quality scores for each sample
        """
        n_samples = data.shape[0]
        quality_scores = np.ones(n_samples)
        
        # Missing data penalty
        missing_rates = np.isnan(data).mean(axis=1)
        missing_penalty = np.clip(missing_rates / self.quality_thresholds['max_missing_rate'], 0, 1)
        quality_scores -= 0.3 * missing_penalty
        
        # Outlier penalty
        try:
            # Mahalanobis distance
            cov_matrix = np.cov(data, rowvar=False)
            if np.linalg.det(cov_matrix) != 0:
                mean_vector = np.mean(data, axis=0)
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                
                mahal_distances = []
                for i in range(n_samples):
                    distance = mahalanobis(data[i], mean_vector, inv_cov_matrix)
                    mahal_distances.append(distance)
                
                mahal_distances = np.array(mahal_distances)
                mahal_scores = 1 / (1 + mahal_distances / np.median(mahal_distances))
                quality_scores *= mahal_scores
        except Exception as e:
            logger.warning(f"Could not compute Mahalanobis distance: {e}")
        
        # Data consistency penalty
        if data.shape[1] > 1:
            # Check for internal consistency
            correlations = np.corrcoef(data.T)
            avg_correlation = np.mean(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
            consistency_score = max(0, (avg_correlation - self.quality_thresholds['min_correlation']) / 
                                   (1 - self.quality_thresholds['min_correlation']))
            quality_scores *= (0.7 + 0.3 * consistency_score)
        
        # Feature range penalty
        feature_ranges = np.nanmax(data, axis=0) - np.nanmin(data, axis=0)
        range_penalty = np.mean(feature_ranges == 0)  # Penalty for constant features
        quality_scores *= (1 - 0.2 * range_penalty)
        
        # Ensure quality scores are in [0, 1]
        quality_scores = np.clip(quality_scores, 0, 1)
        
        return quality_scores
    
    def _detect_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Detect outliers in data.
        
        Args:
            data: Data array
            
        Returns:
            Boolean array indicating outliers
        """
        n_samples = data.shape[0]
        outlier_flags = np.zeros(n_samples, dtype=bool)
        
        # Method 1: Z-score based outlier detection
        if self.config.outlier_method == "zscore":
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            z_outliers = z_scores > self.config.outlier_threshold
            outlier_flags = outlier_flags | np.any(z_outliers, axis=1)
        
        # Method 2: IQR based outlier detection
        elif self.config.outlier_method == "iqr":
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = (data < lower_bound) | (data > upper_bound)
            outlier_flags = outlier_flags | np.any(iqr_outliers, axis=1)
        
        # Method 3: Isolation Forest
        elif self.config.outlier_method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_labels = iso_forest.fit_predict(data)
            outlier_flags = outlier_flags | (iso_labels == -1)
        
        # Method 4: DBSCAN clustering
        try:
            dbscan = DBSCAN(eps=3.0, min_samples=5)
            cluster_labels = dbscan.fit_predict(data)
            dbscan_outliers = cluster_labels == -1
            outlier_flags = outlier_flags | dbscan_outliers
        except Exception as e:
            logger.warning(f"DBSCAN outlier detection failed: {e}")
        
        return outlier_flags
    
    def _generate_quality_flags(
        self, 
        data: np.ndarray, 
        quality_scores: np.ndarray, 
        outlier_flags: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Generate quality flags for data.
        
        Args:
            data: Data array
            quality_scores: Quality scores
            outlier_flags: Outlier flags
            
        Returns:
            Dictionary of quality flags
        """
        flags = {}
        
        # Overall quality flag
        flags['high_quality'] = quality_scores >= self.quality_thresholds['min_data_quality']
        flags['low_quality'] = quality_scores < self.quality_thresholds['min_data_quality']
        flags['outlier'] = outlier_flags
        
        # Missing data flags
        missing_rates = np.isnan(data).mean(axis=1)
        flags['high_missing'] = missing_rates > self.quality_thresholds['max_missing_rate']
        flags['moderate_missing'] = (
            (missing_rates > 0.1) & 
            (missing_rates <= self.quality_thresholds['max_missing_rate'])
        )
        flags['low_missing'] = missing_rates <= 0.1
        
        # Data completeness flags
        flags['complete'] = missing_rates == 0
        flags['incomplete'] = missing_rates > 0
        
        # Statistical flags
        if data.shape[1] > 1:
            # Compute z-scores for each feature
            z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
            flags['extreme_values'] = np.any(z_scores > 5, axis=1)
            flags['moderate_values'] = np.all(z_scores <= 5, axis=1)
        
        # Sample size flag
        flags['sufficient_samples'] = len(data) >= self.quality_thresholds['min_samples']
        
        return flags
    
    def _compute_quality_statistics(
        self, 
        data: np.ndarray, 
        quality_scores: np.ndarray, 
        outlier_flags: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute quality statistics for data.
        
        Args:
            data: Data array
            quality_scores: Quality scores
            outlier_flags: Outlier flags
            
        Returns:
            Dictionary of quality statistics
        """
        stats = {}
        
        # Basic statistics
        stats['n_samples'] = len(data)
        stats['n_features'] = data.shape[1] if len(data.shape) > 1 else 1
        stats['missing_rate'] = np.isnan(data).mean()
        
        # Quality score statistics
        stats['quality_mean'] = np.mean(quality_scores)
        stats['quality_std'] = np.std(quality_scores)
        stats['quality_min'] = np.min(quality_scores)
        stats['quality_max'] = np.max(quality_scores)
        stats['quality_median'] = np.median(quality_scores)
        
        # Outlier statistics
        stats['outlier_count'] = np.sum(outlier_flags)
        stats['outlier_rate'] = np.mean(outlier_flags)
        
        # Quality distribution
        stats['high_quality_count'] = np.sum(quality_scores >= self.quality_thresholds['min_data_quality'])
        stats['high_quality_rate'] = stats['high_quality_count'] / len(data)
        stats['low_quality_count'] = np.sum(quality_scores < self.quality_thresholds['min_data_quality'])
        stats['low_quality_rate'] = stats['low_quality_count'] / len(data)
        
        # Feature statistics
        if len(data.shape) > 1:
            stats['feature_variance'] = np.var(data, axis=0)
            stats['feature_mean'] = np.mean(data, axis=0)
            stats['feature_std'] = np.std(data, axis=0)
            
            # Check for constant features
            constant_features = np.sum(np.var(data, axis=0) == 0)
            stats['constant_features'] = constant_features
            stats['constant_feature_rate'] = constant_features / stats['n_features']
        
        # Data distribution statistics
        if len(data.shape) > 1:
            try:
                # Compute correlation matrix
                correlations = np.corrcoef(data.T)
                avg_correlation = np.mean(np.abs(correlations[np.triu_indices_from(correlations, k=1)]))
                stats['avg_correlation'] = avg_correlation
                
                # Check for multicollinearity
                from statsmodels.stats.outliers_influence import variance_inflation_factor
                vif_scores = []
                for i in range(data.shape[1]):
                    if not np.isnan(data[:, i]).all():
                        vif = variance_inflation_factor(data, i)
                        vif_scores.append(vif)
                
                if vif_scores:
                    stats['max_vif'] = max(vif_scores)
                    stats['high_vif_count'] = sum(1 for vif in vif_scores if vif > self.quality_thresholds['max_variance_inflation'])
            except Exception as e:
                logger.warning(f"Could not compute correlation/VIF statistics: {e}")
        
        return stats
    
    def _compute_overall_quality_stats(self, modality_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall quality statistics across all modalities.
        
        Args:
            modality_stats: Dictionary of modality-specific statistics
            
        Returns:
            Dictionary of overall quality statistics
        """
        overall_stats = {}
        
        # Count modalities
        modalities = [k for k in modality_stats.keys() if k != 'overall']
        overall_stats['n_modalities'] = len(modalities)
        
        # Aggregate statistics
        if modalities:
            overall_stats['total_samples'] = sum(modality_stats[m]['n_samples'] for m in modalities)
            overall_stats['avg_quality'] = np.mean([modality_stats[m]['quality_mean'] for m in modalities])
            overall_stats['total_outliers'] = sum(modality_stats[m]['outlier_count'] for m in modalities)
            overall_stats['overall_outlier_rate'] = overall_stats['total_outliers'] / overall_stats['total_samples']
            
            # Quality distribution
            overall_stats['high_quality_modalities'] = sum(
                1 for m in modalities if modality_stats[m]['high_quality_rate'] >= 0.8
            )
            overall_stats['low_quality_modalities'] = sum(
                1 for m in modalities if modality_stats[m]['low_quality_rate'] >= 0.2
            )
        
        return overall_stats
    
    def _generate_recommendations(self, data: np.ndarray, stats: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations for data quality improvement.
        
        Args:
            data: Data array
            stats: Quality statistics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Sample size recommendations
        if stats['n_samples'] < self.quality_thresholds['min_samples']:
            recommendations.append(f"Consider collecting more samples (current: {stats['n_samples']}, minimum: {self.quality_thresholds['min_samples']})")
        
        # Missing data recommendations
        if stats['missing_rate'] > self.quality_thresholds['max_missing_rate']:
            recommendations.append(f"High missing data rate detected ({stats['missing_rate']:.2%}). Consider imputation or data collection strategies.")
        
        # Outlier recommendations
        if stats['outlier_rate'] > self.quality_thresholds['max_outlier_rate']:
            recommendations.append(f"High outlier rate detected ({stats['outlier_rate']:.2%}). Consider outlier investigation and removal.")
        
        # Quality score recommendations
        if stats['quality_mean'] < self.quality_thresholds['min_data_quality']:
            recommendations.append(f"Low overall data quality (mean: {stats['quality_mean']:.2f}). Consider data cleaning and validation.")
        
        # Feature recommendations
        if 'constant_features' in stats and stats['constant_features'] > 0:
            recommendations.append(f"Found {stats['constant_features']} constant features. Consider removing them.")
        
        # Multicollinearity recommendations
        if 'high_vif_count' in stats and stats['high_vif_count'] > 0:
            recommendations.append(f"Found {stats['high_vif_count']} features with high multicollinearity. Consider feature selection or dimensionality reduction.")
        
        # Correlation recommendations
        if 'avg_correlation' in stats and stats['avg_correlation'] < self.quality_thresholds['min_correlation']:
            recommendations.append(f"Low average correlation between features ({stats['avg_correlation']:.2f}). Consider feature engineering.")
        
        return recommendations
    
    def _generate_overall_recommendations(self, quality_stats: Dict[str, Any]) -> List[str]:
        """
        Generate overall recommendations for the dataset.
        
        Args:
            quality_stats: Overall quality statistics
            
        Returns:
            List of overall recommendations
        """
        recommendations = []
        
        if 'overall' in quality_stats:
            overall = quality_stats['overall']
            
            # Modality coverage
            if overall['n_modalities'] < 3:
                recommendations.append(f"Limited modality coverage ({overall['n_modalities']} modalities). Consider collecting additional data types.")
            
            # Overall quality
            if overall['avg_quality'] < self.quality_thresholds['min_data_quality']:
                recommendations.append(f"Low overall data quality across modalities (mean: {overall['avg_quality']:.2f}).")
            
            # Outlier rate
            if overall['overall_outlier_rate'] > self.quality_thresholds['max_outlier_rate']:
                recommendations.append(f"High overall outlier rate ({overall['overall_outlier_rate']:.2%}).")
            
            # Modality quality
            if overall['low_quality_modalities'] > 0:
                recommendations.append(f"Found {overall['low_quality_modalities']} low-quality modalities requiring attention.")
        
        return recommendations
    
    def filter_data(
        self, 
        data: Dict[str, np.ndarray], 
        quality_result: QualityControlResult,
        min_quality: float = None,
        remove_outliers: bool = True,
        max_missing_rate: float = None
    ) -> Dict[str, np.ndarray]:
        """
        Filter data based on quality assessment.
        
        Args:
            data: Dictionary of modality-specific data arrays
            quality_result: Quality control result
            min_quality: Minimum quality score threshold
            remove_outliers: Whether to remove outliers
            max_missing_rate: Maximum missing rate threshold
            
        Returns:
            Filtered data dictionary
        """
        filtered_data = {}
        
        min_quality = min_quality or self.quality_thresholds['min_data_quality']
        max_missing_rate = max_missing_rate or self.quality_thresholds['max_missing_rate']
        
        for modality, modality_data in data.items():
            if modality_data is None or len(modality_data) == 0:
                continue
            
            # Get quality flags
            if modality in quality_result.quality_flags:
                flags = quality_result.quality_flags[modality]
                
                # Create filter mask
                filter_mask = np.ones(len(modality_data), dtype=bool)
                
                # Apply quality score filter
                if modality in quality_result.quality_scores:
                    quality_scores = quality_result.quality_scores[modality]
                    filter_mask &= (quality_scores >= min_quality)
                
                # Apply outlier filter
                if remove_outliers and 'outlier' in flags:
                    filter_mask &= ~flags['outlier']
                
                # Apply missing data filter
                if 'high_missing' in flags:
                    filter_mask &= ~flags['high_missing']
                
                # Filter data
                filtered_data[modality] = modality_data[filter_mask]
                
                logger.info(f"Filtered {modality}: {len(modality_data)} -> {len(filtered_data[modality])} samples")
            else:
                # No quality flags available, keep all data
                filtered_data[modality] = modality_data
        
        return filtered_data
    
    def generate_quality_report(self, quality_result: QualityControlResult) -> str:
        """
        Generate a comprehensive quality report.
        
        Args:
            quality_result: Quality control result
            
        Returns:
            Quality report string
        """
        report = []
        report.append("=" * 60)
        report.append("HAMBAE DATA QUALITY REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        if 'overall' in quality_result.quality_stats:
            overall = quality_result.quality_stats['overall']
            report.append("OVERALL STATISTICS:")
            report.append(f"  Total Modalities: {overall['n_modalities']}")
            report.append(f"  Total Samples: {overall['total_samples']}")
            report.append(f"  Average Quality: {overall['avg_quality']:.3f}")
            report.append(f"  Overall Outlier Rate: {overall['overall_outlier_rate']:.2%}")
            report.append("")
        
        # Modality-specific statistics
        report.append("MODALITY-SPECIFIC STATISTICS:")
        for modality, stats in quality_result.quality_stats.items():
            if modality == 'overall':
                continue
            
            report.append(f"  {modality.upper()}:")
            report.append(f"    Samples: {stats['n_samples']}")
            report.append(f"    Features: {stats['n_features']}")
            report.append(f"    Missing Rate: {stats['missing_rate']:.2%}")
            report.append(f"    Quality Mean: {stats['quality_mean']:.3f}")
            report.append(f"    Outlier Rate: {stats['outlier_rate']:.2%}")
            report.append(f"    High Quality Rate: {stats['high_quality_rate']:.2%}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if 'overall' in quality_result.recommendations:
            for rec in quality_result.recommendations['overall']:
                report.append(f"  - {rec}")
        
        for modality, recs in quality_result.recommendations.items():
            if modality == 'overall':
                continue
            for rec in recs:
                report.append(f"  - {modality}: {rec}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def create_quality_controller(config: DataConfig) -> QualityController:
    """
    Create a quality controller with the given configuration.
    
    Args:
        config: Data configuration
        
    Returns:
        Configured quality controller
    """
    return QualityController(config)


def assess_ukbb_quality(
    data: Dict[str, np.ndarray],
    config: DataConfig,
) -> QualityControlResult:
    """
    Assess quality of UK Biobank data.
    
    Args:
        data: Dictionary of modality-specific data arrays
        config: Data configuration
        
    Returns:
        Quality control result
    """
    controller = QualityController(config)
    return controller.assess_quality(data)