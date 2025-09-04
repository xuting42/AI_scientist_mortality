"""
Feature engineering for HAMBAE algorithm system.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.cluster import KMeans
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import logging
from dataclasses import dataclass
from ..config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureEngineeringResult:
    """Result of feature engineering."""
    engineered_features: Dict[str, np.ndarray]
    feature_names: Dict[str, List[str]]
    feature_importance: Dict[str, np.ndarray]
    engineering_stats: Dict[str, Any]


class FeatureEngineer:
    """
    Feature engineering pipeline for HAMBAE algorithm system.
    
    This class handles feature engineering including ratio creation,
    interaction terms, temporal features, and advanced transformations.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize feature engineer.
        
        Args:
            config: Data configuration
        """
        self.config = config
        self.feature_names = {}
        self.feature_importance = {}
        self.pca_transformers = {}
        self.feature_selectors = {}
        
        # Initialize feature engineering components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize feature engineering components."""
        # Initialize PCA for dimensionality reduction
        self.pca_transformers['metabolomics'] = PCA(n_components=50, random_state=42)
        self.pca_transformers['genetic_features'] = PCA(n_components=100, random_state=42)
        
        # Initialize feature selectors
        self.feature_selectors['blood_biomarkers'] = SelectKBest(score_func=f_regression, k=10)
        self.feature_selectors['metabolomics'] = SelectKBest(score_func=f_regression, k=30)
    
    def fit(self, data: Dict[str, np.ndarray], targets: np.ndarray) -> 'FeatureEngineer':
        """
        Fit feature engineering components on training data.
        
        Args:
            data: Dictionary of modality-specific data arrays
            targets: Target values (age or other)
            
        Returns:
            Fitted feature engineer
        """
        logger.info("Fitting feature engineering components")
        
        engineering_stats = {}
        
        for modality, modality_data in data.items():
            if modality_data is None or len(modality_data) == 0:
                continue
            
            logger.info(f"Fitting feature engineering for {modality}")
            
            # Fit PCA for dimensionality reduction
            if modality in self.pca_transformers:
                logger.info(f"Fitting PCA for {modality}")
                self.pca_transformers[modality].fit(modality_data)
            
            # Fit feature selection
            if modality in self.feature_selectors:
                logger.info(f"Fitting feature selection for {modality}")
                self.feature_selectors[modality].fit(modality_data, targets)
            
            # Compute feature statistics
            engineering_stats[modality] = {
                'n_samples': len(modality_data),
                'n_features': modality_data.shape[1] if len(modality_data.shape) > 1 else 1,
                'mean': np.nanmean(modality_data),
                'std': np.nanstd(modality_data),
                'variance_explained': {},
            }
            
            # Compute variance explained by PCA
            if modality in self.pca_transformers:
                pca = self.pca_transformers[modality]
                engineering_stats[modality]['variance_explained'] = {
                    'total': np.sum(pca.explained_variance_ratio_),
                    'components': pca.explained_variance_ratio_.tolist(),
                }
        
        self.engineering_stats = engineering_stats
        logger.info("Feature engineering components fitted successfully")
        
        return self
    
    def transform(self, data: Dict[str, np.ndarray]) -> FeatureEngineeringResult:
        """
        Transform data using fitted feature engineering components.
        
        Args:
            data: Dictionary of modality-specific data arrays
            
        Returns:
            Feature engineering result with engineered features
        """
        logger.info("Engineering features")
        
        engineered_features = {}
        feature_names = {}
        feature_importance = {}
        
        for modality, modality_data in data.items():
            if modality_data is None or len(modality_data) == 0:
                continue
            
            logger.info(f"Engineering features for {modality}")
            
            # Start with original features
            engineered_features[modality] = modality_data.copy()
            feature_names[modality] = [f"{modality}_{i}" for i in range(modality_data.shape[1])]
            
            # Create ratio features
            if self.config.create_ratios:
                ratio_features, ratio_names = self._create_ratios(modality_data, modality)
                if ratio_features is not None:
                    engineered_features[modality] = np.hstack([
                        engineered_features[modality], ratio_features
                    ])
                    feature_names[modality].extend(ratio_names)
            
            # Create interaction features
            if self.config.create_interactions:
                interaction_features, interaction_names = self._create_interactions(
                    modality_data, modality
                )
                if interaction_features is not None:
                    engineered_features[modality] = np.hstack([
                        engineered_features[modality], interaction_features
                    ])
                    feature_names[modality].extend(interaction_names)
            
            # Create temporal features
            if self.config.temporal_features and 'timestamp' in data:
                temporal_features, temporal_names = self._create_temporal_features(
                    modality_data, data['timestamp'], modality
                )
                if temporal_features is not None:
                    engineered_features[modality] = np.hstack([
                        engineered_features[modality], temporal_features
                    ])
                    feature_names[modality].extend(temporal_names)
            
            # Apply dimensionality reduction
            if modality in self.pca_transformers:
                pca_features = self.pca_transformers[modality].transform(modality_data)
                engineered_features[f"{modality}_pca"] = pca_features
                feature_names[f"{modality}_pca"] = [
                    f"{modality}_pca_{i}" for i in range(pca_features.shape[1])
                ]
            
            # Apply feature selection
            if modality in self.feature_selectors:
                selected_features = self.feature_selectors[modality].transform(modality_data)
                selected_names = [feature_names[modality][i] for i in 
                                self.feature_selectors[modality].get_support(indices=True)]
                
                engineered_features[f"{modality}_selected"] = selected_features
                feature_names[f"{modality}_selected"] = selected_names
                
                # Get feature importance scores
                feature_importance[modality] = self.feature_selectors[modality].scores_
            
            # Create modality-specific features
            if modality == 'blood_biomarkers':
                bio_features, bio_names = self._create_biological_features(modality_data)
                if bio_features is not None:
                    engineered_features[f"{modality}_biological"] = bio_features
                    feature_names[f"{modality}_biological"] = bio_names
            
            elif modality == 'metabolomics':
                metab_features, metab_names = self._create_metabolomic_features(modality_data)
                if metab_features is not None:
                    engineered_features[f"{modality}_metabolic"] = metab_features
                    feature_names[f"{modality}_metabolic"] = metab_names
        
        # Create cross-modality features
        if len(engineered_features) > 1:
            cross_features, cross_names = self._create_cross_modality_features(
                engineered_features, feature_names
            )
            if cross_features is not None:
                engineered_features['cross_modality'] = cross_features
                feature_names['cross_modality'] = cross_names
        
        result = FeatureEngineeringResult(
            engineered_features=engineered_features,
            feature_names=feature_names,
            feature_importance=feature_importance,
            engineering_stats=getattr(self, 'engineering_stats', {}),
        )
        
        logger.info("Feature engineering completed")
        
        return result
    
    def fit_transform(self, data: Dict[str, np.ndarray], targets: np.ndarray) -> FeatureEngineeringResult:
        """
        Fit feature engineering components and transform data.
        
        Args:
            data: Dictionary of modality-specific data arrays
            targets: Target values (age or other)
            
        Returns:
            Feature engineering result with engineered features
        """
        return self.fit(data, targets).transform(data)
    
    def _create_ratios(self, data: np.ndarray, modality: str) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Create ratio features from data.
        
        Args:
            data: Data array
            modality: Modality name
            
        Returns:
            Tuple of (ratio_features, ratio_names)
        """
        if data.shape[1] < 2:
            return None, []
        
        ratio_features = []
        ratio_names = []
        
        # Create ratios between different features
        for i in range(data.shape[1]):
            for j in range(i + 1, data.shape[1]):
                # Avoid division by zero
                denominator = data[:, j] + 1e-8
                ratio = data[:, i] / denominator
                
                ratio_features.append(ratio.reshape(-1, 1))
                ratio_names.append(f"{modality}_ratio_{i}_{j}")
        
        if ratio_features:
            return np.hstack(ratio_features), ratio_names
        else:
            return None, []
    
    def _create_interactions(self, data: np.ndarray, modality: str) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Create interaction features from data.
        
        Args:
            data: Data array
            modality: Modality name
            
        Returns:
            Tuple of (interaction_features, interaction_names)
        """
        if data.shape[1] < 2:
            return None, []
        
        interaction_features = []
        interaction_names = []
        
        # Create pairwise interactions
        for i in range(data.shape[1]):
            for j in range(i + 1, data.shape[1]):
                interaction = data[:, i] * data[:, j]
                
                interaction_features.append(interaction.reshape(-1, 1))
                interaction_names.append(f"{modality}_interaction_{i}_{j}")
        
        if interaction_features:
            return np.hstack(interaction_features), interaction_names
        else:
            return None, []
    
    def _create_temporal_features(
        self, 
        data: np.ndarray, 
        timestamps: np.ndarray, 
        modality: str
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Create temporal features from data.
        
        Args:
            data: Data array
            timestamps: Timestamp array
            modality: Modality name
            
        Returns:
            Tuple of (temporal_features, temporal_names)
        """
        if len(timestamps) != len(data):
            logger.warning("Timestamp length mismatch, skipping temporal features")
            return None, []
        
        temporal_features = []
        temporal_names = []
        
        # Create time-based features
        for i in range(data.shape[1]):
            # Rate of change (approximate derivative)
            if len(data) > 1:
                sorted_indices = np.argsort(timestamps)
                sorted_data = data[sorted_indices]
                sorted_timestamps = timestamps[sorted_indices]
                
                # Compute rate of change
                rates = np.gradient(sorted_data[:, i], sorted_timestamps)
                
                # Sort back to original order
                original_indices = np.argsort(sorted_indices)
                rates = rates[original_indices]
                
                temporal_features.append(rates.reshape(-1, 1))
                temporal_names.append(f"{modality}_rate_{i}")
        
        if temporal_features:
            return np.hstack(temporal_features), temporal_names
        else:
            return None, []
    
    def _create_biological_features(self, data: np.ndarray) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Create biological features from blood biomarkers.
        
        Args:
            data: Blood biomarker data
            
        Returns:
            Tuple of (biological_features, biological_names)
        """
        if data.shape[1] < 13:  # Expected 13 blood biomarkers
            return None, []
        
        biological_features = []
        biological_names = []
        
        # Create biologically meaningful ratios
        # Example: Albumin/Globulin ratio
        if data.shape[1] >= 2:
            albumin_globulin_ratio = data[:, 0] / (data[:, 1] + 1e-8)
            biological_features.append(albumin_globulin_ratio.reshape(-1, 1))
            biological_names.append("albumin_globulin_ratio")
        
        # Create inflammation score
        if data.shape[1] >= 3:
            inflammation_markers = data[:, 2:5]  # CRP, ESR, etc.
            inflammation_score = np.mean(inflammation_markers, axis=1)
            biological_features.append(inflammation_score.reshape(-1, 1))
            biological_names.append("inflammation_score")
        
        # Create metabolic syndrome score
        if data.shape[1] >= 5:
            metabolic_features = data[:, 5:10]  # Glucose, triglycerides, etc.
            metabolic_score = np.mean(metabolic_features, axis=1)
            biological_features.append(metabolic_score.reshape(-1, 1))
            biological_names.append("metabolic_score")
        
        # Create immune system score
        if data.shape[1] >= 8:
            immune_features = data[:, 8:13]  # WBC, lymphocytes, etc.
            immune_score = np.mean(immune_features, axis=1)
            biological_features.append(immune_score.reshape(-1, 1))
            biological_names.append("immune_score")
        
        if biological_features:
            return np.hstack(biological_features), biological_names
        else:
            return None, []
    
    def _create_metabolomic_features(self, data: np.ndarray) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Create metabolomic features from metabolomics data.
        
        Args:
            data: Metabolomics data
            
        Returns:
            Tuple of (metabolomic_features, metabolomic_names)
        """
        if data.shape[1] < 10:
            return None, []
        
        metabolomic_features = []
        metabolomic_names = []
        
        # Create pathway-level features
        # This is a simplified implementation
        # In practice, you would use actual pathway information
        
        # Group features by clusters
        n_clusters = min(10, data.shape[1] // 10)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(data.T)
            
            for cluster_id in range(n_clusters):
                cluster_features = data[:, cluster_labels == cluster_id]
                if cluster_features.shape[1] > 0:
                    cluster_mean = np.mean(cluster_features, axis=1)
                    metabolomic_features.append(cluster_mean.reshape(-1, 1))
                    metabolomic_names.append(f"metabolomic_cluster_{cluster_id}")
        
        # Create metabolite diversity score
        metabolite_diversity = np.std(data, axis=1)
        metabolomic_features.append(metabolite_diversity.reshape(-1, 1))
        metabolomic_names.append("metabolite_diversity")
        
        if metabolomic_features:
            return np.hstack(metabolomic_features), metabolomic_names
        else:
            return None, []
    
    def _create_cross_modality_features(
        self, 
        data: Dict[str, np.ndarray], 
        feature_names: Dict[str, List[str]]
    ) -> Tuple[Optional[np.ndarray], List[str]]:
        """
        Create cross-modality features.
        
        Args:
            data: Dictionary of modality-specific data
            feature_names: Dictionary of feature names
            
        Returns:
            Tuple of (cross_features, cross_names)
        """
        cross_features = []
        cross_names = []
        
        # Get modalities
        modalities = list(data.keys())
        
        # Create cross-modality correlations
        if len(modalities) >= 2:
            for i, modality1 in enumerate(modalities):
                for j, modality2 in enumerate(modalities[i+1:], i+1):
                    if (data[modality1].shape[1] > 0 and 
                        data[modality2].shape[1] > 0):
                        
                        # Compute correlation between first features of each modality
                        corr = np.corrcoef(data[modality1][:, 0], data[modality2][:, 0])[0, 1]
                        
                        # Create cross-modality feature
                        cross_feature = data[modality1][:, 0] * data[modality2][:, 0]
                        cross_features.append(cross_feature.reshape(-1, 1))
                        cross_names.append(f"cross_{modality1}_{modality2}")
        
        if cross_features:
            return np.hstack(cross_features), cross_names
        else:
            return None, []
    
    def get_feature_importance(self, modality: str) -> Optional[np.ndarray]:
        """
        Get feature importance scores for a modality.
        
        Args:
            modality: Modality name
            
        Returns:
            Feature importance scores or None
        """
        return self.feature_importance.get(modality)
    
    def get_feature_names(self, modality: str) -> Optional[List[str]]:
        """
        Get feature names for a modality.
        
        Args:
            modality: Modality name
            
        Returns:
            Feature names or None
        """
        return self.feature_names.get(modality)
    
    def save_feature_engineer(self, save_path: str) -> None:
        """
        Save fitted feature engineering components.
        
        Args:
            save_path: Path to save feature engineering components
        """
        import joblib
        
        components = {
            'pca_transformers': self.pca_transformers,
            'feature_selectors': self.feature_selectors,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'config': self.config,
        }
        
        joblib.dump(components, save_path)
        logger.info(f"Feature engineering components saved to {save_path}")
    
    def load_feature_engineer(self, load_path: str) -> None:
        """
        Load fitted feature engineering components.
        
        Args:
            load_path: Path to load feature engineering components
        """
        import joblib
        
        components = joblib.load(load_path)
        self.pca_transformers = components['pca_transformers']
        self.feature_selectors = components['feature_selectors']
        self.feature_names = components['feature_names']
        self.feature_importance = components['feature_importance']
        
        logger.info(f"Feature engineering components loaded from {load_path}")


def create_feature_engineer(config: DataConfig) -> FeatureEngineer:
    """
    Create a feature engineer with the given configuration.
    
    Args:
        config: Data configuration
        
    Returns:
        Configured feature engineer
    """
    return FeatureEngineer(config)


def engineer_ukbb_features(
    data: Dict[str, np.ndarray],
    targets: np.ndarray,
    config: DataConfig,
    fit: bool = True,
) -> FeatureEngineeringResult:
    """
    Engineer features for UK Biobank data.
    
    Args:
        data: Dictionary of modality-specific data arrays
        targets: Target values (age or other)
        config: Data configuration
        fit: Whether to fit feature engineering components
        
    Returns:
        Feature engineering result
    """
    engineer = FeatureEngineer(config)
    
    if fit:
        result = engineer.fit_transform(data, targets)
    else:
        result = engineer.transform(data)
    
    return result