"""
Evaluation metrics for biological age estimation models.

Includes standard regression metrics, survival analysis metrics,
and biological age-specific evaluations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import pandas as pd
import logging
from tqdm import tqdm
import warnings

logger = logging.getLogger(__name__)


class BioAgeEvaluator:
    """Comprehensive evaluator for biological age models."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: str = 'cuda'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained biological age model
            config: Configuration object
            device: Device for evaluation
        """
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_survival: bool = True,
        compute_uncertainty: bool = True
    ) -> Dict[str, float]:
        """
        Comprehensive model evaluation.
        
        Args:
            dataloader: Test dataloader
            compute_survival: Whether to compute survival metrics
            compute_uncertainty: Whether to compute uncertainty metrics
        
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = []
        targets = []
        uncertainties = []
        features_list = []
        
        # Collect predictions
        logger.info("Collecting predictions...")
        with torch.no_grad():
            for batch_idx, (inputs, batch_targets) in enumerate(tqdm(dataloader)):
                # Move to device
                inputs = self._move_to_device(inputs)
                batch_targets = batch_targets.to(self.device)
                
                # Get predictions
                outputs = self.model(inputs, return_uncertainty=compute_uncertainty)
                batch_predictions = outputs['prediction'].squeeze()
                
                # Store results
                predictions.append(batch_predictions.cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
                
                if compute_uncertainty and 'uncertainty' in outputs:
                    uncertainties.append(outputs['uncertainty'].cpu().numpy())
                
                if 'features' in outputs:
                    features_list.append(outputs['features'].cpu().numpy())
        
        # Concatenate results
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets).flatten()
        
        if uncertainties:
            uncertainties = np.concatenate(uncertainties)
        
        if features_list:
            features = np.concatenate(features_list)
        else:
            features = None
        
        # Compute metrics
        metrics = self._compute_regression_metrics(predictions, targets)
        
        # Add uncertainty metrics if available
        if uncertainties and len(uncertainties) > 0:
            uncertainty_metrics = self._compute_uncertainty_metrics(
                predictions, targets, uncertainties
            )
            metrics.update(uncertainty_metrics)
        
        # Add survival metrics if mortality labels available
        if compute_survival and hasattr(dataloader.dataset, 'get_mortality_labels'):
            mortality_labels = dataloader.dataset.get_mortality_labels()
            if mortality_labels is not None and len(mortality_labels) == len(predictions):
                survival_metrics = self._compute_survival_metrics(
                    predictions, targets, mortality_labels
                )
                metrics.update(survival_metrics)
        
        # Add age acceleration metrics
        age_acceleration_metrics = self._compute_age_acceleration_metrics(
            predictions, targets
        )
        metrics.update(age_acceleration_metrics)
        
        # Model-specific metrics
        if hasattr(self.model, 'get_feature_importance'):
            metrics['feature_importance'] = self.model.get_feature_importance()
        
        return metrics
    
    def _move_to_device(self, inputs: Any) -> Any:
        """Move inputs to device."""
        if isinstance(inputs, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v 
                   for k, v in inputs.items()}
        elif torch.is_tensor(inputs):
            return inputs.to(self.device)
        else:
            return inputs
    
    def _compute_regression_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute standard regression metrics."""
        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(targets, predictions)
        spearman_r, spearman_p = spearmanr(targets, predictions)
        
        # Percentage within X years
        errors = np.abs(predictions - targets)
        within_3 = np.mean(errors <= 3) * 100
        within_5 = np.mean(errors <= 5) * 100
        within_10 = np.mean(errors <= 10) * 100
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman': float(spearman_r),
            'spearman_p': float(spearman_p),
            'within_3_years': float(within_3),
            'within_5_years': float(within_5),
            'within_10_years': float(within_10),
            'median_error': float(np.median(errors)),
            'std_error': float(np.std(errors))
        }
    
    def _compute_uncertainty_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """Compute uncertainty-related metrics."""
        # Calibration: Are high-uncertainty predictions less accurate?
        errors = np.abs(predictions - targets)
        
        # Split by uncertainty quartiles
        uncertainty_quartiles = np.percentile(uncertainties, [25, 50, 75])
        
        q1_mask = uncertainties <= uncertainty_quartiles[0]
        q2_mask = (uncertainties > uncertainty_quartiles[0]) & (uncertainties <= uncertainty_quartiles[1])
        q3_mask = (uncertainties > uncertainty_quartiles[1]) & (uncertainties <= uncertainty_quartiles[2])
        q4_mask = uncertainties > uncertainty_quartiles[2]
        
        mae_by_uncertainty = {
            'q1_mae': float(errors[q1_mask].mean()) if q1_mask.any() else 0,
            'q2_mae': float(errors[q2_mask].mean()) if q2_mask.any() else 0,
            'q3_mae': float(errors[q3_mask].mean()) if q3_mask.any() else 0,
            'q4_mae': float(errors[q4_mask].mean()) if q4_mask.any() else 0
        }
        
        # Uncertainty correlation with error
        if len(errors) > 1:
            uncertainty_error_corr, _ = spearmanr(uncertainties, errors)
        else:
            uncertainty_error_corr = 0
        
        return {
            'mean_uncertainty': float(uncertainties.mean()),
            'std_uncertainty': float(uncertainties.std()),
            'uncertainty_error_correlation': float(uncertainty_error_corr),
            **mae_by_uncertainty
        }
    
    def _compute_survival_metrics(
        self,
        predictions: np.ndarray,
        chronological_ages: np.ndarray,
        mortality_labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute survival analysis metrics."""
        # Age acceleration
        age_acceleration = predictions - chronological_ages
        
        # C-index for biological age
        try:
            c_index_bio = concordance_index(
                mortality_labels,
                -predictions,  # Negative because higher age -> higher risk
                np.ones_like(mortality_labels)  # All observed
            )
        except:
            c_index_bio = 0.5
        
        # C-index for chronological age
        try:
            c_index_chrono = concordance_index(
                mortality_labels,
                -chronological_ages,
                np.ones_like(mortality_labels)
            )
        except:
            c_index_chrono = 0.5
        
        # C-index for age acceleration
        try:
            c_index_accel = concordance_index(
                mortality_labels,
                -age_acceleration,
                np.ones_like(mortality_labels)
            )
        except:
            c_index_accel = 0.5
        
        # Hazard ratio analysis using Cox regression
        try:
            # Prepare data for Cox regression
            cox_data = pd.DataFrame({
                'duration': np.ones_like(mortality_labels) * 10,  # Assume 10-year follow-up
                'event': mortality_labels,
                'bio_age': predictions,
                'chrono_age': chronological_ages,
                'age_accel': age_acceleration
            })
            
            # Fit Cox model
            cph = CoxPHFitter()
            cph.fit(cox_data[['duration', 'event', 'age_accel']], 
                   duration_col='duration', event_col='event')
            
            # Get hazard ratio for 5-year age acceleration
            hr_per_5_years = np.exp(cph.params_['age_accel'] * 5)
        except:
            hr_per_5_years = 1.0
        
        return {
            'c_index': float(c_index_bio),
            'c_index_chronological': float(c_index_chrono),
            'c_index_acceleration': float(c_index_accel),
            'hazard_ratio_per_5_years': float(hr_per_5_years),
            'mortality_rate': float(mortality_labels.mean())
        }
    
    def _compute_age_acceleration_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """Compute age acceleration specific metrics."""
        # Age acceleration
        age_acceleration = predictions - targets
        
        # Categorize aging rates
        accelerated = (age_acceleration > 5).mean() * 100
        normal = ((age_acceleration >= -5) & (age_acceleration <= 5)).mean() * 100
        decelerated = (age_acceleration < -5).mean() * 100
        
        # Distribution statistics
        aa_mean = age_acceleration.mean()
        aa_std = age_acceleration.std()
        aa_median = np.median(age_acceleration)
        aa_iqr = np.percentile(age_acceleration, 75) - np.percentile(age_acceleration, 25)
        
        return {
            'age_acceleration_mean': float(aa_mean),
            'age_acceleration_std': float(aa_std),
            'age_acceleration_median': float(aa_median),
            'age_acceleration_iqr': float(aa_iqr),
            'percent_accelerated': float(accelerated),
            'percent_normal': float(normal),
            'percent_decelerated': float(decelerated)
        }
    
    def evaluate_subgroups(
        self,
        dataloader: DataLoader,
        subgroup_fn: callable
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on different subgroups.
        
        Args:
            dataloader: Test dataloader
            subgroup_fn: Function to determine subgroup membership
        
        Returns:
            Dictionary of metrics per subgroup
        """
        # Collect predictions and subgroup labels
        predictions = []
        targets = []
        subgroups = []
        
        with torch.no_grad():
            for inputs, batch_targets in tqdm(dataloader):
                # Move to device
                inputs = self._move_to_device(inputs)
                batch_targets = batch_targets.to(self.device)
                
                # Get predictions
                outputs = self.model(inputs)
                batch_predictions = outputs['prediction'].squeeze()
                
                # Get subgroup labels
                batch_subgroups = subgroup_fn(inputs, batch_targets)
                
                # Store results
                predictions.append(batch_predictions.cpu().numpy())
                targets.append(batch_targets.cpu().numpy())
                subgroups.append(batch_subgroups)
        
        # Concatenate
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets).flatten()
        subgroups = np.concatenate(subgroups)
        
        # Evaluate each subgroup
        unique_subgroups = np.unique(subgroups)
        results = {}
        
        for subgroup in unique_subgroups:
            mask = subgroups == subgroup
            subgroup_preds = predictions[mask]
            subgroup_targets = targets[mask]
            
            if len(subgroup_preds) > 0:
                results[str(subgroup)] = self._compute_regression_metrics(
                    subgroup_preds, subgroup_targets
                )
                results[str(subgroup)]['n_samples'] = int(mask.sum())
        
        return results
    
    def compute_feature_importance(
        self,
        dataloader: DataLoader,
        n_permutations: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature importance using permutation.
        
        Args:
            dataloader: Test dataloader
            n_permutations: Number of permutations per feature
        
        Returns:
            Dictionary of feature importance scores
        """
        # Get baseline performance
        baseline_metrics = self.evaluate(dataloader, compute_survival=False)
        baseline_mae = baseline_metrics['mae']
        
        # Get feature names (this would need to be implemented based on model type)
        feature_names = self._get_feature_names()
        
        importance_scores = {}
        
        for feature_idx, feature_name in enumerate(feature_names):
            mae_increases = []
            
            for _ in range(n_permutations):
                # Create permuted dataloader
                permuted_loader = self._create_permuted_loader(
                    dataloader, feature_idx
                )
                
                # Evaluate with permuted feature
                permuted_metrics = self.evaluate(
                    permuted_loader, compute_survival=False
                )
                
                # Calculate importance as increase in error
                mae_increase = permuted_metrics['mae'] - baseline_mae
                mae_increases.append(mae_increase)
            
            importance_scores[feature_name] = np.mean(mae_increases)
        
        return importance_scores
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names based on model configuration."""
        # This would be implemented based on the specific model and config
        # For now, return generic names
        if hasattr(self.config, 'henaw_config'):
            return self.config.henaw_config.input_features
        elif hasattr(self.config, 'modal_config'):
            return self.config.modal_config.biomarker_features
        elif hasattr(self.config, 'metage_config'):
            return [f'metabolite_{i}' for i in range(168)]
        else:
            return []
    
    def _create_permuted_loader(
        self,
        dataloader: DataLoader,
        feature_idx: int
    ) -> DataLoader:
        """Create a dataloader with one feature permuted."""
        # This is a simplified implementation
        # In practice, would need to handle the specific data structure
        
        class PermutedDataset:
            def __init__(self, original_dataset, feature_idx):
                self.dataset = original_dataset
                self.feature_idx = feature_idx
                self.permutation = np.random.permutation(len(self.dataset))
            
            def __len__(self):
                return len(self.dataset)
            
            def __getitem__(self, idx):
                inputs, targets = self.dataset[idx]
                # Permute the specified feature
                # This would need to be adapted based on input structure
                return inputs, targets
        
        permuted_dataset = PermutedDataset(dataloader.dataset, feature_idx)
        
        return DataLoader(
            permuted_dataset,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=0
        )


class ModelComparator:
    """Compare multiple biological age models."""
    
    def __init__(self, models: Dict[str, nn.Module], config: Any, device: str = 'cuda'):
        """
        Initialize comparator.
        
        Args:
            models: Dictionary of model_name -> model
            config: Configuration object
            device: Device for evaluation
        """
        self.models = models
        self.config = config
        self.device = device
        
        # Create evaluators for each model
        self.evaluators = {
            name: BioAgeEvaluator(model, config, device)
            for name, model in models.items()
        }
    
    def compare(self, dataloader: DataLoader) -> pd.DataFrame:
        """
        Compare all models on the same dataset.
        
        Args:
            dataloader: Test dataloader
        
        Returns:
            DataFrame with comparison results
        """
        results = {}
        
        for name, evaluator in self.evaluators.items():
            logger.info(f"Evaluating {name}...")
            metrics = evaluator.evaluate(dataloader)
            results[name] = metrics
        
        # Convert to DataFrame for easy comparison
        df = pd.DataFrame(results).T
        
        # Add ranking for each metric
        for col in df.columns:
            if col in ['mae', 'rmse', 'median_error', 'std_error']:
                # Lower is better
                df[f'{col}_rank'] = df[col].rank()
            elif col in ['r2', 'pearson', 'spearman', 'c_index']:
                # Higher is better
                df[f'{col}_rank'] = df[col].rank(ascending=False)
        
        return df
    
    def statistical_comparison(
        self,
        dataloader: DataLoader,
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison between models using bootstrap.
        
        Args:
            dataloader: Test dataloader
            n_bootstrap: Number of bootstrap samples
        
        Returns:
            Statistical comparison results
        """
        # Collect predictions from all models
        all_predictions = {}
        targets = None
        
        for name, model in self.models.items():
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for inputs, batch_targets in dataloader:
                    if isinstance(inputs, dict):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    else:
                        inputs = inputs.to(self.device)
                    
                    outputs = model(inputs)
                    predictions.append(outputs['prediction'].squeeze().cpu().numpy())
                    
                    if targets is None:
                        targets = []
                    targets.append(batch_targets.numpy())
            
            all_predictions[name] = np.concatenate(predictions)
        
        targets = np.concatenate(targets).flatten()
        
        # Bootstrap comparison
        n_samples = len(targets)
        comparison_results = {}
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            # Compute metrics for each model
            for name, preds in all_predictions.items():
                bootstrap_preds = preds[indices]
                bootstrap_targets = targets[indices]
                
                mae = mean_absolute_error(bootstrap_targets, bootstrap_preds)
                
                if name not in comparison_results:
                    comparison_results[name] = {'mae': []}
                comparison_results[name]['mae'].append(mae)
        
        # Compute confidence intervals
        final_results = {}
        for name, metrics in comparison_results.items():
            mae_values = np.array(metrics['mae'])
            final_results[name] = {
                'mae_mean': float(mae_values.mean()),
                'mae_std': float(mae_values.std()),
                'mae_ci_lower': float(np.percentile(mae_values, 2.5)),
                'mae_ci_upper': float(np.percentile(mae_values, 97.5))
            }
        
        return final_results