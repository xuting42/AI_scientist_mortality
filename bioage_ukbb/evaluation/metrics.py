"""
Evaluation metrics for biological age algorithms.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
import warnings


class BioAgeMetrics:
    """Comprehensive metrics for biological age evaluation."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        self.results = {}
    
    def calculate_basic_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate basic regression metrics.
        
        Args:
            predictions: Predicted ages
            targets: True ages
            sample_weights: Optional sample weights
        
        Returns:
            Dictionary of metrics
        """
        # Ensure numpy arrays
        predictions = np.asarray(predictions).flatten()
        targets = np.asarray(targets).flatten()
        
        # Basic metrics
        mae = mean_absolute_error(targets, predictions, sample_weight=sample_weights)
        mse = mean_squared_error(targets, predictions, sample_weight=sample_weights)
        rmse = np.sqrt(mse)
        
        # R-squared
        r2 = r2_score(targets, predictions, sample_weight=sample_weights)
        
        # Correlation coefficients
        pearson_r, pearson_p = pearsonr(predictions, targets)
        spearman_r, spearman_p = spearmanr(predictions, targets)
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # Bias (systematic error)
        bias = np.mean(predictions - targets)
        
        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'mape': mape,
            'bias': bias
        }
        
        return metrics
    
    def calculate_age_acceleration(
        self,
        biological_age: np.ndarray,
        chronological_age: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate age acceleration metrics.
        
        Args:
            biological_age: Predicted biological age
            chronological_age: Chronological age
        
        Returns:
            Dictionary with age acceleration metrics
        """
        # Age acceleration (residuals from regression)
        z = np.polyfit(chronological_age, biological_age, 1)
        p = np.poly1d(z)
        expected_bio_age = p(chronological_age)
        
        age_acceleration = biological_age - expected_bio_age
        
        # Alternative: simple difference
        age_gap = biological_age - chronological_age
        
        # Statistics
        metrics = {
            'age_acceleration_mean': np.mean(age_acceleration),
            'age_acceleration_std': np.std(age_acceleration),
            'age_acceleration_median': np.median(age_acceleration),
            'age_gap_mean': np.mean(age_gap),
            'age_gap_std': np.std(age_gap),
            'accelerated_aging_proportion': np.mean(age_acceleration > 0),
            'regression_slope': z[0],
            'regression_intercept': z[1],
            'age_acceleration_values': age_acceleration,
            'age_gap_values': age_gap
        }
        
        return metrics
    
    def calculate_survival_metrics(
        self,
        biological_age: np.ndarray,
        event_times: np.ndarray,
        event_indicators: np.ndarray,
        chronological_age: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate survival analysis metrics.
        
        Args:
            biological_age: Predicted biological age
            event_times: Time to event (e.g., mortality)
            event_indicators: Binary event indicators (1=event, 0=censored)
            chronological_age: Optional chronological age for comparison
        
        Returns:
            Dictionary of survival metrics
        """
        # C-index for biological age
        c_index_bio = concordance_index(
            event_times,
            -biological_age,  # Negative because higher age = higher risk
            event_indicators
        )
        
        metrics = {'c_index_biological': c_index_bio}
        
        # Compare with chronological age if provided
        if chronological_age is not None:
            c_index_chrono = concordance_index(
                event_times,
                -chronological_age,
                event_indicators
            )
            metrics['c_index_chronological'] = c_index_chrono
            metrics['c_index_improvement'] = c_index_bio - c_index_chrono
        
        # Hazard ratio estimation (simplified)
        if chronological_age is not None:
            age_acceleration = biological_age - chronological_age
            # Stratify by age acceleration
            accelerated = age_acceleration > np.median(age_acceleration)
            
            # Calculate event rates
            event_rate_accelerated = np.mean(event_indicators[accelerated])
            event_rate_normal = np.mean(event_indicators[~accelerated])
            
            if event_rate_normal > 0:
                hazard_ratio = event_rate_accelerated / event_rate_normal
            else:
                hazard_ratio = np.nan
            
            metrics['hazard_ratio'] = hazard_ratio
            metrics['event_rate_accelerated'] = event_rate_accelerated
            metrics['event_rate_normal'] = event_rate_normal
        
        return metrics
    
    def calculate_subgroup_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        groups: np.ndarray,
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for different subgroups.
        
        Args:
            predictions: Predicted ages
            targets: True ages
            groups: Group labels for each sample
            group_names: Optional names for groups
        
        Returns:
            Dictionary of metrics per group
        """
        unique_groups = np.unique(groups)
        
        if group_names is None:
            group_names = [f"Group_{g}" for g in unique_groups]
        
        subgroup_metrics = {}
        
        for group_id, group_name in zip(unique_groups, group_names):
            mask = groups == group_id
            group_preds = predictions[mask]
            group_targets = targets[mask]
            
            if len(group_preds) > 1:
                metrics = self.calculate_basic_metrics(group_preds, group_targets)
                subgroup_metrics[group_name] = metrics
            else:
                subgroup_metrics[group_name] = {'n_samples': len(group_preds)}
        
        return subgroup_metrics
    
    def calculate_calibration_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, Any]:
        """
        Calculate calibration metrics.
        
        Args:
            predictions: Predicted ages
            targets: True ages
            n_bins: Number of bins for calibration
        
        Returns:
            Dictionary with calibration metrics
        """
        # Bin predictions
        bin_edges = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(predictions, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate mean prediction and target per bin
        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                mean_pred = np.mean(predictions[mask])
                mean_true = np.mean(targets[mask])
                n_samples = np.sum(mask)
                calibration_data.append({
                    'bin': i,
                    'mean_prediction': mean_pred,
                    'mean_target': mean_true,
                    'n_samples': n_samples,
                    'calibration_error': mean_pred - mean_true
                })
        
        calibration_df = pd.DataFrame(calibration_data)
        
        # Expected Calibration Error (ECE)
        if len(calibration_df) > 0:
            weights = calibration_df['n_samples'] / len(predictions)
            ece = np.sum(weights * np.abs(calibration_df['calibration_error']))
        else:
            ece = np.nan
        
        # Maximum Calibration Error (MCE)
        if len(calibration_df) > 0:
            mce = np.max(np.abs(calibration_df['calibration_error']))
        else:
            mce = np.nan
        
        return {
            'ece': ece,
            'mce': mce,
            'calibration_data': calibration_df,
            'n_bins': n_bins
        }
    
    def calculate_uncertainty_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate uncertainty quantification metrics.
        
        Args:
            predictions: Predicted ages
            targets: True ages
            uncertainties: Uncertainty estimates
        
        Returns:
            Dictionary of uncertainty metrics
        """
        errors = np.abs(predictions - targets)
        
        # Correlation between uncertainty and error
        uncertainty_error_corr, _ = pearsonr(uncertainties, errors)
        
        # Uncertainty calibration (are uncertainties well-calibrated?)
        # Sort by uncertainty and check if errors increase
        sorted_idx = np.argsort(uncertainties)
        sorted_errors = errors[sorted_idx]
        sorted_uncertainties = uncertainties[sorted_idx]
        
        # Bin into quantiles
        n_bins = 10
        bin_size = len(predictions) // n_bins
        
        calibration_scores = []
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(predictions)
            
            bin_errors = sorted_errors[start_idx:end_idx]
            bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
            
            if len(bin_errors) > 0:
                # Check if average error matches average uncertainty
                calibration_score = np.mean(bin_errors) / (np.mean(bin_uncertainties) + 1e-8)
                calibration_scores.append(calibration_score)
        
        # Uncertainty sharpness (lower is better)
        sharpness = np.mean(uncertainties)
        
        # Coverage at different confidence levels
        coverage_90 = np.mean(errors <= 1.645 * uncertainties)
        coverage_95 = np.mean(errors <= 1.96 * uncertainties)
        coverage_99 = np.mean(errors <= 2.576 * uncertainties)
        
        return {
            'uncertainty_error_correlation': uncertainty_error_corr,
            'uncertainty_calibration': np.mean(calibration_scores),
            'uncertainty_sharpness': sharpness,
            'coverage_90': coverage_90,
            'coverage_95': coverage_95,
            'coverage_99': coverage_99
        }
    
    def calculate_bootstrap_ci(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        metric_fn: callable,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate bootstrap confidence intervals for a metric.
        
        Args:
            predictions: Predicted ages
            targets: True ages
            metric_fn: Function to calculate metric
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI
        
        Returns:
            Dictionary with metric value and CI
        """
        n_samples = len(predictions)
        bootstrap_metrics = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_preds = predictions[indices]
            boot_targets = targets[indices]
            
            # Calculate metric
            metric_value = metric_fn(boot_targets, boot_preds)
            bootstrap_metrics.append(metric_value)
        
        # Calculate percentile CI
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        
        # Original metric
        original_metric = metric_fn(targets, predictions)
        
        return {
            'metric': original_metric,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std': np.std(bootstrap_metrics),
            'confidence_level': confidence_level
        }
    
    def comprehensive_evaluation(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        chronological_age: Optional[np.ndarray] = None,
        uncertainties: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        event_times: Optional[np.ndarray] = None,
        event_indicators: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.
        
        Args:
            predictions: Predicted biological ages
            targets: Target ages (usually chronological)
            chronological_age: Chronological ages
            uncertainties: Uncertainty estimates
            groups: Group labels for subgroup analysis
            event_times: Survival times
            event_indicators: Event indicators
        
        Returns:
            Comprehensive evaluation results
        """
        results = {}
        
        # Basic metrics
        results['basic_metrics'] = self.calculate_basic_metrics(predictions, targets)
        
        # Age acceleration
        if chronological_age is not None:
            results['age_acceleration'] = self.calculate_age_acceleration(
                predictions, chronological_age
            )
        
        # Survival metrics
        if event_times is not None and event_indicators is not None:
            results['survival_metrics'] = self.calculate_survival_metrics(
                predictions, event_times, event_indicators, chronological_age
            )
        
        # Subgroup analysis
        if groups is not None:
            results['subgroup_metrics'] = self.calculate_subgroup_metrics(
                predictions, targets, groups
            )
        
        # Calibration
        results['calibration'] = self.calculate_calibration_metrics(predictions, targets)
        
        # Uncertainty metrics
        if uncertainties is not None:
            results['uncertainty_metrics'] = self.calculate_uncertainty_metrics(
                predictions, targets, uncertainties
            )
        
        # Bootstrap CI for key metrics
        results['mae_ci'] = self.calculate_bootstrap_ci(
            predictions, targets, mean_absolute_error
        )
        
        self.results = results
        return results
    
    def print_summary(self, results: Optional[Dict] = None) -> None:
        """Print summary of evaluation results."""
        if results is None:
            results = self.results
        
        print("\n" + "="*60)
        print("BIOLOGICAL AGE EVALUATION SUMMARY")
        print("="*60)
        
        # Basic metrics
        if 'basic_metrics' in results:
            print("\n📊 Basic Metrics:")
            for metric, value in results['basic_metrics'].items():
                if not metric.endswith('_p'):
                    print(f"  {metric.upper()}: {value:.4f}")
        
        # Age acceleration
        if 'age_acceleration' in results:
            print("\n⏱️ Age Acceleration:")
            aa = results['age_acceleration']
            print(f"  Mean acceleration: {aa['age_acceleration_mean']:.2f} years")
            print(f"  Std deviation: {aa['age_acceleration_std']:.2f} years")
            print(f"  Accelerated aging proportion: {aa['accelerated_aging_proportion']:.1%}")
        
        # Survival metrics
        if 'survival_metrics' in results:
            print("\n💀 Survival Analysis:")
            for metric, value in results['survival_metrics'].items():
                if not np.isnan(value):
                    print(f"  {metric}: {value:.4f}")
        
        # Calibration
        if 'calibration' in results:
            print("\n🎯 Calibration:")
            print(f"  ECE: {results['calibration']['ece']:.4f}")
            print(f"  MCE: {results['calibration']['mce']:.4f}")
        
        # Uncertainty
        if 'uncertainty_metrics' in results:
            print("\n❓ Uncertainty Quantification:")
            um = results['uncertainty_metrics']
            print(f"  Error-uncertainty correlation: {um['uncertainty_error_correlation']:.3f}")
            print(f"  95% coverage: {um['coverage_95']:.1%}")
        
        print("\n" + "="*60)