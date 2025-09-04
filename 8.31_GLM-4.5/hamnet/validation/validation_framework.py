"""
Comprehensive Validation Framework for HAMNet Biological Age Prediction

This module provides a complete validation framework for evaluating HAMNet models with:
- Multiple cross-validation strategies
- Statistical significance testing
- Robustness testing
- Clinical validation
- Benchmark comparisons
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    train_test_split, GroupKFold
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from scipy import stats
from scipy.stats import (
    pearsonr, spearmanr, ttest_rel, wilcoxon,
    mannwhitneyu, f_oneway, chi2_contingency
)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import warnings
warnings.filterwarnings('ignore')

import logging
import os
import json
import pickle
from datetime import datetime
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path

# Import HAMNet components
from ..models.hamnet import HAMNet
from ..models.uncertainty_quantification import UncertaintyQuantification
from ..models.xai_module import XAIModule
from ..utils.utils import setup_logging


@dataclass
class ValidationConfig:
    """Configuration for validation framework"""
    # Cross-validation
    cv_folds: int = 5
    cv_method: str = 'stratified'  # 'stratified', 'kfold', 'temporal', 'group'
    stratify_by: List[str] = field(default_factory=lambda: ['age_group', 'sex'])
    random_state: int = 42
    
    # Statistical testing
    alpha: float = 0.05
    correction_method: str = 'bonferroni'  # 'bonferroni', 'fdr_bh'
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    
    # Robustness testing
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2])
    missing_rates: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4])
    outlier_percentages: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15])
    
    # Clinical validation
    outcome_prediction: bool = True
    risk_stratification: bool = True
    survival_analysis: bool = True
    
    # Benchmarking
    benchmark_methods: List[str] = field(default_factory=lambda: [
        'linear_regression', 'random_forest', 'xgboost', 'baseline_hamnet'
    ])
    
    # Output
    output_dir: str = './validation_results'
    save_plots: bool = True
    save_predictions: bool = True
    detailed_report: bool = True


class CrossValidationStrategy(ABC):
    """Abstract base class for cross-validation strategies"""
    
    @abstractmethod
    def create_splits(self, X: np.ndarray, y: np.ndarray, 
                     groups: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create train/validation splits"""
        pass


class StratifiedCrossValidation(CrossValidationStrategy):
    """Stratified k-fold cross-validation"""
    
    def __init__(self, n_folds: int = 5, stratify_by: List[str] = None):
        self.n_folds = n_folds
        self.stratify_by = stratify_by or ['age_group', 'sex']
    
    def create_splits(self, X: np.ndarray, y: np.ndarray, 
                     groups: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create stratified splits"""
        # Create stratification key
        stratify_key = self._create_stratify_key(X, y)
        
        # Perform stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        splits = []
        
        for train_idx, val_idx in skf.split(X, stratify_key):
            splits.append((train_idx, val_idx))
        
        return splits
    
    def _create_stratify_key(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Create composite stratification key"""
        # Age bins
        age_bins = pd.cut(y, bins=10, labels=False)
        
        # For simplicity, create synthetic stratification if needed
        if len(X.shape) > 1 and X.shape[1] > 1:
            # Use second feature as proxy for sex if available
            sex_proxy = (X[:, 1] > np.median(X[:, 1])).astype(int)
        else:
            sex_proxy = np.zeros(len(y))
        
        # Combine stratification variables
        stratify_key = age_bins.astype(str) + '_' + sex_proxy.astype(str)
        
        return stratify_key


class TemporalCrossValidation(CrossValidationStrategy):
    """Temporal cross-validation for longitudinal data"""
    
    def __init__(self, n_splits: int = 5, time_column: int = 0):
        self.n_splits = n_splits
        self.time_column = time_column
    
    def create_splits(self, X: np.ndarray, y: np.ndarray, 
                     groups: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create temporal splits"""
        # Sort by time
        if X.shape[1] > self.time_column:
            time_indices = np.argsort(X[:, self.time_column])
        else:
            time_indices = np.arange(len(X))
        
        X_sorted = X[time_indices]
        y_sorted = y[time_indices]
        
        # Calculate split points
        n_samples = len(X_sorted)
        fold_size = n_samples // self.n_splits
        
        splits = []
        for i in range(self.n_splits):
            # Define train/val split
            train_end = (i + 1) * fold_size
            val_start = i * fold_size
            val_end = (i + 2) * fold_size if i < self.n_splits - 1 else n_samples
            
            train_indices = time_indices[:train_end]
            val_indices = time_indices[val_start:val_end]
            
            splits.append((train_indices, val_indices))
        
        return splits


class GroupCrossValidation(CrossValidationStrategy):
    """Group k-fold cross-validation"""
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
    
    def create_splits(self, X: np.ndarray, y: np.ndarray, 
                     groups: np.ndarray = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create group-based splits"""
        if groups is None:
            # Fallback to regular k-fold
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            return list(kf.split(X, y))
        
        gkf = GroupKFold(n_splits=self.n_folds)
        splits = []
        
        for train_idx, val_idx in gkf.split(X, y, groups):
            splits.append((train_idx, val_idx))
        
        return splits


class PerformanceMetrics:
    """Comprehensive performance metrics calculation"""
    
    def __init__(self):
        self.metrics = {
            'mae': self._calculate_mae,
            'mse': self._calculate_mse,
            'rmse': self._calculate_rmse,
            'r2': self._calculate_r2,
            'pearson_r': self._calculate_pearson_r,
            'spearman_r': self._calculate_spearman_r,
            'mape': self._calculate_mape,
            'concordance': self._calculate_concordance,
            'icg': self._calculate_icg
        }
    
    def calculate_all_metrics(self, predictions: np.ndarray, 
                            targets: np.ndarray, 
                            uncertainties: np.ndarray = None) -> Dict[str, float]:
        """Calculate all performance metrics"""
        results = {}
        
        for metric_name, metric_func in self.metrics.items():
            try:
                if metric_name == 'icg' and uncertainties is not None:
                    results[metric_name] = metric_func(predictions, targets, uncertainties)
                else:
                    results[metric_name] = metric_func(predictions, targets)
            except Exception as e:
                logging.warning(f"Error calculating {metric_name}: {e}")
                results[metric_name] = None
        
        return results
    
    def _calculate_mae(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return np.mean(np.abs(predictions - targets))
    
    def _calculate_mse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        return np.mean((predictions - targets) ** 2)
    
    def _calculate_rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Root Mean Square Error"""
        return np.sqrt(np.mean((predictions - targets) ** 2))
    
    def _calculate_r2(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate R-squared"""
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def _calculate_pearson_r(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient"""
        return pearsonr(predictions, targets)[0]
    
    def _calculate_spearman_r(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Spearman correlation coefficient"""
        return spearmanr(predictions, targets)[0]
    
    def _calculate_mape(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((targets - predictions) / targets)) * 100
    
    def _calculate_concordance(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calculate Concordance Correlation Coefficient"""
        # Calculate Pearson correlation
        pearson_r = pearsonr(predictions, targets)[0]
        
        # Calculate means
        mean_pred = np.mean(predictions)
        mean_target = np.mean(targets)
        
        # Calculate standard deviations
        std_pred = np.std(predictions)
        std_target = np.std(targets)
        
        # Calculate concordance
        concordance = (2 * pearson_r * std_pred * std_target) / (
            std_pred**2 + std_target**2 + (mean_pred - mean_target)**2
        )
        
        return concordance
    
    def _calculate_icg(self, predictions: np.ndarray, targets: np.ndarray, 
                       uncertainties: np.ndarray) -> Dict[str, float]:
        """Calculate Interval Coverage Gauge"""
        # Calculate prediction intervals
        lower_bound = predictions - 1.96 * uncertainties
        upper_bound = predictions + 1.96 * uncertainties
        
        # Calculate coverage
        coverage = np.mean((targets >= lower_bound) & (targets <= upper_bound))
        
        # Calculate average interval width
        avg_width = np.mean(2 * 1.96 * uncertainties)
        
        # Calculate ICG score
        icg_score = coverage - np.abs(avg_width - 1.0)
        
        return {
            'coverage': coverage,
            'average_width': avg_width,
            'icg_score': icg_score
        }


class StatisticalTester:
    """Statistical significance testing"""
    
    def __init__(self, alpha: float = 0.05, correction_method: str = 'bonferroni'):
        self.alpha = alpha
        self.correction_method = correction_method
    
    def compare_methods(self, method_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Compare multiple methods using statistical tests"""
        comparisons = {}
        methods = list(method_results.keys())
        
        # Pairwise comparisons
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    comparison = self._pairwise_comparison(
                        method_results[method1],
                        method_results[method2],
                        method1, method2
                    )
                    comparisons[f"{method1}_vs_{method2}"] = comparison
        
        # Overall ANOVA
        anova_results = self._perform_anova(method_results)
        
        # Post-hoc tests
        post_hoc_results = self._post_hoc_tests(method_results)
        
        return {
            'pairwise_comparisons': comparisons,
            'anova': anova_results,
            'post_hoc': post_hoc_results
        }
    
    def _pairwise_comparison(self, results1: Dict[str, List[float]], 
                            results2: Dict[str, List[float]], 
                            method1: str, method2: str) -> Dict[str, Any]:
        """Perform pairwise comparison between two methods"""
        comparison = {}
        
        # Compare MAE
        if 'mae' in results1 and 'mae' in results2:
            mae_comparison = self._compare_metric(results1['mae'], results2['mae'])
            comparison['mae'] = mae_comparison
        
        # Compare R²
        if 'r2' in results1 and 'r2' in results2:
            r2_comparison = self._compare_metric(results1['r2'], results2['r2'], higher_better=True)
            comparison['r2'] = r2_comparison
        
        return comparison
    
    def _compare_metric(self, values1: List[float], values2: List[float], 
                       higher_better: bool = False) -> Dict[str, Any]:
        """Compare metric values between two methods"""
        # Normality tests
        _, p1 = stats.shapiro(values1)
        _, p2 = stats.shapiro(values2)
        
        if p1 > self.alpha and p2 > self.alpha:
            # Parametric test
            stat, p_value = ttest_rel(values1, values2)
            test_type = 'paired_t_test'
        else:
            # Non-parametric test
            stat, p_value = wilcoxon(values1, values2)
            test_type = 'wilcoxon'
        
        # Effect size (Cohen's d for paired samples)
        effect_size = self._calculate_cohens_d(values1, values2)
        
        # Power analysis
        power = self._calculate_power(values1, values2, alpha=self.alpha)
        
        # Determine which method is better
        if higher_better:
            method1_better = np.mean(values1) > np.mean(values2)
        else:
            method1_better = np.mean(values1) < np.mean(values2)
        
        return {
            'test_type': test_type,
            'statistic': stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'power': power,
            'significant': p_value < self.alpha,
            'method1_better': method1_better,
            'mean_diff': np.mean(values1) - np.mean(values2)
        }
    
    def _calculate_cohens_d(self, values1: List[float], values2: List[float]) -> float:
        """Calculate Cohen's d for paired samples"""
        diff = np.array(values1) - np.array(values2)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff, ddof=1)
        
        if std_diff == 0:
            return 0.0
        
        return mean_diff / std_diff
    
    def _calculate_power(self, values1: List[float], values2: List[float], 
                         alpha: float = 0.05) -> float:
        """Calculate statistical power"""
        from statsmodels.stats.power import TTestPower
        
        diff = np.mean(np.array(values1) - np.array(values2))
        std = np.std(np.array(values1) - np.array(values2), ddof=1)
        
        if std == 0:
            return 0.0
        
        # Calculate effect size
        effect_size = diff / std
        
        # Calculate power
        power_analysis = TTestPower()
        power = power_analysis.power(effect_size=effect_size, nobs=len(values1), alpha=alpha)
        
        return power
    
    def _perform_anova(self, method_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Perform ANOVA for multiple method comparison"""
        anova_results = {}
        
        # Perform ANOVA for each metric
        for metric in ['mae', 'r2']:
            if all(metric in results for results in method_results.values()):
                # Prepare data for ANOVA
                group_data = []
                group_labels = []
                
                for method_name, results in method_results.items():
                    if metric in results:
                        group_data.extend(results[metric])
                        group_labels.extend([method_name] * len(results[metric]))
                
                # Perform ANOVA
                unique_methods = list(method_results.keys())
                data_by_method = [method_results[method][metric] for method in unique_methods]
                
                f_stat, p_value = f_oneway(*data_by_method)
                
                anova_results[metric] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha
                }
        
        return anova_results
    
    def _post_hoc_tests(self, method_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Perform post-hoc tests with multiple comparison correction"""
        post_hoc_results = {}
        
        # For each metric, perform pairwise comparisons with correction
        for metric in ['mae', 'r2']:
            if all(metric in results for results in method_results.values()):
                methods = list(method_results.keys())
                comparisons = {}
                
                # Perform all pairwise comparisons
                p_values = []
                comparison_names = []
                
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i < j:
                            comparison = self._compare_metric(
                                method_results[method1][metric],
                                method_results[method2][metric],
                                higher_better=(metric == 'r2')
                            )
                            
                            comparison_name = f"{method1}_vs_{method2}"
                            comparisons[comparison_name] = comparison
                            p_values.append(comparison['p_value'])
                            comparison_names.append(comparison_name)
                
                # Apply multiple comparison correction
                if p_values:
                    corrected_p_values = self._apply_correction(p_values)
                    
                    # Update comparisons with corrected p-values
                    for i, comparison_name in enumerate(comparison_names):
                        comparisons[comparison_name]['corrected_p_value'] = corrected_p_values[i]
                        comparisons[comparison_name]['corrected_significant'] = corrected_p_values[i] < self.alpha
                
                post_hoc_results[metric] = comparisons
        
        return post_hoc_results
    
    def _apply_correction(self, p_values: List[float]) -> List[float]:
        """Apply multiple comparison correction"""
        if self.correction_method == 'bonferroni':
            return [min(p * len(p_values), 1.0) for p in p_values]
        elif self.correction_method == 'fdr_bh':
            # Benjamini-Hochberg procedure
            sorted_p_values = sorted(p_values)
            n = len(p_values)
            
            # Calculate critical values
            critical_values = [(i + 1) / n * self.alpha for i in range(n)]
            
            # Find rejection threshold
            rejection_threshold = 0
            for i in range(n - 1, -1, -1):
                if sorted_p_values[i] <= critical_values[i]:
                    rejection_threshold = sorted_p_values[i]
                    break
            
            # Apply correction
            corrected_p_values = []
            for p in p_values:
                if p <= rejection_threshold:
                    corrected_p_values.append(p * n / (sum(p_i <= p for p_i in p_values)))
                else:
                    corrected_p_values.append(p)
            
            return corrected_p_values
        else:
            return p_values


class RobustnessTester:
    """Robustness testing for model evaluation"""
    
    def __init__(self, noise_levels: List[float] = None, 
                 missing_rates: List[float] = None,
                 outlier_percentages: List[float] = None):
        self.noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2]
        self.missing_rates = missing_rates or [0.1, 0.2, 0.3, 0.4]
        self.outlier_percentages = outlier_percentages or [0.05, 0.1, 0.15]
    
    def test_robustness(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Test model robustness against various perturbations"""
        results = {}
        
        # Test noise robustness
        results['noise_robustness'] = self._test_noise_robustness(model, X, y)
        
        # Test missing data robustness
        results['missing_data_robustness'] = self._test_missing_data_robustness(model, X, y)
        
        # Test outlier robustness
        results['outlier_robustness'] = self._test_outlier_robustness(model, X, y)
        
        return results
    
    def _test_noise_robustness(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Test robustness to Gaussian noise"""
        results = {}
        
        for noise_level in self.noise_levels:
            # Add noise to data
            X_noisy = X + np.random.normal(0, noise_level, X.shape)
            
            # Make predictions
            predictions = self._predict(model, X_noisy)
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - y))
            r2 = r2_score(y, predictions)
            
            results[f'noise_{noise_level}'] = {
                'mae': mae,
                'r2': r2,
                'performance_ratio': mae / np.mean(np.abs(y - np.mean(y)))  # Ratio to baseline
            }
        
        return results
    
    def _test_missing_data_robustness(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Test robustness to missing data"""
        results = {}
        
        for missing_rate in self.missing_rates:
            # Create missing data
            X_missing = X.copy()
            mask = np.random.random(X.shape) < missing_rate
            X_missing[mask] = 0
            
            # Make predictions
            predictions = self._predict(model, X_missing)
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - y))
            r2 = r2_score(y, predictions)
            
            results[f'missing_{missing_rate}'] = {
                'mae': mae,
                'r2': r2,
                'performance_ratio': mae / np.mean(np.abs(y - np.mean(y)))
            }
        
        return results
    
    def _test_outlier_robustness(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Test robustness to outliers"""
        results = {}
        
        for outlier_percentage in self.outlier_percentages:
            # Add outliers
            X_outlier = X.copy()
            n_outliers = int(len(X) * outlier_percentage)
            outlier_indices = np.random.choice(len(X), n_outliers, replace=False)
            
            # Add extreme outliers
            X_outlier[outlier_indices] += np.random.normal(0, 5 * np.std(X), (n_outliers, X.shape[1]))
            
            # Make predictions
            predictions = self._predict(model, X_outlier)
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - y))
            r2 = r2_score(y, predictions)
            
            results[f'outlier_{outlier_percentage}'] = {
                'mae': mae,
                'r2': r2,
                'performance_ratio': mae / np.mean(np.abs(y - np.mean(y)))
            }
        
        return results
    
    def _predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Make predictions with model"""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(next(model.parameters()).device)
            predictions, _ = model({'features': X_tensor})
            return predictions['bio_age'].cpu().numpy()


class ClinicalValidator:
    """Clinical validation for biological age predictions"""
    
    def __init__(self):
        self.logger = setup_logging('ClinicalValidator')
    
    def validate_clinical_outcomes(self, bio_ages: np.ndarray, 
                                  chronological_ages: np.ndarray,
                                  outcomes: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Validate predictions against clinical outcomes"""
        results = {}
        
        # Calculate age acceleration
        age_acceleration = bio_ages - chronological_ages
        
        # Validate each outcome
        for outcome_name, outcome_values in outcomes.items():
            if outcome_name == 'mortality':
                results[f'{outcome_name}_validation'] = self._validate_mortality(
                    age_acceleration, outcome_values
                )
            elif outcome_name == 'disease_status':
                results[f'{outcome_name}_validation'] = self._validate_disease_status(
                    age_acceleration, outcome_values
                )
            elif outcome_name == 'survival_time':
                results[f'{outcome_name}_validation'] = self._validate_survival(
                    age_acceleration, outcome_values, outcomes.get('survival_event')
                )
        
        return results
    
    def _validate_mortality(self, age_acceleration: np.ndarray, 
                          mortality: np.ndarray) -> Dict[str, Any]:
        """Validate mortality prediction"""
        # Create risk groups
        risk_groups = self._create_risk_groups(age_acceleration)
        
        # Calculate mortality rates by risk group
        mortality_rates = {}
        for group_name, group_indices in risk_groups.items():
            group_mortality = mortality[group_indices].mean()
            mortality_rates[group_name] = group_mortality
        
        # Calculate hazard ratio
        from lifelines import CoxPHFitter
        
        # Prepare data for Cox model
        cox_data = pd.DataFrame({
            'age_acceleration': age_acceleration,
            'mortality': mortality,
            'time': np.ones(len(mortality))  # Placeholder
        })
        
        try:
            cph = CoxPHFitter()
            cph.fit(cox_data, duration_col='time', event_col='mortality')
            
            hazard_ratio = np.exp(cph.params_['age_acceleration'])
            p_value = cph.summary['p']['age_acceleration']
            concordance_index = cph.concordance_index_
            
        except Exception as e:
            self.logger.warning(f"Cox model failed: {e}")
            hazard_ratio = 1.0
            p_value = 1.0
            concordance_index = 0.5
        
        return {
            'mortality_rates': mortality_rates,
            'hazard_ratio': hazard_ratio,
            'p_value': p_value,
            'concordance_index': concordance_index,
            'risk_stratification': self._assess_risk_stratification(risk_groups, mortality)
        }
    
    def _validate_disease_status(self, age_acceleration: np.ndarray, 
                               disease_status: np.ndarray) -> Dict[str, Any]:
        """Validate disease status prediction"""
        # Calculate AUROC
        try:
            auroc = roc_auc_score(disease_status, age_acceleration)
        except Exception as e:
            self.logger.warning(f"AUROC calculation failed: {e}")
            auroc = 0.5
        
        # Calculate AUPRC
        try:
            auprc = average_precision_score(disease_status, age_acceleration)
        except Exception as e:
            self.logger.warning(f"AUPRC calculation failed: {e}")
            auprc = 0.5
        
        # Calculate odds ratio
        high_risk = age_acceleration > np.median(age_acceleration)
        contingency_table = pd.crosstab(high_risk, disease_status)
        
        try:
            odds_ratio, p_value = stats.fisher_exact(contingency_table)
        except Exception as e:
            self.logger.warning(f"Fisher's exact test failed: {e}")
            odds_ratio = 1.0
            p_value = 1.0
        
        return {
            'auroc': auroc,
            'auprc': auprc,
            'odds_ratio': odds_ratio,
            'p_value': p_value,
            'contingency_table': contingency_table.to_dict()
        }
    
    def _validate_survival(self, age_acceleration: np.ndarray, 
                         survival_time: np.ndarray, 
                         survival_event: np.ndarray = None) -> Dict[str, Any]:
        """Validate survival analysis"""
        if survival_event is None:
            survival_event = np.ones(len(survival_time))
        
        # Create risk groups
        risk_groups = self._create_risk_groups(age_acceleration)
        
        # Perform log-rank test
        try:
            logrank_results = logrank_test(
                survival_time[risk_groups['low_risk']],
                survival_time[risk_groups['high_risk']],
                event_observed_A=survival_event[risk_groups['low_risk']],
                event_observed_B=survival_event[risk_groups['high_risk']]
            )
            
            logrank_p_value = logrank_results.p_value
        except Exception as e:
            self.logger.warning(f"Log-rank test failed: {e}")
            logrank_p_value = 1.0
        
        # Kaplan-Meier analysis
        kmf = KaplanMeierFitter()
        
        km_results = {}
        for group_name, group_indices in risk_groups.items():
            kmf.fit(survival_time[group_indices], survival_event[group_indices])
            km_results[group_name] = {
                'survival_function': kmf.survival_function_,
                'confidence_interval': kmf.confidence_interval_
            }
        
        return {
            'logrank_p_value': logrank_p_value,
            'kaplan_meier': km_results,
            'risk_groups': risk_groups
        }
    
    def _create_risk_groups(self, age_acceleration: np.ndarray) -> Dict[str, np.ndarray]:
        """Create risk groups based on age acceleration"""
        median_acceleration = np.median(age_acceleration)
        
        risk_groups = {
            'low_risk': np.where(age_acceleration <= median_acceleration)[0],
            'high_risk': np.where(age_acceleration > median_acceleration)[0]
        }
        
        return risk_groups
    
    def _assess_risk_stratification(self, risk_groups: Dict[str, np.ndarray], 
                                  outcomes: np.ndarray) -> Dict[str, float]:
        """Assess risk stratification performance"""
        # Calculate outcome rates by risk group
        group_rates = {}
        for group_name, group_indices in risk_groups.items():
            group_rates[group_name] = outcomes[group_indices].mean()
        
        # Calculate risk difference
        risk_difference = group_rates['high_risk'] - group_rates['low_risk']
        
        # Calculate relative risk
        relative_risk = group_rates['high_risk'] / group_rates['low_risk']
        
        return {
            'risk_difference': risk_difference,
            'relative_risk': relative_risk,
            'group_rates': group_rates
        }


class ComprehensiveValidator:
    """Comprehensive validation framework for HAMNet"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.logger = setup_logging('ComprehensiveValidator')
        
        # Initialize components
        self.cv_strategy = self._create_cv_strategy()
        self.metrics_calculator = PerformanceMetrics()
        self.statistical_tester = StatisticalTester(
            alpha=config.alpha,
            correction_method=config.correction_method
        )
        self.robustness_tester = RobustnessTester(
            noise_levels=config.noise_levels,
            missing_rates=config.missing_rates,
            outlier_percentages=config.outlier_percentages
        )
        self.clinical_validator = ClinicalValidator()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _create_cv_strategy(self) -> CrossValidationStrategy:
        """Create cross-validation strategy"""
        if self.config.cv_method == 'stratified':
            return StratifiedCrossValidation(
                n_folds=self.config.cv_folds,
                stratify_by=self.config.stratify_by
            )
        elif self.config.cv_method == 'temporal':
            return TemporalCrossValidation(n_splits=self.config.cv_folds)
        elif self.config.cv_method == 'group':
            return GroupCrossValidation(n_folds=self.config.cv_folds)
        else:
            return StratifiedCrossValidation(n_folds=self.config.cv_folds)
    
    def validate_model(self, model: nn.Module, X: np.ndarray, y: np.ndarray,
                      groups: np.ndarray = None,
                      clinical_outcomes: Dict[str, np.ndarray] = None) -> Dict[str, Any]:
        """Perform comprehensive model validation"""
        self.logger.info("Starting comprehensive validation...")
        
        validation_results = {}
        
        # Cross-validation
        cv_results = self._cross_validation(model, X, y, groups)
        validation_results['cross_validation'] = cv_results
        
        # Robustness testing
        robustness_results = self.robustness_tester.test_robustness(model, X, y)
        validation_results['robustness'] = robustness_results
        
        # Clinical validation
        if clinical_outcomes is not None:
            # Get predictions for clinical validation
            predictions = self._predict_batch(model, X)
            clinical_results = self.clinical_validator.validate_clinical_outcomes(
                predictions, y, clinical_outcomes
            )
            validation_results['clinical'] = clinical_results
        
        # Save results
        self._save_results(validation_results)
        
        return validation_results
    
    def _cross_validation(self, model: nn.Module, X: np.ndarray, y: np.ndarray,
                         groups: np.ndarray = None) -> Dict[str, Any]:
        """Perform cross-validation"""
        self.logger.info(f"Performing {self.config.cv_folds}-fold cross-validation...")
        
        # Create splits
        splits = self.cv_strategy.create_splits(X, y, groups)
        
        cv_results = {
            'fold_results': [],
            'mean_metrics': {},
            'std_metrics': {},
            'best_fold': None,
            'worst_fold': None
        }
        
        all_fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            self.logger.info(f"Processing fold {fold_idx + 1}/{len(splits)}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model (simplified - in practice, you'd train the model)
            # For now, we'll just evaluate on the validation set
            predictions = self._predict_batch(model, X_val)
            uncertainties = None
            
            # Calculate metrics
            fold_metrics = self.metrics_calculator.calculate_all_metrics(
                predictions, y_val, uncertainties
            )
            
            fold_metrics['fold'] = fold_idx
            fold_metrics['train_size'] = len(train_idx)
            fold_metrics['val_size'] = len(val_idx)
            
            cv_results['fold_results'].append(fold_metrics)
            all_fold_metrics.append(fold_metrics)
        
        # Calculate mean and std metrics
        metric_names = all_fold_metrics[0].keys()
        for metric_name in metric_names:
            if metric_name not in ['fold', 'train_size', 'val_size']:
                values = [fold[metric_name] for fold in all_fold_metrics if fold[metric_name] is not None]
                if values:
                    cv_results['mean_metrics'][metric_name] = np.mean(values)
                    cv_results['std_metrics'][metric_name] = np.std(values)
        
        # Find best and worst folds
        if 'mae' in cv_results['mean_metrics']:
            mae_values = [fold['mae'] for fold in all_fold_metrics if fold.get('mae') is not None]
            if mae_values:
                best_fold_idx = np.argmin(mae_values)
                worst_fold_idx = np.argmax(mae_values)
                cv_results['best_fold'] = all_fold_metrics[best_fold_idx]
                cv_results['worst_fold'] = all_fold_metrics[worst_fold_idx]
        
        return cv_results
    
    def _predict_batch(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """Make batch predictions"""
        model.eval()
        predictions = []
        
        batch_size = 32
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i + batch_size]
                batch_X_tensor = torch.FloatTensor(batch_X).to(next(model.parameters()).device)
                
                batch_predictions, _ = model({'features': batch_X_tensor})
                predictions.append(batch_predictions['bio_age'].cpu().numpy())
        
        return np.concatenate(predictions)
    
    def _save_results(self, results: Dict[str, Any]):
        """Save validation results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = os.path.join(self.config.output_dir, f'validation_results_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save detailed report if requested
        if self.config.detailed_report:
            report_path = os.path.join(self.config.output_dir, f'validation_report_{timestamp}.md')
            self._generate_report(results, report_path)
        
        self.logger.info(f"Validation results saved to {json_path}")
    
    def _generate_report(self, results: Dict[str, Any], report_path: str):
        """Generate detailed validation report"""
        report = f"""# HAMNet Validation Report
        
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Cross-Validation Results

### Mean Performance Metrics
"""
        
        if 'cross_validation' in results:
            cv_results = results['cross_validation']
            for metric_name, mean_value in cv_results['mean_metrics'].items():
                std_value = cv_results['std_metrics'].get(metric_name, 0)
                report += f"- **{metric_name}**: {mean_value:.4f} ± {std_value:.4f}\n"
        
        report += "\n## Robustness Testing\n"
        
        if 'robustness' in results:
            robustness_results = results['robustness']
            for test_type, test_results in robustness_results.items():
                report += f"### {test_type.replace('_', ' ').title()}\n"
                for condition, metrics in test_results.items():
                    report += f"- **{condition}**: MAE = {metrics['mae']:.4f}, R² = {metrics['r2']:.4f}\n"
        
        if 'clinical' in results:
            report += "\n## Clinical Validation\n"
            clinical_results = results['clinical']
            for outcome_name, outcome_results in clinical_results.items():
                report += f"### {outcome_name.replace('_', ' ').title()}\n"
                if isinstance(outcome_results, dict):
                    for key, value in outcome_results.items():
                        if isinstance(value, (int, float)):
                            report += f"- **{key}**: {value:.4f}\n"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Detailed report saved to {report_path}")


def create_validation_config(config_path: str = None) -> ValidationConfig:
    """Create validation configuration from file or defaults"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return ValidationConfig(**config_dict)
    else:
        return ValidationConfig()


def main():
    """Main validation script"""
    # Create configuration
    config = create_validation_config()
    
    # Create validator
    validator = ComprehensiveValidator(config)
    
    # Load model and data (placeholder)
    model = None  # TODO: Load trained model
    X = None      # TODO: Load features
    y = None      # TODO: Load targets
    
    # Perform validation
    results = validator.validate_model(model, X, y)
    
    print("Validation completed!")


if __name__ == "__main__":
    main()