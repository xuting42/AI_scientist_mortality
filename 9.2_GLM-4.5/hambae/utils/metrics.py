"""
Evaluation metrics for HAMBAE algorithm system.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    calibration_curve,
    brier_score_loss,
)
import scipy.stats as stats
import logging

logger = logging.getLogger(__name__)


def compute_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    uncertainty: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mortality_predictions: Optional[Union[torch.Tensor, np.ndarray]] = None,
    mortality_targets: Optional[Union[torch.Tensor, np.ndarray]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Age predictions
        targets: True age values
        uncertainty: Uncertainty estimates
        mortality_predictions: Mortality predictions
        mortality_targets: True mortality status
        
    Returns:
        Dictionary of computed metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if uncertainty is not None and isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.cpu().numpy()
    if mortality_predictions is not None and isinstance(mortality_predictions, torch.Tensor):
        mortality_predictions = mortality_predictions.cpu().numpy()
    if mortality_targets is not None and isinstance(mortality_targets, torch.Tensor):
        mortality_targets = mortality_targets.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    if uncertainty is not None:
        uncertainty = uncertainty.flatten()
    
    metrics = {}
    
    # Age prediction metrics
    metrics['mae'] = mean_absolute_error(targets, predictions)
    metrics['mse'] = mean_squared_error(targets, predictions)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['r2'] = r2_score(targets, predictions)
    
    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(predictions, targets)
    metrics['pearson_correlation'] = pearson_corr
    metrics['pearson_p_value'] = pearson_p
    
    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(predictions, targets)
    metrics['spearman_correlation'] = spearman_corr
    metrics['spearman_p_value'] = spearman_p
    
    # Mean absolute percentage error
    metrics['mape'] = np.mean(np.abs((targets - predictions) / targets)) * 100
    
    # Median absolute error
    metrics['median_ae'] = np.median(np.abs(targets - predictions))
    
    # Age acceleration metrics
    age_acceleration = predictions - targets
    metrics['mean_age_acceleration'] = np.mean(age_acceleration)
    metrics['std_age_acceleration'] = np.std(age_acceleration)
    metrics['median_age_acceleration'] = np.median(age_acceleration)
    
    # Uncertainty metrics if available
    if uncertainty is not None:
        uncertainty_metrics = compute_uncertainty_metrics(predictions, targets, uncertainty)
        metrics.update(uncertainty_metrics)
    
    # Mortality prediction metrics if available
    if mortality_predictions is not None and mortality_targets is not None:
        mortality_metrics = compute_mortality_metrics(mortality_predictions, mortality_targets)
        metrics.update(mortality_metrics)
    
    return metrics


def compute_uncertainty_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    uncertainty: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """
    Compute uncertainty quantification metrics.
    
    Args:
        predictions: Model predictions
        targets: True values
        uncertainty: Uncertainty estimates
        
    Returns:
        Dictionary of uncertainty metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    uncertainty = uncertainty.flatten()
    
    metrics = {}
    
    # Uncertainty calibration metrics
    metrics['mean_uncertainty'] = np.mean(uncertainty)
    metrics['std_uncertainty'] = np.std(uncertainty)
    metrics['median_uncertainty'] = np.median(uncertainty)
    
    # Uncertainty vs error correlation
    errors = np.abs(predictions - targets)
    uncertainty_error_corr, _ = stats.pearsonr(uncertainty, errors)
    metrics['uncertainty_error_correlation'] = uncertainty_error_corr
    
    # Expected Calibration Error (ECE)
    ece = compute_expected_calibration_error(predictions, targets, uncertainty)
    metrics['expected_calibration_error'] = ece
    
    # Negative log likelihood
    nll = compute_negative_log_likelihood(predictions, targets, uncertainty)
    metrics['negative_log_likelihood'] = nll
    
    # Interval coverage metrics
    for confidence in [0.5, 0.8, 0.9, 0.95]:
        interval_width = 2 * stats.norm.ppf((1 + confidence) / 2) * uncertainty
        coverage = np.mean(np.abs(predictions - targets) <= interval_width)
        metrics[f'coverage_{confidence}'] = coverage
        metrics[f'interval_width_{confidence}'] = np.mean(interval_width)
    
    # Sparsification metrics
    sparsification_metrics = compute_sparsification_metrics(predictions, targets, uncertainty)
    metrics.update(sparsification_metrics)
    
    return metrics


def compute_mortality_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
) -> Dict[str, float]:
    """
    Compute mortality prediction metrics.
    
    Args:
        predictions: Mortality predictions
        targets: True mortality status
        
    Returns:
        Dictionary of mortality metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Convert to binary predictions
    binary_predictions = (predictions > 0.5).astype(int)
    
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(targets, binary_predictions)
    metrics['precision'] = precision_score(targets, binary_predictions, zero_division=0)
    metrics['recall'] = recall_score(targets, binary_predictions, zero_division=0)
    metrics['f1_score'] = f1_score(targets, binary_predictions, zero_division=0)
    
    # ROC AUC
    if len(np.unique(targets)) > 1:
        metrics['roc_auc'] = roc_auc_score(targets, predictions)
    else:
        metrics['roc_auc'] = 0.5
    
    # Brier score
    metrics['brier_score'] = brier_score_loss(targets, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(targets, binary_predictions)
    metrics['true_negatives'] = cm[0, 0]
    metrics['false_positives'] = cm[0, 1]
    metrics['false_negatives'] = cm[1, 0]
    metrics['true_positives'] = cm[1, 1]
    
    # Specificity and sensitivity
    if cm[0, 0] + cm[0, 1] > 0:
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    else:
        metrics['specificity'] = 0.0
    
    if cm[1, 1] + cm[1, 0] > 0:
        metrics['sensitivity'] = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    else:
        metrics['sensitivity'] = 0.0
    
    return metrics


def compute_expected_calibration_error(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    uncertainty: Union[torch.Tensor, np.ndarray],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        predictions: Model predictions
        targets: True values
        uncertainty: Uncertainty estimates
        n_bins: Number of bins for calibration
        
    Returns:
        Expected Calibration Error
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    uncertainty = uncertainty.flatten()
    
    # Compute errors
    errors = np.abs(predictions - targets)
    
    # Create bins based on uncertainty
    bins = np.linspace(0, np.max(uncertainty), n_bins + 1)
    bin_indices = np.digitize(uncertainty, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_confidence = np.mean(uncertainty[mask])
            bin_accuracy = np.mean(errors[mask])
            bin_size = np.sum(mask) / len(uncertainty)
            ece += bin_size * np.abs(bin_confidence - bin_accuracy)
    
    return ece


def compute_negative_log_likelihood(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    uncertainty: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Compute negative log likelihood.
    
    Args:
        predictions: Model predictions
        targets: True values
        uncertainty: Uncertainty estimates
        
    Returns:
        Negative log likelihood
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    uncertainty = uncertainty.flatten()
    
    # Compute negative log likelihood
    variance = np.exp(uncertainty)
    nll = 0.5 * np.mean(np.log(2 * np.pi * variance) + (targets - predictions) ** 2 / variance)
    
    return nll


def compute_sparsification_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    uncertainty: Union[torch.Tensor, np.ndarray],
    n_fractions: int = 20,
) -> Dict[str, float]:
    """
    Compute sparsification metrics.
    
    Args:
        predictions: Model predictions
        targets: True values
        uncertainty: Uncertainty estimates
        n_fractions: Number of sparsification fractions
        
    Returns:
        Dictionary of sparsification metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(uncertainty, torch.Tensor):
        uncertainty = uncertainty.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    uncertainty = uncertainty.flatten()
    
    # Compute errors
    errors = np.abs(predictions - targets)
    
    # Sort by uncertainty (descending)
    sorted_indices = np.argsort(uncertainty)[::-1]
    
    # Compute risk at different sparsification levels
    risks = []
    fractions = np.linspace(0, 1, n_fractions)
    
    for fraction in fractions:
        if fraction == 0:
            risk = np.mean(errors)
        else:
            n_remove = int(fraction * len(errors))
            remaining_errors = errors[sorted_indices[n_remove:]]
            risk = np.mean(remaining_errors)
        risks.append(risk)
    
    # Compute area under risk curve
    auc_risk = np.trapz(risks, fractions)
    
    # Compute oracle risk (perfect ordering)
    oracle_errors = np.sort(errors)[::-1]
    oracle_risks = []
    for fraction in fractions:
        if fraction == 0:
            oracle_risk = np.mean(oracle_errors)
        else:
            n_remove = int(fraction * len(oracle_errors))
            remaining_errors = oracle_errors[n_remove:]
            oracle_risk = np.mean(remaining_errors)
        oracle_risks.append(oracle_risk)
    
    auc_oracle = np.trapz(oracle_risks, fractions)
    
    # Compute random risk
    random_risks = [np.mean(errors)] * n_fractions
    auc_random = np.trapz(random_risks, fractions)
    
    # Compute sparsification metrics
    if auc_random - auc_oracle > 0:
        sparsification_auc = 1 - (auc_risk - auc_oracle) / (auc_random - auc_oracle)
    else:
        sparsification_auc = 0.0
    
    return {
        'sparsification_auc': sparsification_auc,
        'auc_risk': auc_risk,
        'auc_oracle': auc_oracle,
        'auc_random': auc_random,
    }


def compute_fairness_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    sensitive_attributes: Dict[str, Union[torch.Tensor, np.ndarray]],
) -> Dict[str, float]:
    """
    Compute fairness metrics across different demographic groups.
    
    Args:
        predictions: Model predictions
        targets: True values
        sensitive_attributes: Dictionary of sensitive attributes
        
    Returns:
        Dictionary of fairness metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    fairness_metrics = {}
    
    for attr_name, attr_values in sensitive_attributes.items():
        if isinstance(attr_values, torch.Tensor):
            attr_values = attr_values.cpu().numpy()
        
        attr_values = attr_values.flatten()
        unique_values = np.unique(attr_values)
        
        # Compute metrics for each group
        group_metrics = {}
        for value in unique_values:
            mask = attr_values == value
            group_predictions = predictions[mask]
            group_targets = targets[mask]
            
            if len(group_predictions) > 0:
                group_metrics[f'group_{value}_mae'] = mean_absolute_error(group_targets, group_predictions)
                group_metrics[f'group_{value}_r2'] = r2_score(group_targets, group_predictions)
                group_metrics[f'group_{value}_count'] = len(group_predictions)
        
        # Compute fairness metrics
        if len(unique_values) >= 2:
            mae_values = [group_metrics[f'group_{value}_mae'] for value in unique_values]
            r2_values = [group_metrics[f'group_{value}_r2'] for value in unique_values]
            
            fairness_metrics[f'{attr_name}_mae_disparity'] = max(mae_values) - min(mae_values)
            fairness_metrics[f'{attr_name}_r2_disparity'] = max(r2_values) - min(r2_values)
            fairness_metrics[f'{attr_name}_mae_ratio'] = max(mae_values) / min(mae_values)
            fairness_metrics[f'{attr_name}_r2_ratio'] = max(r2_values) / min(r2_values)
        
        fairness_metrics.update(group_metrics)
    
    return fairness_metrics


def compute_clinical_utility_metrics(
    predictions: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    risk_thresholds: List[float] = [5, 10, 15],
) -> Dict[str, float]:
    """
    Compute clinical utility metrics.
    
    Args:
        predictions: Model predictions
        targets: True values
        risk_thresholds: List of risk thresholds
        
    Returns:
        Dictionary of clinical utility metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Compute age acceleration
    age_acceleration = predictions - targets
    
    clinical_metrics = {}
    
    for threshold in risk_thresholds:
        # High risk individuals (age acceleration > threshold)
        high_risk_mask = age_acceleration > threshold
        
        # Risk prevalence
        clinical_metrics[f'risk_prevalence_{threshold}'] = np.mean(high_risk_mask)
        
        # Mean age acceleration for high risk group
        if np.sum(high_risk_mask) > 0:
            clinical_metrics[f'mean_acceleration_high_risk_{threshold}'] = np.mean(age_acceleration[high_risk_mask])
        else:
            clinical_metrics[f'mean_acceleration_high_risk_{threshold}'] = 0.0
    
    # Compute Net Reclassification Index (NRI)
    # This would require baseline predictions for comparison
    # For now, we'll compute a simplified version
    
    # Compute Integrated Discrimination Improvement (IDI)
    # This would also require baseline predictions
    
    return clinical_metrics


def format_metrics_report(metrics: Dict[str, float], title: str = "Model Evaluation Report") -> str:
    """
    Format metrics as a readable report.
    
    Args:
        metrics: Dictionary of metrics
        title: Report title
        
    Returns:
        Formatted report string
    """
    report = [f"{'='*60}", title, f"{'='*60}", ""]
    
    # Group metrics by category
    categories = {
        'Age Prediction Metrics': ['mae', 'mse', 'rmse', 'r2', 'pearson_correlation', 'spearman_correlation'],
        'Age Acceleration Metrics': ['mean_age_acceleration', 'std_age_acceleration', 'median_age_acceleration'],
        'Uncertainty Metrics': ['mean_uncertainty', 'expected_calibration_error', 'negative_log_likelihood'],
        'Mortality Metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
        'Clinical Utility': ['risk_prevalence_5', 'risk_prevalence_10', 'risk_prevalence_15'],
    }
    
    for category, metric_keys in categories.items():
        report.append(f"{category}:")
        for key in metric_keys:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
        report.append("")
    
    # Add remaining metrics
    remaining_keys = set(metrics.keys()) - set(sum(categories.values(), []))
    if remaining_keys:
        report.append("Other Metrics:")
        for key in sorted(remaining_keys):
            value = metrics[key]
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("")
    
    report.append(f"{'='*60}")
    
    return "\n".join(report)