"""
Evaluation Module for HENAW Model
Implements comprehensive evaluation metrics and interpretability analysis
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
import shap
from captum.attr import IntegratedGradients, GradientShap

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluator for HENAW model
    Computes MAE, RMSE, C-statistic, ICC, and other metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_config = config['evaluation']['metrics']
        self.targets = config['evaluation']['targets']
        
    def compute_metrics(self,
                       predictions: List[Dict[str, torch.Tensor]],
                       targets: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Compute all evaluation metrics
        
        Args:
            predictions: List of prediction dictionaries
            targets: List of target dictionaries
        
        Returns:
            Dictionary of computed metrics
        """
        # Concatenate predictions and targets
        all_preds = self._concatenate_predictions(predictions)
        all_targets = self._concatenate_targets(targets)
        
        metrics = {}
        
        # Age prediction metrics
        if 'mae' in self.metrics_config:
            metrics['mae'] = self.compute_mae(
                all_preds['biological_age'],
                all_targets['chronological_age']
            )
        
        if 'rmse' in self.metrics_config:
            metrics['rmse'] = self.compute_rmse(
                all_preds['biological_age'],
                all_targets['chronological_age']
            )
        
        if 'r2' in self.metrics_config:
            metrics['r2'] = self.compute_r2(
                all_preds['biological_age'],
                all_targets['chronological_age']
            )
        
        # Survival metrics
        if 'c_statistic' in self.metrics_config and 'mortality_risk' in all_preds:
            metrics['c_statistic'] = self.compute_c_statistic(
                all_preds['mortality_risk'],
                all_targets.get('survival_time'),
                all_targets.get('event_indicator')
            )
        
        # ICC (Intraclass Correlation Coefficient)
        if 'icc' in self.metrics_config:
            metrics['icc'] = self.compute_icc(
                all_preds['biological_age'],
                all_targets['chronological_age']
            )
        
        # Age acceleration metrics
        age_gap = all_preds['biological_age'] - all_targets['chronological_age']
        metrics['age_gap_mean'] = float(age_gap.mean())
        metrics['age_gap_std'] = float(age_gap.std())
        
        # Check performance targets
        self._check_performance_targets(metrics)
        
        return metrics
    
    def _concatenate_predictions(self, predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
        """Concatenate batch predictions"""
        concatenated = {}
        
        for key in predictions[0].keys():
            if predictions[0][key] is not None:
                values = [p[key].numpy() if isinstance(p[key], torch.Tensor) else p[key] 
                         for p in predictions]
                concatenated[key] = np.concatenate(values).squeeze()
        
        return concatenated
    
    def _concatenate_targets(self, targets: List[Dict[str, torch.Tensor]]) -> Dict[str, np.ndarray]:
        """Concatenate batch targets"""
        concatenated = {}
        
        all_keys = set()
        for t in targets:
            all_keys.update(t.keys())
        
        for key in all_keys:
            values = []
            for t in targets:
                if key in t:
                    val = t[key].cpu().numpy() if isinstance(t[key], torch.Tensor) else t[key]
                    values.append(val)
            
            if values:
                concatenated[key] = np.concatenate(values).squeeze()
        
        return concatenated
    
    def compute_mae(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Mean Absolute Error"""
        return mean_absolute_error(targets, predictions)
    
    def compute_rmse(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Root Mean Square Error"""
        return np.sqrt(mean_squared_error(targets, predictions))
    
    def compute_r2(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute R-squared score"""
        return r2_score(targets, predictions)
    
    def compute_c_statistic(self,
                           risk_scores: np.ndarray,
                           survival_times: Optional[np.ndarray],
                           event_indicators: Optional[np.ndarray]) -> float:
        """
        Compute concordance statistic (C-statistic) for survival analysis
        
        Args:
            risk_scores: Predicted risk scores
            survival_times: Time to event
            event_indicators: Binary event indicators
        
        Returns:
            C-statistic value
        """
        if survival_times is None or event_indicators is None:
            return 0.0
        
        try:
            # Use lifelines concordance index
            c_index = concordance_index(
                survival_times,
                -risk_scores,  # Negative because higher risk = shorter survival
                event_indicators
            )
            return c_index
        except Exception as e:
            logger.warning(f"Could not compute C-statistic: {e}")
            return 0.0
    
    def compute_icc(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute Intraclass Correlation Coefficient (ICC)
        Measures reliability/agreement between biological and chronological age
        
        Args:
            predictions: Predicted biological ages
            targets: Chronological ages
        
        Returns:
            ICC value
        """
        # Reshape for ICC calculation
        n = len(predictions)
        
        # Create data frame for ICC
        data = pd.DataFrame({
            'subject': np.repeat(np.arange(n), 2),
            'rater': np.tile([0, 1], n),
            'score': np.concatenate([predictions, targets])
        })
        
        # Compute ICC(2,1) - two-way random effects, single measurement, absolute agreement
        # Using formula: ICC = (MSR - MSE) / (MSR + MSE + 2*(MSC - MSE)/n)
        
        # Between-subject variance
        subject_means = data.groupby('subject')['score'].mean()
        grand_mean = data['score'].mean()
        ss_between = 2 * np.sum((subject_means - grand_mean) ** 2)
        
        # Within-subject variance
        ss_within = np.sum((data.groupby('subject')['score'].transform('mean') - data['score']) ** 2)
        
        # Mean squares
        ms_between = ss_between / (n - 1)
        ms_within = ss_within / n
        
        # ICC calculation
        icc = (ms_between - ms_within) / (ms_between + ms_within)
        
        return max(0, min(1, icc))  # Bound between 0 and 1
    
    def _check_performance_targets(self, metrics: Dict[str, float]) -> None:
        """Check if performance targets are met"""
        targets_met = []
        targets_missed = []
        
        for metric, target_value in self.targets.items():
            if metric in metrics:
                actual_value = metrics[metric]
                
                if metric in ['mae', 'rmse']:  # Lower is better
                    if actual_value < target_value:
                        targets_met.append(f"{metric}: {actual_value:.3f} < {target_value}")
                    else:
                        targets_missed.append(f"{metric}: {actual_value:.3f} >= {target_value}")
                else:  # Higher is better
                    if actual_value > target_value:
                        targets_met.append(f"{metric}: {actual_value:.3f} > {target_value}")
                    else:
                        targets_missed.append(f"{metric}: {actual_value:.3f} <= {target_value}")
        
        if targets_met:
            logger.info(f"Performance targets MET: {', '.join(targets_met)}")
        if targets_missed:
            logger.warning(f"Performance targets MISSED: {', '.join(targets_missed)}")


class InterpretabilityAnalyzer:
    """
    Interpretability analysis for HENAW model
    Includes SHAP values, feature importance, and attention visualization
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.biomarker_names = list(config['ukbb_fields']['biomarkers'].keys())
        
    def compute_shap_values(self,
                           data_loader: torch.utils.data.DataLoader,
                           n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Compute SHAP values for model interpretability
        
        Args:
            data_loader: Data loader
            n_samples: Number of samples to use
        
        Returns:
            Dictionary with SHAP values and feature importance
        """
        self.model.eval()
        
        # Collect samples
        X_list = []
        y_list = []
        
        for batch_idx, batch in enumerate(data_loader):
            if len(X_list) * batch['biomarkers'].size(0) >= n_samples:
                break
            
            X_list.append(batch['biomarkers'])
            y_list.append(batch['chronological_age'])
        
        X = torch.cat(X_list)[:n_samples]
        y = torch.cat(y_list)[:n_samples]
        
        # Create explainer
        def model_predict(x):
            with torch.no_grad():
                output = self.model(x, age=None)
                return output.biological_age
        
        # Use DeepExplainer for neural networks
        background = X[:10]  # Use subset as background
        explainer = shap.DeepExplainer(model_predict, background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X)
        
        # Compute feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'feature_names': self.biomarker_names
        }
    
    def compute_gradient_importance(self,
                                   data_loader: torch.utils.data.DataLoader,
                                   n_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Compute gradient-based feature importance
        
        Args:
            data_loader: Data loader
            n_samples: Number of samples
        
        Returns:
            Feature importance scores
        """
        self.model.eval()
        
        # Initialize integrated gradients
        ig = IntegratedGradients(lambda x: self.model(x, age=None).biological_age)
        
        importance_scores = []
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx * batch['biomarkers'].size(0) >= n_samples:
                break
            
            X = batch['biomarkers'].requires_grad_(True)
            
            # Compute attributions
            attributions = ig.attribute(X, n_steps=50)
            
            # Aggregate importance
            importance = torch.abs(attributions).mean(dim=0)
            importance_scores.append(importance.detach().numpy())
        
        # Average across samples
        avg_importance = np.mean(importance_scores, axis=0)
        
        return {
            'importance_scores': avg_importance,
            'feature_names': self.biomarker_names
        }
    
    def analyze_age_gap_distribution(self,
                                    predictions: np.ndarray,
                                    chronological_ages: np.ndarray,
                                    save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Analyze the distribution of age gaps (biological - chronological age)
        
        Args:
            predictions: Predicted biological ages
            chronological_ages: True chronological ages
            save_path: Optional path to save plots
        
        Returns:
            Dictionary with age gap statistics
        """
        age_gaps = predictions - chronological_ages
        
        # Compute statistics
        stats_dict = {
            'mean': float(np.mean(age_gaps)),
            'std': float(np.std(age_gaps)),
            'median': float(np.median(age_gaps)),
            'q25': float(np.percentile(age_gaps, 25)),
            'q75': float(np.percentile(age_gaps, 75)),
            'min': float(np.min(age_gaps)),
            'max': float(np.max(age_gaps))
        }
        
        # Test for normality
        _, p_value = stats.normaltest(age_gaps)
        stats_dict['normality_p_value'] = float(p_value)
        
        # Create visualizations
        if save_path:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Distribution plot
            axes[0, 0].hist(age_gaps, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(0, color='red', linestyle='--', label='Zero gap')
            axes[0, 0].set_xlabel('Age Gap (years)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Age Gap Distribution')
            axes[0, 0].legend()
            
            # Q-Q plot
            stats.probplot(age_gaps, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title('Q-Q Plot')
            
            # Age gap vs chronological age
            axes[1, 0].scatter(chronological_ages, age_gaps, alpha=0.5, s=1)
            axes[1, 0].axhline(0, color='red', linestyle='--')
            axes[1, 0].set_xlabel('Chronological Age (years)')
            axes[1, 0].set_ylabel('Age Gap (years)')
            axes[1, 0].set_title('Age Gap vs Chronological Age')
            
            # Bland-Altman plot
            mean_ages = (predictions + chronological_ages) / 2
            axes[1, 1].scatter(mean_ages, age_gaps, alpha=0.5, s=1)
            axes[1, 1].axhline(stats_dict['mean'], color='red', linestyle='-', label=f"Mean: {stats_dict['mean']:.2f}")
            axes[1, 1].axhline(stats_dict['mean'] + 1.96 * stats_dict['std'], 
                              color='red', linestyle='--', label=f"±1.96 SD")
            axes[1, 1].axhline(stats_dict['mean'] - 1.96 * stats_dict['std'], 
                              color='red', linestyle='--')
            axes[1, 1].set_xlabel('Mean Age (years)')
            axes[1, 1].set_ylabel('Age Gap (years)')
            axes[1, 1].set_title('Bland-Altman Plot')
            axes[1, 1].legend()
            
            plt.tight_layout()
            plt.savefig(save_path / 'age_gap_analysis.png', dpi=300)
            plt.close()
        
        return stats_dict
    
    def create_feature_importance_plot(self,
                                      importance_scores: np.ndarray,
                                      feature_names: List[str],
                                      save_path: Optional[Path] = None) -> None:
        """
        Create feature importance visualization
        
        Args:
            importance_scores: Feature importance scores
            feature_names: Names of features
            save_path: Optional path to save plot
        """
        # Sort features by importance
        indices = np.argsort(importance_scores)[::-1]
        sorted_scores = importance_scores[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_scores)), sorted_scores)
        plt.yticks(range(len(sorted_scores)), sorted_names)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance for Biological Age Prediction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path / 'feature_importance.png', dpi=300)
            plt.close()
        else:
            plt.show()
    
    def analyze_system_contributions(self,
                                    model: nn.Module,
                                    data_loader: torch.utils.data.DataLoader,
                                    save_path: Optional[Path] = None) -> Dict[str, float]:
        """
        Analyze contributions of different biological systems
        
        Args:
            model: HENAW model
            data_loader: Data loader
            save_path: Optional path to save results
        
        Returns:
            Dictionary with system-level contributions
        """
        model.eval()
        
        system_contributions = {system: [] for system in self.config['biological_systems'].keys()}
        
        with torch.no_grad():
            for batch in data_loader:
                X = batch['biomarkers']
                age = batch['chronological_age']
                
                # Get system embeddings
                output = model(X, age, return_intermediates=True)
                
                if output.system_embeddings is not None:
                    # Analyze system embeddings
                    # This is simplified - in practice, you'd analyze the actual contributions
                    embeddings = output.system_embeddings
                    
                    # Compute L2 norm as proxy for contribution
                    for i, system in enumerate(system_contributions.keys()):
                        start_idx = i * model.system_embedding_dim
                        end_idx = (i + 1) * model.system_embedding_dim
                        system_norm = torch.norm(embeddings[:, start_idx:end_idx], dim=1).mean()
                        system_contributions[system].append(system_norm.item())
        
        # Average contributions
        avg_contributions = {
            system: np.mean(values) for system, values in system_contributions.items()
        }
        
        # Normalize to percentages
        total = sum(avg_contributions.values())
        system_percentages = {
            system: (value / total * 100) for system, value in avg_contributions.items()
        }
        
        # Create visualization
        if save_path:
            plt.figure(figsize=(10, 6))
            systems = list(system_percentages.keys())
            percentages = list(system_percentages.values())
            
            plt.pie(percentages, labels=systems, autopct='%1.1f%%')
            plt.title('Biological System Contributions to Age Prediction')
            plt.savefig(save_path / 'system_contributions.png', dpi=300)
            plt.close()
        
        return system_percentages


class ClinicalReportGenerator:
    """
    Generate clinical reports for biological age predictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.biomarker_names = list(config['ukbb_fields']['biomarkers'].keys())
        
    def generate_individual_report(self,
                                  biomarkers: np.ndarray,
                                  biological_age: float,
                                  chronological_age: float,
                                  mortality_risk: Optional[float] = None,
                                  morbidity_risks: Optional[Dict[str, float]] = None,
                                  feature_importance: Optional[np.ndarray] = None) -> str:
        """
        Generate a clinical report for an individual
        
        Args:
            biomarkers: Individual's biomarker values
            biological_age: Predicted biological age
            chronological_age: Chronological age
            mortality_risk: Optional mortality risk score
            morbidity_risks: Optional disease risk scores
            feature_importance: Optional feature importance scores
        
        Returns:
            Clinical report as string
        """
        report = []
        report.append("=" * 60)
        report.append("BIOLOGICAL AGE ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Age assessment
        report.append("AGE ASSESSMENT")
        report.append("-" * 30)
        report.append(f"Chronological Age: {chronological_age:.1f} years")
        report.append(f"Biological Age: {biological_age:.1f} years")
        
        age_gap = biological_age - chronological_age
        if age_gap > 0:
            report.append(f"Age Acceleration: +{age_gap:.1f} years")
            report.append("Interpretation: Biological aging faster than chronological")
        else:
            report.append(f"Age Deceleration: {age_gap:.1f} years")
            report.append("Interpretation: Biological aging slower than chronological")
        report.append("")
        
        # Biomarker analysis
        report.append("BIOMARKER ANALYSIS")
        report.append("-" * 30)
        
        if feature_importance is not None:
            # Sort biomarkers by importance
            importance_indices = np.argsort(feature_importance)[::-1]
            
            report.append("Key Contributing Factors:")
            for i in range(min(5, len(importance_indices))):
                idx = importance_indices[i]
                biomarker = self.biomarker_names[idx]
                value = biomarkers[idx]
                importance = feature_importance[idx]
                report.append(f"  - {biomarker}: {value:.2f} (importance: {importance:.3f})")
        else:
            report.append("Biomarker Values:")
            for i, name in enumerate(self.biomarker_names):
                report.append(f"  - {name}: {biomarkers[i]:.2f}")
        report.append("")
        
        # Risk assessment
        if mortality_risk is not None or morbidity_risks is not None:
            report.append("RISK ASSESSMENT")
            report.append("-" * 30)
            
            if mortality_risk is not None:
                risk_percentile = stats.norm.cdf(mortality_risk) * 100
                report.append(f"Mortality Risk Score: {mortality_risk:.3f}")
                report.append(f"Risk Percentile: {risk_percentile:.1f}%")
            
            if morbidity_risks:
                report.append("\nDisease Risk Predictions:")
                for disease, risk in morbidity_risks.items():
                    risk_pct = risk * 100
                    report.append(f"  - {disease.capitalize()}: {risk_pct:.1f}%")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        
        if age_gap > 5:
            report.append("• Consider comprehensive health assessment")
            report.append("• Focus on lifestyle modifications")
            report.append("• Regular monitoring of biomarkers")
        elif age_gap > 0:
            report.append("• Maintain healthy lifestyle habits")
            report.append("• Annual biomarker monitoring")
        else:
            report.append("• Continue current health practices")
            report.append("• Regular preventive screenings")
        
        report.append("")
        report.append("=" * 60)
        report.append("Note: This assessment is for research purposes only.")
        report.append("Consult healthcare provider for medical advice.")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def generate_population_report(self,
                                  predictions: np.ndarray,
                                  chronological_ages: np.ndarray,
                                  save_path: Optional[Path] = None) -> str:
        """
        Generate population-level summary report
        
        Args:
            predictions: All biological age predictions
            chronological_ages: All chronological ages
            save_path: Optional path to save report
        
        Returns:
            Population report as string
        """
        age_gaps = predictions - chronological_ages
        
        report = []
        report.append("=" * 60)
        report.append("POPULATION BIOLOGICAL AGE SUMMARY")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Total Participants: {len(predictions):,}")
        report.append("")
        
        report.append("AGE STATISTICS")
        report.append("-" * 30)
        report.append(f"Chronological Age: {chronological_ages.mean():.1f} ± {chronological_ages.std():.1f} years")
        report.append(f"Biological Age: {predictions.mean():.1f} ± {predictions.std():.1f} years")
        report.append(f"Age Gap: {age_gaps.mean():.2f} ± {age_gaps.std():.2f} years")
        report.append("")
        
        report.append("AGE GAP DISTRIBUTION")
        report.append("-" * 30)
        
        # Categorize participants
        accelerated = np.sum(age_gaps > 5)
        normal = np.sum((age_gaps >= -5) & (age_gaps <= 5))
        decelerated = np.sum(age_gaps < -5)
        
        report.append(f"Accelerated Aging (>5 years): {accelerated:,} ({accelerated/len(predictions)*100:.1f}%)")
        report.append(f"Normal Aging (±5 years): {normal:,} ({normal/len(predictions)*100:.1f}%)")
        report.append(f"Decelerated Aging (<-5 years): {decelerated:,} ({decelerated/len(predictions)*100:.1f}%)")
        report.append("")
        
        report.append("AGE GROUP ANALYSIS")
        report.append("-" * 30)
        
        age_bins = [40, 50, 60, 70]
        for i in range(len(age_bins) - 1):
            mask = (chronological_ages >= age_bins[i]) & (chronological_ages < age_bins[i + 1])
            if mask.sum() > 0:
                group_gap = age_gaps[mask].mean()
                group_std = age_gaps[mask].std()
                report.append(f"Age {age_bins[i]}-{age_bins[i+1]}: Gap = {group_gap:.2f} ± {group_std:.2f} years")
        
        report.append("")
        report.append("=" * 60)
        
        final_report = "\n".join(report)
        
        if save_path:
            with open(save_path / 'population_report.txt', 'w') as f:
                f.write(final_report)
        
        return final_report


if __name__ == "__main__":
    # Test evaluation metrics
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create evaluator
    evaluator = Evaluator(config)
    
    # Generate synthetic data for testing
    n_samples = 1000
    predictions = [{
        'biological_age': torch.randn(100, 1) * 10 + 55,
        'chronological_age': torch.randn(100, 1) * 10 + 55,
        'mortality_risk': torch.randn(100, 1)
    } for _ in range(10)]
    
    targets = [{
        'chronological_age': torch.randn(100) * 10 + 55,
        'survival_time': torch.randn(100) * 5 + 10,
        'event_indicator': torch.randint(0, 2, (100,))
    } for _ in range(10)]
    
    # Compute metrics
    metrics = evaluator.compute_metrics(predictions, targets)
    
    print("Evaluation Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test clinical report generation
    report_gen = ClinicalReportGenerator(config)
    
    # Generate individual report
    sample_biomarkers = np.random.randn(9)
    sample_report = report_gen.generate_individual_report(
        biomarkers=sample_biomarkers,
        biological_age=58.5,
        chronological_age=55.0,
        mortality_risk=0.15,
        morbidity_risks={'cardiovascular': 0.25, 'diabetes': 0.18}
    )
    
    print("\nSample Clinical Report:")
    print(sample_report)