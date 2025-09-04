"""
Comprehensive Uncertainty Quantification Module for HAMNet

This module implements various uncertainty quantification methods for biological age prediction:
- Bayesian neural networks with Monte Carlo dropout
- Heteroscedastic uncertainty modeling
- Deep ensemble methods
- Evidential deep learning
- Uncertainty calibration and evaluation

Author: Claude AI Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass
import math
from scipy import stats


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification methods"""
    # Monte Carlo Dropout
    enable_mc_dropout: bool = True
    mc_dropout_rate: float = 0.1
    num_mc_samples: int = 50
    
    # Heteroscedastic Uncertainty
    enable_heteroscedastic: bool = True
    heteroscedastic_loss_weight: float = 0.1
    
    # Deep Ensemble
    enable_deep_ensemble: bool = True
    num_ensemble_models: int = 5
    ensemble_diversity_weight: float = 0.01
    
    # Evidential Deep Learning
    enable_evidential: bool = True
    evidential_coefficient: float = 1.0
    
    # Uncertainty Calibration
    enable_calibration: bool = True
    temperature_scaling: bool = True
    isotonic_regression: bool = False
    
    # Evaluation
    eval_metrics: List[str] = None
    
    def __post_init__(self):
        if self.eval_metrics is None:
            self.eval_metrics = ['mse', 'mae', 'nll', 'calibration_error', 'reliability_diagram']


class BayesianLinear(nn.Module):
    """Bayesian linear layer with Monte Carlo dropout"""
    
    def __init__(self, in_features: int, out_features: int, 
                 dropout_rate: float = 0.1, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        
        # Weight and bias parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize weight parameters
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, -3)
        
        # Initialize bias parameters
        if self.bias_mu is not None:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_rho, -3)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Monte Carlo dropout"""
        # Sample weights from Gaussian distribution
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        
        # Sample bias if present
        if self.bias_mu is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            bias = None
            
        # Apply dropout during training
        if self.training:
            weight = F.dropout(weight, p=self.dropout_rate, training=True)
            
        return F.linear(x, weight, bias)
        
    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL divergence between learned and prior distributions"""
        # Prior: N(0, 1)
        # Variational: N(μ, σ²)
        
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_kl = -0.5 * torch.sum(
            1 + torch.log(weight_sigma**2) - self.weight_mu**2 - weight_sigma**2
        )
        
        if self.bias_mu is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_kl = -0.5 * torch.sum(
                1 + torch.log(bias_sigma**2) - self.bias_mu**2 - bias_sigma**2
            )
        else:
            bias_kl = 0
            
        return weight_kl + bias_kl


class HeteroscedasticLoss(nn.Module):
    """Heteroscedastic loss function for aleatoric uncertainty"""
    
    def __init__(self, loss_weight: float = 0.1):
        super().__init__()
        self.loss_weight = loss_weight
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                log_variance: torch.Tensor) -> torch.Tensor:
        """
        Calculate heteroscedastic loss
        
        Args:
            predictions: Predicted values (batch_size, 1)
            targets: Target values (batch_size, 1)
            log_variance: Log variance predictions (batch_size, 1)
            
        Returns:
            Combined loss
        """
        # Negative log likelihood with learned variance
        nll = 0.5 * torch.exp(-log_variance) * (predictions - targets)**2 + 0.5 * log_variance
        
        # Mean squared error
        mse = F.mse_loss(predictions, targets, reduction='none')
        
        # Combined loss
        loss = torch.mean(nll + self.loss_weight * mse)
        
        return loss


class EvidentialOutput(nn.Module):
    """Evidential deep learning output layer"""
    
    def __init__(self, input_dim: int, coefficient: float = 1.0):
        super().__init__()
        self.coefficient = coefficient
        
        # Output layers for evidential parameters
        self.gamma = nn.Linear(input_dim, 1)  # Mean
        self.nu = nn.Linear(input_dim, 1)      # Precision
        self.alpha = nn.Linear(input_dim, 1)   # Shape
        self.beta = nn.Linear(input_dim, 1)    # Scale
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to generate evidential parameters
        
        Args:
            x: Input features
            
        Returns:
            Dictionary with evidential parameters
        """
        gamma = self.gamma(x)
        nu = F.softplus(self.nu(x)) + 1.0
        alpha = F.softplus(self.alpha(x)) + 1.0
        beta = F.softplus(self.beta(x))
        
        return {
            'gamma': gamma,      # Mean
            'nu': nu,            # Precision
            'alpha': alpha,      # Shape
            'beta': beta         # Scale
        }
        
    def loss(self, evidential_params: Dict[str, torch.Tensor], 
             targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate evidential loss
        
        Args:
            evidential_params: Dictionary with evidential parameters
            targets: Target values
            
        Returns:
            Evidential loss
        """
        gamma = evidential_params['gamma']
        nu = evidential_params['nu']
        alpha = evidential_params['alpha']
        beta = evidential_params['beta']
        
        # Two-sided loss
        two_bl = 0.5 * torch.log(torch.pi / nu) \
                 - alpha * torch.log(2 * beta * (1 + nu)) \
                 + (alpha + 0.5) * torch.log(nu * (targets - gamma)**2 + 2 * beta) \
                 + torch.lgamma(alpha) \
                 - torch.lgamma(alpha + 0.5)
        
        # Regularization term
        reg = self.coefficient * torch.abs(targets - gamma) * (2 * nu + alpha)
        
        return torch.mean(two_bl + reg)
        
    def uncertainty(self, evidential_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate uncertainty from evidential parameters"""
        gamma = evidential_params['gamma']
        nu = evidential_params['nu']
        alpha = evidential_params['alpha']
        beta = evidential_params['beta']
        
        # Total uncertainty = aleatoric + epistemic
        aleatoric = beta / (nu * (alpha - 1))
        epistemic = beta / (nu * (alpha - 1))
        
        return aleatoric + epistemic


class DeepEnsemble(nn.Module):
    """Deep ensemble for uncertainty quantification"""
    
    def __init__(self, base_model_class, model_config: Any, 
                 num_models: int = 5, diversity_weight: float = 0.01):
        super().__init__()
        self.num_models = num_models
        self.diversity_weight = diversity_weight
        
        # Create ensemble of models with diverse initializations
        self.models = nn.ModuleList()
        for i in range(num_models):
            model = base_model_class(model_config)
            # Apply different initialization for diversity
            self._initialize_model_differently(model, i)
            self.models.append(model)
            
    def _initialize_model_differently(self, model: nn.Module, seed: int):
        """Initialize model with different seed for diversity"""
        torch.manual_seed(seed + 42)
        for param in model.parameters():
            if param.dim() > 1:
                nn.init.orthogonal_(param)
            else:
                nn.init.normal_(param, std=0.01)
                
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred['predictions'] if isinstance(pred, dict) else pred)
            
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate ensemble statistics
        mean_pred = predictions.mean(dim=0)
        variance = predictions.var(dim=0)
        uncertainty = torch.sqrt(variance + 1e-8)
        
        return {
            'mean_prediction': mean_pred,
            'variance': variance,
            'uncertainty': uncertainty,
            'ensemble_predictions': predictions,
            'ensemble_std': predictions.std(dim=0)
        }
        
    def diversity_loss(self) -> torch.Tensor:
        """Calculate diversity loss for ensemble"""
        diversity_loss = 0
        count = 0
        
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                # Calculate parameter differences
                for (name1, param1), (name2, param2) in zip(
                    self.models[i].named_parameters(),
                    self.models[j].named_parameters()
                ):
                    if name1 == name2:
                        diversity_loss += torch.norm(param1 - param2)**2
                        count += 1
                        
        return self.diversity_weight * diversity_loss / count if count > 0 else torch.tensor(0.0)


class UncertaintyCalibration(nn.Module):
    """Uncertainty calibration methods"""
    
    def __init__(self, method: str = 'temperature_scaling'):
        super().__init__()
        self.method = method
        
        if method == 'temperature_scaling':
            self.temperature = nn.Parameter(torch.ones(1))
        elif method == 'isotonic_regression':
            self.isotonic_model = None
            
    def forward(self, predictions: torch.Tensor, uncertainties: torch.Tensor) -> torch.Tensor:
        """Calibrate uncertainties"""
        if self.method == 'temperature_scaling':
            return uncertainties * self.temperature
        elif self.method == 'isotonic_regression':
            # Implementation would require sklearn isotonic regression
            return uncertainties
        else:
            return uncertainties
            
    def fit(self, predictions: torch.Tensor, uncertainties: torch.Tensor, 
            targets: torch.Tensor):
        """Fit calibration parameters"""
        if self.method == 'isotonic_regression':
            # Implementation would fit isotonic regression model
            pass


class ComprehensiveUncertainty(nn.Module):
    """Comprehensive uncertainty quantification module"""
    
    def __init__(self, config: UncertaintyConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # Bayesian layers
        if config.enable_mc_dropout:
            self.bayesian_layers = nn.ModuleList([
                BayesianLinear(input_dim, input_dim // 2, config.mc_dropout_rate),
                BayesianLinear(input_dim // 2, input_dim // 4, config.mc_dropout_rate),
                BayesianLinear(input_dim // 4, 2)  # Mean and log variance
            ])
            
        # Heteroscedastic loss
        if config.enable_heteroscedastic:
            self.heteroscedastic_loss = HeteroscedasticLoss(config.heteroscedastic_loss_weight)
            
        # Evidential output
        if config.enable_evidential:
            self.evidential_output = EvidentialOutput(input_dim, config.evidential_coefficient)
            
        # Calibration
        if config.enable_calibration:
            self.calibration = UncertaintyCalibration(
                method='temperature_scaling' if config.temperature_scaling else 'isotonic_regression'
            )
            
    def forward(self, x: torch.Tensor, training: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with comprehensive uncertainty quantification
        
        Args:
            x: Input features
            training: Whether in training mode
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        output = {}
        
        # Monte Carlo dropout predictions
        if self.config.enable_mc_dropout:
            mc_predictions = self._monte_carlo_forward(x)
            output.update(mc_predictions)
            
        # Evidential predictions
        if self.config.enable_evidential:
            evidential_params = self.evidential_output(x)
            output['evidential_params'] = evidential_params
            output['evidential_uncertainty'] = self.evidential_output.uncertainty(evidential_params)
            
        # Calibration
        if self.config.enable_calibration and 'uncertainty' in output:
            output['calibrated_uncertainty'] = self.calibration(
                output.get('predictions', torch.zeros_like(x[:, :1])),
                output['uncertainty']
            )
            
        return output
        
    def _monte_carlo_forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Monte Carlo dropout forward pass"""
        num_samples = self.config.num_mc_samples if self.training else 1
        predictions = []
        
        for _ in range(num_samples):
            h = x
            for layer in self.bayesian_layers[:-1]:
                h = F.relu(layer(h))
            pred = self.bayesian_layers[-1](h)
            predictions.append(pred)
            
        predictions = torch.stack(predictions, dim=0)
        
        # Separate mean and log variance
        mean_pred = predictions[..., 0]
        log_var = predictions[..., 1]
        
        # Calculate statistics
        mean = mean_pred.mean(dim=0)
        variance = mean_pred.var(dim=0)
        uncertainty = torch.sqrt(variance + 1e-8)
        
        return {
            'predictions': mean,
            'log_variance': log_var.mean(dim=0),
            'uncertainty': uncertainty,
            'mc_samples': predictions,
            'aleatoric_uncertainty': torch.exp(log_var.mean(dim=0)),
            'epistemic_uncertainty': variance
        }
        
    def loss(self, predictions: Dict[str, torch.Tensor], 
             targets: torch.Tensor) -> torch.Tensor:
        """Calculate comprehensive loss"""
        total_loss = 0
        
        # MSE loss
        if 'predictions' in predictions:
            mse_loss = F.mse_loss(predictions['predictions'], targets)
            total_loss += mse_loss
            
        # Heteroscedastic loss
        if self.config.enable_heteroscedastic and 'log_variance' in predictions:
            hetero_loss = self.heteroscedastic_loss(
                predictions['predictions'], targets, predictions['log_variance']
            )
            total_loss += hetero_loss
            
        # Evidential loss
        if self.config.enable_evidential and 'evidential_params' in predictions:
            evidential_loss = self.evidential_output.loss(
                predictions['evidential_params'], targets
            )
            total_loss += evidential_loss
            
        # KL divergence for Bayesian layers
        if self.config.enable_mc_dropout:
            kl_loss = sum(layer.kl_divergence() for layer in self.bayesian_layers)
            total_loss += kl_loss / len(targets)  # Normalize by batch size
            
        return total_loss


class UncertaintyMetrics:
    """Evaluation metrics for uncertainty quantification"""
    
    @staticmethod
    def negative_log_likelihood(predictions: torch.Tensor, 
                              targets: torch.Tensor, 
                              variance: torch.Tensor) -> float:
        """Calculate negative log likelihood"""
        nll = 0.5 * torch.log(2 * torch.pi * variance) + \
              0.5 * (predictions - targets)**2 / variance
        return torch.mean(nll).item()
        
    @staticmethod
    def calibration_error(predictions: torch.Tensor, 
                         targets: torch.Tensor, 
                         uncertainties: torch.Tensor, 
                         num_bins: int = 10) -> float:
        """Calculate calibration error"""
        # Sort by uncertainty
        sorted_indices = torch.argsort(uncertainties.flatten())
        sorted_uncertainties = uncertainties.flatten()[sorted_indices]
        sorted_errors = torch.abs(predictions.flatten() - targets.flatten())[sorted_indices]
        
        # Bin into quantiles
        bin_size = len(sorted_uncertainties) // num_bins
        calibration_errors = []
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_uncertainties)
            
            bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
            bin_errors = sorted_errors[start_idx:end_idx]
            
            expected_uncertainty = torch.mean(bin_uncertainties)
            actual_error = torch.mean(bin_errors)
            
            calibration_errors.append(torch.abs(expected_uncertainty - actual_error))
            
        return torch.mean(torch.stack(calibration_errors)).item()
        
    @staticmethod
    def reliability_diagram(predictions: torch.Tensor, 
                          targets: torch.Tensor, 
                          uncertainties: torch.Tensor, 
                          num_bins: int = 10) -> Tuple[List[float], List[float]]:
        """Generate reliability diagram data"""
        # Calculate absolute errors
        abs_errors = torch.abs(predictions - targets)
        
        # Sort by confidence (inverse of uncertainty)
        confidences = 1.0 / (uncertainties + 1e-8)
        sorted_indices = torch.argsort(confidences.flatten())
        
        # Bin into quantiles
        bin_size = len(confidences) // num_bins
        expected_confidences = []
        actual_accuracies = []
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(confidences)
            
            bin_confidences = confidences.flatten()[sorted_indices[start_idx:end_idx]]
            bin_errors = abs_errors.flatten()[sorted_indices[start_idx:end_idx]]
            
            expected_conf = torch.mean(bin_confidences).item()
            actual_acc = torch.mean(1.0 - bin_errors).item()
            
            expected_confidences.append(expected_conf)
            actual_accuracies.append(actual_acc)
            
        return expected_confidences, actual_accuracies
        
    @staticmethod
    def sharpness(uncertainties: torch.Tensor) -> float:
        """Calculate sharpness (average uncertainty)"""
        return torch.mean(uncertainties).item()
        
    @staticmethod
    def evaluate_uncertainty_quality(predictions: torch.Tensor, 
                                 targets: torch.Tensor, 
                                 uncertainties: torch.Tensor,
                                 variance: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Comprehensive evaluation of uncertainty quality"""
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = F.mse_loss(predictions, targets).item()
        metrics['mae'] = F.l1_loss(predictions, targets).item()
        
        # Uncertainty-specific metrics
        metrics['nll'] = UncertaintyMetrics.negative_log_likelihood(
            predictions, targets, uncertainties
        )
        metrics['calibration_error'] = UncertaintyMetrics.calibration_error(
            predictions, targets, uncertainties
        )
        metrics['sharpness'] = UncertaintyMetrics.sharpness(uncertainties)
        
        # Additional metrics if variance is available
        if variance is not None:
            metrics['aleatoric_uncertainty'] = torch.mean(variance).item()
            
        return metrics


# Utility functions
def create_uncertainty_config(**kwargs) -> UncertaintyConfig:
    """Create uncertainty configuration with default values"""
    return UncertaintyConfig(**kwargs)


def uncertainty_aware_training_step(model: nn.Module, 
                                   optimizer: torch.optim.Optimizer,
                                   data: Tuple[torch.Tensor, torch.Tensor],
                                   uncertainty_module: ComprehensiveUncertainty) -> Dict[str, float]:
    """Single training step with uncertainty quantification"""
    x, y = data
    
    # Forward pass
    predictions = model(x)
    if isinstance(predictions, dict):
        predictions = predictions['predictions']
    
    uncertainty_output = uncertainty_module(x, training=True)
    
    # Calculate loss
    loss = uncertainty_module.loss(uncertainty_output, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate metrics
    with torch.no_grad():
        metrics = UncertaintyMetrics.evaluate_uncertainty_quality(
            predictions, y, uncertainty_output.get('uncertainty', torch.zeros_like(y))
        )
        metrics['loss'] = loss.item()
        
    return metrics


def monte_carlo_prediction(model: nn.Module, 
                         x: torch.Tensor, 
                         num_samples: int = 100) -> Dict[str, torch.Tensor]:
    """Monte Carlo prediction with dropout"""
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            # Enable dropout during inference
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
                    
            pred = model(x)
            if isinstance(pred, dict):
                pred = pred['predictions']
            predictions.append(pred)
            
    predictions = torch.stack(predictions, dim=0)
    
    return {
        'mean': predictions.mean(dim=0),
        'variance': predictions.var(dim=0),
        'std': predictions.std(dim=0),
        'samples': predictions
    }