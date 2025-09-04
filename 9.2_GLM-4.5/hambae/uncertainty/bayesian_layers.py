"""
Bayesian neural network layers for uncertainty quantification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.
    
    This layer implements Bayesian linear transformation with Gaussian
    priors on weights and biases, enabling uncertainty quantification.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float = 1.0,
        posterior_sigma_init: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize Bayesian linear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            prior_sigma: Standard deviation of prior distribution
            posterior_sigma_init: Initial standard deviation of posterior
            bias: Whether to include bias term
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.bias = bias
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        # Initialize parameters
        self.reset_parameters(posterior_sigma_init)
        
        # Prior distribution
        self.weight_prior = torch.distributions.Normal(0, prior_sigma)
        if bias:
            self.bias_prior = torch.distributions.Normal(0, prior_sigma)
        
        logger.info(f"Initialized BayesianLinear({in_features}, {out_features})")
    
    def reset_parameters(self, posterior_sigma_init: float) -> None:
        """Reset layer parameters."""
        # Initialize weight parameters
        nn.init.xavier_normal_(self.weight_mu)
        nn.init.constant_(self.weight_rho, math.log(math.expm1(posterior_sigma_init)))
        
        # Initialize bias parameters
        if self.bias is not None:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_rho, math.log(math.expm1(posterior_sigma_init)))
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass through Bayesian linear layer.
        
        Args:
            x: Input tensor
            sample: Whether to sample from posterior distribution
            
        Returns:
            Output tensor
        """
        if self.training or sample:
            # Sample weights from posterior
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            
            # Sample bias from posterior
            if self.bias is not None:
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
            else:
                bias = None
        else:
            # Use mean values
            weight = self.weight_mu
            bias = self.bias_mu if self.bias is not None else None
        
        # Compute KL divergence
        kl_divergence = self._compute_kl_divergence(weight, bias)
        
        # Store KL divergence for regularization
        self.kl_divergence = kl_divergence
        
        # Linear transformation
        output = F.linear(x, weight, bias)
        
        return output
    
    def _compute_kl_divergence(
        self, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        # Weight KL divergence
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_posterior = torch.distributions.Normal(self.weight_mu, weight_sigma)
        weight_kl = torch.distributions.kl.kl_divergence(weight_posterior, self.weight_prior)
        weight_kl = torch.sum(weight_kl)
        
        # Bias KL divergence
        bias_kl = torch.tensor(0.0, device=weight.device)
        if bias is not None and self.bias is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_posterior = torch.distributions.Normal(self.bias_mu, bias_sigma)
            bias_kl = torch.distributions.kl.kl_divergence(bias_posterior, self.bias_prior)
            bias_kl = torch.sum(bias_kl)
        
        return weight_kl + bias_kl
    
    def get_kl_divergence(self) -> torch.Tensor:
        """Get KL divergence from last forward pass."""
        return getattr(self, 'kl_divergence', torch.tensor(0.0))
    
    def get_weight_uncertainty(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get weight uncertainty (mean and std)."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        return self.weight_mu, weight_sigma
    
    def get_bias_uncertainty(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get bias uncertainty (mean and std)."""
        if self.bias is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            return self.bias_mu, bias_sigma
        return None


class BayesianDropout(nn.Module):
    """
    Bayesian dropout layer with concrete dropout.
    
    This layer implements concrete dropout, which learns the dropout rate
    during training and provides uncertainty estimates.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        p_init: float = 0.1,
        p_min: float = 0.01,
        p_max: float = 0.9,
    ):
        """
        Initialize Bayesian dropout layer.
        
        Args:
            p: Initial dropout probability
            p_init: Initial learned dropout probability
            p_min: Minimum dropout probability
            p_max: Maximum dropout probability
        """
        super().__init__()
        self.p_init = p_init
        self.p_min = p_min
        self.p_max = p_max
        
        # Learnable dropout parameter
        self.p_logit = nn.Parameter(torch.tensor(math.log(p_init / (1 - p_init))))
        
        # Regular dropout for training
        self.dropout = nn.Dropout(p)
        
        logger.info(f"Initialized BayesianDropout with p_init={p_init}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Bayesian dropout layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.training:
            # Use concrete dropout during training
            p = torch.sigmoid(self.p_logit)
            p = torch.clamp(p, self.p_min, self.p_max)
            
            # Concrete dropout
            noise = torch.rand_like(x)
            dropout_mask = (torch.log(p + 1e-8) - torch.log(1 - p + 1e-8) + 
                          torch.log(noise + 1e-8) - torch.log(1 - noise + 1e-8))
            dropout_mask = torch.sigmoid(dropout_mask / 0.1)
            
            output = x * dropout_mask / (1 - p + 1e-8)
            
            # Store KL divergence
            self.kl_divergence = self._compute_kl_divergence(p)
        else:
            # Use mean during inference
            output = x
            self.kl_divergence = torch.tensor(0.0, device=x.device)
        
        return output
    
    def _compute_kl_divergence(self, p: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for concrete dropout."""
        # Simplified KL divergence for concrete dropout
        # This is an approximation of the true KL divergence
        kl = p * torch.log(p / 0.5 + 1e-8) + (1 - p) * torch.log((1 - p) / 0.5 + 1e-8)
        return torch.sum(kl)
    
    def get_dropout_probability(self) -> torch.Tensor:
        """Get current dropout probability."""
        return torch.sigmoid(self.p_logit)
    
    def get_kl_divergence(self) -> torch.Tensor:
        """Get KL divergence from last forward pass."""
        return getattr(self, 'kl_divergence', torch.tensor(0.0))


class BayesianConv2d(nn.Module):
    """
    Bayesian 2D convolution layer with weight uncertainty.
    
    This layer implements Bayesian 2D convolution with Gaussian
    priors on weights and biases.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        prior_sigma: float = 1.0,
        posterior_sigma_init: float = 0.1,
        bias: bool = True,
    ):
        """
        Initialize Bayesian 2D convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolution kernel
            stride: Stride of the convolution
            padding: Padding added to all sides of input
            prior_sigma: Standard deviation of prior distribution
            posterior_sigma_init: Initial standard deviation of posterior
            bias: Whether to include bias term
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior_sigma = prior_sigma
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        
        # Bias parameters
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_channels))
            self.bias_rho = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        
        # Initialize parameters
        self.reset_parameters(posterior_sigma_init)
        
        # Prior distributions
        self.weight_prior = torch.distributions.Normal(0, prior_sigma)
        if bias:
            self.bias_prior = torch.distributions.Normal(0, prior_sigma)
        
        logger.info(f"Initialized BayesianConv2d({in_channels}, {out_channels}, kernel_size={kernel_size})")
    
    def reset_parameters(self, posterior_sigma_init: float) -> None:
        """Reset layer parameters."""
        # Initialize weight parameters
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.weight_rho, math.log(math.expm1(posterior_sigma_init)))
        
        # Initialize bias parameters
        if self.bias is not None:
            nn.init.zeros_(self.bias_mu)
            nn.init.constant_(self.bias_rho, math.log(math.expm1(posterior_sigma_init)))
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Forward pass through Bayesian 2D convolution layer.
        
        Args:
            x: Input tensor
            sample: Whether to sample from posterior distribution
            
        Returns:
            Output tensor
        """
        if self.training or sample:
            # Sample weights from posterior
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            
            # Sample bias from posterior
            if self.bias is not None:
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
            else:
                bias = None
        else:
            # Use mean values
            weight = self.weight_mu
            bias = self.bias_mu if self.bias is not None else None
        
        # Compute KL divergence
        kl_divergence = self._compute_kl_divergence(weight, bias)
        
        # Store KL divergence for regularization
        self.kl_divergence = kl_divergence
        
        # Convolution operation
        output = F.conv2d(x, weight, bias, stride=self.stride, padding=self.padding)
        
        return output
    
    def _compute_kl_divergence(
        self, 
        weight: torch.Tensor, 
        bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        # Weight KL divergence
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        weight_posterior = torch.distributions.Normal(self.weight_mu, weight_sigma)
        weight_kl = torch.distributions.kl.kl_divergence(weight_posterior, self.weight_prior)
        weight_kl = torch.sum(weight_kl)
        
        # Bias KL divergence
        bias_kl = torch.tensor(0.0, device=weight.device)
        if bias is not None and self.bias is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            bias_posterior = torch.distributions.Normal(self.bias_mu, bias_sigma)
            bias_kl = torch.distributions.kl.kl_divergence(bias_posterior, self.bias_prior)
            bias_kl = torch.sum(bias_kl)
        
        return weight_kl + bias_kl
    
    def get_kl_divergence(self) -> torch.Tensor:
        """Get KL divergence from last forward pass."""
        return getattr(self, 'kl_divergence', torch.tensor(0.0))
    
    def get_weight_uncertainty(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get weight uncertainty (mean and std)."""
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        return self.weight_mu, weight_sigma
    
    def get_bias_uncertainty(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get bias uncertainty (mean and std)."""
        if self.bias is not None:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho))
            return self.bias_mu, bias_sigma
        return None


def compute_bayesian_kl_divergence(model: nn.Module) -> torch.Tensor:
    """
    Compute total KL divergence for a Bayesian neural network.
    
    Args:
        model: Bayesian neural network model
        
    Returns:
        Total KL divergence
    """
    total_kl = torch.tensor(0.0, device=next(model.parameters()).device)
    
    for module in model.modules():
        if isinstance(module, (BayesianLinear, BayesianConv2d, BayesianDropout)):
            total_kl += module.get_kl_divergence()
    
    return total_kl


def enable_bayesian_sampling(model: nn.Module, sample: bool = True) -> None:
    """
    Enable or disable sampling in Bayesian layers.
    
    Args:
        model: Bayesian neural network model
        sample: Whether to enable sampling
    """
    for module in model.modules():
        if isinstance(module, (BayesianLinear, BayesianConv2d)):
            module.sample = sample