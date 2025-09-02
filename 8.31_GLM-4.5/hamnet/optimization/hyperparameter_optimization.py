"""
Hyperparameter Optimization Framework for HAMNet

This module provides comprehensive hyperparameter optimization capabilities
using Bayesian optimization, grid search, and other advanced techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import random
import time
from collections import defaultdict
import json
import pickle
from pathlib import Path

from ..models.hamnet import HAMNet, HAMNetConfig
from ..utils.metrics import ModelMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """Supported optimization strategies."""
    BAYESIAN = "bayesian"
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    HYPERBAND = "hyperband"
    BOHB = "bohb"
    CMA_ES = "cma_es"
    TPE = "tpe"


class ParameterType(Enum):
    """Parameter types for hyperparameter optimization."""
    FLOAT = "float"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"


@dataclass
class Hyperparameter:
    """Represents a hyperparameter to be optimized."""
    name: str
    param_type: ParameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    default: Any = None
    
    def __post_init__(self):
        """Validate hyperparameter configuration."""
        if self.param_type in [ParameterType.FLOAT, ParameterType.INTEGER]:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"Min and max values required for {self.param_type}")
            if self.min_value >= self.max_value:
                raise ValueError("Min value must be less than max value")
        
        if self.param_type == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError("Choices required for categorical parameter")
        
        if self.default is None:
            if self.param_type == ParameterType.FLOAT:
                self.default = (self.min_value + self.max_value) / 2
            elif self.param_type == ParameterType.INTEGER:
                self.default = int((self.min_value + self.max_value) / 2)
            elif self.param_type == ParameterType.CATEGORICAL:
                self.default = self.choices[0]
            elif self.param_type == ParameterType.BOOLEAN:
                self.default = True
    
    def sample(self) -> Any:
        """Sample a value from the parameter space."""
        if self.param_type == ParameterType.FLOAT:
            if self.log_scale:
                return np.exp(np.random.uniform(
                    np.log(self.min_value), np.log(self.max_value)
                ))
            else:
                return np.random.uniform(self.min_value, self.max_value)
        
        elif self.param_type == ParameterType.INTEGER:
            return random.randint(int(self.min_value), int(self.max_value))
        
        elif self.param_type == ParameterType.CATEGORICAL:
            return random.choice(self.choices)
        
        elif self.param_type == ParameterType.BOOLEAN:
            return random.choice([True, False])
    
    def to_skopt_space(self):
        """Convert to scikit-optimize space format."""
        try:
            from skopt.space import Real, Integer, Categorical
            
            if self.param_type == ParameterType.FLOAT:
                if self.log_scale:
                    return Real(
                        np.log(self.min_value), np.log(self.max_value),
                        name=self.name, prior='log-uniform'
                    )
                else:
                    return Real(self.min_value, self.max_value, name=self.name)
            
            elif self.param_type == ParameterType.INTEGER:
                return Integer(self.min_value, self.max_value, name=self.name)
            
            elif self.param_type == ParameterType.CATEGORICAL:
                return Categorical(self.choices, name=self.name)
            
        except ImportError:
            raise ImportError("scikit-optimize required for Bayesian optimization")


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter optimization."""
    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    max_trials: int = 100
    max_epochs_per_trial: int = 50
    early_stop_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    n_jobs: int = 1
    random_state: int = 42
    metric: str = "val_loss"
    metric_mode: str = "min"  # "min" or "max"
    hyperparameters: List[Hyperparameter] = field(default_factory=list)
    
    def __post_init__(self):
        """Set default hyperparameters if not provided."""
        if not self.hyperparameters:
            self.hyperparameters = [
                Hyperparameter("learning_rate", ParameterType.FLOAT, 1e-5, 1e-1, log_scale=True),
                Hyperparameter("batch_size", ParameterType.INTEGER, 16, 256),
                Hyperparameter("hidden_size", ParameterType.INTEGER, 64, 1024),
                Hyperparameter("num_layers", ParameterType.INTEGER, 2, 8),
                Hyperparameter("dropout_rate", ParameterType.FLOAT, 0.0, 0.5),
                Hyperparameter("weight_decay", ParameterType.FLOAT, 1e-6, 1e-2, log_scale=True),
                Hyperparameter("optimizer", ParameterType.CATEGORICAL, choices=["adam", "adamw", "sgd"]),
                Hyperparameter("scheduler", ParameterType.CATEGORICAL, choices=["cosine", "linear", "step"]),
                Hyperparameter("attention_heads", ParameterType.INTEGER, 2, 16),
                Hyperparameter("activation", ParameterType.CATEGORICAL, choices=["gelu", "relu", "swish"]),
                Hyperparameter("normalization", ParameterType.CATEGORICAL, choices=["layer_norm", "batch_norm"])
            ]


@dataclass
class TrialResult:
    """Results from a single optimization trial."""
    trial_id: int
    hyperparameters: Dict[str, Any]
    train_loss: float
    val_loss: float
    test_loss: Optional[float] = None
    train_metrics: Optional[Dict[str, float]] = None
    val_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    training_time: float = 0.0
    model_size: int = 0
    flops: float = 0.0
    epoch: int = 0
    status: str = "completed"  # "completed", "failed", "early_stopped"


class BaseOptimizer(ABC):
    """Base class for hyperparameter optimizers."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.trial_results = []
        self.best_trial = None
        self.best_score = float('inf') if config.metric_mode == "min" else float('-inf')
        self.current_trial = 0
        
    @abstractmethod
    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest next set of hyperparameters to evaluate."""
        pass
    
    @abstractmethod
    def update_results(self, result: TrialResult):
        """Update optimizer with trial results."""
        pass
    
    def create_trial_config(self, hyperparameters: Dict[str, Any]) -> HAMNetConfig:
        """Create HAMNet configuration from hyperparameters."""
        # Extract relevant hyperparameters
        hidden_size = hyperparameters.get("hidden_size", 256)
        num_layers = hyperparameters.get("num_layers", 4)
        dropout_rate = hyperparameters.get("dropout_rate", 0.1)
        attention_heads = hyperparameters.get("attention_heads", 8)
        activation = hyperparameters.get("activation", "gelu")
        normalization = hyperparameters.get("normalization", "layer_norm")
        
        # Create layer configurations
        layers = []
        for i in range(num_layers):
            layer_config = {
                "type": "attention" if i % 2 == 0 else "conv1d",
                "hidden_size": hidden_size,
                "dropout": dropout_rate,
                "activation": activation,
                "normalization": normalization,
                "num_heads": attention_heads
            }
            layers.append(layer_config)
        
        return HAMNetConfig(
            input_size=512,  # Will be updated based on actual input
            hidden_size=hidden_size,
            num_layers=num_layers,
            layers=layers,
            dropout_rate=dropout_rate,
            attention_heads=attention_heads,
            activation=activation,
            normalization=normalization
        )
    
    def evaluate_trial(self, hyperparameters: Dict[str, Any], 
                     train_fn: Callable, test_fn: Optional[Callable] = None) -> TrialResult:
        """Evaluate a single trial."""
        trial_id = self.current_trial
        self.current_trial += 1
        
        start_time = time.time()
        
        try:
            # Create model configuration
            model_config = self.create_trial_config(hyperparameters)
            
            # Create and train model
            model = HAMNet(model_config)
            
            # Train model
            train_result = train_fn(model, hyperparameters)
            
            # Test model if test function provided
            test_result = None
            if test_fn is not None:
                test_result = test_fn(model, hyperparameters)
            
            # Create trial result
            result = TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters.copy(),
                train_loss=train_result.get("train_loss", float('inf')),
                val_loss=train_result.get("val_loss", float('inf')),
                test_loss=test_result.get("test_loss") if test_result else None,
                train_metrics=train_result.get("train_metrics"),
                val_metrics=train_result.get("val_metrics"),
                test_metrics=test_result.get("test_metrics") if test_result else None,
                training_time=time.time() - start_time,
                model_size=sum(p.numel() for p in model.parameters()),
                epoch=train_result.get("epoch", 0),
                status=train_result.get("status", "completed")
            )
            
            # Update best trial
            score = result.val_loss if self.config.metric == "val_loss" else result.train_loss
            if self.config.metric_mode == "min":
                if score < self.best_score:
                    self.best_score = score
                    self.best_trial = result
            else:
                if score > self.best_score:
                    self.best_score = score
                    self.best_trial = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in trial {trial_id}: {e}")
            return TrialResult(
                trial_id=trial_id,
                hyperparameters=hyperparameters.copy(),
                train_loss=float('inf'),
                val_loss=float('inf'),
                training_time=time.time() - start_time,
                status="failed"
            )


class RandomOptimizer(BaseOptimizer):
    """Random search optimizer."""
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest random hyperparameters."""
        params = {}
        for param in self.config.hyperparameters:
            params[param.name] = param.sample()
        return params
    
    def update_results(self, result: TrialResult):
        """Update with trial results."""
        self.trial_results.append(result)


class GridSearchOptimizer(BaseOptimizer):
    """Grid search optimizer."""
    
    def __init__(self, config: HyperparameterConfig):
        super().__init__(config)
        self.param_grid = self._create_param_grid()
        self.grid_iterator = iter(self._generate_grid_combinations())
    
    def _create_param_grid(self) -> Dict[str, List[Any]]:
        """Create parameter grid."""
        grid = {}
        for param in self.config.hyperparameters:
            if param.param_type == ParameterType.FLOAT:
                # Sample 10 values for continuous parameters
                if param.log_scale:
                    grid[param.name] = np.logspace(
                        np.log10(param.min_value), np.log10(param.max_value), 10
                    ).tolist()
                else:
                    grid[param.name] = np.linspace(
                        param.min_value, param.max_value, 10
                    ).tolist()
            elif param.param_type == ParameterType.INTEGER:
                grid[param.name] = list(range(
                    int(param.min_value), int(param.max_value) + 1
                ))
            elif param.param_type == ParameterType.CATEGORICAL:
                grid[param.name] = param.choices
            elif param.param_type == ParameterType.BOOLEAN:
                grid[param.name] = [True, False]
        return grid
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """Generate all combinations of grid parameters."""
        import itertools
        
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest next grid combination."""
        try:
            return next(self.grid_iterator)
        except StopIteration:
            return None
    
    def update_results(self, result: TrialResult):
        """Update with trial results."""
        self.trial_results.append(result)


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using scikit-optimize."""
    
    def __init__(self, config: HyperparameterConfig):
        super().__init__(config)
        
        try:
            from skopt import gp_minimize
            from skopt.utils import use_named_args
            self.gp_minimize = gp_minimize
            self.use_named_args = use_named_args
            self.skopt_available = True
        except ImportError:
            logger.warning("scikit-optimize not available. Using random search instead.")
            self.skopt_available = False
        
        if self.skopt_available:
            self.dimensions = [param.to_skopt_space() for param in config.hyperparameters]
            self.optimization_results = []
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters using Bayesian optimization."""
        if not self.skopt_available:
            # Fall back to random search
            optimizer = RandomOptimizer(self.config)
            return optimizer.suggest_parameters()
        
        if len(self.optimization_results) == 0:
            # First suggestion is random
            params = {}
            for param in self.config.hyperparameters:
                params[param.name] = param.sample()
            return params
        else:
            # Use Bayesian optimization to suggest next parameters
            @self.use_named_args(self.dimensions)
            def objective(**params):
                # Find corresponding trial result
                for result in self.trial_results:
                    match = True
                    for param_name, param_value in params.items():
                        if result.hyperparameters.get(param_name) != param_value:
                            match = False
                            break
                    if match:
                        return result.val_loss
                
                # If no match found, return high loss
                return float('inf')
            
            # Run one step of optimization
            result = self.gp_minimize(
                objective,
                self.dimensions,
                n_calls=1,
                n_initial_points=0,
                x0=[r.hyperparameters[p.name] for p in self.config.hyperparameters 
                    for r in self.trial_results[-1:]],
                y0=[r.val_loss for r in self.trial_results[-1:]],
                random_state=self.config.random_state
            )
            
            # Convert result to parameter dictionary
            params = {}
            for i, param in enumerate(self.config.hyperparameters):
                params[param.name] = result.x[i]
            
            return params
    
    def update_results(self, result: TrialResult):
        """Update with trial results."""
        self.trial_results.append(result)
        if self.skopt_available:
            self.optimization_results.append({
                'params': result.hyperparameters,
                'loss': result.val_loss
            })


class HyperbandOptimizer(BaseOptimizer):
    """Hyperband optimization algorithm."""
    
    def __init__(self, config: HyperparameterConfig):
        super().__init__(config)
        self.max_resources = config.max_epochs_per_trial
        self.min_resources = 1
        self.eta = 3  # Elimination factor
        self.s_max = int(np.log(self.max_resources / self.min_resources) / np.log(self.eta))
        self.bracket_level = 0
        self.current_bracket = []
        self.completed_brackets = []
    
    def suggest_parameters(self) -> Dict[str, Any]:
        """Suggest hyperparameters using Hyperband."""
        if not self.current_bracket:
            # Start new bracket
            self._start_new_bracket()
        
        if self.current_bracket:
            return self.current_bracket.pop(0)
        else:
            return None
    
    def _start_new_bracket(self):
        """Start new Hyperband bracket."""
        if self.bracket_level > self.s_max:
            self.bracket_level = 0
        
        # Calculate bracket parameters
        n = int(np.ceil((self.s_max + 1) / (self.bracket_level + 1)) * (self.eta ** self.bracket_level))
        r = self.max_resources * (self.eta ** (-self.bracket_level))
        
        # Generate configurations for this bracket
        self.current_bracket = []
        for _ in range(n):
            params = {}
            for param in self.config.hyperparameters:
                params[param.name] = param.sample()
            self.current_bracket.append(params)
        
        self.bracket_resources = r
        self.bracket_level += 1
    
    def update_results(self, result: TrialResult):
        """Update with trial results."""
        self.trial_results.append(result)
        
        # Check if bracket is complete
        if len(self.trial_results) % len(self.current_bracket) == 0:
            # Perform successive halving
            self._perform_successive_halving()
    
    def _perform_successive_halving(self):
        """Perform successive halving within bracket."""
        if not self.current_bracket:
            return
        
        # Get results for current bracket
        bracket_size = len(self.current_bracket)
        bracket_results = self.trial_results[-bracket_size:]
        
        # Sort by performance
        bracket_results.sort(key=lambda x: x.val_loss)
        
        # Keep top 1/eta configurations
        keep_count = max(1, int(bracket_size / self.eta))
        kept_configs = [r.hyperparameters for r in bracket_results[:keep_count]]
        
        # Double resources for next iteration
        self.bracket_resources *= self.eta
        
        # Prepare for next iteration
        self.current_bracket = kept_configs


class HAMNetHyperparameterOptimizer:
    """Main hyperparameter optimization framework for HAMNet."""
    
    def __init__(self, config: HyperparameterConfig):
        self.config = config
        self.optimizer = self._create_optimizer()
        self.optimization_history = []
        
    def _create_optimizer(self) -> BaseOptimizer:
        """Create optimizer based on configuration."""
        if self.config.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return RandomOptimizer(self.config)
        elif self.config.strategy == OptimizationStrategy.GRID_SEARCH:
            return GridSearchOptimizer(self.config)
        elif self.config.strategy == OptimizationStrategy.BAYESIAN:
            return BayesianOptimizer(self.config)
        elif self.config.strategy == OptimizationStrategy.HYPERBAND:
            return HyperbandOptimizer(self.config)
        else:
            raise ValueError(f"Unsupported optimization strategy: {self.config.strategy}")
    
    def optimize(self, train_fn: Callable, test_fn: Optional[Callable] = None) -> TrialResult:
        """Run hyperparameter optimization."""
        logger.info(f"Starting hyperparameter optimization with {self.config.strategy.value}")
        
        for trial in range(self.config.max_trials):
            # Suggest parameters
            params = self.optimizer.suggest_parameters()
            if params is None:
                logger.info("No more parameters to evaluate")
                break
            
            # Evaluate trial
            result = self.optimizer.evaluate_trial(params, train_fn, test_fn)
            
            # Update optimizer
            self.optimizer.update_results(result)
            
            # Store in history
            self.optimization_history.append(result)
            
            # Log progress
            logger.info(f"Trial {trial + 1}/{self.config.max_trials}: "
                       f"val_loss={result.val_loss:.4f}, "
                       f"train_loss={result.train_loss:.4f}, "
                       f"time={result.training_time:.2f}s")
            
            # Early stopping if no improvement
            if trial > 20 and self._should_early_stop():
                logger.info("Early stopping due to no improvement")
                break
        
        logger.info(f"Optimization completed. Best val_loss: {self.optimizer.best_score:.4f}")
        return self.optimizer.best_trial
    
    def _should_early_stop(self) -> bool:
        """Check if early stopping criteria are met."""
        if len(self.optimization_history) < self.config.early_stop_patience:
            return False
        
        recent_trials = self.optimization_history[-self.config.early_stop_patience:]
        recent_scores = [r.val_loss for r in recent_trials]
        
        best_recent_score = min(recent_scores)
        improvement_threshold = 0.001  # 0.1% improvement
        
        return best_recent_score > (self.optimizer.best_score * (1 + improvement_threshold))
    
    def get_best_hyperparameters(self) -> Dict[str, Any]:
        """Get best hyperparameters found."""
        if self.optimizer.best_trial:
            return self.optimizer.best_trial.hyperparameters
        return {}
    
    def get_optimization_results(self) -> List[TrialResult]:
        """Get all optimization results."""
        return self.optimization_history
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results."""
        if not self.optimization_history:
            return {}
        
        successful_trials = [r for r in self.optimization_history if r.status == "completed"]
        
        if not successful_trials:
            return {}
        
        val_losses = [r.val_loss for r in successful_trials]
        train_losses = [r.train_loss for r in successful_trials]
        training_times = [r.training_time for r in successful_trials]
        
        analysis = {
            "total_trials": len(self.optimization_history),
            "successful_trials": len(successful_trials),
            "best_val_loss": min(val_losses),
            "mean_val_loss": np.mean(val_losses),
            "std_val_loss": np.std(val_losses),
            "best_train_loss": min(train_losses),
            "mean_train_loss": np.mean(train_losses),
            "mean_training_time": np.mean(training_times),
            "convergence_trend": self._analyze_convergence(),
            "parameter_importance": self._analyze_parameter_importance(),
            "best_hyperparameters": self.get_best_hyperparameters()
        }
        
        return analysis
    
    def _analyze_convergence(self) -> List[float]:
        """Analyze convergence trend."""
        if not self.optimization_history:
            return []
        
        # Calculate running minimum
        convergence = []
        current_min = float('inf')
        
        for result in self.optimization_history:
            if result.status == "completed":
                current_min = min(current_min, result.val_loss)
            convergence.append(current_min)
        
        return convergence
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze parameter importance using correlation analysis."""
        if not self.optimization_history:
            return {}
        
        successful_trials = [r for r in self.optimization_history if r.status == "completed"]
        
        if len(successful_trials) < 5:
            return {}
        
        # Extract parameter values and corresponding losses
        param_values = defaultdict(list)
        losses = []
        
        for trial in successful_trials:
            losses.append(trial.val_loss)
            for param_name, param_value in trial.hyperparameters.items():
                param_values[param_name].append(param_value)
        
        # Calculate correlations
        correlations = {}
        for param_name, values in param_values.items():
            try:
                if len(set(values)) > 1:  # Only if parameter varies
                    correlation = np.corrcoef(values, losses)[0, 1]
                    correlations[param_name] = abs(correlation)
                else:
                    correlations[param_name] = 0.0
            except:
                correlations[param_name] = 0.0
        
        return correlations
    
    def save_results(self, filepath: str):
        """Save optimization results to file."""
        results = {
            "config": self.config,
            "optimization_history": self.optimization_history,
            "best_trial": self.optimizer.best_trial,
            "best_score": self.optimizer.best_score
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """Load optimization results from file."""
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.config = results["config"]
        self.optimization_history = results["optimization_history"]
        self.optimizer.best_trial = results["best_trial"]
        self.optimizer.best_score = results["best_score"]
        
        logger.info(f"Optimization results loaded from {filepath}")


def create_train_fn(train_loader, val_loader, device: str = 'cuda') -> Callable:
    """Create training function for hyperparameter optimization."""
    
    def train_fn(model: HAMNet, hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Train model with given hyperparameters."""
        # Extract hyperparameters
        learning_rate = hyperparameters.get("learning_rate", 0.001)
        batch_size = hyperparameters.get("batch_size", 32)
        weight_decay = hyperparameters.get("weight_decay", 1e-4)
        optimizer_name = hyperparameters.get("optimizer", "adam")
        scheduler_name = hyperparameters.get("scheduler", "cosine")
        
        # Setup optimizer
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup scheduler
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        elif scheduler_name == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Training loop
        model = model.to(device)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(100):  # Max epochs
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 10:
                break
        
        return {
            "train_loss": best_val_loss,
            "val_loss": best_val_loss,
            "epoch": epoch,
            "status": "completed"
        }
    
    return train_fn


# Example usage
def example_hyperparameter_optimization():
    """Example of hyperparameter optimization."""
    # Configuration
    config = HyperparameterConfig(
        strategy=OptimizationStrategy.BAYESIAN,
        max_trials=50,
        max_epochs_per_trial=50,
        early_stop_patience=10
    )
    
    # Create optimizer
    optimizer = HAMNetHyperparameterOptimizer(config)
    
    # Mock training function
    # train_fn = create_train_fn(train_loader, val_loader)
    
    # Run optimization
    # best_trial = optimizer.optimize(train_fn)
    
    # Analyze results
    # analysis = optimizer.analyze_results()
    # print(f"Best val_loss: {analysis['best_val_loss']:.4f}")
    
    return optimizer


if __name__ == "__main__":
    # Run example
    optimizer = example_hyperparameter_optimization()
    print("Hyperparameter optimization framework initialized successfully")