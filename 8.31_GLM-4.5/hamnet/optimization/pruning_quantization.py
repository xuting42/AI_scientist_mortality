"""
Model Pruning and Quantization Framework for HAMNet

This module provides comprehensive model pruning and quantization capabilities
for optimizing HAMNet models for deployment and inference efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
import time
import copy
from collections import defaultdict
import json
import pickle
from pathlib import Path

from ..models.hamnet import HAMNet, HAMNetConfig
from ..utils.metrics import ModelMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PruningMethod(Enum):
    """Supported pruning methods."""
    MAGNITUDE = "magnitude"
    GRADIENT = "gradient"
    STRUCTURED = "structured"
    MOVEMENT = "movement"
    FIRST_ORDER = "first_order"
    SECOND_ORDER = "second_order"
    GLOBAL = "global"
    LAYER_WISE = "layer_wise"


class QuantizationMode(Enum):
    """Supported quantization modes."""
    DYNAMIC = "dynamic"
    STATIC = "static"
    QUANTIZATION_AWARE_TRAINING = "quantization_aware_training"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    MIXED = "mixed"


@dataclass
class PruningConfig:
    """Configuration for model pruning."""
    method: PruningMethod = PruningMethod.MAGNITUDE
    target_sparsity: float = 0.5
    pruning_schedule: str = "linear"  # "linear", "exponential", "cosine"
    pruning_frequency: int = 100
    start_epoch: int = 0
    end_epoch: int = 100
    layer_wise_sparsity: bool = False
    exclude_layers: List[str] = field(default_factory=list)
    prune_biases: bool = False
    prune_norms: bool = False
    prune_embeddings: bool = False
    importance_score: str = "l1"  # "l1", "l2", "gradient", "fisher"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    mode: QuantizationMode = QuantizationMode.STATIC
    precision: str = "int8"  # "int8", "fp16", "bf16", "mixed"
    calibration_dataset_size: int = 1000
    quantization_aware_training_epochs: int = 10
    backend: str = "fbgemm"  # "fbgemm", "qnnpack"
    observer: str = "min_max"  # "min_max", "moving_average", "histogram"
    fuse_modules: bool = True
    preserve_accuracy: bool = True
    accuracy_threshold: float = 0.01
    mixed_precision_config: Optional[Dict[str, str]] = None


@dataclass
class OptimizationResult:
    """Results from model optimization."""
    original_model_size: int
    optimized_model_size: int
    compression_ratio: float
    original_flops: float
    optimized_flops: float
    speedup_ratio: float
    original_accuracy: float
    optimized_accuracy: float
    accuracy_drop: float
    inference_time_original: float
    inference_time_optimized: float
    speedup_factor: float
    memory_usage_original: float
    memory_usage_optimized: float
    memory_reduction: float
    optimization_details: Dict[str, Any]


class Pruner(ABC):
    """Base class for model pruners."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.mask = {}
        self.importance_scores = {}
        self.pruning_history = []
        
    @abstractmethod
    def calculate_importance(self, model: nn.Module, data_loader: Optional = None) -> Dict[str, torch.Tensor]:
        """Calculate importance scores for model parameters."""
        pass
    
    @abstractmethod
    def prune_model(self, model: nn.Module, current_sparsity: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Prune model based on importance scores."""
        pass
    
    def apply_mask(self, model: nn.Module, mask: Dict[str, torch.Tensor]):
        """Apply pruning mask to model parameters."""
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name]
    
    def get_sparsity(self, model: nn.Module) -> Dict[str, float]:
        """Calculate sparsity for each layer."""
        sparsity = {}
        for name, param in model.named_parameters():
            if param.dim() > 1:  # Only consider weights, not biases
                total_params = param.numel()
                zero_params = torch.sum(param == 0).item()
                sparsity[name] = zero_params / total_params if total_params > 0 else 0.0
        return sparsity
    
    def get_global_sparsity(self, model: nn.Module) -> float:
        """Calculate global sparsity of the model."""
        total_params = 0
        zero_params = 0
        
        for name, param in model.named_parameters():
            if param.dim() > 1 and not any(excl in name for excl in self.config.exclude_layers):
                total_params += param.numel()
                zero_params += torch.sum(param == 0).item()
        
        return zero_params / total_params if total_params > 0 else 0.0


class MagnitudePruner(Pruner):
    """Magnitude-based pruning."""
    
    def calculate_importance(self, model: nn.Module, data_loader: Optional = None) -> Dict[str, torch.Tensor]:
        """Calculate importance based on parameter magnitudes."""
        importance_scores = {}
        
        for name, param in model.named_parameters():
            if param.dim() > 1 and not any(excl in name for excl in self.config.exclude_layers):
                if self.config.importance_score == "l1":
                    importance_scores[name] = torch.abs(param.data)
                elif self.config.importance_score == "l2":
                    importance_scores[name] = param.data ** 2
        
        return importance_scores
    
    def prune_model(self, model: nn.Module, current_sparsity: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Prune model based on magnitude importance."""
        importance_scores = self.calculate_importance(model)
        
        # Calculate threshold
        if self.config.layer_wise_sparsity:
            mask = {}
            for name, scores in importance_scores.items():
                threshold = torch.quantile(scores.flatten(), current_sparsity)
                mask[name] = (scores > threshold).float()
        else:
            # Global pruning
            all_scores = torch.cat([scores.flatten() for scores in importance_scores.values()])
            threshold = torch.quantile(all_scores, current_sparsity)
            
            mask = {}
            for name, scores in importance_scores.items():
                mask[name] = (scores > threshold).float()
        
        # Apply mask
        self.apply_mask(model, mask)
        
        return model, mask


class GradientPruner(Pruner):
    """Gradient-based pruning."""
    
    def calculate_importance(self, model: nn.Module, data_loader: Optional = None) -> Dict[str, torch.Tensor]:
        """Calculate importance based on gradient magnitudes."""
        if data_loader is None:
            raise ValueError("Data loader required for gradient-based pruning")
        
        importance_scores = {}
        model.train()
        
        # Collect gradients
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 10:  # Limit batches for efficiency
                break
            
            model.zero_grad()
            output = model(data)
            loss = F.mse_loss(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.dim() > 1 and not any(excl in name for excl in self.config.exclude_layers):
                    if name not in importance_scores:
                        importance_scores[name] = torch.zeros_like(param.data)
                    importance_scores[name] += torch.abs(param.grad)
        
        # Average gradients
        for name in importance_scores:
            importance_scores[name] /= min(10, len(data_loader))
        
        return importance_scores
    
    def prune_model(self, model: nn.Module, current_sparsity: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Prune model based on gradient importance."""
        importance_scores = self.calculate_importance(model)
        
        # Calculate threshold and create mask
        if self.config.layer_wise_sparsity:
            mask = {}
            for name, scores in importance_scores.items():
                threshold = torch.quantile(scores.flatten(), current_sparsity)
                mask[name] = (scores > threshold).float()
        else:
            all_scores = torch.cat([scores.flatten() for scores in importance_scores.values()])
            threshold = torch.quantile(all_scores, current_sparsity)
            
            mask = {}
            for name, scores in importance_scores.items():
                mask[name] = (scores > threshold).float()
        
        # Apply mask
        self.apply_mask(model, mask)
        
        return model, mask


class StructuredPruner(Pruner):
    """Structured pruning (prune entire channels/neurons)."""
    
    def calculate_importance(self, model: nn.Module, data_loader: Optional = None) -> Dict[str, torch.Tensor]:
        """Calculate importance for structured pruning."""
        importance_scores = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Calculate importance for each output channel
                if isinstance(module, nn.Linear):
                    weights = module.weight.data
                    importance = torch.norm(weights, dim=1)
                else:  # Conv layers
                    weights = module.weight.data
                    importance = torch.norm(weights.view(weights.size(0), -1), dim=1)
                
                importance_scores[name] = importance
        
        return importance_scores
    
    def prune_model(self, model: nn.Module, current_sparsity: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Prune model using structured pruning."""
        importance_scores = self.calculate_importance(model)
        
        # Create pruned model
        pruned_model = copy.deepcopy(model)
        
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if name in importance_scores:
                    importance = importance_scores[name]
                    threshold = torch.quantile(importance, current_sparsity)
                    
                    # Identify channels to keep
                    keep_mask = importance > threshold
                    keep_indices = torch.where(keep_mask)[0]
                    
                    if len(keep_indices) > 0:
                        # Prune the layer
                        if isinstance(module, nn.Linear):
                            module.weight.data = module.weight.data[keep_indices]
                            if module.bias is not None:
                                module.bias.data = module.bias.data[keep_indices]
                        else:  # Conv layers
                            module.weight.data = module.weight.data[keep_indices]
                            if module.bias is not None:
                                module.bias.data = module.bias.data[keep_indices]
        
        return pruned_model, {}


class MovementPruner(Pruner):
    """Movement pruning (prune based on weight movement during training)."""
    
    def __init__(self, config: PruningConfig):
        super().__init__(config)
        self.initial_weights = {}
        self.weight_movement = {}
    
    def calculate_importance(self, model: nn.Module, data_loader: Optional = None) -> Dict[str, torch.Tensor]:
        """Calculate importance based on weight movement."""
        importance_scores = {}
        
        for name, param in model.named_parameters():
            if param.dim() > 1 and not any(excl in name for excl in self.config.exclude_layers):
                if name in self.initial_weights:
                    movement = torch.abs(param.data - self.initial_weights[name])
                    importance_scores[name] = movement
                else:
                    # Store initial weights
                    self.initial_weights[name] = param.data.clone()
                    importance_scores[name] = torch.zeros_like(param.data)
        
        return importance_scores
    
    def prune_model(self, model: nn.Module, current_sparsity: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Prune model based on movement importance."""
        importance_scores = self.calculate_importance(model)
        
        # Calculate threshold and create mask
        if self.config.layer_wise_sparsity:
            mask = {}
            for name, scores in importance_scores.items():
                threshold = torch.quantile(scores.flatten(), current_sparsity)
                mask[name] = (scores > threshold).float()
        else:
            all_scores = torch.cat([scores.flatten() for scores in importance_scores.values()])
            threshold = torch.quantile(all_scores, current_sparsity)
            
            mask = {}
            for name, scores in importance_scores.items():
                mask[name] = (scores > threshold).float()
        
        # Apply mask
        self.apply_mask(model, mask)
        
        return model, mask


class Quantizer:
    """Model quantization utilities."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_data = None
        
    def prepare_model_for_quantization(self, model: nn.Module) -> nn.Module:
        """Prepare model for quantization."""
        model.eval()
        
        # Fuse modules if requested
        if self.config.fuse_modules:
            model = self._fuse_modules(model)
        
        # Set quantization backend
        torch.backends.quantized.engine = self.config.backend
        
        return model
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse modules for quantization."""
        # Common fusion patterns
        fusion_patterns = [
            (nn.Conv1d, nn.ReLU),
            (nn.Linear, nn.ReLU),
            (nn.Conv1d, nn.BatchNorm1d, nn.ReLU),
            (nn.Linear, nn.BatchNorm1d, nn.ReLU)
        ]
        
        modules_to_fuse = []
        
        for name, module in model.named_modules():
            for pattern in fusion_patterns:
                if isinstance(module, pattern[0]):
                    # Check if next module matches pattern
                    modules_to_fuse.append(name)
        
        # Actually fuse modules (this is a simplified version)
        # In practice, you'd need to handle the actual fusion logic
        return model
    
    def calibrate_model(self, model: nn.Module, data_loader):
        """Calibrate model for static quantization."""
        model.eval()
        
        calibration_data = []
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= self.config.calibration_dataset_size:
                break
            calibration_data.append(data)
        
        self.calibration_data = calibration_data
        
        # Run calibration
        with torch.no_grad():
            for data in calibration_data:
                _ = model(data)
    
    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        if self.config.precision == "int8":
            # Dynamic quantization for linear layers
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.GRU, nn.LSTM},
                dtype=torch.qint8
            )
        elif self.config.precision == "fp16":
            quantized_model = model.half()
        else:
            raise ValueError(f"Unsupported precision for dynamic quantization: {self.config.precision}")
        
        return quantized_model
    
    def quantize_static(self, model: nn.Module, data_loader) -> nn.Module:
        """Apply static quantization."""
        # Prepare model
        model = self.prepare_model_for_quantization(model)
        
        # Calibrate
        self.calibrate_model(model, data_loader)
        
        # Quantize
        quantized_model = torch.quantization.convert(model)
        
        return quantized_model
    
    def quantize_aware_training(self, model: nn.Module, train_loader, val_loader, epochs: int = 10):
        """Apply quantization-aware training."""
        # Prepare model
        model = self.prepare_model_for_quantization(model)
        
        # Add quantization stubs
        model = torch.quantization.prepare_qat(model)
        
        # Train with quantization awareness
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model.eval())
        
        return quantized_model
    
    def mixed_precision_quantization(self, model: nn.Module) -> nn.Module:
        """Apply mixed precision quantization."""
        if self.config.mixed_precision_config is None:
            # Default mixed precision configuration
            self.config.mixed_precision_config = {
                "embedding": "fp16",
                "linear": "int8",
                "conv": "int8",
                "attention": "fp16"
            }
        
        # Apply mixed precision based on configuration
        for name, module in model.named_modules():
            for layer_type, precision in self.config.mixed_precision_config.items():
                if layer_type in name.lower():
                    if precision == "fp16":
                        module.half()
                    elif precision == "int8":
                        # This would require more complex handling
                        pass
        
        return model


class HAMNetOptimizer:
    """Main optimization framework for HAMNet models."""
    
    def __init__(self, pruning_config: Optional[PruningConfig] = None, 
                 quantization_config: Optional[QuantizationConfig] = None):
        self.pruning_config = pruning_config or PruningConfig()
        self.quantization_config = quantization_config or QuantizationConfig()
        self.pruner = self._create_pruner()
        self.quantizer = Quantizer(self.quantization_config)
        self.optimization_results = []
        
    def _create_pruner(self) -> Pruner:
        """Create pruner based on configuration."""
        if self.pruning_config.method == PruningMethod.MAGNITUDE:
            return MagnitudePruner(self.pruning_config)
        elif self.pruning_config.method == PruningMethod.GRADIENT:
            return GradientPruner(self.pruning_config)
        elif self.pruning_config.method == PruningMethod.STRUCTURED:
            return StructuredPruner(self.pruning_config)
        elif self.pruning_config.method == PruningMethod.MOVEMENT:
            return MovementPruner(self.pruning_config)
        else:
            raise ValueError(f"Unsupported pruning method: {self.pruning_config.method}")
    
    def prune_model(self, model: nn.Module, train_loader = None, 
                   current_sparsity: float = None) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """Prune model using configured method."""
        if current_sparsity is None:
            current_sparsity = self.pruning_config.target_sparsity
        
        logger.info(f"Pruning model to {current_sparsity:.2f} sparsity using {self.pruning_config.method.value}")
        
        # Calculate importance scores
        if self.pruning_config.method in [PruningMethod.GRADIENT, PruningMethod.MOVEMENT]:
            importance_scores = self.pruner.calculate_importance(model, train_loader)
        else:
            importance_scores = self.pruner.calculate_importance(model)
        
        # Prune model
        pruned_model, mask = self.pruner.prune_model(model, current_sparsity)
        
        # Log results
        sparsity = self.pruner.get_global_sparsity(pruned_model)
        logger.info(f"Model pruned. Global sparsity: {sparsity:.2f}")
        
        return pruned_model, mask
    
    def iterative_pruning(self, model: nn.Module, train_loader, val_loader, 
                         epochs: int = 100) -> nn.Module:
        """Perform iterative pruning during training."""
        pruned_model = copy.deepcopy(model)
        
        for epoch in range(epochs):
            # Calculate current sparsity based on schedule
            if self.pruning_config.pruning_schedule == "linear":
                progress = (epoch - self.pruning_config.start_epoch) / (self.pruning_config.end_epoch - self.pruning_config.start_epoch)
                current_sparsity = self.pruning_config.target_sparsity * max(0, min(1, progress))
            elif self.pruning_config.pruning_schedule == "exponential":
                progress = (epoch - self.pruning_config.start_epoch) / (self.pruning_config.end_epoch - self.pruning_config.start_epoch)
                current_sparsity = self.pruning_config.target_sparsity * (1 - np.exp(-5 * progress))
            elif self.pruning_config.pruning_schedule == "cosine":
                progress = (epoch - self.pruning_config.start_epoch) / (self.pruning_config.end_epoch - self.pruning_config.start_epoch)
                current_sparsity = self.pruning_config.target_sparsity * (1 - np.cos(np.pi * progress / 2))
            else:
                current_sparsity = self.pruning_config.target_sparsity
            
            # Prune if it's time
            if (epoch >= self.pruning_config.start_epoch and 
                epoch <= self.pruning_config.end_epoch and 
                epoch % self.pruning_config.pruning_frequency == 0):
                
                pruned_model, mask = self.prune_model(pruned_model, train_loader, current_sparsity)
                
                # Fine-tune after pruning
                pruned_model = self._fine_tune_model(pruned_model, train_loader, val_loader, epochs=1)
        
        return pruned_model
    
    def _fine_tune_model(self, model: nn.Module, train_loader, val_loader, epochs: int = 1) -> nn.Module:
        """Fine-tune model after pruning."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        
        return model
    
    def quantize_model(self, model: nn.Module, train_loader = None, val_loader = None) -> nn.Module:
        """Quantize model using configured method."""
        logger.info(f"Quantizing model using {self.quantization_config.mode.value}")
        
        if self.quantization_config.mode == QuantizationMode.DYNAMIC:
            quantized_model = self.quantizer.quantize_dynamic(model)
        elif self.quantization_config.mode == QuantizationMode.STATIC:
            quantized_model = self.quantizer.quantize_static(model, val_loader)
        elif self.quantization_config.mode == QuantizationMode.QUANTIZATION_AWARE_TRAINING:
            quantized_model = self.quantizer.quantize_aware_training(
                model, train_loader, val_loader, 
                self.quantization_config.quantization_aware_training_epochs
            )
        elif self.quantization_config.mode == QuantizationMode.FP16:
            quantized_model = model.half()
        elif self.quantization_config.mode == QuantizationMode.MIXED:
            quantized_model = self.quantizer.mixed_precision_quantization(model)
        else:
            raise ValueError(f"Unsupported quantization mode: {self.quantization_config.mode}")
        
        return quantized_model
    
    def optimize_model(self, model: nn.Module, train_loader = None, val_loader = None,
                     test_loader = None) -> Tuple[nn.Module, OptimizationResult]:
        """Comprehensive model optimization (pruning + quantization)."""
        logger.info("Starting comprehensive model optimization")
        
        # Get baseline metrics
        original_metrics = self._measure_model_performance(model, test_loader)
        
        optimized_model = copy.deepcopy(model)
        
        # Prune model
        if self.pruning_config.target_sparsity > 0:
            optimized_model = self.iterative_pruning(optimized_model, train_loader, val_loader)
        
        # Quantize model
        optimized_model = self.quantize_model(optimized_model, train_loader, val_loader)
        
        # Measure optimized performance
        optimized_metrics = self._measure_model_performance(optimized_model, test_loader)
        
        # Create optimization result
        result = OptimizationResult(
            original_model_size=original_metrics["model_size"],
            optimized_model_size=optimized_metrics["model_size"],
            compression_ratio=original_metrics["model_size"] / optimized_metrics["model_size"],
            original_flops=original_metrics["flops"],
            optimized_flops=optimized_metrics["flops"],
            speedup_ratio=original_metrics["flops"] / optimized_metrics["flops"],
            original_accuracy=original_metrics["accuracy"],
            optimized_accuracy=optimized_metrics["accuracy"],
            accuracy_drop=original_metrics["accuracy"] - optimized_metrics["accuracy"],
            inference_time_original=original_metrics["inference_time"],
            inference_time_optimized=optimized_metrics["inference_time"],
            speedup_factor=original_metrics["inference_time"] / optimized_metrics["inference_time"],
            memory_usage_original=original_metrics["memory_usage"],
            memory_usage_optimized=optimized_metrics["memory_usage"],
            memory_reduction=original_metrics["memory_usage"] / optimized_metrics["memory_usage"],
            optimization_details={
                "pruning_method": self.pruning_config.method.value,
                "target_sparsity": self.pruning_config.target_sparsity,
                "quantization_mode": self.quantization_config.mode.value,
                "quantization_precision": self.quantization_config.precision
            }
        )
        
        self.optimization_results.append(result)
        
        logger.info(f"Optimization completed. Compression ratio: {result.compression_ratio:.2f}x, "
                   f"Speedup: {result.speedup_factor:.2f}x, Accuracy drop: {result.accuracy_drop:.4f}")
        
        return optimized_model, result
    
    def _measure_model_performance(self, model: nn.Module, data_loader = None) -> Dict[str, float]:
        """Measure model performance metrics."""
        metrics = {}
        
        # Model size
        model_size = sum(p.numel() for p in model.parameters())
        metrics["model_size"] = model_size
        
        # FLOPs (simplified estimation)
        metrics["flops"] = self._estimate_flops(model)
        
        # Inference time
        if data_loader is not None:
            model.eval()
            start_time = time.time()
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(data_loader):
                    if batch_idx >= 10:  # Measure first 10 batches
                        break
                    _ = model(data)
            metrics["inference_time"] = time.time() - start_time
            
            # Accuracy
            total_loss = 0
            total_samples = 0
            with torch.no_grad():
                for data, target in data_loader:
                    output = model(data)
                    loss = F.mse_loss(output, target)
                    total_loss += loss.item() * len(data)
                    total_samples += len(data)
            
            metrics["accuracy"] = 1.0 / (1.0 + total_loss / total_samples)  # Convert loss to accuracy-like metric
        else:
            metrics["inference_time"] = 0.0
            metrics["accuracy"] = 0.0
        
        # Memory usage (simplified)
        metrics["memory_usage"] = model_size * 4  # Assuming 4 bytes per parameter
        
        return metrics
    
    def _estimate_flops(self, model: nn.Module) -> float:
        """Estimate FLOPs for the model."""
        # This is a simplified estimation
        # In practice, you'd use tools like torch.profiler or thop
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                input_size = module.in_features
                output_size = module.out_features
                total_flops += 2 * input_size * output_size  # 2 operations per multiply-accumulate
            elif isinstance(module, nn.Conv1d):
                input_channels = module.in_channels
                output_channels = module.out_channels
                kernel_size = module.kernel_size[0]
                output_length = 100  # Assumed output length
                total_flops += 2 * input_channels * output_channels * kernel_size * output_length
            elif isinstance(module, nn.MultiheadAttention):
                # Simplified attention FLOPs
                embed_dim = module.embed_dim
                seq_len = 100  # Assumed sequence length
                total_flops += 4 * seq_len * embed_dim * embed_dim  # Q, K, V projections + output
        
        return total_flops
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.optimization_results:
            return {}
        
        summary = {
            "total_optimizations": len(self.optimization_results),
            "best_compression_ratio": max(r.compression_ratio for r in self.optimization_results),
            "best_speedup_factor": max(r.speedup_factor for r in self.optimization_results),
            "average_accuracy_drop": np.mean([r.accuracy_drop for r in self.optimization_results]),
            "optimization_methods": {
                "pruning": self.pruning_config.method.value,
                "quantization": self.quantization_config.mode.value
            }
        }
        
        return summary
    
    def save_optimized_model(self, model: nn.Module, filepath: str):
        """Save optimized model to file."""
        torch.save(model.state_dict(), filepath)
        logger.info(f"Optimized model saved to {filepath}")
    
    def load_optimized_model(self, model: nn.Module, filepath: str) -> nn.Module:
        """Load optimized model from file."""
        model.load_state_dict(torch.load(filepath))
        logger.info(f"Optimized model loaded from {filepath}")
        return model


# Example usage
def example_model_optimization():
    """Example of model optimization."""
    # Configuration
    pruning_config = PruningConfig(
        method=PruningMethod.MAGNITUDE,
        target_sparsity=0.5,
        pruning_schedule="linear",
        pruning_frequency=10,
        start_epoch=10,
        end_epoch=90
    )
    
    quantization_config = QuantizationConfig(
        mode=QuantizationMode.DYNAMIC,
        precision="int8"
    )
    
    # Create optimizer
    optimizer = HAMNetOptimizer(pruning_config, quantization_config)
    
    # Create model (example)
    # model = HAMNet(HAMNetConfig())
    
    # Optimize model
    # optimized_model, result = optimizer.optimize_model(model, train_loader, val_loader, test_loader)
    
    return optimizer


if __name__ == "__main__":
    # Run example
    optimizer = example_model_optimization()
    print("Model optimization framework initialized successfully")