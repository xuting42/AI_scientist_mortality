"""
Advanced Regularization Techniques for HAMNet

This module provides comprehensive regularization techniques for improving
model generalization and preventing overfitting in HAMNet models.
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
import math
from collections import defaultdict
import time

from ..models.hamnet import HAMNet, HAMNetConfig
from ..utils.metrics import ModelMetrics
from ..utils.logger import get_logger

logger = get_logger(__name__)


class RegularizationType(Enum):
    """Supported regularization types."""
    DROPOUT = "dropout"
    WEIGHT_DECAY = "weight_decay"
    BATCH_NORM = "batch_norm"
    LAYER_NORM = "layer_norm"
    SPECTRAL_NORM = "spectral_norm"
    DROPCONNECT = "dropconnect"
    DROPATH = "dropath"
    STOCHASTIC_DEPTH = "stochastic_depth"
    LABEL_SMOOTHING = "label_smoothing"
    MIXUP = "mixup"
    CUTMIX = "cutmix"
    AUTOAUGMENT = "autoaugment"
    RANDAUGMENT = "randaugment"
    ADVERSARIAL_TRAINING = "adversarial_training"
    VIRTUAL_ADVERSARIAL_TRAINING = "virtual_adversarial_training"
    CONSISTENCY_REGULARIZATION = "consistency_regularization"
    TEMPERATURE_SCALING = "temperature_scaling"
    MANIFOLD_REGULARIZATION = "manifold_regularization"
    CONTRASTIVE_REGULARIZATION = "contrastive_regularization"
    INFORMATION_BOTTLENECK = "information_bottleneck"
    VARIATIONAL_REGULARIZATION = "variational_regularization"


@dataclass
class RegularizationConfig:
    """Configuration for regularization techniques."""
    dropout_rate: float = 0.1
    weight_decay: float = 1e-4
    dropout_spatial: bool = True
    dropout_type: str = "bernoulli"  # "bernoulli", "gaussian", "alpha"
    weight_decay_type: str = "l2"  # "l1", "l2", "elastic_net"
    elastic_net_alpha: float = 0.5
    spectral_norm_coefficient: float = 1.0
    dropconnect_rate: float = 0.1
    dropath_rate: float = 0.1
    stochastic_depth_rate: float = 0.1
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    autoaugment_policy: str = "imagenet"
    randaugment_n: int = 2
    randaugment_m: int = 10
    adversarial_epsilon: float = 0.01
    adversarial_alpha: float = 0.01
    adversarial_iterations: int = 1
    virtual_adversarial_epsilon: float = 1.0
    virtual_adversarial_xi: float = 1e-6
    virtual_adversarial_iterations: int = 1
    consistency_temperature: float = 0.5
    consistency_weight: float = 1.0
    temperature_scaling: float = 1.0
    manifold_coefficient: float = 0.1
    contrastive_temperature: float = 0.1
    contrastive_weight: float = 0.1
    information_beta: float = 0.1
    variational_beta: float = 0.1
    enabled_techniques: List[RegularizationType] = field(default_factory=lambda: [
        RegularizationType.DROPOUT,
        RegularizationType.WEIGHT_DECAY,
        RegularizationType.LAYER_NORM
    ])


class Regularizer(ABC):
    """Base class for regularizers."""
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss."""
        pass


class DropoutRegularizer(Regularizer):
    """Dropout regularization."""
    
    def __init__(self, config: RegularizationConfig):
        super().__init__(config)
        self.dropout_layers = nn.ModuleDict()
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply dropout regularization."""
        # Dropout is applied during forward pass, so no additional loss here
        return torch.tensor(0.0, device=self.device)
    
    def apply_dropout(self, x: torch.Tensor, p: float = None) -> torch.Tensor:
        """Apply dropout to input."""
        if p is None:
            p = self.config.dropout_rate
        
        if self.config.dropout_type == "bernoulli":
            return F.dropout(x, p=p, training=True)
        elif self.config.dropout_type == "gaussian":
            noise = torch.randn_like(x) * p
            return x + noise
        elif self.config.dropout_type == "alpha":
            # Alpha dropout
            alpha = torch.sqrt(torch.tensor(p / (1 - p)))
            return x + alpha * torch.randn_like(x)
        else:
            return F.dropout(x, p=p, training=True)


class WeightDecayRegularizer(Regularizer):
    """Weight decay regularization."""
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute weight decay loss."""
        total_loss = torch.tensor(0.0, device=self.device)
        
        for param in model.parameters():
            if param.dim() > 1:  # Only apply to weights, not biases
                if self.config.weight_decay_type == "l1":
                    total_loss += torch.sum(torch.abs(param))
                elif self.config.weight_decay_type == "l2":
                    total_loss += torch.sum(param ** 2)
                elif self.config.weight_decay_type == "elastic_net":
                    l1_loss = torch.sum(torch.abs(param))
                    l2_loss = torch.sum(param ** 2)
                    total_loss += (
                        self.config.elastic_net_alpha * l1_loss + 
                        (1 - self.config.elastic_net_alpha) * l2_loss
                    )
        
        return self.config.weight_decay * total_loss


class SpectralNormRegularizer(Regularizer):
    """Spectral normalization regularization."""
    
    def __init__(self, config: RegularizationConfig):
        super().__init__(config)
        self.spectral_norm_layers = {}
    
    def apply_spectral_norm(self, module: nn.Module, name: str = 'weight'):
        """Apply spectral normalization to module."""
        if hasattr(module, name):
            nn.utils.spectral_norm(module, name=name, n_power_iterations=1)
            self.spectral_norm_layers[module] = name
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute spectral norm regularization loss."""
        total_loss = torch.tensor(0.0, device=self.device)
        
        for module, weight_name in self.spectral_norm_layers.items():
            if hasattr(module, weight_name):
                weight = getattr(module, weight_name)
                # Compute spectral norm
                weight_mat = weight.view(weight.size(0), -1)
                sigma = torch.linalg.norm(weight_mat, 2)
                total_loss += torch.relu(sigma - self.config.spectral_norm_coefficient)
        
        return total_loss


class DropConnectRegularizer(Regularizer):
    """DropConnect regularization."""
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply DropConnect to model weights."""
        total_loss = torch.tensor(0.0, device=self.device)
        
        for param in model.parameters():
            if param.dim() > 1 and random.random() < self.config.dropconnect_rate:
                mask = torch.bernoulli(torch.ones_like(param) * (1 - self.config.dropconnect_rate))
                param.data *= mask
        
        return total_loss


class DropPathRegularizer(Regularizer):
    """DropPath regularization."""
    
    def __init__(self, config: RegularizationConfig):
        super().__init__(config)
        self.drop_path_rate = config.dropath_rate
    
    def drop_path(self, x: torch.Tensor, drop_prob: float = 0.0) -> torch.Tensor:
        """Drop path operation."""
        if drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        output = x.div(keep_prob) * random_tensor
        return output
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply DropPath regularization."""
        # DropPath is applied during forward pass
        return torch.tensor(0.0, device=self.device)


class LabelSmoothingRegularizer(Regularizer):
    """Label smoothing regularization."""
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing."""
        num_classes = outputs.size(-1)
        smooth_targets = torch.zeros_like(outputs)
        smooth_targets.fill_(self.config.label_smoothing / num_classes)
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.config.label_smoothing)
        
        return F.kl_div(F.log_softmax(outputs, dim=-1), smooth_targets, reduction='batchmean')


class MixupRegularizer(Regularizer):
    """Mixup regularization."""
    
    def mixup_data(self, x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Apply mixup to data."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion: Callable, pred: torch.Tensor, 
                       y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        """Mixup criterion."""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply mixup regularization."""
        # Mixup is applied during training, not as a loss
        return torch.tensor(0.0, device=self.device)


class CutMixRegularizer(Regularizer):
    """CutMix regularization."""
    
    def rand_bbox(self, size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
        """Generate random bounding box."""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def cutmix_data(self, x: torch.Tensor, y: torch.Tensor, 
                    beta: float = 1.0, prob: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply CutMix to data."""
        if np.random.rand() > prob:
            return x, y, y, 1.0
        
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(x.size()[0]).to(x.device)
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
        
        return x, y, y[rand_index], lam
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply CutMix regularization."""
        # CutMix is applied during training, not as a loss
        return torch.tensor(0.0, device=self.device)


class AdversarialRegularizer(Regularizer):
    """Adversarial training regularization."""
    
    def fgsm_attack(self, model: nn.Module, inputs: torch.Tensor, 
                    targets: torch.Tensor, epsilon: float) -> torch.Tensor:
        """FGSM attack."""
        inputs.requires_grad = True
        
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        
        model.zero_grad()
        loss.backward()
        
        data_grad = inputs.grad.data
        perturbed_inputs = inputs + epsilon * data_grad.sign()
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        return perturbed_inputs.detach()
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply adversarial training."""
        # Generate adversarial examples
        perturbed_inputs = self.fgsm_attack(model, inputs, targets, self.config.adversarial_epsilon)
        
        # Compute adversarial loss
        adv_outputs = model(perturbed_inputs)
        adv_loss = F.cross_entropy(adv_outputs, targets)
        
        return adv_loss


class VirtualAdversarialRegularizer(Regularizer):
    """Virtual adversarial training regularization."""
    
    def get_virtual_adversarial_perturbation(self, model: nn.Module, inputs: torch.Tensor, 
                                            targets: torch.Tensor) -> torch.Tensor:
        """Get virtual adversarial perturbation."""
        d = torch.randn_like(inputs) * self.config.virtual_adversarial_xi
        d = F.normalize(d.view(d.size(0), -1), dim=1).view(d.size())
        
        for _ in range(self.config.virtual_adversarial_iterations):
            d.requires_grad = True
            
            outputs = model(inputs)
            perturbed_outputs = model(inputs + d)
            
            kl_div = F.kl_div(
                F.log_softmax(perturbed_outputs, dim=-1),
                F.softmax(outputs, dim=-1),
                reduction='batchmean'
            )
            
            model.zero_grad()
            kl_div.backward()
            
            d = d.grad.data
            d = F.normalize(d.view(d.size(0), -1), dim=1).view(d.size())
            d = self.config.virtual_adversarial_epsilon * d
        
        return d
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply virtual adversarial training."""
        # Get virtual adversarial perturbation
        d = self.get_virtual_adversarial_perturbation(model, inputs, targets)
        
        # Compute virtual adversarial loss
        perturbed_outputs = model(inputs + d)
        vat_loss = F.kl_div(
            F.log_softmax(perturbed_outputs, dim=-1),
            F.softmax(outputs, dim=-1),
            reduction='batchmean'
        )
        
        return vat_loss


class ConsistencyRegularizer(Regularizer):
    """Consistency regularization."""
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply consistency regularization."""
        # Apply data augmentation
        augmented_inputs = self._apply_augmentation(inputs)
        
        # Get predictions for original and augmented inputs
        with torch.no_grad():
            original_outputs = model(inputs)
        
        augmented_outputs = model(augmented_inputs)
        
        # Compute consistency loss
        consistency_loss = F.kl_div(
            F.log_softmax(augmented_outputs / self.config.consistency_temperature, dim=-1),
            F.softmax(original_outputs / self.config.consistency_temperature, dim=-1),
            reduction='batchmean'
        )
        
        return self.config.consistency_weight * consistency_loss
    
    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply simple data augmentation."""
        # Random noise
        noise = torch.randn_like(x) * 0.01
        return x + noise


class ManifoldRegularizer(Regularizer):
    """Manifold regularization."""
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply manifold regularization."""
        # Extract features
        features = model.get_features(inputs)
        
        # Compute graph Laplacian regularization
        batch_size = features.size(0)
        distance_matrix = torch.cdist(features, features)
        adjacency_matrix = torch.exp(-distance_matrix ** 2 / (2 * 0.1 ** 2))
        
        degree_matrix = torch.diag(adjacency_matrix.sum(dim=1))
        laplacian_matrix = degree_matrix - adjacency_matrix
        
        manifold_loss = torch.trace(features @ laplacian_matrix @ features.T) / batch_size
        
        return self.config.manifold_coefficient * manifold_loss


class ContrastiveRegularizer(Regularizer):
    """Contrastive regularization."""
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply contrastive regularization."""
        # Extract features
        features = model.get_features(inputs)
        
        # Normalize features
        features = F.normalize(features, dim=-1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.T) / self.config.contrastive_temperature
        
        # Create positive pairs (same class)
        labels = targets.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal
        mask = mask * (1 - torch.eye(mask.size(0), device=mask.device))
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        positive_loss = -torch.log(
            exp_sim * mask / (exp_sim.sum(dim=1, keepdim=True) - torch.diag(exp_sim))
        )
        
        contrastive_loss = (positive_loss * mask).sum() / mask.sum()
        
        return self.config.contrastive_weight * contrastive_loss


class InformationBottleneckRegularizer(Regularizer):
    """Information bottleneck regularization."""
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply information bottleneck regularization."""
        # Get intermediate representations
        intermediate_features = model.get_intermediate_features(inputs)
        
        # Compute mutual information approximation
        ib_loss = torch.tensor(0.0, device=self.device)
        
        for features in intermediate_features:
            # Approximate I(X; Z) using variance
            feature_variance = torch.var(features, dim=0).mean()
            
            # Approximate I(Z; Y) using cross-entropy
            z_pred = model.features_to_prediction(features)
            z_loss = F.cross_entropy(z_pred, targets)
            
            # Information bottleneck loss
            ib_loss += feature_variance - self.config.information_beta * z_loss
        
        return ib_loss


class VariationalRegularizer(Regularizer):
    """Variational regularization."""
    
    def __call__(self, model: nn.Module, inputs: torch.Tensor, 
                targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply variational regularization."""
        # Get variational parameters
        mu, logvar = model.encode(inputs)
        
        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return self.config.variational_beta * kl_loss


class RegularizationManager:
    """Manages multiple regularization techniques."""
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.regularizers = {}
        self._initialize_regularizers()
    
    def _initialize_regularizers(self):
        """Initialize regularizers based on configuration."""
        for technique in self.config.enabled_techniques:
            if technique == RegularizationType.DROPOUT:
                self.regularizers[technique] = DropoutRegularizer(self.config)
            elif technique == RegularizationType.WEIGHT_DECAY:
                self.regularizers[technique] = WeightDecayRegularizer(self.config)
            elif technique == RegularizationType.SPECTRAL_NORM:
                self.regularizers[technique] = SpectralNormRegularizer(self.config)
            elif technique == RegularizationType.DROPCONNECT:
                self.regularizers[technique] = DropConnectRegularizer(self.config)
            elif technique == RegularizationType.DROPATH:
                self.regularizers[technique] = DropPathRegularizer(self.config)
            elif technique == RegularizationType.LABEL_SMOOTHING:
                self.regularizers[technique] = LabelSmoothingRegularizer(self.config)
            elif technique == RegularizationType.MIXUP:
                self.regularizers[technique] = MixupRegularizer(self.config)
            elif technique == RegularizationType.CUTMIX:
                self.regularizers[technique] = CutMixRegularizer(self.config)
            elif technique == RegularizationType.ADVERSARIAL_TRAINING:
                self.regularizers[technique] = AdversarialRegularizer(self.config)
            elif technique == RegularizationType.VIRTUAL_ADVERSARIAL_TRAINING:
                self.regularizers[technique] = VirtualAdversarialRegularizer(self.config)
            elif technique == RegularizationType.CONSISTENCY_REGULARIZATION:
                self.regularizers[technique] = ConsistencyRegularizer(self.config)
            elif technique == RegularizationType.MANIFOLD_REGULARIZATION:
                self.regularizers[technique] = ManifoldRegularizer(self.config)
            elif technique == RegularizationType.CONTRASTIVE_REGULARIZATION:
                self.regularizers[technique] = ContrastiveRegularizer(self.config)
            elif technique == RegularizationType.INFORMATION_BOTTLENECK:
                self.regularizers[technique] = InformationBottleneckRegularizer(self.config)
            elif technique == RegularizationType.VARIATIONAL_REGULARIZATION:
                self.regularizers[technique] = VariationalRegularizer(self.config)
    
    def compute_regularization_loss(self, model: nn.Module, inputs: torch.Tensor, 
                                  targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Compute total regularization loss."""
        total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for technique, regularizer in self.regularizers.items():
            loss = regularizer(model, inputs, targets, outputs)
            total_loss += loss
        
        return total_loss
    
    def apply_data_augmentation(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply data augmentation techniques."""
        augmented_inputs = inputs.clone()
        augmented_targets = targets.clone()
        
        # Apply Mixup
        if RegularizationType.MIXUP in self.regularizers:
            mixup_reg = self.regularizers[RegularizationType.MIXUP]
            augmented_inputs, y_a, y_b, lam = mixup_reg.mixup_data(
                augmented_inputs, augmented_targets, self.config.mixup_alpha
            )
            return augmented_inputs, (y_a, y_b, lam)
        
        # Apply CutMix
        if RegularizationType.CUTMIX in self.regularizers:
            cutmix_reg = self.regularizers[RegularizationType.CUTMIX]
            augmented_inputs, y_a, y_b, lam = cutmix_reg.cutmix_data(
                augmented_inputs, augmented_targets, 
                self.config.cutmix_alpha, self.config.cutmix_prob
            )
            return augmented_inputs, (y_a, y_b, lam)
        
        return augmented_inputs, augmented_targets
    
    def apply_model_modifications(self, model: nn.Module):
        """Apply model modifications for regularization."""
        # Apply spectral norm
        if RegularizationType.SPECTRAL_NORM in self.regularizers:
            spectral_reg = self.regularizers[RegularizationType.SPECTRAL_NORM]
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    spectral_reg.apply_spectral_norm(module)
        
        # Add dropout layers
        if RegularizationType.DROPOUT in self.regularizers:
            dropout_reg = self.regularizers[RegularizationType.DROPOUT]
            self._add_dropout_layers(model, dropout_reg)
    
    def _add_dropout_layers(self, model: nn.Module, dropout_reg: DropoutRegularizer):
        """Add dropout layers to model."""
        # This would modify the model architecture to include dropout
        pass


class HAMNetRegularizer:
    """Main regularization framework for HAMNet."""
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.regularization_manager = RegularizationManager(config)
        self.regularization_history = []
    
    def apply_regularization(self, model: nn.Module, inputs: torch.Tensor, 
                           targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Apply all regularization techniques."""
        # Compute regularization loss
        reg_loss = self.regularization_manager.compute_regularization_loss(
            model, inputs, targets, outputs
        )
        
        # Store regularization metrics
        reg_metrics = {
            "total_reg_loss": reg_loss.item(),
            "techniques_used": [t.value for t in self.config.enabled_techniques]
        }
        
        self.regularization_history.append(reg_metrics)
        
        return reg_loss
    
    def prepare_model(self, model: nn.Module) -> nn.Module:
        """Prepare model with regularization modifications."""
        self.regularization_manager.apply_model_modifications(model)
        return model
    
    def prepare_data(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Prepare data with augmentation."""
        return self.regularization_manager.apply_data_augmentation(inputs, targets)
    
    def get_regularization_summary(self) -> Dict[str, Any]:
        """Get regularization summary."""
        if not self.regularization_history:
            return {}
        
        summary = {
            "total_regularizations": len(self.regularization_history),
            "average_reg_loss": np.mean([r["total_reg_loss"] for r in self.regularization_history]),
            "enabled_techniques": [t.value for t in self.config.enabled_techniques],
            "technique_count": len(self.config.enabled_techniques)
        }
        
        return summary
    
    def save_regularization_config(self, filepath: str):
        """Save regularization configuration."""
        config_dict = {
            "config": self.config.__dict__,
            "regularization_history": self.regularization_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Regularization configuration saved to {filepath}")
    
    def load_regularization_config(self, filepath: str):
        """Load regularization configuration."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        self.config = RegularizationConfig(**config_dict["config"])
        self.regularization_history = config_dict["regularization_history"]
        self.regularization_manager = RegularizationManager(self.config)
        
        logger.info(f"Regularization configuration loaded from {filepath}")


# Example usage
def example_regularization():
    """Example of regularization usage."""
    # Configuration
    config = RegularizationConfig(
        dropout_rate=0.1,
        weight_decay=1e-4,
        label_smoothing=0.1,
        mixup_alpha=0.2,
        enabled_techniques=[
            RegularizationType.DROPOUT,
            RegularizationType.WEIGHT_DECAY,
            RegularizationType.LABEL_SMOOTHING,
            RegularizationType.MIXUP
        ]
    )
    
    # Create regularizer
    regularizer = HAMNetRegularizer(config)
    
    # Create model (example)
    # model = HAMNet(HAMNetConfig())
    
    # Prepare model
    # model = regularizer.prepare_model(model)
    
    return regularizer


if __name__ == "__main__":
    # Run example
    regularizer = example_regularization()
    print("Regularization framework initialized successfully")