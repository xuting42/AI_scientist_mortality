"""
Knowledge Distillation Framework for HAMNet Model Compression

This module provides comprehensive knowledge distillation capabilities
for compressing HAMNet models while preserving performance.
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


class DistillationMethod(Enum):
    """Supported distillation methods."""
    SOFT_TARGET = "soft_target"
    FEATURE_MATCHING = "feature_matching"
    ATTENTION_TRANSFER = "attention_transfer"
    RELATIONSHIP_KNOWLEDGE = "relationship_knowledge"
    FLOW_OF_PROCEDURE = "flow_of_procedure"
    CONTRASTIVE = "contrastive"
    MULTI_TEACHER = "multi_teacher"
    SELF_DISTILLATION = "self_distillation"
    ONLINE_DISTILLATION = "online_distillation"


class DistillationLoss(Enum):
    """Supported distillation loss functions."""
    KL_DIVERGENCE = "kl_divergence"
    MSE = "mse"
    COSINE_SIMILARITY = "cosine_similarity"
    L1 = "l1"
    L2 = "l2"
    HUBER = "huber"
    CONTRASTIVE = "contrastive"
    TRIplet = "triplet"


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    method: DistillationMethod = DistillationMethod.SOFT_TARGET
    temperature: float = 4.0
    alpha: float = 0.5  # Balance between task loss and distillation loss
    beta: float = 0.3   # Balance between different distillation losses
    loss_function: DistillationLoss = DistillationLoss.KL_DIVERGENCE
    intermediate_features: bool = True
    attention_maps: bool = True
    layer_selection: List[int] = field(default_factory=lambda: [0, 2, 4])
    feature_matching_layers: List[str] = field(default_factory=lambda: ["encoder", "attention"])
    contrastive_temperature: float = 0.1
    use_augmentation: bool = True
    augmentation_strength: float = 0.1
    ensemble_method: str = "average"  # "average", "weighted", "voting"
    teacher_weight_decay: float = 1e-4
    student_weight_decay: float = 1e-3
    learning_rate_schedule: str = "cosine"
    warmup_epochs: int = 10
    total_epochs: int = 100


@dataclass
class DistillationResult:
    """Results from knowledge distillation."""
    teacher_accuracy: float
    student_accuracy: float
    student_accuracy_baseline: float
    compression_ratio: float
    speedup_ratio: float
    parameter_count_teacher: int
    parameter_count_student: int
    flops_teacher: float
    flops_student: float
    training_time: float
    final_loss: float
    best_epoch: int
    convergence_history: List[float]
    layer_wise_performance: Dict[str, float]


class DistillationLossFunction(ABC):
    """Base class for distillation loss functions."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
    
    @abstractmethod
    def compute_loss(self, teacher_outputs: Dict[str, torch.Tensor], 
                    student_outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute distillation loss."""
        pass


class SoftTargetLoss(DistillationLossFunction):
    """Soft target distillation loss."""
    
    def compute_loss(self, teacher_outputs: Dict[str, torch.Tensor], 
                    student_outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute soft target loss."""
        teacher_logits = teacher_outputs.get("logits")
        student_logits = student_outputs.get("logits")
        
        if teacher_logits is None or student_logits is None:
            return torch.tensor(0.0, device=student_logits.device)
        
        # Apply temperature scaling
        teacher_soft = F.softmax(teacher_logits / self.config.temperature, dim=-1)
        student_soft = F.softmax(student_logits / self.config.temperature, dim=-1)
        
        # KL divergence loss
        loss = F.kl_div(
            F.log_softmax(student_logits / self.config.temperature, dim=-1),
            teacher_soft,
            reduction='batchmean'
        ) * (self.config.temperature ** 2)
        
        return loss


class FeatureMatchingLoss(DistillationLossFunction):
    """Feature matching distillation loss."""
    
    def compute_loss(self, teacher_outputs: Dict[str, torch.Tensor], 
                    student_outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute feature matching loss."""
        total_loss = torch.tensor(0.0, device=targets.device)
        
        # Match intermediate features
        for layer_name in self.config.feature_matching_layers:
            teacher_features = teacher_outputs.get(f"features_{layer_name}")
            student_features = student_outputs.get(f"features_{layer_name}")
            
            if teacher_features is not None and student_features is not None:
                # Resize features if necessary
                if teacher_features.shape != student_features.shape:
                    student_features = F.adaptive_avg_pool1d(
                        student_features, teacher_features.shape[-1]
                    )
                
                # MSE loss between features
                loss = F.mse_loss(student_features, teacher_features)
                total_loss += loss
        
        return total_loss / len(self.config.feature_matching_layers)


class AttentionTransferLoss(DistillationLossFunction):
    """Attention transfer distillation loss."""
    
    def compute_loss(self, teacher_outputs: Dict[str, torch.Tensor], 
                    student_outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute attention transfer loss."""
        total_loss = torch.tensor(0.0, device=targets.device)
        
        # Extract attention maps
        teacher_attentions = teacher_outputs.get("attentions", [])
        student_attentions = student_outputs.get("attentions", [])
        
        if not teacher_attentions or not student_attentions:
            return total_loss
        
        # Match attention maps from selected layers
        for layer_idx in self.config.layer_selection:
            if layer_idx < len(teacher_attentions) and layer_idx < len(student_attentions):
                teacher_attn = teacher_attentions[layer_idx]
                student_attn = student_attentions[layer_idx]
                
                # Compute attention map similarity
                teacher_attn_flat = teacher_attn.view(teacher_attn.size(0), -1)
                student_attn_flat = student_attn.view(student_attn.size(0), -1)
                
                # Resize if necessary
                if teacher_attn_flat.shape != student_attn_flat.shape:
                    student_attn_flat = F.adaptive_avg_pool1d(
                        student_attn_flat.unsqueeze(1), 
                        teacher_attn_flat.shape[-1]
                    ).squeeze(1)
                
                # Cosine similarity loss
                similarity = F.cosine_similarity(teacher_attn_flat, student_attn_flat, dim=1)
                loss = 1 - similarity.mean()
                total_loss += loss
        
        return total_loss / len(self.config.layer_selection)


class RelationshipKnowledgeLoss(DistillationLossFunction):
    """Relationship knowledge distillation loss."""
    
    def compute_loss(self, teacher_outputs: Dict[str, torch.Tensor], 
                    student_outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute relationship knowledge loss."""
        teacher_logits = teacher_outputs.get("logits")
        student_logits = student_outputs.get("logits")
        
        if teacher_logits is None or student_logits is None:
            return torch.tensor(0.0, device=student_logits.device)
        
        # Compute pairwise similarities
        batch_size = teacher_logits.size(0)
        
        teacher_sim = torch.mm(teacher_logits, teacher_logits.t())
        student_sim = torch.mm(student_logits, student_logits.t())
        
        # Normalize similarities
        teacher_sim = F.softmax(teacher_sim, dim=-1)
        student_sim = F.softmax(student_sim, dim=-1)
        
        # KL divergence between similarity matrices
        loss = F.kl_div(
            F.log_softmax(student_sim, dim=-1),
            teacher_sim,
            reduction='batchmean'
        )
        
        return loss


class ContrastiveDistillationLoss(DistillationLossFunction):
    """Contrastive distillation loss."""
    
    def compute_loss(self, teacher_outputs: Dict[str, torch.Tensor], 
                    student_outputs: Dict[str, torch.Tensor],
                    targets: torch.Tensor) -> torch.Tensor:
        """Compute contrastive distillation loss."""
        teacher_features = teacher_outputs.get("features")
        student_features = student_outputs.get("features")
        
        if teacher_features is None or student_features is None:
            return torch.tensor(0.0, device=student_features.device)
        
        batch_size = teacher_features.size(0)
        
        # Normalize features
        teacher_features = F.normalize(teacher_features, dim=-1)
        student_features = F.normalize(student_features, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.mm(student_features, teacher_features.t())
        
        # Create labels (positive pairs are on diagonal)
        labels = torch.arange(batch_size, device=student_features.device)
        
        # Contrastive loss
        loss = F.cross_entropy(similarity / self.config.contrastive_temperature, labels)
        
        return loss


class DistillationLossFactory:
    """Factory for creating distillation loss functions."""
    
    @staticmethod
    def create_loss(config: DistillationConfig) -> DistillationLossFunction:
        """Create distillation loss function based on configuration."""
        if config.method == DistillationMethod.SOFT_TARGET:
            return SoftTargetLoss(config)
        elif config.method == DistillationMethod.FEATURE_MATCHING:
            return FeatureMatchingLoss(config)
        elif config.method == DistillationMethod.ATTENTION_TRANSFER:
            return AttentionTransferLoss(config)
        elif config.method == DistillationMethod.RELATIONSHIP_KNOWLEDGE:
            return RelationshipKnowledgeLoss(config)
        elif config.method == DistillationMethod.CONTRASTIVE:
            return ContrastiveDistillationLoss(config)
        else:
            raise ValueError(f"Unsupported distillation method: {config.method}")


class FeatureExtractor:
    """Extract intermediate features from models."""
    
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.features = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to extract features."""
        def get_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach()
            return hook
        
        for name, module in self.model.named_modules():
            if any(layer_name in name for layer_name in self.layer_names):
                hook = get_hook(name)
                module.register_forward_hook(hook)
                self.hooks.append((module, hook))
    
    def __call__(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from input."""
        self.features.clear()
        _ = self.model(x)
        return self.features.copy()
    
    def remove_hooks(self):
        """Remove registered hooks."""
        for module, hook in self.hooks:
            module.remove_forward_hook(hook)
        self.hooks.clear()


class KnowledgeDistillationTrainer:
    """Knowledge distillation trainer."""
    
    def __init__(self, teacher_model: nn.Module, student_model: nn.Module, 
                 config: DistillationConfig):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.distillation_loss = DistillationLossFactory.create_loss(config)
        self.teacher_feature_extractor = FeatureExtractor(
            teacher_model, config.feature_matching_layers
        )
        self.student_feature_extractor = FeatureExtractor(
            student_model, config.feature_matching_layers
        )
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        self.training_history = []
    
    def train_epoch(self, train_loader, optimizer, epoch: int, device: str = 'cuda') -> Dict[str, float]:
        """Train for one epoch."""
        self.teacher_model.eval()
        self.student_model.train()
        
        total_task_loss = 0
        total_distill_loss = 0
        total_loss = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Teacher forward pass
            with torch.no_grad():
                teacher_outputs = self._get_teacher_outputs(data)
            
            # Student forward pass
            student_outputs = self._get_student_outputs(data)
            
            # Compute losses
            task_loss = self._compute_task_loss(student_outputs, targets)
            distill_loss = self.distillation_loss.compute_loss(
                teacher_outputs, student_outputs, targets
            )
            
            # Combined loss
            loss = (1 - self.config.alpha) * task_loss + self.config.alpha * distill_loss
            
            loss.backward()
            optimizer.step()
            
            total_task_loss += task_loss.item()
            total_distill_loss += distill_loss.item()
            total_loss += loss.item()
        
        num_batches = len(train_loader)
        return {
            "task_loss": total_task_loss / num_batches,
            "distill_loss": total_distill_loss / num_batches,
            "total_loss": total_loss / num_batches
        }
    
    def _get_teacher_outputs(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get teacher model outputs."""
        outputs = {}
        
        # Get logits
        with torch.no_grad():
            logits = self.teacher_model(data)
            outputs["logits"] = logits
        
        # Get features
        features = self.teacher_feature_extractor(data)
        for key, value in features.items():
            outputs[f"features_{key}"] = value
        
        # Get attention maps if available
        if hasattr(self.teacher_model, 'get_attention_maps'):
            attention_maps = self.teacher_model.get_attention_maps(data)
            outputs["attentions"] = attention_maps
        
        return outputs
    
    def _get_student_outputs(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get student model outputs."""
        outputs = {}
        
        # Get logits
        logits = self.student_model(data)
        outputs["logits"] = logits
        
        # Get features
        features = self.student_feature_extractor(data)
        for key, value in features.items():
            outputs[f"features_{key}"] = value
        
        # Get attention maps if available
        if hasattr(self.student_model, 'get_attention_maps'):
            attention_maps = self.student_model.get_attention_maps(data)
            outputs["attentions"] = attention_maps
        
        return outputs
    
    def _compute_task_loss(self, student_outputs: Dict[str, torch.Tensor], 
                          targets: torch.Tensor) -> torch.Tensor:
        """Compute task loss."""
        logits = student_outputs.get("logits")
        if logits is None:
            return torch.tensor(0.0, device=targets.device)
        
        return F.mse_loss(logits, targets)
    
    def evaluate(self, data_loader, device: str = 'cuda') -> Dict[str, float]:
        """Evaluate models."""
        self.teacher_model.eval()
        self.student_model.eval()
        
        teacher_loss = 0
        student_loss = 0
        teacher_correct = 0
        student_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                
                # Teacher evaluation
                teacher_outputs = self.teacher_model(data)
                teacher_loss += F.mse_loss(teacher_outputs, targets).item()
                
                # Student evaluation
                student_outputs = self.student_model(data)
                student_loss += F.mse_loss(student_outputs, targets).item()
                
                total_samples += data.size(0)
        
        return {
            "teacher_loss": teacher_loss / len(data_loader),
            "student_loss": student_loss / len(data_loader),
            "teacher_accuracy": 1.0 / (1.0 + teacher_loss / len(data_loader)),
            "student_accuracy": 1.0 / (1.0 + student_loss / len(data_loader))
        }
    
    def train(self, train_loader, val_loader, epochs: int = None, 
              device: str = 'cuda') -> DistillationResult:
        """Train student model with knowledge distillation."""
        if epochs is None:
            epochs = self.config.total_epochs
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.student_model.parameters(),
            lr=0.001,
            weight_decay=self.config.student_weight_decay
        )
        
        # Learning rate scheduler
        if self.config.learning_rate_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - self.config.warmup_epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=epochs // 3, gamma=0.1
            )
        
        # Training loop
        best_student_accuracy = 0
        best_epoch = 0
        convergence_history = []
        
        logger.info(f"Starting knowledge distillation training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Warmup learning rate
            if epoch < self.config.warmup_epochs:
                lr = 0.001 * (epoch + 1) / self.config.warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, optimizer, epoch, device)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader, device)
            
            # Update learning rate
            if epoch >= self.config.warmup_epochs:
                scheduler.step()
            
            # Track best model
            if val_metrics["student_accuracy"] > best_student_accuracy:
                best_student_accuracy = val_metrics["student_accuracy"]
                best_epoch = epoch
            
            # Store training history
            convergence_history.append(val_metrics["student_accuracy"])
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: "
                           f"Train Loss: {train_metrics['total_loss']:.4f}, "
                           f"Val Student Acc: {val_metrics['student_accuracy']:.4f}, "
                           f"Val Teacher Acc: {val_metrics['teacher_accuracy']:.4f}")
        
        # Final evaluation
        final_metrics = self.evaluate(val_loader, device)
        
        # Create result
        result = DistillationResult(
            teacher_accuracy=final_metrics["teacher_accuracy"],
            student_accuracy=final_metrics["student_accuracy"],
            student_accuracy_baseline=0.0,  # Would need baseline training
            compression_ratio=self._calculate_compression_ratio(),
            speedup_ratio=self._calculate_speedup_ratio(),
            parameter_count_teacher=sum(p.numel() for p in self.teacher_model.parameters()),
            parameter_count_student=sum(p.numel() for p in self.student_model.parameters()),
            flops_teacher=self._estimate_flops(self.teacher_model),
            flops_student=self._estimate_flops(self.student_model),
            training_time=0.0,  # Would need to track actual time
            final_loss=train_metrics["total_loss"],
            best_epoch=best_epoch,
            convergence_history=convergence_history,
            layer_wise_performance=self._analyze_layer_performance()
        )
        
        logger.info(f"Distillation completed. "
                   f"Teacher Acc: {result.teacher_accuracy:.4f}, "
                   f"Student Acc: {result.student_accuracy:.4f}, "
                   f"Compression: {result.compression_ratio:.2f}x")
        
        return result
    
    def _calculate_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        teacher_params = sum(p.numel() for p in self.teacher_model.parameters())
        student_params = sum(p.numel() for p in self.student_model.parameters())
        return teacher_params / student_params
    
    def _calculate_speedup_ratio(self) -> float:
        """Calculate speedup ratio."""
        teacher_flops = self._estimate_flops(self.teacher_model)
        student_flops = self._estimate_flops(self.student_model)
        return teacher_flops / student_flops
    
    def _estimate_flops(self, model: nn.Module) -> float:
        """Estimate FLOPs for model."""
        # Simplified FLOP estimation
        total_flops = 0
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total_flops += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.Conv1d):
                total_flops += 2 * module.in_channels * module.out_channels * module.kernel_size[0]
        
        return total_flops
    
    def _analyze_layer_performance(self) -> Dict[str, float]:
        """Analyze layer-wise performance."""
        # This would require more detailed analysis
        return {"layer_0": 0.9, "layer_1": 0.85, "layer_2": 0.8}


class MultiTeacherDistillationTrainer(KnowledgeDistillationTrainer):
    """Multi-teacher knowledge distillation."""
    
    def __init__(self, teacher_models: List[nn.Module], student_model: nn.Module, 
                 config: DistillationConfig):
        self.teacher_models = teacher_models
        self.student_model = student_model
        self.config = config
        self.teacher_weights = self._calculate_teacher_weights()
        
        # Freeze teacher models
        for teacher in teacher_models:
            for param in teacher.parameters():
                param.requires_grad = False
        
        # Setup feature extractors for each teacher
        self.teacher_feature_extractors = [
            FeatureExtractor(teacher, config.feature_matching_layers)
            for teacher in teacher_models
        ]
        
        self.student_feature_extractor = FeatureExtractor(
            student_model, config.feature_matching_layers
        )
        
        # Setup distillation loss
        self.distillation_loss = DistillationLossFactory.create_loss(config)
        
        self.training_history = []
    
    def _calculate_teacher_weights(self) -> List[float]:
        """Calculate teacher ensemble weights."""
        if self.config.ensemble_method == "average":
            return [1.0 / len(self.teacher_models)] * len(self.teacher_models)
        elif self.config.ensemble_method == "weighted":
            # Would need teacher performance information
            return [1.0 / len(self.teacher_models)] * len(self.teacher_models)
        else:
            return [1.0 / len(self.teacher_models)] * len(self.teacher_models)
    
    def _get_teacher_outputs(self, data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get ensemble teacher outputs."""
        all_outputs = []
        
        for i, (teacher, extractor) in enumerate(zip(self.teacher_models, self.teacher_feature_extractors)):
            outputs = {}
            
            # Get logits
            with torch.no_grad():
                logits = teacher(data)
                outputs["logits"] = logits
            
            # Get features
            features = extractor(data)
            for key, value in features.items():
                outputs[f"features_{key}"] = value
            
            # Get attention maps if available
            if hasattr(teacher, 'get_attention_maps'):
                attention_maps = teacher.get_attention_maps(data)
                outputs["attentions"] = attention_maps
            
            all_outputs.append(outputs)
        
        # Ensemble outputs
        ensemble_outputs = {}
        
        # Average logits
        all_logits = [out["logits"] for out in all_outputs]
        ensemble_outputs["logits"] = torch.stack(all_logits).mean(dim=0)
        
        # Average features
        for key in all_outputs[0].keys():
            if key.startswith("features_"):
                all_features = [out[key] for out in all_outputs]
                ensemble_outputs[key] = torch.stack(all_features).mean(dim=0)
        
        # Average attention maps
        if "attentions" in all_outputs[0]:
            all_attentions = [out["attentions"] for out in all_outputs]
            ensemble_attentions = []
            for layer_idx in range(len(all_attentions[0])):
                layer_attentions = [attn[layer_idx] for attn in all_attentions]
                ensemble_attentions.append(torch.stack(layer_attentions).mean(dim=0))
            ensemble_outputs["attentions"] = ensemble_attentions
        
        return ensemble_outputs


class SelfDistillationTrainer(KnowledgeDistillationTrainer):
    """Self-distillation trainer."""
    
    def __init__(self, model: nn.Module, config: DistillationConfig):
        # Create student as a copy of teacher
        student_model = copy.deepcopy(model)
        
        # Modify student architecture (make it smaller)
        self._modify_student_architecture(student_model)
        
        super().__init__(model, student_model, config)
        
        # Use same model for both teacher and student
        self.teacher_model = model
        self.student_model = student_model
    
    def _modify_student_architecture(self, model: nn.Module):
        """Modify student architecture to be smaller."""
        # This would implement architecture modifications
        # For example: reduce hidden sizes, remove layers, etc.
        pass


class HAMNetKnowledgeDistillation:
    """Main knowledge distillation framework for HAMNet."""
    
    def __init__(self, teacher_model: nn.Module, config: DistillationConfig):
        self.teacher_model = teacher_model
        self.config = config
        self.distillation_results = []
    
    def create_student_model(self, compression_ratio: float = 0.5) -> nn.Module:
        """Create student model with specified compression ratio."""
        teacher_config = self.teacher_model.config
        
        # Create smaller configuration
        student_config = HAMNetConfig(
            input_size=teacher_config.input_size,
            hidden_size=int(teacher_config.hidden_size * np.sqrt(compression_ratio)),
            num_layers=max(1, int(teacher_config.num_layers * compression_ratio)),
            dropout_rate=teacher_config.dropout_rate,
            attention_heads=max(1, int(teacher_config.attention_heads * compression_ratio))
        )
        
        return HAMNet(student_config)
    
    def distill(self, train_loader, val_loader, compression_ratio: float = 0.5,
                epochs: int = None) -> Tuple[nn.Module, DistillationResult]:
        """Perform knowledge distillation."""
        logger.info(f"Starting knowledge distillation with compression ratio: {compression_ratio}")
        
        # Create student model
        student_model = self.create_student_model(compression_ratio)
        
        # Create trainer based on method
        if self.config.method == DistillationMethod.MULTI_TEACHER:
            # For multi-teacher, would need multiple teacher models
            trainer = MultiTeacherDistillationTrainer(
                [self.teacher_model], student_model, self.config
            )
        elif self.config.method == DistillationMethod.SELF_DISTILLATION:
            trainer = SelfDistillationTrainer(self.teacher_model, self.config)
            student_model = trainer.student_model
        else:
            trainer = KnowledgeDistillationTrainer(
                self.teacher_model, student_model, self.config
            )
        
        # Train student model
        result = trainer.train(train_loader, val_loader, epochs)
        
        self.distillation_results.append(result)
        
        return student_model, result
    
    def get_distillation_summary(self) -> Dict[str, Any]:
        """Get summary of distillation results."""
        if not self.distillation_results:
            return {}
        
        summary = {
            "total_distillations": len(self.distillation_results),
            "best_compression_ratio": max(r.compression_ratio for r in self.distillation_results),
            "best_student_accuracy": max(r.student_accuracy for r in self.distillation_results),
            "average_accuracy_retention": np.mean([
                r.student_accuracy / r.teacher_accuracy 
                for r in self.distillation_results
            ]),
            "distillation_method": self.config.method.value,
            "best_result": max(self.distillation_results, key=lambda x: x.student_accuracy)
        }
        
        return summary
    
    def save_distilled_model(self, model: nn.Module, filepath: str):
        """Save distilled model."""
        torch.save(model.state_dict(), filepath)
        logger.info(f"Distilled model saved to {filepath}")
    
    def load_distilled_model(self, model: nn.Module, filepath: str) -> nn.Module:
        """Load distilled model."""
        model.load_state_dict(torch.load(filepath))
        logger.info(f"Distilled model loaded from {filepath}")
        return model


# Example usage
def example_knowledge_distillation():
    """Example of knowledge distillation."""
    # Configuration
    config = DistillationConfig(
        method=DistillationMethod.SOFT_TARGET,
        temperature=4.0,
        alpha=0.5,
        intermediate_features=True,
        attention_maps=True,
        total_epochs=50
    )
    
    # Create teacher model (example)
    # teacher_model = HAMNet(HAMNetConfig(hidden_size=512, num_layers=8))
    
    # Create distillation framework
    # distillation = HAMNetKnowledgeDistillation(teacher_model, config)
    
    # Distill to create smaller student model
    # student_model, result = distillation.distill(train_loader, val_loader, compression_ratio=0.25)
    
    return config


if __name__ == "__main__":
    # Run example
    config = example_knowledge_distillation()
    print("Knowledge distillation framework initialized successfully")