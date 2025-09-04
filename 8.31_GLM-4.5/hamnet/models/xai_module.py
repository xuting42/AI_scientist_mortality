"""
Comprehensive Explainable AI (XAI) Module for HAMNet

This module implements various explainable AI methods for biological age prediction:
- SHAP value computation for biomarker importance
- Integrated gradients for feature attribution
- Attention visualization and interpretation
- Layer-wise relevance propagation (LRP)
- Local interpretable model-agnostic explanations (LIME)
- Clinical interpretability tools

Author: Claude AI Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
from dataclasses import dataclass
import math
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


@dataclass
class XAIConfig:
    """Configuration for XAI methods"""
    # SHAP configuration
    enable_shap: bool = True
    shap_samples: int = 100
    shap_background_size: int = 50
    
    # Integrated gradients configuration
    enable_integrated_gradients: bool = True
    ig_steps: int = 50
    ig_batch_size: int = 32
    
    # Attention visualization
    enable_attention_viz: bool = True
    attention_threshold: float = 0.1
    
    # LRP configuration
    enable_lrp: bool = True
    lrp_rule: str = 'epsilon'  # epsilon, alpha1beta0, alpha2beta1
    
    # LIME configuration
    enable_lime: bool = True
    lime_samples: int = 1000
    lime_features: int = 10
    
    # Clinical interpretability
    enable_clinical_rules: bool = True
    enable_nl_explanations: bool = True
    enable_counterfactuals: bool = True
    
    # Visualization
    enable_visualization: bool = True
    plot_dpi: int = 300


class SHAPExplainer:
    """SHAP value computation for neural networks"""
    
    def __init__(self, model: nn.Module, background_data: torch.Tensor, 
                 config: XAIConfig):
        self.model = model
        self.background_data = background_data
        self.config = config
        self.device = next(model.parameters()).device
        
    def explain(self, input_data: torch.Tensor, 
                target_index: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute SHAP values for input data
        
        Args:
            input_data: Input tensor to explain
            target_index: Target output index
            
        Returns:
            Dictionary with SHAP values and base values
        """
        self.model.eval()
        
        # Randomly sample background data
        if len(self.background_data) > self.config.shap_background_size:
            background_indices = torch.randperm(len(self.background_data))[:self.config.shap_background_size]
            background = self.background_data[background_indices]
        else:
            background = self.background_data
            
        # Compute expected value (base value)
        with torch.no_grad():
            background_predictions = []
            for i in range(0, len(background), self.config.ig_batch_size):
                batch = background[i:i + self.config.ig_batch_size].to(self.device)
                pred = self.model(batch)
                if isinstance(pred, dict):
                    pred = pred['predictions']
                background_predictions.append(pred[:, target_index])
            base_value = torch.cat(background_predictions).mean()
            
        # Compute SHAP values using sampling
        shap_values = self._compute_shap_values(input_data, background, target_index)
        
        return {
            'shap_values': shap_values,
            'base_value': base_value,
            'input_data': input_data
        }
        
    def _compute_shap_values(self, input_data: torch.Tensor, 
                           background: torch.Tensor, 
                           target_index: int) -> torch.Tensor:
        """Compute SHAP values using sampling approximation"""
        batch_size = input_data.shape[0]
        input_dim = input_data.shape[1]
        shap_values = torch.zeros(batch_size, input_dim, device=self.device)
        
        for i in range(batch_size):
            single_input = input_data[i:i+1]
            
            # Generate perturbed samples
            for _ in range(self.config.shap_samples):
                # Randomly choose background instance
                bg_idx = torch.randint(0, len(background), (1,))
                bg_instance = background[bg_idx]
                
                # Randomly choose features to perturb
                num_features = input_dim
                z = torch.randint(0, 2, (num_features,))
                
                # Create perturbed instance
                perturbed = bg_instance.clone()
                perturbed[z == 1] = single_input[0, z == 1]
                
                # Forward pass
                with torch.no_grad():
                    pred_input = self.model(single_input)
                    pred_perturbed = self.model(perturbed.unsqueeze(0))
                    
                    if isinstance(pred_input, dict):
                        pred_input = pred_input['predictions']
                    if isinstance(pred_perturbed, dict):
                        pred_perturbed = pred_perturbed['predictions']
                        
                    pred_input = pred_input[0, target_index]
                    pred_perturbed = pred_perturbed[0, target_index]
                
                # Update SHAP values
                contribution = (pred_input - pred_perturbed) * z.float()
                shap_values[i] += contribution
                
        shap_values /= self.config.shap_samples
        
        return shap_values
        
    def feature_importance(self, shap_values: torch.Tensor) -> torch.Tensor:
        """Calculate global feature importance from SHAP values"""
        return torch.mean(torch.abs(shap_values), dim=0)


class IntegratedGradients:
    """Integrated gradients for feature attribution"""
    
    def __init__(self, model: nn.Module, config: XAIConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def explain(self, input_data: torch.Tensor, 
                target_index: int = 0,
                baseline: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute integrated gradients for input data
        
        Args:
            input_data: Input tensor to explain
            target_index: Target output index
            baseline: Baseline tensor (default: zeros)
            
        Returns:
            Dictionary with integrated gradients and attributions
        """
        self.model.eval()
        
        if baseline is None:
            baseline = torch.zeros_like(input_data)
            
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, self.config.ig_steps)
        path = baseline.unsqueeze(0) + alphas.view(-1, 1, 1) * (input_data - baseline).unsqueeze(0)
        
        # Compute gradients along path
        gradients = []
        for alpha_path in path:
            alpha_path = alpha_path.to(self.device)
            
            # Enable gradient computation
            alpha_path.requires_grad_(True)
            
            # Forward pass
            output = self.model(alpha_path)
            if isinstance(output, dict):
                output = output['predictions']
                
            # Select target
            target = output[:, target_index]
            
            # Backward pass
            self.model.zero_grad()
            torch.autograd.backward(target, alpha_path)
            
            # Store gradients
            gradients.append(alpha_path.grad.cpu())
            
        gradients = torch.stack(gradients, dim=0)
        
        # Compute integrated gradients
        integrated_gradients = torch.mean(gradients, dim=0) * (input_data - baseline)
        
        # Compute attributions
        attributions = integrated_gradients.sum(dim=-1) if len(integrated_gradients.shape) > 2 else integrated_gradients
        
        return {
            'integrated_gradients': integrated_gradients,
            'attributions': attributions,
            'input_data': input_data,
            'baseline': baseline
        }
        
    def visualize(self, attributions: torch.Tensor, feature_names: List[str] = None):
        """Visualize integrated gradients attributions"""
        if not self.config.enable_visualization:
            return
            
        import matplotlib.pyplot as plt
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(attributions.shape[-1])]
            
        # Create bar plot
        plt.figure(figsize=(12, 6))
        
        if len(attributions.shape) == 1:
            plt.bar(feature_names, attributions.cpu().numpy())
        else:
            # Average across batch
            avg_attributions = attributions.mean(dim=0)
            plt.bar(feature_names, avg_attributions.cpu().numpy())
            
        plt.title('Integrated Gradients Feature Attribution')
        plt.xlabel('Features')
        plt.ylabel('Attribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


class AttentionVisualizer:
    """Attention visualization and interpretation"""
    
    def __init__(self, model: nn.Module, config: XAIConfig):
        self.model = model
        self.config = config
        self.attention_weights = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention weights"""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    self.attention_weights[name] = output[1]
            return hook
            
        # Find attention modules and register hooks
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() and hasattr(module, 'forward'):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}
        
    def explain(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract and analyze attention weights
        
        Args:
            input_data: Input tensor
            
        Returns:
            Dictionary with attention weights and analysis
        """
        self.register_hooks()
        
        # Forward pass to capture attention
        with torch.no_grad():
            output = self.model(input_data)
            
        # Process attention weights
        attention_analysis = {}
        for name, weights in self.attention_weights.items():
            if weights is not None:
                attention_analysis[name] = {
                    'weights': weights,
                    'entropy': self._calculate_attention_entropy(weights),
                    'sparsity': self._calculate_attention_sparsity(weights),
                    'max_attention': torch.max(weights),
                    'mean_attention': torch.mean(weights)
                }
                
        self.remove_hooks()
        
        return attention_analysis
        
    def _calculate_attention_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of attention weights"""
        # Add small epsilon to avoid log(0)
        weights = weights + 1e-8
        weights = weights / weights.sum(dim=-1, keepdim=True)
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)
        return entropy.mean()
        
    def _calculate_attention_sparsity(self, weights: torch.Tensor) -> torch.Tensor:
        """Calculate sparsity of attention weights"""
        threshold = self.config.attention_threshold
        sparse_ratio = (weights < threshold).float().mean()
        return sparse_ratio
        
    def visualize_attention(self, attention_weights: torch.Tensor, 
                          layer_name: str = None):
        """Visualize attention weights"""
        if not self.config.enable_visualization:
            return
            
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(10, 8))
        
        if len(attention_weights.shape) == 3:
            # Multi-head attention
            num_heads = attention_weights.shape[1]
            for i in range(min(num_heads, 8)):  # Show max 8 heads
                plt.subplot(2, 4, i + 1)
                sns.heatmap(attention_weights[0, i].cpu().numpy(), 
                           cmap='Blues', cbar=False)
                plt.title(f'Head {i+1}')
        else:
            # Single attention matrix
            sns.heatmap(attention_weights.cpu().numpy(), cmap='Blues')
            
        title = f'Attention Weights - {layer_name}' if layer_name else 'Attention Weights'
        plt.title(title)
        plt.tight_layout()
        plt.show()


class LRPExplainer:
    """Layer-wise Relevance Propagation (LRP)"""
    
    def __init__(self, model: nn.Module, config: XAIConfig):
        self.model = model
        self.config = config
        self.relevance = {}
        
    def explain(self, input_data: torch.Tensor, 
                target_index: int = 0) -> Dict[str, torch.Tensor]:
        """
        Compute LRP relevance scores
        
        Args:
            input_data: Input tensor
            target_index: Target output index
            
        Returns:
            Dictionary with relevance scores
        """
        self.model.eval()
        
        # Forward pass to compute activations
        activations = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
            
        # Register hooks for all layers
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
                
        # Forward pass
        with torch.no_grad():
            output = self.model(input_data)
            if isinstance(output, dict):
                output = output['predictions']
                
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Initialize relevance at output
        relevance = torch.zeros_like(output)
        relevance[:, target_index] = output[:, target_index]
        
        # Backward propagation of relevance
        relevance_scores = self._propagate_relevance(relevance, activations)
        
        return {
            'relevance_scores': relevance_scores,
            'input_relevance': relevance_scores.get('input', torch.zeros_like(input_data)),
            'activations': activations
        }
        
    def _propagate_relevance(self, relevance: torch.Tensor, 
                           activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Propagate relevance scores backward through network"""
        # This is a simplified implementation
        # Full LRP implementation would require layer-specific rules
        
        relevance_scores = {}
        
        # Get layer names in reverse order
        layer_names = list(activations.keys())
        layer_names.reverse()
        
        current_relevance = relevance
        
        for layer_name in layer_names:
            activation = activations[layer_name]
            
            if self.config.lrp_rule == 'epsilon':
                # Epsilon rule
                epsilon = 1e-9
                z = activation + epsilon * torch.sign(activation)
                s = current_relevance / z
                c = s * activation
                relevance_scores[layer_name] = c
                
            elif self.config.lrp_rule == 'alpha1beta0':
                # Alpha-1-Beta-0 rule (positive contributions only)
                z = activation.clamp(min=0)
                s = current_relevance / (z + 1e-9)
                c = s * activation.clamp(min=0)
                relevance_scores[layer_name] = c
                
            elif self.config.lrp_rule == 'alpha2beta1':
                # Alpha-2-Beta-1 rule
                z_pos = activation.clamp(min=0)
                z_neg = activation.clamp(max=0)
                z = z_pos + 2 * z_neg + 1e-9
                s = current_relevance / z
                c = s * (z_pos + 2 * z_neg)
                relevance_scores[layer_name] = c
                
            current_relevance = c
            
        return relevance_scores


class LIMEExplainer:
    """Local Interpretable Model-agnostic Explanations (LIME)"""
    
    def __init__(self, model: nn.Module, config: XAIConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
    def explain(self, input_data: torch.Tensor, 
                target_index: int = 0,
                feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Generate LIME explanation for input data
        
        Args:
            input_data: Input tensor
            target_index: Target output index
            feature_names: Names of features
            
        Returns:
            Dictionary with LIME explanation
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(input_data.shape[1])]
            
        # Generate perturbed samples
        perturbed_data, masks = self._generate_perturbed_samples(input_data)
        
        # Get predictions for perturbed samples
        predictions = self._get_predictions(perturbed_data, target_index)
        
        # Train interpretable model
        interpretable_model = self._train_interpretable_model(masks, predictions)
        
        # Get feature importance
        feature_importance = interpretable_model.coef_
        
        # Generate explanation
        explanation = {
            'feature_importance': feature_importance,
            'intercept': interpretable_model.intercept_,
            'feature_names': feature_names,
            'perturbed_data': perturbed_data,
            'masks': masks,
            'predictions': predictions,
            'interpretable_model': interpretable_model
        }
        
        return explanation
        
    def _generate_perturbed_samples(self, input_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate perturbed samples for LIME"""
        num_samples = self.config.lime_samples
        num_features = input_data.shape[1]
        
        # Generate binary masks
        masks = torch.randint(0, 2, (num_samples, num_features))
        
        # Create perturbed data
        perturbed_data = input_data.repeat(num_samples, 1)
        perturbed_data = perturbed_data * masks
        
        return perturbed_data, masks
        
    def _get_predictions(self, perturbed_data: torch.Tensor, 
                        target_index: int) -> torch.Tensor:
        """Get predictions for perturbed samples"""
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(perturbed_data), self.config.ig_batch_size):
                batch = perturbed_data[i:i + self.config.ig_batch_size].to(self.device)
                pred = self.model(batch)
                if isinstance(pred, dict):
                    pred = pred['predictions']
                predictions.append(pred[:, target_index].cpu())
                
        return torch.cat(predictions)
        
    def _train_interpretable_model(self, masks: torch.Tensor, 
                                 predictions: torch.Tensor) -> Ridge:
        """Train interpretable model (Ridge regression)"""
        # Normalize features
        scaler = StandardScaler()
        masks_scaled = scaler.fit_transform(masks.numpy())
        
        # Train Ridge regression
        model = Ridge(alpha=1.0)
        model.fit(masks_scaled, predictions.numpy())
        
        return model


class ClinicalInterpretability:
    """Clinical interpretability tools for biological age prediction"""
    
    def __init__(self, model: nn.Module, config: XAIConfig):
        self.model = model
        self.config = config
        
    def extract_clinical_rules(self, input_data: torch.Tensor, 
                             predictions: torch.Tensor,
                             feature_names: List[str]) -> List[Dict[str, Any]]:
        """
        Extract clinical rules from neural network predictions
        
        Args:
            input_data: Input data
            predictions: Model predictions
            feature_names: Names of clinical features
            
        Returns:
            List of clinical rules
        """
        if not self.config.enable_clinical_rules:
            return []
            
        rules = []
        
        # Analyze feature contributions
        feature_importance = self._analyze_feature_contributions(input_data, predictions)
        
        # Extract rules based on important features
        for i, importance in enumerate(feature_importance):
            if abs(importance) > 0.1:  # Threshold for important features
                feature_name = feature_names[i]
                feature_values = input_data[:, i]
                
                # Calculate statistics
                mean_val = torch.mean(feature_values).item()
                std_val = torch.std(feature_values).item()
                
                # Determine rule type based on importance sign
                if importance > 0:
                    rule_type = "increases_biological_age"
                    condition = f"{feature_name} > {mean_val + std_val:.2f}"
                else:
                    rule_type = "decreases_biological_age"
                    condition = f"{feature_name} < {mean_val - std_val:.2f}"
                    
                rules.append({
                    'feature': feature_name,
                    'importance': importance.item(),
                    'rule_type': rule_type,
                    'condition': condition,
                    'confidence': abs(importance).item(),
                    'mean_value': mean_val,
                    'std_value': std_val
                })
                
        # Sort by importance
        rules.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        return rules
        
    def _analyze_feature_contributions(self, input_data: torch.Tensor, 
                                      predictions: torch.Tensor) -> torch.Tensor:
        """Analyze feature contributions using gradient-based method"""
        self.model.eval()
        
        # Enable gradient computation
        input_data.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_data)
        if isinstance(output, dict):
            output = output['predictions']
            
        # Backward pass
        self.model.zero_grad()
        torch.autograd.backward(output, input_data)
        
        # Get gradients
        gradients = input_data.grad
        
        # Calculate feature importance
        importance = torch.mean(torch.abs(gradients), dim=0)
        
        return importance
        
    def generate_nl_explanation(self, input_data: torch.Tensor, 
                              predictions: torch.Tensor,
                              feature_names: List[str]) -> str:
        """
        Generate natural language explanation for predictions
        
        Args:
            input_data: Input data
            predictions: Model predictions
            feature_names: Names of features
            
        Returns:
            Natural language explanation
        """
        if not self.config.enable_nl_explanations:
            return "Natural language explanations disabled."
            
        # Extract clinical rules
        rules = self.extract_clinical_rules(input_data, predictions, feature_names)
        
        # Generate explanation
        explanation = "Biological Age Prediction Explanation:\n\n"
        
        # Overall prediction
        predicted_age = predictions.mean().item()
        explanation += f"Predicted biological age: {predicted_age:.1f} years\n\n"
        
        # Key contributing factors
        explanation += "Key contributing factors:\n"
        for rule in rules[:5]:  # Top 5 rules
            feature = rule['feature']
            importance = rule['importance']
            confidence = rule['confidence']
            
            if importance > 0:
                explanation += f"• {feature} increases biological age (confidence: {confidence:.2f})\n"
            else:
                explanation += f"• {feature} decreases biological age (confidence: {confidence:.2f})\n"
                
        # Add summary
        explanation += f"\nSummary: The model predicts a biological age of {predicted_age:.1f} years "
        explanation += f"based on {len(rules)} significant biomarkers. "
        
        if rules:
            most_important = rules[0]['feature']
            explanation += f"The most influential factor is {most_important}."
            
        return explanation
        
    def generate_counterfactual(self, input_data: torch.Tensor, 
                              predictions: torch.Tensor,
                              target_age: float,
                              feature_names: List[str]) -> Dict[str, Any]:
        """
        Generate counterfactual explanations
        
        Args:
            input_data: Original input data
            predictions: Original predictions
            target_age: Target biological age
            feature_names: Names of features
            
        Returns:
            Counterfactual explanation
        """
        if not self.config.enable_counterfactuals:
            return {'error': 'Counterfactual explanations disabled.'}
            
        # Calculate required change
        current_age = predictions.mean().item()
        required_change = target_age - current_age
        
        # Find features to modify
        feature_importance = self._analyze_feature_contributions(input_data, predictions)
        
        # Select top features for modification
        top_features = torch.argsort(torch.abs(feature_importance), descending=True)[:5]
        
        counterfactual_changes = []
        
        for feature_idx in top_features:
            feature_name = feature_names[feature_idx]
            current_value = input_data[0, feature_idx].item()
            importance = feature_importance[feature_idx].item()
            
            # Calculate required change in feature
            if abs(importance) > 1e-6:
                feature_change = required_change / importance
                new_value = current_value + feature_change
                
                counterfactual_changes.append({
                    'feature': feature_name,
                    'current_value': current_value,
                    'new_value': new_value,
                    'change': feature_change,
                    'importance': importance
                })
                
        return {
            'current_age': current_age,
            'target_age': target_age,
            'required_change': required_change,
            'counterfactual_changes': counterfactual_changes,
            'feasibility': self._assess_counterfactual_feasibility(counterfactual_changes)
        }
        
    def _assess_counterfactual_feasibility(self, changes: List[Dict[str, Any]]) -> float:
        """Assess feasibility of counterfactual changes"""
        if not changes:
            return 0.0
            
        # Simple feasibility score based on magnitude of changes
        total_change = sum(abs(change['change']) for change in changes)
        avg_change = total_change / len(changes)
        
        # Normalize feasibility (0 = not feasible, 1 = very feasible)
        feasibility = 1.0 / (1.0 + avg_change)
        
        return feasibility


class ComprehensiveXAI:
    """Comprehensive XAI module combining all interpretability methods"""
    
    def __init__(self, model: nn.Module, config: XAIConfig, 
                 background_data: Optional[torch.Tensor] = None,
                 feature_names: Optional[List[str]] = None):
        self.model = model
        self.config = config
        self.background_data = background_data
        self.feature_names = feature_names
        
        # Initialize explainers
        self.explainers = {}
        
        if config.enable_shap:
            if background_data is not None:
                self.explainers['shap'] = SHAPExplainer(model, background_data, config)
            else:
                print("Warning: SHAP requires background data. Disabling SHAP.")
                
        if config.enable_integrated_gradients:
            self.explainers['integrated_gradients'] = IntegratedGradients(model, config)
            
        if config.enable_attention_viz:
            self.explainers['attention'] = AttentionVisualizer(model, config)
            
        if config.enable_lrp:
            self.explainers['lrp'] = LRPExplainer(model, config)
            
        if config.enable_lime:
            self.explainers['lime'] = LIMEExplainer(model, config)
            
        if config.enable_clinical_rules or config.enable_nl_explanations or config.enable_counterfactuals:
            self.explainers['clinical'] = ClinicalInterpretability(model, config)
            
    def explain(self, input_data: torch.Tensor, 
                target_index: int = 0) -> Dict[str, Any]:
        """
        Generate comprehensive explanation using all available methods
        
        Args:
            input_data: Input tensor to explain
            target_index: Target output index
            
        Returns:
            Dictionary with explanations from all methods
        """
        explanations = {}
        
        # Get model prediction
        with torch.no_grad():
            prediction = self.model(input_data)
            if isinstance(prediction, dict):
                prediction = prediction['predictions']
                
        explanations['prediction'] = prediction
        
        # SHAP explanation
        if 'shap' in self.explainers:
            explanations['shap'] = self.explainers['shap'].explain(input_data, target_index)
            
        # Integrated gradients
        if 'integrated_gradients' in self.explainers:
            explanations['integrated_gradients'] = self.explainers['integrated_gradients'].explain(
                input_data, target_index
            )
            
        # Attention visualization
        if 'attention' in self.explainers:
            explanations['attention'] = self.explainers['attention'].explain(input_data)
            
        # LRP explanation
        if 'lrp' in self.explainers:
            explanations['lrp'] = self.explainers['lrp'].explain(input_data, target_index)
            
        # LIME explanation
        if 'lime' in self.explainers:
            explanations['lime'] = self.explainers['lime'].explain(
                input_data, target_index, self.feature_names
            )
            
        # Clinical interpretability
        if 'clinical' in self.explainers:
            clinical = self.explainers['clinical']
            
            if self.config.enable_clinical_rules:
                explanations['clinical_rules'] = clinical.extract_clinical_rules(
                    input_data, prediction, self.feature_names or []
                )
                
            if self.config.enable_nl_explanations:
                explanations['nl_explanation'] = clinical.generate_nl_explanation(
                    input_data, prediction, self.feature_names or []
                )
                
            if self.config.enable_counterfactuals:
                target_age = prediction.mean().item() + 5.0  # Example: 5 years younger
                explanations['counterfactual'] = clinical.generate_counterfactual(
                    input_data, prediction, target_age, self.feature_names or []
                )
                
        return explanations
        
    def visualize_explanations(self, explanations: Dict[str, Any]):
        """Visualize all explanations"""
        if not self.config.enable_visualization:
            return
            
        import matplotlib.pyplot as plt
        
        # SHAP visualization
        if 'shap' in explanations:
            plt.figure(figsize=(12, 6))
            shap_values = explanations['shap']['shap_values']
            feature_names = self.feature_names or [f'Feature_{i}' for i in range(shap_values.shape[1])]
            
            # Summary plot
            plt.subplot(1, 2, 1)
            plt.bar(feature_names, torch.mean(torch.abs(shap_values), dim=0).cpu().numpy())
            plt.title('SHAP Feature Importance')
            plt.xticks(rotation=45)
            
            # Individual explanation
            plt.subplot(1, 2, 2)
            plt.bar(feature_names, shap_values[0].cpu().numpy())
            plt.title('SHAP Values - Individual Sample')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.show()
            
        # Integrated gradients visualization
        if 'integrated_gradients' in explanations:
            self.explainers['integrated_gradients'].visualize(
                explanations['integrated_gradients']['attributions'],
                self.feature_names
            )
            
        # Attention visualization
        if 'attention' in explanations:
            for layer_name, attention_data in explanations['attention'].items():
                if 'weights' in attention_data:
                    self.explainers['attention'].visualize_attention(
                        attention_data['weights'], layer_name
                    )
                    
    def generate_report(self, explanations: Dict[str, Any]) -> str:
        """Generate comprehensive explanation report"""
        report = "=== Comprehensive XAI Report ===\n\n"
        
        # Prediction summary
        prediction = explanations['prediction']
        report += f"Prediction Summary:\n"
        report += f"  Predicted biological age: {prediction.mean().item():.1f} years\n"
        report += f"  Prediction range: [{prediction.min().item():.1f}, {prediction.max().item():.1f}]\n\n"
        
        # Feature importance from different methods
        report += "Feature Importance Analysis:\n"
        
        if 'shap' in explanations:
            shap_importance = torch.mean(torch.abs(explanations['shap']['shap_values']), dim=0)
            report += "  SHAP-based importance:\n"
            for i, imp in enumerate(shap_importance.topk(5).values):
                feature_name = self.feature_names[i] if self.feature_names else f'Feature_{i}'
                report += f"    {feature_name}: {imp.item():.3f}\n"
                
        if 'integrated_gradients' in explanations:
            ig_attributions = explanations['integrated_gradients']['attributions']
            report += "  Integrated Gradients importance:\n"
            for i, imp in enumerate(torch.abs(ig_attributions).topk(5).values):
                feature_name = self.feature_names[i] if self.feature_names else f'Feature_{i}'
                report += f"    {feature_name}: {imp.item():.3f}\n"
                
        # Clinical rules
        if 'clinical_rules' in explanations:
            report += "\nClinical Rules:\n"
            for rule in explanations['clinical_rules'][:5]:
                report += f"  • {rule['feature']} {rule['rule_type']} (confidence: {rule['confidence']:.2f})\n"
                report += f"    Condition: {rule['condition']}\n"
                
        # Natural language explanation
        if 'nl_explanation' in explanations:
            report += f"\nNatural Language Explanation:\n{explanations['nl_explanation']}\n"
            
        # Counterfactual explanation
        if 'counterfactual' in explanations:
            cf = explanations['counterfactual']
            report += f"\nCounterfactual Explanation:\n"
            report += f"  To achieve biological age of {cf['target_age']:.1f} years:\n"
            for change in cf['counterfactual_changes'][:3]:
                report += f"    • Change {change['feature']} from {change['current_value']:.2f} to {change['new_value']:.2f}\n"
            report += f"  Feasibility score: {cf['feasibility']:.2f}\n"
            
        return report


# Utility functions
def create_xai_config(**kwargs) -> XAIConfig:
    """Create XAI configuration with default values"""
    return XAIConfig(**kwargs)


def explain_hamnet_prediction(model: nn.Module, 
                           input_data: torch.Tensor,
                           background_data: Optional[torch.Tensor] = None,
                           feature_names: Optional[List[str]] = None,
                           **kwargs) -> Dict[str, Any]:
    """
    Convenience function to explain HAMNet predictions
    
    Args:
        model: HAMNet model
        input_data: Input data to explain
        background_data: Background data for SHAP
        feature_names: Names of features
        **kwargs: Additional configuration parameters
        
    Returns:
        Comprehensive explanation dictionary
    """
    config = create_xai_config(**kwargs)
    xai = ComprehensiveXAI(model, config, background_data, feature_names)
    return xai.explain(input_data)