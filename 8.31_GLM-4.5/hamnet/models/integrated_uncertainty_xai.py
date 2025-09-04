"""
HAMNet Integration Module: Uncertainty-Aware Explainable AI

This module integrates uncertainty quantification with explainable AI methods
to create a comprehensive system for interpretable biological age prediction.

Key Features:
- Uncertainty-aware attention mechanisms
- Explainable multimodal fusion
- Confidence-weighted predictions
- Uncertainty propagation through model layers
- Interactive visualization tools

Author: Claude AI Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import seaborn as sns
from .uncertainty_quantification import (
    ComprehensiveUncertainty, UncertaintyConfig, UncertaintyMetrics
)
from .xai_module import (
    ComprehensiveXAI, XAIConfig, ClinicalInterpretability
)


@dataclass
class IntegrationConfig:
    """Configuration for integrated uncertainty-XAI system"""
    # Uncertainty configuration
    uncertainty_config: UncertaintyConfig = None
    
    # XAI configuration
    xai_config: XAIConfig = None
    
    # Integration parameters
    enable_uncertainty_attention: bool = True
    enable_confidence_weighting: bool = True
    enable_uncertainty_propagation: bool = True
    enable_interactive_viz: bool = True
    
    # Visualization parameters
    plot_style: str = 'seaborn'
    color_palette: str = 'viridis'
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    def __post_init__(self):
        if self.uncertainty_config is None:
            self.uncertainty_config = UncertaintyConfig()
        if self.xai_config is None:
            self.xai_config = XAIConfig()


class UncertaintyAwareAttention(nn.Module):
    """Attention mechanism that incorporates uncertainty information"""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Uncertainty-aware projections
        self.uncertainty_q_proj = nn.Linear(1, embedding_dim)
        self.uncertainty_k_proj = nn.Linear(1, embedding_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Uncertainty gating
        self.uncertainty_gate = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                uncertainty: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Uncertainty-aware attention forward pass
        
        Args:
            query: Query tensor (batch_size, seq_len, embedding_dim)
            key: Key tensor (batch_size, seq_len, embedding_dim)
            value: Value tensor (batch_size, seq_len, embedding_dim)
            uncertainty: Uncertainty tensor (batch_size, seq_len, 1)
            key_padding_mask: Optional mask for key padding
            
        Returns:
            Tuple of (attended_output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Incorporate uncertainty information
        if uncertainty is not None:
            # Project uncertainty to embedding dimension
            unc_q = self.uncertainty_q_proj(uncertainty).unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
            unc_k = self.uncertainty_k_proj(uncertainty).unsqueeze(1)  # (batch_size, 1, seq_len, embedding_dim)
            
            # Reshape for multi-head
            unc_q = unc_q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            unc_k = unc_k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
            
            # Add uncertainty-modulated attention
            unc_scores = torch.matmul(unc_q, unc_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + unc_scores.squeeze(1)
            
        # Apply key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, v)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        
        # Apply output projection
        output = self.out_proj(attended)
        
        # Apply uncertainty gating
        if uncertainty is not None:
            gate = self.uncertainty_gate(output)
            output = gate * output + (1 - gate) * query
            
        return output, attention_weights


class ExplainableMultimodalFusion(nn.Module):
    """Explainable multimodal fusion with uncertainty awareness"""
    
    def __init__(self, config: IntegrationConfig, modalities: List[str]):
        super().__init__()
        self.config = config
        self.modalities = modalities
        self.num_modalities = len(modalities)
        
        # Modality encoders
        self.modality_encoders = nn.ModuleDict({
            modality: nn.Linear(256, 256) for modality in modalities
        })
        
        # Uncertainty-aware cross-modal attention
        self.cross_attention = UncertaintyAwareAttention(
            embedding_dim=256, num_heads=8, dropout=0.1
        )
        
        # Modality importance weights
        self.modality_importance = nn.Parameter(torch.ones(self.num_modalities) / self.num_modalities)
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
        
        # Uncertainty estimation for fusion
        self.fusion_uncertainty = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Mean and variance
        )
        
    def forward(self, modality_embeddings: Dict[str, torch.Tensor],
                modality_uncertainties: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Explainable multimodal fusion
        
        Args:
            modality_embeddings: Dictionary of modality embeddings
            modality_uncertainties: Dictionary of modality uncertainties
            
        Returns:
            Dictionary with fused output and explanations
        """
        if modality_uncertainties is None:
            modality_uncertainties = {}
            
        # Encode each modality
        encoded_embeddings = {}
        encoded_uncertainties = {}
        
        for modality in self.modalities:
            if modality in modality_embeddings:
                encoded_embeddings[modality] = self.modality_encoders[modality](modality_embeddings[modality])
                if modality in modality_uncertainties:
                    encoded_uncertainties[modality] = modality_uncertainties[modality]
                    
        # Cross-modal attention with uncertainty
        if len(encoded_embeddings) > 1:
            fused_embeddings = []
            attention_weights = {}
            
            # Apply cross-modal attention for each modality
            for i, query_modality in enumerate(self.modalities):
                if query_modality in encoded_embeddings:
                    query = encoded_embeddings[query_modality]
                    query_uncertainty = encoded_uncertainties.get(query_modality, None)
                    
                    # Create key and value from all modalities
                    keys = []
                    values = []
                    uncertainties = []
                    
                    for modality in self.modalities:
                        if modality in encoded_embeddings:
                            keys.append(encoded_embeddings[modality])
                            values.append(encoded_embeddings[modality])
                            uncertainties.append(encoded_uncertainties.get(modality, None))
                            
                    keys = torch.stack(keys, dim=1)  # (batch_size, num_modalities, embedding_dim)
                    values = torch.stack(values, dim=1)
                    
                    if any(unc is not None for unc in uncertainties):
                        uncertainties = torch.stack(uncertainties, dim=1)
                    else:
                        uncertainties = None
                        
                    # Apply cross-modal attention
                    attended, weights = self.cross_attention(
                        query.unsqueeze(1), keys, values, uncertainties
                    )
                    
                    fused_embeddings.append(attended.squeeze(1))
                    attention_weights[query_modality] = weights
                    
            # Combine fused embeddings
            if fused_embeddings:
                fused_embeddings = torch.stack(fused_embeddings, dim=1)
                
                # Apply modality importance weights
                importance_weights = F.softmax(self.modality_importance, dim=0)
                fused_output = (fused_embeddings * importance_weights.view(1, -1, 1)).sum(dim=1)
            else:
                fused_output = torch.zeros(1, 256).to(next(self.parameters()).device)
                
        else:
            # Single modality case
            fused_output = list(encoded_embeddings.values())[0]
            attention_weights = {}
            
        # Apply fusion layers
        final_output = self.fusion_layers(fused_output)
        
        # Estimate fusion uncertainty
        fusion_params = self.fusion_uncertainty(final_output)
        fusion_mean = fusion_params[:, 0:1]
        fusion_log_var = fusion_params[:, 1:2]
        fusion_uncertainty = torch.exp(fusion_log_var)
        
        return {
            'fused_output': final_output,
            'fusion_mean': fusion_mean,
            'fusion_log_variance': fusion_log_var,
            'fusion_uncertainty': fusion_uncertainty,
            'attention_weights': attention_weights,
            'modality_importance': importance_weights,
            'encoded_embeddings': encoded_embeddings
        }


class ConfidenceWeightedPredictor(nn.Module):
    """Confidence-weighted prediction module"""
    
    def __init__(self, input_dim: int, config: IntegrationConfig):
        super().__init__()
        self.config = config
        
        # Prediction network
        self.prediction_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Confidence estimation
        self.confidence_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Uncertainty-weighted combination
        self.uncertainty_weight = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Confidence-weighted prediction
        
        Args:
            x: Input features
            uncertainty: Uncertainty estimates
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        # Base prediction
        prediction = self.prediction_network(x)
        
        # Confidence score
        confidence = self.confidence_network(x)
        
        # Uncertainty-based weighting
        if uncertainty is not None and self.config.enable_confidence_weighting:
            # Lower uncertainty -> higher weight
            uncertainty_weight = 1.0 - self.uncertainty_weight(uncertainty)
            
            # Adjust prediction based on confidence
            adjusted_prediction = prediction * confidence * uncertainty_weight
        else:
            adjusted_prediction = prediction * confidence
            uncertainty_weight = torch.ones_like(prediction)
            
        return {
            'prediction': adjusted_prediction,
            'base_prediction': prediction,
            'confidence': confidence,
            'uncertainty_weight': uncertainty_weight,
            'adjusted_confidence': confidence * uncertainty_weight
        }


class UncertaintyPropagation(nn.Module):
    """Uncertainty propagation through network layers"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__()
        self.config = config
        
    def propagate_uncertainty(self, x: torch.Tensor, uncertainty: torch.Tensor,
                            layers: List[nn.Module]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Propagate uncertainty through network layers
        
        Args:
            x: Input tensor
            uncertainty: Input uncertainty
            layers: List of network layers
            
        Returns:
            Tuple of (output, propagated_uncertainty)
        """
        if not self.config.enable_uncertainty_propagation:
            # Simple forward pass without uncertainty propagation
            for layer in layers:
                x = layer(x)
            return x, uncertainty
            
        current_x = x
        current_uncertainty = uncertainty
        
        for layer in layers:
            if isinstance(layer, nn.Linear):
                # Propagate uncertainty through linear layer
                weight = layer.weight
                bias = layer.bias
                
                # Uncertainty propagation using first-order Taylor approximation
                # σ²_out ≈ J * σ²_in * J^T where J is the Jacobian
                propagated_uncertainty = torch.matmul(
                    current_uncertainty, weight.T ** 2
                )
                
                # Apply activation
                current_x = layer(current_x)
                current_x = F.relu(current_x)
                
                # Adjust uncertainty based on activation
                if hasattr(F, 'relu'):
                    # For ReLU: uncertainty is preserved for positive inputs
                    relu_mask = (current_x > 0).float()
                    propagated_uncertainty = propagated_uncertainty * relu_mask
                    
                current_uncertainty = propagated_uncertainty
                
            elif isinstance(layer, nn.Dropout):
                # Dropout increases uncertainty
                current_x = layer(current_x)
                current_uncertainty = current_uncertainty / (1 - layer.p)
                
            else:
                # Other layers
                current_x = layer(current_x)
                
        return current_x, current_uncertainty


class InteractiveVisualization:
    """Interactive visualization tools for uncertainty and explanations"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup visualization parameters"""
        if self.config.plot_style == 'seaborn':
            sns.set_style("whitegrid")
        plt.style.use(self.config.plot_style)
        
    def plot_uncertainty_vs_accuracy(self, predictions: torch.Tensor, 
                                   targets: torch.Tensor, 
                                   uncertainties: torch.Tensor,
                                   title: str = "Uncertainty vs Accuracy"):
        """Plot uncertainty vs accuracy relationship"""
        if not self.config.enable_interactive_viz:
            return
            
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Calculate errors
        errors = torch.abs(predictions - targets)
        
        # Create scatter plot
        scatter = ax.scatter(uncertainties.cpu().numpy(), errors.cpu().numpy(), 
                          alpha=0.6, c=errors.cpu().numpy(), cmap='viridis')
        
        # Add trend line
        z = np.polyfit(uncertainties.cpu().numpy().flatten(), 
                      errors.cpu().numpy().flatten(), 1)
        p = np.poly1d(z)
        ax.plot(uncertainties.cpu().numpy().flatten(), 
               p(uncertainties.cpu().numpy().flatten()), 
               "r--", alpha=0.8)
        
        ax.set_xlabel('Predicted Uncertainty')
        ax.set_ylabel('Absolute Error')
        ax.set_title(title)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Absolute Error')
        
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance_comparison(self, shap_importance: torch.Tensor,
                                        ig_importance: torch.Tensor,
                                        feature_names: List[str],
                                        title: str = "Feature Importance Comparison"):
        """Compare feature importance from different methods"""
        if not self.config.enable_interactive_viz:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # SHAP importance
        shap_imp = shap_importance.cpu().numpy()
        indices1 = np.argsort(shap_imp)[::-1][:10]
        
        ax1.barh([feature_names[i] for i in indices1], shap_imp[indices1])
        ax1.set_xlabel('SHAP Importance')
        ax1.set_title('SHAP Feature Importance')
        
        # Integrated Gradients importance
        ig_imp = ig_importance.cpu().numpy()
        indices2 = np.argsort(ig_imp)[::-1][:10]
        
        ax2.barh([feature_names[i] for i in indices2], ig_imp[indices2])
        ax2.set_xlabel('Integrated Gradients Importance')
        ax2.set_title('Integrated Gradients Feature Importance')
        
        plt.tight_layout()
        plt.show()
        
    def plot_attention_heatmap(self, attention_weights: torch.Tensor,
                              modality_names: List[str],
                              title: str = "Cross-Modal Attention Heatmap"):
        """Plot attention weights as heatmap"""
        if not self.config.enable_interactive_viz:
            return
            
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Average attention weights across batch and heads
        if len(attention_weights.shape) == 4:  # (batch, heads, query, key)
            attention_map = attention_weights.mean(dim=(0, 1)).cpu().numpy()
        else:
            attention_map = attention_weights.cpu().numpy()
            
        sns.heatmap(attention_map, annot=True, cmap='Blues', 
                   xticklabels=modality_names, yticklabels=modality_names,
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Key Modalities')
        ax.set_ylabel('Query Modalities')
        
        plt.tight_layout()
        plt.show()
        
    def plot_uncertainty_calibration(self, predictions: torch.Tensor,
                                   targets: torch.Tensor,
                                   uncertainties: torch.Tensor,
                                   title: str = "Uncertainty Calibration"):
        """Plot uncertainty calibration curve"""
        if not self.config.enable_interactive_viz:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Reliability diagram
        expected_conf, actual_acc = UncertaintyMetrics.reliability_diagram(
            predictions, targets, uncertainties
        )
        
        ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
        ax1.plot(expected_conf, actual_acc, 'bo-', label='Model calibration')
        ax1.set_xlabel('Expected Confidence')
        ax1.set_ylabel('Actual Accuracy')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty vs Error scatter plot
        errors = torch.abs(predictions - targets)
        ax2.scatter(uncertainties.cpu().numpy(), errors.cpu().numpy(), alpha=0.6)
        ax2.set_xlabel('Predicted Uncertainty')
        ax2.set_ylabel('Absolute Error')
        ax2.set_title('Uncertainty vs Error')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def create_interactive_dashboard(self, explanations: Dict[str, Any]):
        """Create interactive dashboard for explanations"""
        if not self.config.enable_interactive_viz:
            return
            
        # This would typically use libraries like Plotly or Dash
        # For now, we'll create a comprehensive static visualization
        
        fig = plt.figure(figsize=(16, 12), dpi=self.config.dpi)
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Prediction summary
        ax1 = fig.add_subplot(gs[0, 0])
        prediction = explanations['prediction']
        ax1.hist(prediction.cpu().numpy(), bins=30, alpha=0.7, color='skyblue')
        ax1.set_title('Prediction Distribution')
        ax1.set_xlabel('Predicted Age')
        ax1.set_ylabel('Frequency')
        
        # 2. SHAP importance
        if 'shap' in explanations:
            ax2 = fig.add_subplot(gs[0, 1])
            shap_imp = torch.mean(torch.abs(explanations['shap']['shap_values']), dim=0)
            top_features = torch.argsort(shap_imp, descending=True)[:10]
            ax2.barh(range(10), shap_imp[top_features].cpu().numpy())
            ax2.set_yticks(range(10))
            ax2.set_yticklabels([f'Feature_{i}' for i in top_features.cpu().numpy()])
            ax2.set_title('Top SHAP Features')
            ax2.set_xlabel('SHAP Importance')
            
        # 3. Attention weights
        if 'attention' in explanations:
            ax3 = fig.add_subplot(gs[0, 2])
            attention_data = list(explanations['attention'].values())[0]
            if 'weights' in attention_data:
                attention_map = attention_data['weights'].mean(dim=(0, 1)).cpu().numpy()
                sns.heatmap(attention_map, ax=ax3, cmap='Blues')
                ax3.set_title('Attention Weights')
                
        # 4. Feature attribution comparison
        if 'shap' in explanations and 'integrated_gradients' in explanations:
            ax4 = fig.add_subplot(gs[1, :])
            shap_imp = torch.mean(torch.abs(explanations['shap']['shap_values']), dim=0)
            ig_imp = torch.abs(explanations['integrated_gradients']['attributions']).mean(dim=0)
            
            x_pos = np.arange(len(shap_imp))
            width = 0.35
            
            ax4.bar(x_pos - width/2, shap_imp.cpu().numpy(), width, label='SHAP', alpha=0.8)
            ax4.bar(x_pos + width/2, ig_imp.cpu().numpy(), width, label='IG', alpha=0.8)
            
            ax4.set_xlabel('Features')
            ax4.set_ylabel('Importance')
            ax4.set_title('Feature Attribution Comparison')
            ax4.legend()
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'F{i}' for i in range(len(shap_imp))])
            
        # 5. Clinical rules
        if 'clinical_rules' in explanations:
            ax5 = fig.add_subplot(gs[2, :])
            rules = explanations['clinical_rules'][:10]
            
            rule_texts = []
            confidences = []
            colors = []
            
            for rule in rules:
                rule_texts.append(f"{rule['feature'][:20]}...")
                confidences.append(rule['confidence'])
                colors.append('green' if rule['importance'] > 0 else 'red')
                
            y_pos = np.arange(len(rule_texts))
            ax5.barh(y_pos, confidences, color=colors, alpha=0.7)
            ax5.set_yticks(y_pos)
            ax5.set_yticklabels(rule_texts)
            ax5.set_xlabel('Confidence')
            ax5.set_title('Clinical Rules')
            
        plt.suptitle('HAMNet Explainable AI Dashboard', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()


class IntegratedHAMNet(nn.Module):
    """Integrated HAMNet with uncertainty quantification and XAI"""
    
    def __init__(self, base_model: nn.Module, config: IntegrationConfig):
        super().__init__()
        self.config = config
        self.base_model = base_model
        
        # Initialize uncertainty quantification
        self.uncertainty_module = ComprehensiveUncertainty(
            config.uncertainty_config, input_dim=256
        )
        
        # Initialize XAI module
        self.xai_module = ComprehensiveXAI(
            base_model, config.xai_config, 
            background_data=None, feature_names=None
        )
        
        # Uncertainty-aware components
        self.uncertainty_attention = UncertaintyAwareAttention(
            embedding_dim=256, num_heads=8, dropout=0.1
        )
        
        # Explainable multimodal fusion
        self.multimodal_fusion = ExplainableMultimodalFusion(
            config, modalities=['clinical', 'imaging', 'genetic', 'lifestyle']
        )
        
        # Confidence-weighted predictor
        self.confidence_predictor = ConfidenceWeightedPredictor(
            input_dim=256, config=config
        )
        
        # Uncertainty propagation
        self.uncertainty_propagation = UncertaintyPropagation(config)
        
        # Interactive visualization
        self.visualizer = InteractiveVisualization(config)
        
    def forward(self, inputs: Dict[str, torch.Tensor],
                masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with integrated uncertainty and explainability
        
        Args:
            inputs: Dictionary of modality inputs
            masks: Optional dictionary of modality masks
            
        Returns:
            Dictionary with predictions, uncertainties, and explanations
        """
        if masks is None:
            masks = {}
            
        # Base model forward pass
        base_output = self.base_model(inputs, masks)
        
        # Extract modality embeddings and uncertainties
        modality_embeddings = {}
        modality_uncertainties = {}
        
        # This would be populated by the base model's intermediate representations
        # For now, we'll use a simplified approach
        
        # Apply explainable multimodal fusion
        fusion_output = self.multimodal_fusion(modality_embeddings, modality_uncertainties)
        
        # Uncertainty quantification
        uncertainty_output = self.uncertainty_module(fusion_output['fused_output'])
        
        # Confidence-weighted prediction
        prediction_output = self.confidence_predictor(
            fusion_output['fused_output'], 
            uncertainty_output.get('uncertainty', None)
        )
        
        # Combine all outputs
        final_output = {
            'predictions': prediction_output['prediction'],
            'base_predictions': prediction_output['base_prediction'],
            'confidence': prediction_output['confidence'],
            'uncertainty_weight': prediction_output['uncertainty_weight'],
            'adjusted_confidence': prediction_output['adjusted_confidence'],
            'uncertainty': uncertainty_output.get('uncertainty', None),
            'fusion_output': fusion_output,
            'uncertainty_output': uncertainty_output,
            'prediction_output': prediction_output
        }
        
        return final_output
        
    def explain(self, inputs: Dict[str, torch.Tensor],
                target_index: int = 0) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for predictions
        
        Args:
            inputs: Input dictionary
            target_index: Target output index
            
        Returns:
            Comprehensive explanation dictionary
        """
        # Get predictions
        with torch.no_grad():
            predictions = self.forward(inputs)
            
        # Generate explanations using XAI module
        explanations = self.xai_module.explain(
            inputs.get('clinical', torch.zeros(1, 100)),  # Example input
            target_index
        )
        
        # Add uncertainty-specific explanations
        explanations['uncertainty_explanation'] = {
            'total_uncertainty': predictions['uncertainty'],
            'confidence_score': predictions['confidence'],
            'uncertainty_weight': predictions['uncertainty_weight'],
            'adjusted_confidence': predictions['adjusted_confidence']
        }
        
        # Add fusion-specific explanations
        explanations['fusion_explanation'] = {
            'modality_importance': predictions['fusion_output']['modality_importance'],
            'attention_weights': predictions['fusion_output']['attention_weights']
        }
        
        return explanations
        
    def visualize_explanations(self, explanations: Dict[str, Any]):
        """Visualize all explanations"""
        self.visualizer.create_interactive_dashboard(explanations)
        
    def evaluate_uncertainty_quality(self, test_data: torch.Tensor, 
                                   test_targets: torch.Tensor) -> Dict[str, float]:
        """Evaluate uncertainty quality"""
        with torch.no_grad():
            predictions = self.forward({'clinical': test_data})
            
        metrics = UncertaintyMetrics.evaluate_uncertainty_quality(
            predictions['predictions'], test_targets, predictions['uncertainty']
        )
        
        return metrics


# Factory functions
def create_integration_config(**kwargs) -> IntegrationConfig:
    """Create integration configuration"""
    return IntegrationConfig(**kwargs)


def create_integrated_hamnet(base_model: nn.Module, 
                           config: IntegrationConfig) -> IntegratedHAMNet:
    """Create integrated HAMNet model"""
    return IntegratedHAMNet(base_model, config)