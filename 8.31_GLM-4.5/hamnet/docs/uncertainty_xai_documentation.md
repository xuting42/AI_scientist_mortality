# HAMNet Uncertainty Quantification and Explainable AI Documentation

## Overview

This document provides comprehensive documentation for the uncertainty quantification and explainable AI (XAI) components implemented for the HAMNet (Hierarchical Attention-based Multimodal Network) biological age prediction system.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Uncertainty Quantification](#uncertainty-quantification)
3. [Explainable AI (XAI)](#explainable-ai-xai)
4. [Integrated System](#integrated-system)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Best Practices](#best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Requirements

```bash
# Core dependencies
torch>=1.9.0
numpy>=1.19.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
seaborn>=0.11.0

# Optional for enhanced visualization
plotly>=5.0.0
dash>=2.0.0
```

### Installation

```python
# Clone the repository
git clone https://github.com/your-repo/hamnet.git
cd hamnet

# Install the package
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

## Uncertainty Quantification

The uncertainty quantification module provides multiple methods to estimate and quantify uncertainty in biological age predictions.

### Methods Implemented

1. **Bayesian Neural Networks with Monte Carlo Dropout**
   - Probabilistic modeling of network weights
   - Monte Carlo sampling for uncertainty estimation
   - KL divergence regularization

2. **Heteroscedastic Loss Functions**
   - Aleatoric uncertainty modeling
   - Input-dependent variance estimation
   - Combined prediction and uncertainty loss

3. **Deep Ensemble Methods**
   - Multiple models with diverse initialization
   - Ensemble uncertainty estimation
   - Diversity regularization

4. **Evidential Deep Learning**
   - Higher-order uncertainty modeling
   - Evidence theory for uncertainty
   - Robust uncertainty estimation

### Basic Usage

```python
import torch
from models.uncertainty_quantification import (
    ComprehensiveUncertainty, UncertaintyConfig, UncertaintyMetrics
)

# Configure uncertainty quantification
config = UncertaintyConfig(
    enable_mc_dropout=True,
    enable_heteroscedastic=True,
    enable_evidential=True,
    num_mc_samples=50,
    mc_dropout_rate=0.1
)

# Create uncertainty module
uncertainty_module = ComprehensiveUncertainty(config, input_dim=256)

# Forward pass with uncertainty
input_data = torch.randn(32, 256)
output = uncertainty_module(input_data, training=True)

# Access uncertainty estimates
prediction = output['predictions']
uncertainty = output['uncertainty']
aleatoric = output['aleatoric_uncertainty']
epistemic = output['epistemic_uncertainty']

# Calculate loss
targets = torch.randn(32, 1)
loss = uncertainty_module.loss(output, targets)
```

### Advanced Usage

```python
# Monte Carlo prediction for uncertainty estimation
from models.uncertainty_quantification import monte_carlo_prediction

# Enable dropout during inference
model.eval()
mc_predictions = monte_carlo_prediction(model, input_data, num_samples=100)

# Access ensemble statistics
mean_prediction = mc_predictions['mean']
prediction_variance = mc_predictions['variance']
prediction_std = mc_predictions['std']

# Evaluate uncertainty quality
metrics = UncertaintyMetrics.evaluate_uncertainty_quality(
    mean_prediction, targets, prediction_std
)

print(f"Negative Log Likelihood: {metrics['nll']:.4f}")
print(f"Calibration Error: {metrics['calibration_error']:.4f}")
print(f"Sharpness: {metrics['sharpness']:.4f}")
```

## Explainable AI (XAI)

The XAI module provides comprehensive interpretability methods for understanding HAMNet predictions.

### Methods Implemented

1. **SHAP Value Computation**
   - Model-agnostic feature attribution
   - Global and local feature importance
   - Interaction effects analysis

2. **Integrated Gradients**
   - Gradient-based feature attribution
   - Axiomatic attribution methods
   - Satisfies sensitivity and implementation invariance

3. **Attention Visualization**
   - Multi-head attention analysis
   - Cross-modal attention interpretation
   - Attention entropy and sparsity metrics

4. **Layer-wise Relevance Propagation (LRP)**
   - Backward propagation of relevance
   - Multiple propagation rules
   - Layer-specific relevance distribution

5. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Local surrogate models
   - Perturbation-based explanations
   - Model-agnostic local interpretability

### Basic Usage

```python
from models.xai_module import ComprehensiveXAI, XAIConfig

# Configure XAI methods
xai_config = XAIConfig(
    enable_integrated_gradients=True,
    enable_lime=True,
    enable_clinical_rules=True,
    enable_nl_explanations=True
)

# Create XAI module
background_data = torch.randn(100, 256)  # Background data for SHAP
feature_names = [f'biomarker_{i}' for i in range(256)]

xai_module = ComprehensiveXAI(
    model, xai_config, background_data, feature_names
)

# Generate explanations
input_data = torch.randn(1, 256)
explanations = xai_module.explain(input_data)

# Access different explanation methods
shap_values = explanations['shap']['shap_values']
ig_attributions = explanations['integrated_gradients']['attributions']
clinical_rules = explanations['clinical_rules']
nl_explanation = explanations['nl_explanation']
```

### Clinical Interpretability

```python
# Generate clinical rules
for rule in clinical_rules[:5]:
    print(f"Rule: {rule['feature']} {rule['rule_type']}")
    print(f"Condition: {rule['condition']}")
    print(f"Confidence: {rule['confidence']:.3f}")
    print()

# Natural language explanation
print("Natural Language Explanation:")
print(nl_explanation)

# Counterfactual explanation
counterfactual = explanations['counterfactual']
print(f"Counterfactual to achieve {counterfactual['target_age']:.1f} years:")
for change in counterfactual['counterfactual_changes'][:3]:
    print(f"  - Change {change['feature']} from {change['current_value']:.2f} to {change['new_value']:.2f}")
```

## Integrated System

The integrated system combines uncertainty quantification with XAI methods to provide comprehensive interpretability.

### Key Features

1. **Uncertainty-Aware Attention**
   - Attention mechanisms incorporating uncertainty
   - Confidence-weighted attention scores
   - Uncertainty-modulated information flow

2. **Explainable Multimodal Fusion**
   - Interpretable fusion of multimodal data
   - Modality importance estimation
   - Cross-modal attention analysis

3. **Confidence-Weighted Predictions**
   - Uncertainty-informed prediction weighting
   - Confidence score estimation
   - Reliable prediction filtering

4. **Interactive Visualization**
   - Comprehensive dashboard creation
   - Uncertainty vs accuracy plots
   - Feature importance comparison
   - Attention weight visualization

### Usage Example

```python
from models.integrated_uncertainty_xai import IntegratedHAMNet, IntegrationConfig

# Configure integrated system
integration_config = IntegrationConfig(
    enable_uncertainty_attention=True,
    enable_confidence_weighting=True,
    enable_uncertainty_propagation=True,
    enable_interactive_viz=True
)

# Create integrated HAMNet
integrated_model = IntegratedHAMNet(base_model, integration_config)

# Forward pass with uncertainty and explainability
inputs = {
    'clinical': torch.randn(32, 100),
    'imaging': torch.randn(32, 512),
    'genetic': torch.randn(32, 1000),
    'lifestyle': torch.randn(32, 50)
}

output = integrated_model(inputs)

# Access comprehensive output
prediction = output['predictions']
confidence = output['confidence']
uncertainty = output['uncertainty']
adjusted_confidence = output['adjusted_confidence']

# Generate explanations
explanations = integrated_model.explain(inputs)

# Visualize explanations
integrated_model.visualize_explanations(explanations)

# Evaluate uncertainty quality
test_targets = torch.randn(32, 1)
metrics = integrated_model.evaluate_uncertainty_quality(
    inputs['clinical'], test_targets
)
```

## Usage Examples

### Example 1: Basic Uncertainty Quantification

```python
import torch
import torch.nn as nn
from models.uncertainty_quantification import (
    ComprehensiveUncertainty, UncertaintyConfig
)

# Create a simple model
model = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Configure uncertainty
config = UncertaintyConfig(
    enable_mc_dropout=True,
    enable_heteroscedastic=True,
    num_mc_samples=20
)

# Create uncertainty module
uncertainty_module = ComprehensiveUncertainty(config, input_dim=64)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    # Forward pass
    features = torch.randn(32, 256)
    targets = torch.randn(32, 1)
    
    # Get model predictions
    with torch.no_grad():
        intermediate_features = model[:-1](features)
    
    # Uncertainty quantification
    uncertainty_output = uncertainty_module(intermediate_features, training=True)
    predictions = uncertainty_output['predictions']
    
    # Calculate loss
    loss = uncertainty_module.loss(uncertainty_output, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Monte Carlo prediction at inference
model.eval()
mc_output = monte_carlo_prediction(model, features, num_samples=50)
print(f"Prediction: {mc_output['mean'].item():.2f} ± {mc_output['std'].item():.2f}")
```

### Example 2: Comprehensive XAI Analysis

```python
from models.xai_module import ComprehensiveXAI, XAIConfig
import matplotlib.pyplot as plt

# Configure XAI
xai_config = XAIConfig(
    enable_integrated_gradients=True,
    enable_lime=True,
    enable_clinical_rules=True,
    enable_nl_explanations=True,
    enable_visualization=True
)

# Background data for explanations
background_data = torch.randn(100, 256)
feature_names = [f'biomarker_{i}' for i in range(256)]

# Create XAI module
xai = ComprehensiveXAI(model, xai_config, background_data, feature_names)

# Analyze specific prediction
test_input = torch.randn(1, 256)
explanations = xai.explain(test_input)

# Generate comprehensive report
report = xai.generate_report(explanations)
print(report)

# Visualize explanations
xai.visualize_explanations(explanations)

# Compare feature importance across methods
if 'shap' in explanations and 'integrated_gradients' in explanations:
    shap_importance = torch.mean(torch.abs(explanations['shap']['shap_values']), dim=0)
    ig_importance = torch.abs(explanations['integrated_gradients']['attributions'])
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(10), shap_importance[:10].cpu().numpy())
    plt.title('SHAP Importance (Top 10)')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), ig_importance[:10].cpu().numpy())
    plt.title('Integrated Gradients (Top 10)')
    
    plt.tight_layout()
    plt.show()
```

### Example 3: Clinical Decision Support

```python
from models.integrated_uncertainty_xai import IntegratedHAMNet, IntegrationConfig

# Create integrated system for clinical use
integration_config = IntegrationConfig(
    enable_uncertainty_attention=True,
    enable_confidence_weighting=True,
    enable_counterfactuals=True,
    enable_nl_explanations=True
)

integrated_model = IntegratedHAMNet(base_model, integration_config)

# Patient data analysis
patient_data = {
    'clinical': torch.randn(1, 100),
    'imaging': torch.randn(1, 512),
    'genetic': torch.randn(1, 1000),
    'lifestyle': torch.randn(1, 50)
}

# Get prediction with uncertainty
output = integrated_model(patient_data)
predicted_age = output['predictions'].item()
confidence = output['confidence'].item()
uncertainty = output['uncertainty'].item()

print(f"Predicted Biological Age: {predicted_age:.1f} years")
print(f"Confidence: {confidence:.2f}")
print(f"Uncertainty: {uncertainty:.2f}")

# Generate clinical explanation
explanations = integrated_model.explain(patient_data)

# Clinical rules
print("\nKey Clinical Factors:")
for rule in explanations['clinical_rules'][:3]:
    print(f"• {rule['feature']}: {rule['rule_type']} (confidence: {rule['confidence']:.2f})")

# Natural language explanation
print(f"\n{explanations['nl_explanation']}")

# Counterfactual analysis for intervention
target_age = predicted_age - 5.0  # 5 years younger
counterfactual = explanations['counterfactual']
print(f"\nIntervention Recommendations:")
print(f"To achieve biological age of {target_age:.1f} years:")
for change in counterfactual['counterfactual_changes'][:3]:
    if abs(change['change']) > 0.1:  # Significant changes only
        print(f"  • Modify {change['feature']}: {change['current_value']:.2f} → {change['new_value']:.2f}")
```

## API Reference

### UncertaintyQuantification Classes

#### `ComprehensiveUncertainty`
Main uncertainty quantification module.

```python
class ComprehensiveUncertainty(nn.Module):
    def __init__(self, config: UncertaintyConfig, input_dim: int)
    def forward(self, x: torch.Tensor, training: bool = False) -> Dict[str, torch.Tensor]
    def loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor
```

#### `BayesianLinear`
Bayesian linear layer with Monte Carlo dropout.

```python
class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout_rate: float = 0.1)
    def forward(self, x: torch.Tensor) -> torch.Tensor
    def kl_divergence(self) -> torch.Tensor
```

#### `EvidentialOutput`
Evidential deep learning output layer.

```python
class EvidentialOutput(nn.Module):
    def __init__(self, input_dim: int, coefficient: float = 1.0)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]
    def loss(self, evidential_params: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor
    def uncertainty(self, evidential_params: Dict[str, torch.Tensor]) -> torch.Tensor
```

### XAI Classes

#### `ComprehensiveXAI`
Comprehensive XAI module combining all methods.

```python
class ComprehensiveXAI:
    def __init__(self, model: nn.Module, config: XAIConfig, background_data: torch.Tensor, feature_names: List[str])
    def explain(self, input_data: torch.Tensor, target_index: int = 0) -> Dict[str, Any]
    def visualize_explanations(self, explanations: Dict[str, Any])
    def generate_report(self, explanations: Dict[str, Any]) -> str
```

#### `IntegratedGradients`
Integrated gradients implementation.

```python
class IntegratedGradients:
    def __init__(self, model: nn.Module, config: XAIConfig)
    def explain(self, input_data: torch.Tensor, target_index: int = 0, baseline: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]
    def visualize(self, attributions: torch.Tensor, feature_names: List[str] = None)
```

#### `ClinicalInterpretability`
Clinical interpretability tools.

```python
class ClinicalInterpretability:
    def __init__(self, model: nn.Module, config: XAIConfig)
    def extract_clinical_rules(self, input_data: torch.Tensor, predictions: torch.Tensor, feature_names: List[str]) -> List[Dict[str, Any]]
    def generate_nl_explanation(self, input_data: torch.Tensor, predictions: torch.Tensor, feature_names: List[str]) -> str
    def generate_counterfactual(self, input_data: torch.Tensor, predictions: torch.Tensor, target_age: float, feature_names: List[str]) -> Dict[str, Any]
```

### Integration Classes

#### `IntegratedHAMNet`
Integrated HAMNet with uncertainty and XAI.

```python
class IntegratedHAMNet(nn.Module):
    def __init__(self, base_model: nn.Module, config: IntegrationConfig)
    def forward(self, inputs: Dict[str, torch.Tensor], masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]
    def explain(self, inputs: Dict[str, torch.Tensor], target_index: int = 0) -> Dict[str, Any]
    def visualize_explanations(self, explanations: Dict[str, Any])
    def evaluate_uncertainty_quality(self, test_data: torch.Tensor, test_targets: torch.Tensor) -> Dict[str, float]
```

## Best Practices

### Uncertainty Quantification

1. **Choose appropriate methods based on data size**:
   - Small datasets: Use Bayesian methods with strong regularization
   - Medium datasets: Use ensemble methods
   - Large datasets: Use heteroscedastic or evidential methods

2. **Calibrate uncertainty estimates**:
   - Use temperature scaling for better calibration
   - Monitor calibration error during training
   - Apply isotonic regression if needed

3. **Monitor uncertainty quality**:
   - Track negative log likelihood
   - Monitor calibration error
   - Ensure uncertainty correlates with error

### XAI Implementation

1. **Select appropriate XAI methods**:
   - Global explanations: SHAP summary plots
   - Local explanations: LIME or Integrated Gradients
   - Clinical interpretability: Rule extraction

2. **Validate explanations**:
   - Compare multiple XAI methods
   - Check for consistency
   - Validate with domain experts

3. **Consider computational cost**:
   - SHAP: Computationally expensive
   - Integrated Gradients: Moderate cost
   - Attention-based: Low cost

### Integration Strategies

1. **Uncertainty-aware attention**:
   - Use uncertainty to modulate attention
   - Implement confidence weighting
   - Enable uncertainty propagation

2. **Clinical deployment**:
   - Provide confidence thresholds
   - Generate actionable insights
   - Support clinical decision-making

## Performance Optimization

### Computational Efficiency

1. **Batch processing**:
   ```python
   # Process multiple samples efficiently
   batch_size = 32
   for i in range(0, len(data), batch_size):
       batch = data[i:i+batch_size]
       explanations = xai_module.explain(batch)
   ```

2. **GPU acceleration**:
   ```python
   # Move models to GPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = model.to(device)
   uncertainty_module = uncertainty_module.to(device)
   ```

3. **Caching mechanisms**:
   ```python
   # Cache background computations
   from functools import lru_cache

   @lru_cache(maxsize=100)
   def cached_explanation(input_hash):
       return xai_module.explain(input_data)
   ```

### Memory Optimization

1. **Gradient checkpointing**:
   ```python
   # Enable gradient checkpointing for large models
   from torch.utils.checkpoint import checkpoint

   def forward(self, x):
       return checkpoint(self._forward, x)
   ```

2. **Mixed precision training**:
   ```python
   # Use mixed precision for faster training
   scaler = torch.cuda.amp.GradScaler()

   with torch.cuda.amp.autocast():
       output = model(input_data)
   ```

3. **Memory-efficient attention**:
   ```python
   # Use memory-efficient attention for long sequences
   from transformers import BertModel

   model = BertModel.from_pretrained('bert-base-uncased')
   ```

## Troubleshooting

### Common Issues

1. **Uncertainty estimates too high/low**:
   - Check dropout rates and Monte Carlo samples
   - Verify loss function weights
   - Monitor uncertainty calibration

2. **XAI explanations inconsistent**:
   - Compare multiple XAI methods
   - Check input data preprocessing
   - Verify model architecture

3. **Memory issues during training**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

4. **Slow inference**:
   - Optimize model architecture
   - Use model quantization
   - Implement batch processing

### Debugging Tips

1. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Visualize intermediate outputs**:
   ```python
   # Add hooks to inspect intermediate layers
   activations = {}
   def hook_fn(name):
       def hook(module, input, output):
           activations[name] = output.detach()
       return hook
   ```

3. **Validate with synthetic data**:
   ```python
   # Create test data with known properties
   synthetic_data = torch.randn(100, 256)
   synthetic_targets = synthetic_data.sum(dim=1, keepdim=True)
   ```

### Performance Monitoring

1. **Track key metrics**:
   ```python
   metrics = {
       'nll': [],
       'calibration_error': [],
       'sharpness': []
   }

   for epoch in range(epochs):
       # ... training code ...
       metrics['nll'].append(current_nll)
       metrics['calibration_error'].append(current_cal_error)
   ```

2. **Visualize training progress**:
   ```python
   import matplotlib.pyplot as plt

   plt.figure(figsize=(12, 4))
   plt.subplot(1, 3, 1)
   plt.plot(metrics['nll'])
   plt.title('Negative Log Likelihood')

   plt.subplot(1, 3, 2)
   plt.plot(metrics['calibration_error'])
   plt.title('Calibration Error')

   plt.subplot(1, 3, 3)
   plt.plot(metrics['sharpness'])
   plt.title('Sharpness')
   plt.show()
   ```

## Conclusion

This comprehensive uncertainty quantification and XAI system provides robust tools for interpretable biological age prediction. By combining multiple uncertainty estimation methods with various explainability techniques, the system offers clinicians and researchers reliable and interpretable predictions.

The modular design allows for easy customization and extension, while the integrated approach ensures consistency between uncertainty estimates and explanations. With proper configuration and optimization, the system can handle large-scale biological data while providing actionable insights for clinical decision-making.

For further questions or support, please refer to the project repository or contact the development team.