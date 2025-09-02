"""
Comprehensive Usage Example for HAMNet Uncertainty Quantification and XAI

This script demonstrates the complete workflow for uncertainty quantification
and explainable AI analysis using the HAMNet biological age prediction system.

Author: Claude AI Assistant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import HAMNet components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.uncertainty_quantification import (
    ComprehensiveUncertainty, UncertaintyConfig, UncertaintyMetrics,
    BayesianLinear, EvidentialOutput, DeepEnsemble
)
from models.xai_module import (
    ComprehensiveXAI, XAIConfig, ClinicalInterpretability,
    IntegratedGradients, LIMEExplainer
)
from models.integrated_uncertainty_xai import (
    IntegratedHAMNet, IntegrationConfig, UncertaintyAwareAttention,
    ExplainableMultimodalFusion, ConfidenceWeightedPredictor
)
from models.hamnet import HAMNet, HAMNetConfig


class ExampleBiologicalAgeModel(nn.Module):
    """Example biological age prediction model for demonstration"""
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Biological age prediction layers
        self.age_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Uncertainty estimation layers
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Mean and log variance
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation"""
        # Extract features
        features = self.feature_extractor(x)
        
        # Predict biological age
        age_prediction = self.age_predictor(features)
        
        # Estimate uncertainty
        uncertainty_params = self.uncertainty_estimator(features)
        mean = uncertainty_params[:, 0:1]
        log_var = uncertainty_params[:, 1:2]
        
        return {
            'predictions': age_prediction,
            'features': features,
            'uncertainty_mean': mean,
            'uncertainty_log_var': log_var,
            'uncertainty': torch.exp(log_var)
        }


def generate_synthetic_data(num_samples: int = 1000, input_dim: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic biological age data for demonstration"""
    
    # Generate random biomarker data
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create synthetic biomarkers with some structure
    biomarkers = torch.randn(num_samples, input_dim)
    
    # Add some age-related patterns
    age_factor = torch.linspace(20, 80, num_samples).unsqueeze(1)
    
    # Age-related biomarkers (some increase, some decrease with age)
    age_biomarkers = input_dim // 4
    biomarkers[:, :age_biomarkers] += 0.01 * age_factor * torch.randn(age_biomarkers)
    biomarkers[:, age_biomarkers:2*age_biomarkers] -= 0.005 * age_factor * torch.randn(age_biomarkers)
    
    # Add some noise
    biomarkers += 0.1 * torch.randn(num_samples, input_dim)
    
    # Generate target biological ages with some relationship to biomarkers
    biological_age = 30 + 0.5 * age_factor.squeeze() + 0.1 * biomarkers.sum(dim=1)
    biological_age += 5 * torch.randn(num_samples)  # Add noise
    
    # Normalize to reasonable age range
    biological_age = torch.clamp(biological_age, 20, 90)
    
    return biomarkers, biological_age.unsqueeze(1)


def example_uncertainty_quantification():
    """Example 1: Basic uncertainty quantification"""
    print("="*80)
    print("EXAMPLE 1: Uncertainty Quantification")
    print("="*80)
    
    # Generate synthetic data
    print("Generating synthetic data...")
    train_data, train_targets = generate_synthetic_data(800, 256)
    test_data, test_targets = generate_synthetic_data(200, 256)
    
    # Configure uncertainty quantification
    uncertainty_config = UncertaintyConfig(
        enable_mc_dropout=True,
        enable_heteroscedastic=True,
        enable_evidential=True,
        enable_deep_ensemble=True,
        num_mc_samples=30,
        mc_dropout_rate=0.1,
        num_ensemble_models=3
    )
    
    print(f"Uncertainty Configuration:")
    print(f"  - Monte Carlo Dropout: {uncertainty_config.enable_mc_dropout}")
    print(f"  - Heteroscedastic Loss: {uncertainty_config.enable_heteroscedastic}")
    print(f"  - Evidential Deep Learning: {uncertainty_config.enable_evidential}")
    print(f"  - Deep Ensemble: {uncertainty_config.enable_deep_ensemble}")
    
    # Create model and uncertainty module
    model = ExampleBiologicalAgeModel(input_dim=256)
    uncertainty_module = ComprehensiveUncertainty(uncertainty_config, input_dim=128)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50
    
    print(f"\nTraining model with uncertainty quantification...")
    print(f"Training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        uncertainty_module.train()
        
        # Forward pass
        batch_output = model(train_data)
        features = batch_output['features']
        
        # Uncertainty quantification
        uncertainty_output = uncertainty_module(features, training=True)
        predictions = uncertainty_output['predictions']
        
        # Calculate loss
        loss = uncertainty_module.loss(uncertainty_output, train_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")
    
    print("\nTraining completed!")
    
    # Evaluation
    print("\nEvaluating uncertainty quality...")
    model.eval()
    uncertainty_module.eval()
    
    with torch.no_grad():
        test_output = model(test_data)
        test_features = test_output['features']
        test_uncertainty = uncertainty_module(test_features, training=False)
        test_predictions = test_uncertainty['predictions']
        test_uncertainty_est = test_uncertainty['uncertainty']
    
    # Calculate metrics
    metrics = UncertaintyMetrics.evaluate_uncertainty_quality(
        test_predictions, test_targets, test_uncertainty_est
    )
    
    print("\nUncertainty Quality Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:20s}: {value:.4f}")
    
    # Monte Carlo prediction example
    print("\nMonte Carlo Prediction Example:")
    sample = test_data[0:1]
    mc_predictions = []
    
    model.eval()
    for _ in range(100):
        # Enable dropout for MC sampling
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        
        with torch.no_grad():
            pred = model(sample)['predictions']
            mc_predictions.append(pred)
    
    mc_predictions = torch.stack(mc_predictions)
    mc_mean = mc_predictions.mean()
    mc_std = mc_predictions.std()
    
    print(f"  MC Prediction: {mc_mean.item():.1f} ± {mc_std.item():.1f} years")
    print(f"  True Age: {test_targets[0].item():.1f} years")
    print(f"  Prediction Error: {abs(mc_mean.item() - test_targets[0].item()):.1f} years")
    
    return model, uncertainty_module, test_data, test_targets


def example_xai_analysis(model, test_data, test_targets):
    """Example 2: Comprehensive XAI analysis"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Explainable AI Analysis")
    print("="*80)
    
    # Configure XAI methods
    xai_config = XAIConfig(
        enable_integrated_gradients=True,
        enable_lime=True,
        enable_attention_viz=True,
        enable_lrp=True,
        enable_clinical_rules=True,
        enable_nl_explanations=True,
        enable_counterfactuals=True,
        enable_visualization=True
    )
    
    print(f"XAI Configuration:")
    print(f"  - Integrated Gradients: {xai_config.enable_integrated_gradients}")
    print(f"  - LIME: {xai_config.enable_lime}")
    print(f"  - Clinical Rules: {xai_config.enable_clinical_rules}")
    print(f"  - NL Explanations: {xai_config.enable_nl_explanations}")
    
    # Create background data for SHAP
    background_data = test_data[:50]  # Use subset for background
    
    # Feature names for interpretability
    feature_names = []
    biomarker_types = ['clinical', 'blood', 'genetic', 'lifestyle', 'imaging']
    for i in range(256):
        biomarker_type = biomarker_types[i % len(biomarker_types)]
        feature_names.append(f'{biomarker_type}_{i//len(biomarker_types)+1:02d}')
    
    # Create XAI module
    xai_module = ComprehensiveXAI(model, xai_config, background_data, feature_names)
    
    print("\nGenerating explanations for sample prediction...")
    
    # Select a sample for detailed analysis
    sample_idx = 0
    sample_input = test_data[sample_idx:sample_idx+1]
    sample_target = test_targets[sample_idx]
    
    # Get model prediction
    with torch.no_grad():
        sample_output = model(sample_input)
        sample_prediction = sample_output['predictions'].item()
    
    print(f"Sample Prediction: {sample_prediction:.1f} years")
    print(f"True Age: {sample_target.item():.1f} years")
    print(f"Prediction Error: {abs(sample_prediction - sample_target.item()):.1f} years")
    
    # Generate comprehensive explanations
    explanations = xai_module.explain(sample_input)
    
    # Integrated Gradients Analysis
    if 'integrated_gradients' in explanations:
        print("\nIntegrated Gradients Analysis:")
        ig_attributions = explanations['integrated_gradients']['attributions']
        top_features = torch.argsort(torch.abs(ig_attributions.squeeze()), descending=True)[:10]
        
        print("Top 10 most important features:")
        for i, feature_idx in enumerate(top_features):
            feature_name = feature_names[feature_idx]
            attribution = ig_attributions[0, feature_idx].item()
            print(f"  {i+1:2d}. {feature_name:20s}: {attribution:+.4f}")
    
    # LIME Analysis
    if 'lime' in explanations:
        print("\nLIME Analysis:")
        lime_explanation = explanations['lime']
        lime_importance = lime_explanation['feature_importance']
        top_lime_features = np.argsort(np.abs(lime_importance))[-10:][::-1]
        
        print("Top 10 features according to LIME:")
        for i, feature_idx in enumerate(top_lime_features):
            feature_name = feature_names[feature_idx]
            importance = lime_importance[feature_idx]
            print(f"  {i+1:2d}. {feature_name:20s}: {importance:+.4f}")
    
    # Clinical Rules
    if 'clinical_rules' in explanations:
        print("\nExtracted Clinical Rules:")
        rules = explanations['clinical_rules']
        
        for i, rule in enumerate(rules[:5]):
            print(f"  Rule {i+1}:")
            print(f"    Feature: {rule['feature']}")
            print(f"    Effect: {rule['rule_type']}")
            print(f"    Condition: {rule['condition']}")
            print(f"    Confidence: {rule['confidence']:.3f}")
            print()
    
    # Natural Language Explanation
    if 'nl_explanation' in explanations:
        print("\nNatural Language Explanation:")
        nl_explanation = explanations['nl_explanation']
        print(nl_explanation)
    
    # Counterfactual Analysis
    if 'counterfactual' in explanations:
        print("\nCounterfactual Analysis:")
        counterfactual = explanations['counterfactual']
        
        print(f"Current biological age: {counterfactual['current_age']:.1f} years")
        print(f"Target biological age: {counterfactual['target_age']:.1f} years")
        print(f"Required change: {counterfactual['required_change']:+.1f} years")
        print("\nRecommended interventions:")
        
        for change in counterfactual['counterfactual_changes'][:5]:
            if abs(change['change']) > 0.01:  # Show significant changes
                print(f"  • {change['feature']}: {change['current_value']:.2f} → {change['new_value']:.2f}")
                print(f"    Change: {change['change']:+.3f}")
        
        print(f"\nFeasibility score: {counterfactual['feasibility']:.3f}")
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("COMPREHENSIVE XAI REPORT")
    print("="*60)
    
    report = xai_module.generate_report(explanations)
    print(report)
    
    # Visualize explanations
    if xai_config.enable_visualization:
        print("\nGenerating visualizations...")
        xai_module.visualize_explanations(explanations)
    
    return xai_module, explanations


def example_integrated_system():
    """Example 3: Integrated uncertainty-XAI system"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Integrated Uncertainty-XAI System")
    print("="*80)
    
    # Create HAMNet configuration
    hamnet_config = HAMNetConfig(
        model_tier="standard",
        embedding_dim=128,
        hidden_dim=256,
        clinical_dim=64,
        imaging_dim=128,
        genetic_dim=256,
        lifestyle_dim=32,
        enable_uncertainty=True,
        num_monte_carlo=20
    )
    
    # Create base HAMNet model
    base_model = HAMNet(hamnet_config)
    
    # Configure integrated system
    integration_config = IntegrationConfig(
        enable_uncertainty_attention=True,
        enable_confidence_weighting=True,
        enable_uncertainty_propagation=True,
        enable_interactive_viz=True,
        plot_style='seaborn',
        color_palette='viridis'
    )
    
    print(f"Integration Configuration:")
    print(f"  - Uncertainty-aware Attention: {integration_config.enable_uncertainty_attention}")
    print(f"  - Confidence Weighting: {integration_config.enable_confidence_weighting}")
    print(f"  - Uncertainty Propagation: {integration_config.enable_uncertainty_propagation}")
    
    # Create integrated model
    integrated_model = IntegratedHAMNet(base_model, integration_config)
    
    # Generate multimodal test data
    print("\nGenerating multimodal test data...")
    batch_size = 16
    
    test_inputs = {
        'clinical': torch.randn(batch_size, 64),
        'imaging': torch.randn(batch_size, 128),
        'genetic': torch.randn(batch_size, 256),
        'lifestyle': torch.randn(batch_size, 32)
    }
    
    test_targets = torch.randn(batch_size, 1) * 20 + 50  # Ages around 50 ± 20
    
    print(f"Test data shapes:")
    for modality, data in test_inputs.items():
        print(f"  {modality:10s}: {data.shape}")
    
    # Forward pass with integrated system
    print("\nRunning forward pass with integrated system...")
    output = integrated_model(test_inputs)
    
    print("\nIntegrated System Output:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:25s}: {value.shape} (mean: {value.mean().item():.3f})")
        else:
            print(f"  {key:25s}: {type(value)}")
    
    # Extract key metrics
    predictions = output['predictions']
    confidence = output['confidence']
    uncertainty = output['uncertainty']
    adjusted_confidence = output['adjusted_confidence']
    
    print(f"\nPrediction Statistics:")
    print(f"  Mean Prediction: {predictions.mean().item():.1f} years")
    print(f"  Mean Confidence: {confidence.mean().item():.3f}")
    print(f"  Mean Uncertainty: {uncertainty.mean().item():.3f}")
    print(f"  Mean Adjusted Confidence: {adjusted_confidence.mean().item():.3f}")
    
    # Generate explanations
    print("\nGenerating comprehensive explanations...")
    explanations = integrated_model.explain(test_inputs)
    
    print("\nExplanation Components:")
    for key in explanations.keys():
        print(f"  - {key}")
    
    # Analyze uncertainty-explanation relationship
    print("\nUncertainty-Explanation Analysis:")
    
    if 'uncertainty_explanation' in explanations:
        unc_exp = explanations['uncertainty_explanation']
        print(f"  Uncertainty range: {unc_exp['total_uncertainty'].min().item():.3f} - {unc_exp['total_uncertainty'].max().item():.3f}")
        print(f"  Confidence range: {unc_exp['confidence_score'].min().item():.3f} - {unc_exp['confidence_score'].max().item():.3f}")
    
    if 'fusion_explanation' in explanations:
        fusion_exp = explanations['fusion_explanation']
        if 'modality_importance' in fusion_exp:
            mod_importance = fusion_exp['modality_importance']
            print(f"  Modality Importance:")
            for i, modality in enumerate(['clinical', 'imaging', 'genetic', 'lifestyle']):
                if i < len(mod_importance):
                    print(f"    {modality:10s}: {mod_importance[i].item():.3f}")
    
    # Evaluate uncertainty quality
    print("\nEvaluating uncertainty quality...")
    metrics = integrated_model.evaluate_uncertainty_quality(
        test_inputs['clinical'], test_targets
    )
    
    print("Uncertainty Quality Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:20s}: {value:.4f}")
    
    # Create interactive visualization
    if integration_config.enable_interactive_viz:
        print("\nGenerating interactive visualizations...")
        integrated_model.visualize_explanations(explanations)
    
    return integrated_model, explanations


def example_clinical_decision_support():
    """Example 4: Clinical decision support system"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Clinical Decision Support System")
    print("="*80)
    
    # Create a more clinically realistic model
    class ClinicalModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Clinical biomarkers (e.g., blood pressure, cholesterol, etc.)
            self.clinical_encoder = nn.Sequential(
                nn.Linear(50, 100),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(100, 50)
            )
            
            # Prediction head
            self.predictor = nn.Sequential(
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 1)
            )
            
            # Uncertainty head
            self.uncertainty_head = nn.Sequential(
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 2)
            )
            
        def forward(self, x):
            features = self.clinical_encoder(x)
            prediction = self.predictor(features)
            uncertainty_params = self.uncertainty_head(features)
            
            return {
                'prediction': prediction,
                'features': features,
                'uncertainty_mean': uncertainty_params[:, 0:1],
                'uncertainty_log_var': uncertainty_params[:, 1:2]
            }
    
    # Clinical biomarker names
    clinical_biomarkers = [
        'systolic_bp', 'diastolic_bp', 'total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol',
        'triglycerides', 'fasting_glucose', 'hba1c', 'creatinine', 'egfr',
        'crp', 'albumin', 'hemoglobin', 'white_blood_cells', 'red_blood_cells',
        'platelets', 'ast', 'alt', 'alkaline_phosphatase', 'bilirubin',
        'sodium', 'potassium', 'chloride', 'bicarbonate', 'calcium',
        'phosphorus', 'magnesium', 'urate', 'vitamin_d', 'vitamin_b12',
        'folate', 'ferritin', 'tsh', 'ft4', 'testosterone',
        'estrogen', 'cortisol', 'igf1', 'dhea_s', 'homocysteine',
        'fibrinogen', 'd_dimer', 'nt_pro_bnp', 'troponin', 'ck_mb',
        'ldh', 'amylase', 'lipase', 'uric_acid', 'microalbumin'
    ]
    
    # Create model
    model = ClinicalModel()
    
    # Generate realistic patient data
    print("Generating patient data...")
    num_patients = 5
    
    # Base patient data (normal ranges)
    patient_data = torch.zeros(num_patients, 50)
    
    # Add some realistic variation and abnormalities
    np.random.seed(123)
    torch.manual_seed(123)
    
    for i in range(num_patients):
        # Each patient has different biomarker patterns
        age_factor = 40 + i * 10  # Ages 40, 50, 60, 70, 80
        
        # Age-related changes
        patient_data[i, 0] = 120 + age_factor * 0.5 + torch.randn(1) * 10  # Systolic BP
        patient_data[i, 1] = 80 + age_factor * 0.3 + torch.randn(1) * 8   # Diastolic BP
        patient_data[i, 2] = 180 + age_factor * 0.8 + torch.randn(1) * 30  # Total cholesterol
        
        # Add some abnormalities for older patients
        if i >= 2:  # Patients 60+
            patient_data[i, 6] = 100 + torch.randn(1) * 20  # Elevated glucose
            patient_data[i, 7] = 6.5 + torch.randn(1) * 1.0  # Elevated HbA1c
    
    # Calculate target biological ages
    biological_ages = torch.tensor([[45.], [52.], [61.], [68.], [75.]])
    
    print(f"Generated data for {num_patients} patients")
    print(f"Patient ages: {biological_ages.squeeze().tolist()}")
    
    # Configure clinical XAI system
    xai_config = XAIConfig(
        enable_integrated_gradients=True,
        enable_lime=True,
        enable_clinical_rules=True,
        enable_nl_explanations=True,
        enable_counterfactuals=True,
        lime_samples=500,
        lime_features=10
    )
    
    # Create clinical interpretability module
    clinical_xai = ClinicalInterpretability(model, xai_config)
    
    # Analyze each patient
    for patient_idx in range(num_patients):
        print(f"\n{'='*60}")
        print(f"PATIENT {patient_idx + 1} ANALYSIS")
        print(f"{'='*60}")
        
        patient_input = patient_data[patient_idx:patient_idx+1]
        patient_age = biological_ages[patient_idx].item()
        
        # Get model prediction
        with torch.no_grad():
            patient_output = model(patient_input)
            patient_prediction = patient_output['prediction'].item()
            patient_features = patient_output['features']
        
        print(f"Chronological Age: {patient_age:.1f} years")
        print(f"Predicted Biological Age: {patient_prediction:.1f} years")
        print(f"Age Acceleration: {patient_prediction - patient_age:+.1f} years")
        
        # Extract clinical rules
        clinical_rules = clinical_xai.extract_clinical_rules(
            patient_input, patient_output['prediction'], clinical_biomarkers
        )
        
        print(f"\nKey Clinical Factors:")
        for rule in clinical_rules[:5]:
            print(f"  • {rule['feature']}: {rule['rule_type'].replace('_', ' ').title()}")
            print(f"    Condition: {rule['condition']}")
            print(f"    Confidence: {rule['confidence']:.2f}")
            print()
        
        # Generate natural language explanation
        nl_explanation = clinical_xai.generate_nl_explanation(
            patient_input, patient_output['prediction'], clinical_biomarkers
        )
        print(f"Clinical Summary:")
        print(nl_explanation)
        
        # Counterfactual analysis (what if patient were 5 years younger biologically?)
        target_age = patient_prediction - 5.0
        counterfactual = clinical_xai.generate_counterfactual(
            patient_input, patient_output['prediction'], target_age, clinical_biomarkers
        )
        
        print(f"\nIntervention Recommendations:")
        print(f"To achieve biological age of {target_age:.1f} years:")
        
        feasible_changes = [change for change in counterfactual['counterfactual_changes'] 
                          if abs(change['change']) > 0.1 and change['feasibility'] > 0.5]
        
        if feasible_changes:
            for change in feasible_changes[:3]:
                print(f"  • Modify {change['feature']}: {change['current_value']:.2f} → {change['new_value']:.2f}")
        else:
            print("  No significant feasible interventions identified")
        
        print(f"Overall feasibility: {counterfactual['feasibility']:.2f}")
        
        # Risk assessment
        age_acceleration = patient_prediction - patient_age
        if age_acceleration > 5:
            risk_level = "High"
            recommendations = ["Comprehensive lifestyle intervention", "Medical consultation recommended"]
        elif age_acceleration > 2:
            risk_level = "Moderate"
            recommendations = ["Lifestyle modifications", "Regular monitoring"]
        else:
            risk_level = "Low"
            recommendations = ["Maintain healthy lifestyle", "Annual check-up"]
        
        print(f"\nRisk Assessment: {risk_level} risk")
        print("Recommendations:")
        for rec in recommendations:
            print(f"  • {rec}")


def main():
    """Main function to run all examples"""
    print("HAMNet Uncertainty Quantification and XAI - Comprehensive Examples")
    print("=" * 80)
    
    try:
        # Example 1: Uncertainty Quantification
        model, uncertainty_module, test_data, test_targets = example_uncertainty_quantification()
        
        # Example 2: XAI Analysis
        xai_module, explanations = example_xai_analysis(model, test_data, test_targets)
        
        # Example 3: Integrated System
        integrated_model, integrated_explanations = example_integrated_system()
        
        # Example 4: Clinical Decision Support
        example_clinical_decision_support()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSummary:")
        print("✓ Uncertainty quantification with multiple methods")
        print("✓ Comprehensive XAI analysis with various techniques")
        print("✓ Integrated uncertainty-XAI system")
        print("✓ Clinical decision support system")
        print("\nThe system provides robust uncertainty estimates and interpretable")
        print("explanations suitable for clinical applications.")
        
    except Exception as e:
        print(f"\nError running examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()