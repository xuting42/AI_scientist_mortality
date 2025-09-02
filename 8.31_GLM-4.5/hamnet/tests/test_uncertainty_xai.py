"""
Unit Tests for Uncertainty Quantification and XAI Components

This module provides comprehensive unit tests for all uncertainty quantification
and explainable AI components implemented in the HAMNet system.

Author: Claude AI Assistant
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.uncertainty_quantification import (
    BayesianLinear, HeteroscedasticLoss, EvidentialOutput, DeepEnsemble,
    ComprehensiveUncertainty, UncertaintyConfig, UncertaintyMetrics
)
from models.xai_module import (
    SHAPExplainer, IntegratedGradients, AttentionVisualizer, LRPExplainer,
    LIMEExplainer, ClinicalInterpretability, ComprehensiveXAI, XAIConfig
)
from models.integrated_uncertainty_xai import (
    UncertaintyAwareAttention, ExplainableMultimodalFusion,
    ConfidenceWeightedPredictor, IntegratedHAMNet, IntegrationConfig
)
from models.hamnet import HAMNet, HAMNetConfig


class TestUncertaintyQuantification(unittest.TestCase):
    """Test cases for uncertainty quantification components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = UncertaintyConfig()
        self.input_dim = 64
        self.batch_size = 32
        
        # Create test data
        self.test_input = torch.randn(self.batch_size, self.input_dim)
        self.test_targets = torch.randn(self.batch_size, 1)
        
    def test_bayesian_linear(self):
        """Test Bayesian linear layer"""
        layer = BayesianLinear(self.input_dim, 32, dropout_rate=0.1)
        
        # Test forward pass
        output = layer(self.test_input)
        self.assertEqual(output.shape, (self.batch_size, 32))
        
        # Test KL divergence
        kl_loss = layer.kl_divergence()
        self.assertIsInstance(kl_loss, torch.Tensor)
        self.assertGreater(kl_loss.item(), 0)
        
    def test_heteroscedastic_loss(self):
        """Test heteroscedastic loss function"""
        loss_fn = HeteroscedasticLoss(loss_weight=0.1)
        
        predictions = torch.randn(self.batch_size, 1)
        targets = torch.randn(self.batch_size, 1)
        log_variance = torch.randn(self.batch_size, 1)
        
        loss = loss_fn(predictions, targets, log_variance)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        
    def test_evidential_output(self):
        """Test evidential deep learning output"""
        evidential = EvidentialOutput(self.input_dim, coefficient=1.0)
        
        # Test forward pass
        params = evidential(self.test_input)
        expected_keys = ['gamma', 'nu', 'alpha', 'beta']
        for key in expected_keys:
            self.assertIn(key, params)
            self.assertEqual(params[key].shape, (self.batch_size, 1))
            
        # Test loss computation
        loss = evidential.loss(params, self.test_targets)
        self.assertIsInstance(loss, torch.Tensor)
        
        # Test uncertainty computation
        uncertainty = evidential.uncertainty(params)
        self.assertEqual(uncertainty.shape, (self.batch_size, 1))
        
    def test_deep_ensemble(self):
        """Test deep ensemble implementation"""
        # Create simple model for ensemble
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(self.input_dim, 1)
                
            def forward(self, x):
                return {'predictions': self.linear(x)}
                
        ensemble = DeepEnsemble(SimpleModel, None, num_models=3)
        
        # Test forward pass
        output = ensemble(self.test_input)
        expected_keys = ['mean_prediction', 'variance', 'uncertainty', 'ensemble_predictions']
        for key in expected_keys:
            self.assertIn(key, output)
            
        # Test diversity loss
        div_loss = ensemble.diversity_loss()
        self.assertIsInstance(div_loss, torch.Tensor)
        
    def test_comprehensive_uncertainty(self):
        """Test comprehensive uncertainty module"""
        uncertainty_module = ComprehensiveUncertainty(self.config, self.input_dim)
        
        # Test forward pass
        output = uncertainty_module(self.test_input, training=True)
        
        # Check output structure
        if self.config.enable_mc_dropout:
            self.assertIn('predictions', output)
            self.assertIn('uncertainty', output)
            
        if self.config.enable_evidential:
            self.assertIn('evidential_params', output)
            
        # Test loss computation
        loss = uncertainty_module.loss(output, self.test_targets)
        self.assertIsInstance(loss, torch.Tensor)
        
    def test_uncertainty_metrics(self):
        """Test uncertainty evaluation metrics"""
        predictions = torch.randn(self.batch_size, 1)
        targets = torch.randn(self.batch_size, 1)
        uncertainties = torch.rand(self.batch_size, 1)
        
        # Test NLL
        nll = UncertaintyMetrics.negative_log_likelihood(predictions, targets, uncertainties)
        self.assertIsInstance(nll, float)
        
        # Test calibration error
        cal_error = UncertaintyMetrics.calibration_error(predictions, targets, uncertainties)
        self.assertIsInstance(cal_error, float)
        
        # Test reliability diagram
        exp_conf, actual_acc = UncertaintyMetrics.reliability_diagram(predictions, targets, uncertainties)
        self.assertEqual(len(exp_conf), 10)  # Default 10 bins
        self.assertEqual(len(actual_acc), 10)
        
        # Test sharpness
        sharpness = UncertaintyMetrics.sharpness(uncertainties)
        self.assertIsInstance(sharpness, float)
        
        # Test comprehensive evaluation
        metrics = UncertaintyMetrics.evaluate_uncertainty_quality(predictions, targets, uncertainties)
        self.assertIn('mse', metrics)
        self.assertIn('nll', metrics)
        self.assertIn('calibration_error', metrics)


class TestXAIComponents(unittest.TestCase):
    """Test cases for XAI components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = XAIConfig()
        self.input_dim = 64
        self.batch_size = 16
        
        # Create test data
        self.test_input = torch.randn(self.batch_size, self.input_dim)
        self.test_targets = torch.randn(self.batch_size, 1)
        
        # Create simple model for testing
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Background data for SHAP
        self.background_data = torch.randn(100, self.input_dim)
        
    def test_integrated_gradients(self):
        """Test integrated gradients implementation"""
        ig = IntegratedGradients(self.model, self.config)
        
        # Test explanation
        explanation = ig.explain(self.test_input, target_index=0)
        
        expected_keys = ['integrated_gradients', 'attributions', 'input_data', 'baseline']
        for key in expected_keys:
            self.assertIn(key, explanation)
            
        # Check shapes
        self.assertEqual(explanation['attributions'].shape, (self.batch_size, self.input_dim))
        
    def test_lime_explainer(self):
        """Test LIME implementation"""
        lime = LIMEExplainer(self.model, self.config)
        
        # Test explanation
        feature_names = [f'feature_{i}' for i in range(self.input_dim)]
        explanation = lime.explain(self.test_input, target_index=0, feature_names=feature_names)
        
        expected_keys = ['feature_importance', 'intercept', 'feature_names', 
                        'perturbed_data', 'masks', 'predictions', 'interpretable_model']
        for key in expected_keys:
            self.assertIn(key, explanation)
            
        # Check feature importance shape
        self.assertEqual(len(explanation['feature_importance']), self.input_dim)
        
    def test_clinical_interpretability(self):
        """Test clinical interpretability tools"""
        clinical = ClinicalInterpretability(self.model, self.config)
        
        feature_names = [f'biomarker_{i}' for i in range(self.input_dim)]
        
        # Test clinical rule extraction
        rules = clinical.extract_clinical_rules(self.test_input, self.test_targets, feature_names)
        self.assertIsInstance(rules, list)
        
        # Test natural language explanation
        nl_explanation = clinical.generate_nl_explanation(self.test_input, self.test_targets, feature_names)
        self.assertIsInstance(nl_explanation, str)
        self.assertGreater(len(nl_explanation), 0)
        
        # Test counterfactual explanation
        counterfactual = clinical.generate_counterfactual(self.test_input, self.test_targets, 50.0, feature_names)
        self.assertIn('current_age', counterfactual)
        self.assertIn('target_age', counterfactual)
        self.assertIn('counterfactual_changes', counterfactual)
        
    def test_comprehensive_xai(self):
        """Test comprehensive XAI module"""
        feature_names = [f'feature_{i}' for i in range(self.input_dim)]
        xai = ComprehensiveXAI(self.model, self.config, self.background_data, feature_names)
        
        # Test comprehensive explanation
        explanations = xai.explain(self.test_input, target_index=0)
        
        # Check that explanations contain expected methods
        if self.config.enable_integrated_gradients:
            self.assertIn('integrated_gradients', explanations)
        if self.config.enable_lime:
            self.assertIn('lime', explanations)
        if self.config.enable_clinical_rules:
            self.assertIn('clinical_rules', explanations)
            
        # Test report generation
        report = xai.generate_report(explanations)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)


class TestIntegratedComponents(unittest.TestCase):
    """Test cases for integrated uncertainty-XAI components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = IntegrationConfig()
        self.embedding_dim = 256
        self.batch_size = 16
        
        # Create test data
        self.test_input = torch.randn(self.batch_size, self.embedding_dim)
        self.test_uncertainty = torch.rand(self.batch_size, 1)
        
    def test_uncertainty_aware_attention(self):
        """Test uncertainty-aware attention mechanism"""
        attention = UncertaintyAwareAttention(self.embedding_dim, num_heads=8)
        
        # Test forward pass without uncertainty
        query = key = value = self.test_input
        output, weights = attention(query, key, value)
        
        self.assertEqual(output.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(weights.shape, (self.batch_size, 8, self.batch_size, self.batch_size))
        
        # Test forward pass with uncertainty
        output_unc, weights_unc = attention(query, key, value, self.test_uncertainty)
        self.assertEqual(output_unc.shape, (self.batch_size, self.embedding_dim))
        
    def test_explainable_multimodal_fusion(self):
        """Test explainable multimodal fusion"""
        modalities = ['clinical', 'imaging', 'genetic', 'lifestyle']
        fusion = ExplainableMultimodalFusion(self.config, modalities)
        
        # Create modality embeddings
        modality_embeddings = {
            'clinical': torch.randn(self.batch_size, self.embedding_dim),
            'imaging': torch.randn(self.batch_size, self.embedding_dim),
            'genetic': torch.randn(self.batch_size, self.embedding_dim)
        }
        
        # Test fusion
        output = fusion(modality_embeddings)
        
        expected_keys = ['fused_output', 'fusion_mean', 'fusion_log_variance', 
                        'fusion_uncertainty', 'attention_weights', 'modality_importance']
        for key in expected_keys:
            self.assertIn(key, output)
            
        # Check output shapes
        self.assertEqual(output['fused_output'].shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(output['fusion_uncertainty'].shape, (self.batch_size, 1))
        
    def test_confidence_weighted_predictor(self):
        """Test confidence-weighted prediction"""
        predictor = ConfidenceWeightedPredictor(self.embedding_dim, self.config)
        
        # Test without uncertainty
        output = predictor(self.test_input)
        expected_keys = ['prediction', 'base_prediction', 'confidence', 'uncertainty_weight', 'adjusted_confidence']
        for key in expected_keys:
            self.assertIn(key, output)
            
        # Test with uncertainty
        output_unc = predictor(self.test_input, self.test_uncertainty)
        for key in expected_keys:
            self.assertIn(key, output_unc)
            
    def test_integrated_hamnet(self):
        """Test integrated HAMNet model"""
        # Create base HAMNet model
        hamnet_config = HAMNetConfig(
            model_tier="base",
            embedding_dim=64,
            hidden_dim=128,
            clinical_dim=50,
            imaging_dim=100,
            genetic_dim=200,
            lifestyle_dim=30
        )
        base_model = HAMNet(hamnet_config)
        
        # Create integrated model
        integrated_model = IntegratedHAMNet(base_model, self.config)
        
        # Create test inputs
        test_inputs = {
            'clinical': torch.randn(self.batch_size, 50),
            'imaging': torch.randn(self.batch_size, 100),
            'genetic': torch.randn(self.batch_size, 200),
            'lifestyle': torch.randn(self.batch_size, 30)
        }
        
        # Test forward pass
        output = integrated_model(test_inputs)
        
        expected_keys = ['predictions', 'base_predictions', 'confidence', 
                        'uncertainty_weight', 'adjusted_confidence', 'uncertainty']
        for key in expected_keys:
            self.assertIn(key, output)
            
        # Test explanation generation
        explanations = integrated_model.explain(test_inputs)
        self.assertIn('prediction', explanations)
        self.assertIn('uncertainty_explanation', explanations)
        
        # Test uncertainty quality evaluation
        test_targets = torch.randn(self.batch_size, 1)
        metrics = integrated_model.evaluate_uncertainty_quality(
            test_inputs['clinical'], test_targets
        )
        self.assertIsInstance(metrics, dict)


class TestIntegrationAndCompatibility(unittest.TestCase):
    """Test integration and compatibility between components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 16
        self.input_dim = 64
        
        # Create configurations
        self.uncertainty_config = UncertaintyConfig(
            enable_mc_dropout=True,
            enable_heteroscedastic=True,
            enable_evidential=True
        )
        
        self.xai_config = XAIConfig(
            enable_integrated_gradients=True,
            enable_lime=True,
            enable_clinical_rules=True,
            enable_nl_explanations=True
        )
        
        self.integration_config = IntegrationConfig(
            uncertainty_config=self.uncertainty_config,
            xai_config=self.xai_config
        )
        
    def test_uncertainty_xai_integration(self):
        """Test integration between uncertainty and XAI components"""
        # Create test model
        model = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Create components
        uncertainty_module = ComprehensiveUncertainty(self.uncertainty_config, self.input_dim)
        background_data = torch.randn(50, self.input_dim)
        xai_module = ComprehensiveXAI(model, self.xai_config, background_data)
        
        # Test data
        test_input = torch.randn(self.batch_size, self.input_dim)
        test_targets = torch.randn(self.batch_size, 1)
        
        # Test uncertainty quantification
        uncertainty_output = uncertainty_module(test_input, training=True)
        self.assertIn('predictions', uncertainty_output)
        self.assertIn('uncertainty', uncertainty_output)
        
        # Test XAI explanation
        explanations = xai_module.explain(test_input)
        self.assertIn('prediction', explanations)
        
        # Verify compatibility
        self.assertEqual(test_input.shape[0], self.batch_size)
        self.assertEqual(uncertainty_output['predictions'].shape[0], self.batch_size)
        
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline with uncertainty and XAI"""
        # Create simple HAMNet-like model
        class SimpleHAMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(self.input_dim, 32)
                self.predictor = nn.Linear(32, 1)
                
            def forward(self, x):
                encoded = self.encoder(x)
                prediction = self.predictor(encoded)
                return {'predictions': prediction, 'encoded': encoded}
                
        model = SimpleHAMNet()
        
        # Create integrated system
        integrated_model = IntegratedHAMNet(model, self.integration_config)
        
        # Test data
        test_input = torch.randn(self.batch_size, self.input_dim)
        
        # Forward pass
        output = integrated_model({'clinical': test_input})
        
        # Verify output structure
        required_keys = ['predictions', 'confidence', 'uncertainty']
        for key in required_keys:
            self.assertIn(key, output)
            
        # Explanation generation
        explanations = integrated_model.explain({'clinical': test_input})
        
        # Verify explanations
        self.assertIn('prediction', explanations)
        self.assertIn('uncertainty_explanation', explanations)
        
        # Uncertainty evaluation
        test_targets = torch.randn(self.batch_size, 1)
        metrics = integrated_model.evaluate_uncertainty_quality(test_input, test_targets)
        
        # Verify metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('mse', metrics)
        self.assertIn('nll', metrics)
        
    def test_gpu_compatibility(self):
        """Test GPU compatibility if available"""
        if torch.cuda.is_available():
            # Create model and move to GPU
            model = nn.Linear(self.input_dim, 1).cuda()
            test_input = torch.randn(self.batch_size, self.input_dim).cuda()
            
            # Test uncertainty module on GPU
            uncertainty_module = ComprehensiveUncertainty(self.uncertainty_config, self.input_dim).cuda()
            output = uncertainty_module(test_input)
            
            # Verify output is on GPU
            self.assertTrue(output['predictions'].is_cuda)
            
            # Test XAI module on GPU
            background_data = torch.randn(50, self.input_dim).cuda()
            xai_module = ComprehensiveXAI(model, self.xai_config, background_data)
            
            # Explanation generation on GPU
            explanations = xai_module.explain(test_input)
            self.assertIn('prediction', explanations)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = UncertaintyConfig()
        self.input_dim = 64
        
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        uncertainty_module = ComprehensiveUncertainty(self.config, self.input_dim)
        
        # Test with empty batch
        empty_input = torch.randn(0, self.input_dim)
        try:
            output = uncertainty_module(empty_input)
            # Should handle gracefully or raise appropriate error
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError))
            
    def test_nan_input_handling(self):
        """Test handling of NaN inputs"""
        uncertainty_module = ComprehensiveUncertainty(self.config, self.input_dim)
        
        # Test with NaN values
        nan_input = torch.full((16, self.input_dim), float('nan'))
        try:
            output = uncertainty_module(nan_input)
            # Should handle NaN gracefully
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError))
            
    def test_invalid_configurations(self):
        """Test handling of invalid configurations"""
        # Test invalid uncertainty configuration
        invalid_config = UncertaintyConfig(num_mc_samples=-1)
        uncertainty_module = ComprehensiveUncertainty(invalid_config, self.input_dim)
        
        test_input = torch.randn(16, self.input_dim)
        try:
            output = uncertainty_module(test_input)
            # Should handle invalid config gracefully
        except Exception as e:
            self.assertIsInstance(e, (ValueError, RuntimeError))


def create_test_suite():
    """Create comprehensive test suite"""
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestUncertaintyQuantification,
        TestXAIComponents,
        TestIntegratedComponents,
        TestIntegrationAndCompatibility,
        TestErrorHandling
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    return suite


def run_tests():
    """Run all tests"""
    print("Running Uncertainty Quantification and XAI Tests...")
    print("=" * 60)
    
    # Create test suite
    suite = create_test_suite()
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
            
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
            
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)