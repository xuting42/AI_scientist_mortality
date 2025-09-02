"""
Comprehensive tests for missing data imputation components

This module provides unit tests and validation procedures for all missing data
handling components in HAMNet.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import tempfile
import os

# Import the modules to test
from .gan_imputation import GANImputationConfig, GANImputer, ConditionalGenerator
from .graph_imputation import GraphImputationConfig, GraphImputer, PatientSimilarityGraph
from .advanced_imputation import AdvancedImputationConfig, AdvancedImputer, VariationalAutoencoder
from .integrated_missing_data import IntegratedMissingDataConfig, IntegratedMissingDataHandler


class TestGANImputation:
    """Test suite for GAN-based imputation"""
    
    @pytest.fixture
    def gan_config(self):
        """Create GAN configuration for testing"""
        return GANImputationConfig(
            latent_dim=64,
            hidden_dim=128,
            batch_size=16,
            num_epochs=2,  # Reduced for testing
            clinical_dim=50,
            imaging_dim=100,
            genetic_dim=200,
            lifestyle_dim=25
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        batch_size = 16
        
        data = {
            'clinical': torch.randn(batch_size, 50),
            'imaging': torch.randn(batch_size, 100),
            'genetic': torch.randn(batch_size, 200),
            'lifestyle': torch.randn(batch_size, 25)
        }
        
        masks = {
            'clinical': torch.bernoulli(0.8 * torch.ones(batch_size, 50)).bool(),
            'imaging': torch.bernoulli(0.7 * torch.ones(batch_size, 100)).bool(),
            'genetic': torch.bernoulli(0.9 * torch.ones(batch_size, 200)).bool(),
            'lifestyle': torch.bernoulli(0.85 * torch.ones(batch_size, 25)).bool()
        }
        
        return data, masks
    
    def test_conditional_generator(self, gan_config):
        """Test conditional generator"""
        generator = ConditionalGenerator(gan_config, "clinical")
        
        batch_size = 16
        observed_data = torch.randn(batch_size, gan_config.clinical_dim)
        condition = torch.randn(batch_size, sum(gan_config.modality_dims.values()) - gan_config.clinical_dim)
        noise = torch.randn(batch_size, gan_config.latent_dim)
        
        # Test forward pass
        output = generator(observed_data, condition, noise)
        
        assert output.shape == (batch_size, gan_config.clinical_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gan_imputer_initialization(self, gan_config):
        """Test GAN imputer initialization"""
        imputer = GANImputer(gan_config)
        
        assert imputer.config == gan_config
        assert imputer.model is not None
        assert imputer.g_optimizer is not None
        assert imputer.d_optimizer is not None
        assert isinstance(imputer.training_history, dict)
    
    def test_gan_imputer_train_step(self, gan_config, sample_data):
        """Test GAN imputer training step"""
        imputer = GANImputer(gan_config)
        data, masks = sample_data
        
        # Test training step
        losses = imputer.train_step(data, masks)
        
        assert isinstance(losses, dict)
        assert 'generator_loss' in losses
        assert 'discriminator_loss' in losses
        assert all(isinstance(v, float) for v in losses.values())
    
    def test_gan_imputer_imputation(self, gan_config, sample_data):
        """Test GAN imputation"""
        imputer = GANImputer(gan_config)
        data, masks = sample_data
        
        # Test imputation
        imputed_data = imputer.impute(data, masks)
        
        assert isinstance(imputed_data, dict)
        assert set(imputed_data.keys()) == set(data.keys())
        
        for modality in imputed_data:
            assert imputed_data[modality].shape == data[modality].shape
            assert not torch.isnan(imputed_data[modality]).any()
    
    def test_gan_imputer_save_load(self, gan_config, sample_data):
        """Test GAN imputer save/load functionality"""
        imputer = GANImputer(gan_config)
        data, masks = sample_data
        
        # Get initial imputation
        initial_imputation = imputer.impute(data, masks)
        
        # Save model
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            imputer.save(tmp.name)
            tmp_path = tmp.name
        
        # Load model
        new_imputer = GANImputer(gan_config)
        new_imputer.load(tmp_path)
        
        # Test loaded model
        loaded_imputation = new_imputer.impute(data, masks)
        
        # Check that imputations are similar
        for modality in initial_imputation:
            assert torch.allclose(initial_imputation[modality], 
                                loaded_imputation[modality], atol=1e-6)
        
        # Clean up
        os.unlink(tmp_path)


class TestGraphImputation:
    """Test suite for graph-based imputation"""
    
    @pytest.fixture
    def graph_config(self):
        """Create graph configuration for testing"""
        return GraphImputationConfig(
            hidden_dim=128,
            num_layers=2,
            similarity_threshold=0.5,
            k_neighbors=5,
            batch_size=16,
            clinical_dim=50,
            imaging_dim=100,
            genetic_dim=200,
            lifestyle_dim=25
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        batch_size = 16
        
        data = {
            'clinical': torch.randn(batch_size, 50),
            'imaging': torch.randn(batch_size, 100),
            'genetic': torch.randn(batch_size, 200),
            'lifestyle': torch.randn(batch_size, 25)
        }
        
        masks = {
            'clinical': torch.bernoulli(0.8 * torch.ones(batch_size, 50)).bool(),
            'imaging': torch.bernoulli(0.7 * torch.ones(batch_size, 100)).bool(),
            'genetic': torch.bernoulli(0.9 * torch.ones(batch_size, 200)).bool(),
            'lifestyle': torch.bernoulli(0.85 * torch.ones(batch_size, 25)).bool()
        }
        
        return data, masks
    
    def test_patient_similarity_graph(self, graph_config, sample_data):
        """Test patient similarity graph construction"""
        graph_constructor = PatientSimilarityGraph(graph_config)
        data, masks = sample_data
        
        # Test graph construction
        edge_index, edge_weight = graph_constructor.construct_graph(data, masks)
        
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == edge_weight.shape[0]
        assert edge_weight.min() >= 0
        assert edge_weight.max() <= 1
        
        # Test community detection
        assert graph_constructor.communities is not None
        assert len(graph_constructor.communities) == len(data['clinical'])
    
    def test_graph_imputer_initialization(self, graph_config):
        """Test graph imputer initialization"""
        imputer = GraphImputer(graph_config)
        
        assert imputer.config == graph_config
        assert imputer.graph_constructor is not None
        assert imputer.gnn is not None
        assert imputer.optimizer is not None
    
    def test_graph_imputer_train_step(self, graph_config, sample_data):
        """Test graph imputer training step"""
        imputer = GraphImputer(graph_config)
        data, masks = sample_data
        
        # Test training step
        losses = imputer.train_step(data, masks)
        
        assert isinstance(losses, dict)
        assert 'reconstruction_loss' in losses
        assert 'graph_loss' in losses
        assert 'community_loss' in losses
        assert all(isinstance(v, float) for v in losses.values())
    
    def test_graph_imputer_imputation(self, graph_config, sample_data):
        """Test graph imputation"""
        imputer = GraphImputer(graph_config)
        data, masks = sample_data
        
        # Test imputation
        imputed_data = imputer.impute(data, masks)
        
        assert isinstance(imputed_data, dict)
        assert set(imputed_data.keys()) == set(data.keys())
        
        for modality in imputed_data:
            assert imputed_data[modality].shape == data[modality].shape
            assert not torch.isnan(imputed_data[modality]).any()


class TestAdvancedImputation:
    """Test suite for advanced imputation methods"""
    
    @pytest.fixture
    def advanced_config(self):
        """Create advanced configuration for testing"""
        return AdvancedImputationConfig(
            latent_dim=64,
            hidden_dim=128,
            num_layers=2,
            clinical_dim=50,
            imaging_dim=100,
            genetic_dim=200,
            lifestyle_dim=25
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        batch_size = 16
        
        data = torch.randn(batch_size, 375)  # Sum of all modalities
        mask = torch.bernoulli(0.8 * torch.ones(batch_size, 375)).bool()
        
        return {'data': data, 'mask': mask}
    
    def test_variational_autoencoder(self, advanced_config):
        """Test variational autoencoder"""
        vae = VariationalAutoencoder(advanced_config)
        
        batch_size = 16
        total_dim = sum(advanced_config.modality_dims.values())
        x = torch.randn(batch_size, total_dim)
        mask = torch.bernoulli(0.8 * torch.ones(batch_size, total_dim)).bool()
        
        # Test forward pass
        outputs = vae(x, mask)
        
        assert 'mu' in outputs
        assert 'logvar' in outputs
        assert 'clinical' in outputs
        assert 'clinical_uncertainty' in outputs
        
        # Test shapes
        assert outputs['mu'].shape == (batch_size, advanced_config.latent_dim)
        assert outputs['logvar'].shape == (batch_size, advanced_config.latent_dim)
        assert outputs['clinical'].shape == (batch_size, advanced_config.clinical_dim)
    
    def test_vae_kl_loss(self, advanced_config):
        """Test VAE KL divergence computation"""
        vae = VariationalAutoencoder(advanced_config)
        
        batch_size = 16
        mu = torch.randn(batch_size, advanced_config.latent_dim)
        logvar = torch.randn(batch_size, advanced_config.latent_dim)
        
        kl_loss = vae.compute_kl_loss(mu, logvar)
        
        assert isinstance(kl_loss, torch.Tensor)
        assert kl_loss.shape == ()
        assert kl_loss >= 0
    
    def test_advanced_imputer_initialization(self, advanced_config):
        """Test advanced imputer initialization"""
        # Test VAE
        vae_imputer = AdvancedImputer(advanced_config, method="vae")
        assert vae_imputer.method == "vae"
        assert vae_imputer.model is not None
        
        # Test multi-task
        mt_imputer = AdvancedImputer(advanced_config, method="multi_task")
        assert mt_imputer.method == "multi_task"
        assert mt_imputer.model is not None
    
    def test_advanced_imputer_imputation(self, advanced_config, sample_data):
        """Test advanced imputation methods"""
        # Test VAE
        vae_imputer = AdvancedImputer(advanced_config, method="vae")
        vae_output = vae_imputer.impute(sample_data['data'], sample_data['mask'])
        
        assert isinstance(vae_output, dict)
        assert 'clinical' in vae_output
        assert 'clinical_uncertainty' in vae_output
        
        # Test multi-task
        mt_imputer = AdvancedImputer(advanced_config, method="multi_task")
        mt_output = mt_imputer.impute(sample_data['data'], sample_data['mask'])
        
        assert isinstance(mt_output, dict)
        assert 'imputed' in mt_output
        assert 'quality_assessment' in mt_output


class TestIntegratedMissingData:
    """Test suite for integrated missing data handling"""
    
    @pytest.fixture
    def integrated_config(self):
        """Create integrated configuration for testing"""
        return IntegratedMissingDataConfig(
            primary_method="adaptive",
            batch_size=16,
            gan_config=GANImputationConfig(
                latent_dim=32,
                hidden_dim=64,
                clinical_dim=50,
                imaging_dim=100,
                genetic_dim=200,
                lifestyle_dim=25
            ),
            graph_config=GraphImputationConfig(
                hidden_dim=64,
                clinical_dim=50,
                imaging_dim=100,
                genetic_dim=200,
                lifestyle_dim=25
            ),
            advanced_config=AdvancedImputationConfig(
                latent_dim=32,
                hidden_dim=64,
                clinical_dim=50,
                imaging_dim=100,
                genetic_dim=200,
                lifestyle_dim=25
            )
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        batch_size = 16
        
        data = {
            'clinical': torch.randn(batch_size, 50),
            'imaging': torch.randn(batch_size, 100),
            'genetic': torch.randn(batch_size, 200),
            'lifestyle': torch.randn(batch_size, 25)
        }
        
        masks = {
            'clinical': torch.bernoulli(0.8 * torch.ones(batch_size, 50)).bool(),
            'imaging': torch.bernoulli(0.7 * torch.ones(batch_size, 100)).bool(),
            'genetic': torch.bernoulli(0.9 * torch.ones(batch_size, 200)).bool(),
            'lifestyle': torch.bernoulli(0.85 * torch.ones(batch_size, 25)).bool()
        }
        
        return data, masks
    
    def test_missing_data_analyzer(self, integrated_config, sample_data):
        """Test missing data analyzer"""
        from .integrated_missing_data import MissingDataAnalyzer
        
        analyzer = MissingDataAnalyzer(integrated_config)
        data, masks = sample_data
        
        # Test pattern analysis
        analysis = analyzer.analyze_missing_patterns(masks)
        
        assert isinstance(analysis, dict)
        assert 'overall_missing_rate' in analysis
        assert 'modality_stats' in analysis
        assert 'pattern_analysis' in analysis
        assert 'recommended_method' in analysis
        
        # Check types
        assert isinstance(analysis['overall_missing_rate'], float)
        assert isinstance(analysis['modality_stats'], dict)
        assert isinstance(analysis['recommended_method'], str)
    
    def test_integrated_imputation_module(self, integrated_config, sample_data):
        """Test integrated imputation module"""
        from .integrated_missing_data import IntegratedImputationModule
        
        module = IntegratedImputationModule(integrated_config)
        data, masks = sample_data
        
        # Test forward pass
        outputs = module(data, masks, training=False)
        
        assert isinstance(outputs, dict)
        assert 'predictions' in outputs
        assert 'imputed_data' in outputs
        assert 'missing_analysis' in outputs
        assert 'selected_method' in outputs
        
        # Check that imputed data preserves shapes
        for modality in outputs['imputed_data']:
            assert outputs['imputed_data'][modality].shape == data[modality].shape
    
    def test_integrated_missing_data_handler(self, integrated_config, sample_data):
        """Test integrated missing data handler"""
        handler = IntegratedMissingDataHandler(integrated_config)
        data, masks = sample_data
        
        # Test prediction
        outputs = handler.predict(data, masks)
        
        assert isinstance(outputs, dict)
        assert 'predictions' in outputs
        assert 'imputed_data' in outputs
        
        # Test missing data analysis
        analysis = handler.analyze_missing_data(masks)
        
        assert isinstance(analysis, dict)
        assert 'overall_missing_rate' in analysis
        assert 'recommended_method' in analysis


class TestIntegration:
    """Integration tests for missing data components"""
    
    @pytest.fixture
    def integration_config(self):
        """Create configuration for integration tests"""
        return IntegratedMissingDataConfig(
            primary_method="ensemble",
            batch_size=8,
            gan_config=GANImputationConfig(
                latent_dim=32,
                hidden_dim=64,
                num_epochs=1,
                clinical_dim=20,
                imaging_dim=40,
                genetic_dim=80,
                lifestyle_dim=10
            ),
            graph_config=GraphImputationConfig(
                hidden_dim=64,
                num_epochs=1,
                clinical_dim=20,
                imaging_dim=40,
                genetic_dim=80,
                lifestyle_dim=10
            ),
            advanced_config=AdvancedImputationConfig(
                latent_dim=32,
                hidden_dim=64,
                num_epochs=1,
                clinical_dim=20,
                imaging_dim=40,
                genetic_dim=80,
                lifestyle_dim=10
            )
        )
    
    def test_end_to_end_pipeline(self, integration_config):
        """Test end-to-end pipeline with synthetic data"""
        # Create synthetic dataset
        num_samples = 100
        
        dataset = []
        for _ in range(num_samples):
            data = {
                'clinical': torch.randn(20),
                'imaging': torch.randn(40),
                'genetic': torch.randn(80),
                'lifestyle': torch.randn(10)
            }
            
            masks = {
                'clinical': torch.bernoulli(0.8 * torch.ones(20)).bool(),
                'imaging': torch.bernoulli(0.7 * torch.ones(40)).bool(),
                'genetic': torch.bernoulli(0.9 * torch.ones(80)).bool(),
                'lifestyle': torch.bernoulli(0.85 * torch.ones(10)).bool()
            }
            
            dataset.append({'data': data, 'masks': masks})
        
        # Initialize handler
        handler = IntegratedMissingDataHandler(integration_config)
        
        # Test with a sample
        sample = dataset[0]
        outputs = handler.predict(sample['data'], sample['masks'])
        
        assert isinstance(outputs, dict)
        assert 'predictions' in outputs
        assert 'imputed_data' in outputs
        
        # Test that imputed data has correct shapes
        for modality in outputs['imputed_data']:
            assert outputs['imputed_data'][modality].shape == sample['data'][modality].shape
    
    def test_imputation_quality_metrics(self, integration_config):
        """Test imputation quality evaluation"""
        # Create test data with known missing values
        num_samples = 50
        
        # Ground truth data
        true_data = {
            'clinical': torch.randn(num_samples, 20),
            'imaging': torch.randn(num_samples, 40),
            'genetic': torch.randn(num_samples, 80),
            'lifestyle': torch.randn(num_samples, 10)
        }
        
        # Create masks
        masks = {
            'clinical': torch.bernoulli(0.7 * torch.ones(num_samples, 20)).bool(),
            'imaging': torch.bernoulli(0.6 * torch.ones(num_samples, 40)).bool(),
            'genetic': torch.bernoulli(0.8 * torch.ones(num_samples, 80)).bool(),
            'lifestyle': torch.bernoulli(0.75 * torch.ones(num_samples, 10)).bool()
        }
        
        # Create masked data
        masked_data = {}
        for modality in true_data:
            masked_data[modality] = true_data[modality] * masks[modality].float()
        
        # Initialize handler
        handler = IntegratedMissingDataHandler(integration_config)
        
        # Impute
        imputed_data = handler.predict(masked_data, masks)['imputed_data']
        
        # Compute reconstruction errors
        errors = {}
        for modality in true_data:
            mask = masks[modality]
            if mask.any():
                true_values = true_data[modality][mask]
                imputed_values = imputed_data[modality][mask]
                
                mse = F.mse_loss(imputed_values, true_values).item()
                mae = F.l1_loss(imputed_values, true_values).item()
                
                errors[modality] = {'mse': mse, 'mae': mae}
        
        # Check that errors are reasonable
        for modality in errors:
            assert errors[modality]['mse'] > 0
            assert errors[modality]['mae'] > 0
            assert errors[modality]['mse'] < 10  # Should not be too large
    
    def test_scalability_test(self, integration_config):
        """Test scalability with different batch sizes"""
        handler = IntegratedMissingDataHandler(integration_config)
        
        batch_sizes = [1, 8, 16, 32]
        
        for batch_size in batch_sizes:
            # Create data
            data = {
                'clinical': torch.randn(batch_size, 20),
                'imaging': torch.randn(batch_size, 40),
                'genetic': torch.randn(batch_size, 80),
                'lifestyle': torch.randn(batch_size, 10)
            }
            
            masks = {
                'clinical': torch.bernoulli(0.8 * torch.ones(batch_size, 20)).bool(),
                'imaging': torch.bernoulli(0.7 * torch.ones(batch_size, 40)).bool(),
                'genetic': torch.bernoulli(0.9 * torch.ones(batch_size, 80)).bool(),
                'lifestyle': torch.bernoulli(0.85 * torch.ones(batch_size, 10)).bool()
            }
            
            # Test prediction
            outputs = handler.predict(data, masks)
            
            # Check outputs
            assert isinstance(outputs, dict)
            assert 'predictions' in outputs
            assert outputs['predictions'].shape[0] == batch_size


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])