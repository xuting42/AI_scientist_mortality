"""
Comprehensive unit tests for HAMNet

This module provides unit tests for all components of HAMNet:
- Model architecture tests
- Encoder tests
- Attention mechanism tests
- Temporal integration tests
- Uncertainty quantification tests
- Utility function tests
- Integration tests
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional
import tempfile
import os
from pathlib import Path

# Import HAMNet components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hamnet.models.hamnet import (
    HAMNet, HAMNetConfig, ModalityEncoder, ClinicalEncoder,
    ImagingEncoder, GeneticEncoder, LifestyleEncoder,
    CrossModalAttention, TemporalIntegrationLayer,
    UncertaintyQuantification, create_hamnet_model, count_parameters
)

from hamnet.utils.utils import (
    TrainingConfig, HAMNetDataset, CheckpointManager, EarlyStopping,
    MetricsTracker, compute_loss, collate_fn, DataValidator,
    get_model_size, profile_model
)

from hamnet.training import HAMNetTrainer, CrossValidator, ModelEnsemble


class TestHAMNetConfig:
    """Test HAMNet configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = HAMNetConfig()
        
        assert config.model_tier == "standard"
        assert config.embedding_dim == 256
        assert config.hidden_dim == 512
        assert config.num_heads == 8
        assert config.dropout == 0.1
        assert config.enable_uncertainty == True
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = HAMNetConfig(
            model_tier="comprehensive",
            embedding_dim=512,
            hidden_dim=1024,
            num_heads=16,
            dropout=0.2
        )
        
        assert config.model_tier == "comprehensive"
        assert config.embedding_dim == 512
        assert config.hidden_dim == 1024
        assert config.num_heads == 16
        assert config.dropout == 0.2
        
    def test_invalid_config(self):
        """Test invalid configuration"""
        with pytest.raises(Exception):
            # Invalid model tier
            HAMNetConfig(model_tier="invalid")


class TestModalityEncoders:
    """Test modality-specific encoders"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = HAMNetConfig()
        self.batch_size = 32
        
    def test_clinical_encoder(self):
        """Test clinical encoder"""
        encoder = ClinicalEncoder(self.config)
        
        # Test input
        x = torch.randn(self.batch_size, self.config.clinical_dim)
        
        # Forward pass
        output = encoder(x)
        
        assert output.shape == (self.batch_size, self.config.embedding_dim)
        
    def test_clinical_encoder_with_mask(self):
        """Test clinical encoder with missing data mask"""
        encoder = ClinicalEncoder(self.config)
        
        # Test input with mask
        x = torch.randn(self.batch_size, self.config.clinical_dim)
        mask = torch.ones(self.batch_size, self.config.clinical_dim)
        mask[0, :10] = 0  # Simulate missing data
        
        output = encoder(x, mask)
        
        assert output.shape == (self.batch_size, self.config.embedding_dim)
        
    def test_imaging_encoder(self):
        """Test imaging encoder"""
        encoder = ImagingEncoder(self.config)
        
        # Test input
        x = torch.randn(self.batch_size, self.config.imaging_dim)
        
        output = encoder(x)
        
        assert output.shape == (self.batch_size, self.config.embedding_dim)
        
    def test_genetic_encoder(self):
        """Test genetic encoder"""
        encoder = GeneticEncoder(self.config)
        
        # Test input
        x = torch.randn(self.batch_size, self.config.genetic_dim)
        
        output = encoder(x)
        
        assert output.shape == (self.batch_size, self.config.embedding_dim)
        
    def test_lifestyle_encoder(self):
        """Test lifestyle encoder"""
        encoder = LifestyleEncoder(self.config)
        
        # Test input
        x = torch.randn(self.batch_size, self.config.lifestyle_dim)
        
        output = encoder(x)
        
        assert output.shape == (self.batch_size, self.config.embedding_dim)
        
    def test_lifestyle_encoder_categorical(self):
        """Test lifestyle encoder with categorical data"""
        encoder = LifestyleEncoder(self.config)
        
        # Test input with categorical data
        x = torch.randint(0, 10, (self.batch_size, self.config.lifestyle_dim))
        
        output = encoder(x)
        
        assert output.shape == (self.batch_size, self.config.embedding_dim)


class TestCrossModalAttention:
    """Test cross-modal attention mechanism"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = HAMNetConfig()
        self.attention = CrossModalAttention(self.config)
        self.batch_size = 32
        self.seq_len = 10
        
    def test_attention_forward(self):
        """Test attention forward pass"""
        query = torch.randn(self.batch_size, self.seq_len, self.config.embedding_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.config.embedding_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.config.embedding_dim)
        
        output, weights = self.attention(query, key, value)
        
        assert output.shape == (self.batch_size, self.seq_len, self.config.embedding_dim)
        assert weights.shape == (self.batch_size, self.config.num_heads, self.seq_len, self.seq_len)
        
    def test_attention_with_masks(self):
        """Test attention with padding masks"""
        query = torch.randn(self.batch_size, self.seq_len, self.config.embedding_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.config.embedding_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.config.embedding_dim)
        
        # Create padding mask
        key_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        key_mask[:, -2:] = False  # Last two positions are padding
        
        output, weights = self.attention(query, key, value, key_mask=key_mask)
        
        assert output.shape == (self.batch_size, self.seq_len, self.config.embedding_dim)
        assert weights.shape == (self.batch_size, self.config.num_heads, self.seq_len, self.seq_len)


class TestTemporalIntegration:
    """Test temporal integration layer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = HAMNetConfig()
        self.temporal_layer = TemporalIntegrationLayer(self.config)
        self.batch_size = 32
        self.time_steps = 5
        
    def test_temporal_integration(self):
        """Test temporal integration"""
        x = torch.randn(self.batch_size, self.time_steps, self.config.embedding_dim)
        
        output = self.temporal_layer(x)
        
        assert output.shape == (self.batch_size, self.time_steps, self.config.embedding_dim)
        
    def test_temporal_integration_with_mask(self):
        """Test temporal integration with mask"""
        x = torch.randn(self.batch_size, self.time_steps, self.config.embedding_dim)
        temporal_mask = torch.ones(self.batch_size, self.time_steps, dtype=torch.bool)
        temporal_mask[:, -1:] = False  # Last time step is padding
        
        output = self.temporal_layer(x, temporal_mask)
        
        assert output.shape == (self.batch_size, self.time_steps, self.config.embedding_dim)


class TestUncertaintyQuantification:
    """Test uncertainty quantification module"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = HAMNetConfig()
        self.uncertainty_module = UncertaintyQuantification(self.config)
        self.batch_size = 32
        
    def test_uncertainty_forward(self):
        """Test uncertainty quantification forward pass"""
        x = torch.randn(self.batch_size, self.config.embedding_dim)
        
        output = self.uncertainty_module(x, num_samples=10)
        
        assert 'mean' in output
        assert 'variance' in output
        assert 'uncertainty' in output
        
        assert output['mean'].shape == (self.batch_size, 2)
        assert output['variance'].shape == (self.batch_size, 2)
        assert output['uncertainty'].shape == (self.batch_size, 2)
        
    def test_uncertainty_positive_variance(self):
        """Test that variance is always positive"""
        x = torch.randn(self.batch_size, self.config.embedding_dim)
        
        output = self.uncertainty_module(x, num_samples=10)
        
        assert torch.all(output['variance'] >= 0)
        assert torch.all(output['uncertainty'] >= 0)


class TestHAMNet:
    """Test main HAMNet model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = HAMNetConfig()
        self.model = HAMNet(self.config)
        self.batch_size = 32
        
    def test_model_creation(self):
        """Test model creation"""
        assert isinstance(self.model, HAMNet)
        assert count_parameters(self.model) > 0
        
    def test_model_forward_single_modality(self):
        """Test model forward pass with single modality"""
        inputs = {
            'clinical': torch.randn(self.batch_size, self.config.clinical_dim)
        }
        
        output = self.model(inputs)
        
        assert 'predictions' in output
        assert output['predictions'].shape == (self.batch_size, 1)
        
    def test_model_forward_multiple_modalities(self):
        """Test model forward pass with multiple modalities"""
        inputs = {
            'clinical': torch.randn(self.batch_size, self.config.clinical_dim),
            'imaging': torch.randn(self.batch_size, self.config.imaging_dim),
            'genetic': torch.randn(self.batch_size, self.config.genetic_dim),
            'lifestyle': torch.randn(self.batch_size, self.config.lifestyle_dim)
        }
        
        output = self.model(inputs)
        
        assert 'predictions' in output
        assert output['predictions'].shape == (self.batch_size, 1)
        
    def test_model_forward_with_masks(self):
        """Test model forward pass with missing data masks"""
        inputs = {
            'clinical': torch.randn(self.batch_size, self.config.clinical_dim),
            'imaging': torch.randn(self.batch_size, self.config.imaging_dim)
        }
        
        masks = {
            'clinical': torch.ones(self.batch_size, self.config.clinical_dim),
            'imaging': torch.ones(self.batch_size, self.config.imaging_dim)
        }
        
        # Simulate missing data
        masks['clinical'][0, :10] = 0
        masks['imaging'][0, :20] = 0
        
        output = self.model(inputs, masks)
        
        assert 'predictions' in output
        assert output['predictions'].shape == (self.batch_size, 1)
        
    def test_model_forward_with_uncertainty(self):
        """Test model forward pass with uncertainty quantification"""
        self.config.enable_uncertainty = True
        model = HAMNet(self.config)
        
        inputs = {
            'clinical': torch.randn(self.batch_size, self.config.clinical_dim)
        }
        
        output = model(inputs)
        
        assert 'predictions' in output
        assert 'mean' in output
        assert 'variance' in output
        assert 'uncertainty' in output
        
    def test_model_predict_with_uncertainty(self):
        """Test model prediction with uncertainty"""
        inputs = {
            'clinical': torch.randn(self.batch_size, self.config.clinical_dim)
        }
        
        output = self.model.predict_with_uncertainty(inputs, num_samples=10)
        
        assert 'mean_prediction' in output
        assert 'variance' in output
        assert 'uncertainty' in output
        assert 'samples' in output
        
    def test_model_empty_input(self):
        """Test model with empty input"""
        inputs = {}
        
        output = self.model(inputs)
        
        assert 'predictions' in output
        assert output['predictions'].shape == (1, 1)  # Default batch size


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_count_parameters(self):
        """Test parameter counting"""
        config = HAMNetConfig()
        model = HAMNet(config)
        
        param_count = count_parameters(model)
        
        assert param_count > 0
        assert isinstance(param_count, int)
        
    def test_create_hamnet_model(self):
        """Test model factory function"""
        config = HAMNetConfig()
        model = create_hamnet_model(config)
        
        assert isinstance(model, HAMNet)
        
    def test_get_model_size(self):
        """Test model size calculation"""
        config = HAMNetConfig()
        model = HAMNet(config)
        
        size_info = get_model_size(model)
        
        assert 'total_parameters' in size_info
        assert 'trainable_parameters' in size_info
        assert 'model_size_mb' in size_info
        
        assert size_info['total_parameters'] > 0
        assert size_info['trainable_parameters'] > 0
        assert size_info['model_size_mb'] > 0


class TestDataset:
    """Test HAMNet dataset"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.batch_size = 32
        self.sample_size = 100
        
        # Create dummy data
        self.data = {
            'clinical': np.random.randn(self.sample_size, 100),
            'imaging': np.random.randn(self.sample_size, 512),
            'genetic': np.random.randn(self.sample_size, 1000),
            'lifestyle': np.random.randn(self.sample_size, 50)
        }
        
        self.targets = np.random.randn(self.sample_size)
        
        # Create masks
        self.masks = {
            'clinical': np.ones((self.sample_size, 100)),
            'imaging': np.ones((self.sample_size, 512))
        }
        
    def test_dataset_creation(self):
        """Test dataset creation"""
        dataset = HAMNetDataset(self.data, self.targets)
        
        assert len(dataset) == self.sample_size
        
    def test_dataset_with_masks(self):
        """Test dataset with masks"""
        dataset = HAMNetDataset(self.data, self.targets, self.masks)
        
        assert len(dataset) == self.sample_size
        
        sample = dataset[0]
        assert 'clinical_mask' in sample
        assert 'imaging_mask' in sample
        
    def test_dataset_getitem(self):
        """Test dataset item retrieval"""
        dataset = HAMNetDataset(self.data, self.targets)
        
        sample = dataset[0]
        
        assert 'targets' in sample
        assert 'clinical' in sample
        assert 'imaging' in sample
        assert 'genetic' in sample
        assert 'lifestyle' in sample
        assert 'index' in sample
        
    def test_dataset_validation(self):
        """Test dataset validation"""
        # Test with inconsistent sample sizes
        invalid_data = {
            'clinical': np.random.randn(self.sample_size, 100),
            'imaging': np.random.randn(self.sample_size + 1, 512)  # Wrong size
        }
        
        with pytest.raises(ValueError):
            HAMNetDataset(invalid_data, self.targets)
            
    def test_collate_fn(self):
        """Test collate function"""
        dataset = HAMNetDataset(self.data, self.targets)
        
        # Create batch
        batch = [dataset[i] for i in range(self.batch_size)]
        
        collated = collate_fn(batch)
        
        assert 'targets' in collated
        assert 'clinical' in collated
        assert 'imaging' in collated
        assert 'genetic' in collated
        assert 'lifestyle' in collated
        
        assert collated['targets'].shape == (self.batch_size,)


class TestCheckpointManager:
    """Test checkpoint manager"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint_manager = CheckpointManager(self.temp_dir, "test_model")
        
        config = HAMNetConfig()
        self.model = HAMNet(config)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_save_checkpoint(self):
        """Test checkpoint saving"""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, epoch=0, loss=0.1, config={}
        )
        
        assert os.path.exists(checkpoint_path)
        
    def test_save_best_checkpoint(self):
        """Test best checkpoint saving"""
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, epoch=0, loss=0.1, config={}, is_best=True
        )
        
        assert os.path.exists(checkpoint_path)
        best_path = self.checkpoint_manager.get_best_checkpoint()
        assert os.path.exists(best_path)
        
    def test_load_checkpoint(self):
        """Test checkpoint loading"""
        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, epoch=0, loss=0.1, config={}
        )
        
        # Create new model and optimizer
        config = HAMNetConfig()
        new_model = HAMNet(config)
        new_optimizer = torch.optim.Adam(new_model.parameters())
        
        # Load checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(
            checkpoint_path, new_model, new_optimizer
        )
        
        assert checkpoint['epoch'] == 0
        assert checkpoint['loss'] == 0.1
        
    def test_get_latest_checkpoint(self):
        """Test getting latest checkpoint"""
        # Save multiple checkpoints
        self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, epoch=0, loss=0.1, config={}
        )
        self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, epoch=1, loss=0.05, config={}
        )
        
        latest_path = self.checkpoint_manager.get_latest_checkpoint()
        assert os.path.exists(latest_path)
        assert "epoch_1" in latest_path


class TestEarlyStopping:
    """Test early stopping"""
    
    def test_early_stopping_basic(self):
        """Test basic early stopping"""
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        
        # Simulate improving loss
        assert not early_stopping(0.5)
        assert not early_stopping(0.4)
        assert not early_stopping(0.3)
        
        # Simulate plateau
        assert not early_stopping(0.3)
        assert not early_stopping(0.3)
        assert not early_stopping(0.3)
        
        # Should trigger early stopping
        assert early_stopping(0.3)
        
    def test_early_stopping_min_delta(self):
        """Test early stopping with minimum delta"""
        early_stopping = EarlyStopping(patience=2, min_delta=0.1)
        
        # Simulate small improvements
        assert not early_stopping(0.5)
        assert not early_stopping(0.45)  # Only 0.05 improvement
        assert not early_stopping(0.42)  # Only 0.03 improvement
        
        # Should trigger early stopping
        assert early_stopping(0.42)


class TestMetricsTracker:
    """Test metrics tracker"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.metrics_tracker = MetricsTracker()
        self.batch_size = 32
        
    def test_metrics_tracker_update(self):
        """Test metrics update"""
        predictions = torch.randn(self.batch_size)
        targets = torch.randn(self.batch_size)
        loss = 0.1
        
        self.metrics_tracker.update(predictions, targets, loss)
        
        assert len(self.metrics_tracker.predictions) == self.batch_size
        assert len(self.metrics_tracker.targets) == self.batch_size
        assert len(self.metrics_tracker.losses) == 1
        
    def test_metrics_tracker_compute(self):
        """Test metrics computation"""
        predictions = torch.randn(self.batch_size)
        targets = torch.randn(self.batch_size)
        loss = 0.1
        
        self.metrics_tracker.update(predictions, targets, loss)
        metrics = self.metrics_tracker.compute_metrics()
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mean_loss' in metrics
        
    def test_metrics_tracker_with_uncertainty(self):
        """Test metrics tracking with uncertainty"""
        predictions = torch.randn(self.batch_size)
        targets = torch.randn(self.batch_size)
        loss = 0.1
        uncertainties = torch.rand(self.batch_size)
        
        self.metrics_tracker.update(predictions, targets, loss, uncertainties)
        metrics = self.metrics_tracker.compute_metrics()
        
        assert 'mean_uncertainty' in metrics
        assert 'uncertainty_calibration' in metrics
        
    def test_metrics_tracker_reset(self):
        """Test metrics reset"""
        predictions = torch.randn(self.batch_size)
        targets = torch.randn(self.batch_size)
        loss = 0.1
        
        self.metrics_tracker.update(predictions, targets, loss)
        self.metrics_tracker.reset()
        
        assert len(self.metrics_tracker.predictions) == 0
        assert len(self.metrics_tracker.targets) == 0
        assert len(self.metrics_tracker.losses) == 0


class TestDataValidator:
    """Test data validator"""
    
    def test_validate_input_data_valid(self):
        """Test validation of valid input data"""
        sample_size = 100
        
        data = {
            'clinical': np.random.randn(sample_size, 100),
            'imaging': np.random.randn(sample_size, 512)
        }
        
        targets = np.random.randn(sample_size)
        
        errors = DataValidator.validate_input_data(data, targets)
        
        assert len(errors) == 0
        
    def test_validate_input_data_invalid(self):
        """Test validation of invalid input data"""
        sample_size = 100
        
        # Inconsistent sample sizes
        data = {
            'clinical': np.random.randn(sample_size, 100),
            'imaging': np.random.randn(sample_size + 1, 512)  # Wrong size
        }
        
        targets = np.random.randn(sample_size)
        
        errors = DataValidator.validate_input_data(data, targets)
        
        assert len(errors) > 0
        
    def test_validate_input_data_with_nan(self):
        """Test validation with NaN values"""
        sample_size = 100
        
        data = {
            'clinical': np.random.randn(sample_size, 100),
            'imaging': np.random.randn(sample_size, 512)
        }
        
        # Add NaN values
        data['clinical'][0, 0] = np.nan
        targets = np.random.randn(sample_size)
        targets[0] = np.nan
        
        errors = DataValidator.validate_input_data(data, targets)
        
        assert len(errors) > 0
        
    def test_validate_config_valid(self):
        """Test validation of valid config"""
        config = HAMNetConfig()
        
        errors = DataValidator.validate_config(config)
        
        assert len(errors) == 0
        
    def test_validate_config_invalid(self):
        """Test validation of invalid config"""
        config = HAMNetConfig(model_tier="invalid")
        
        errors = DataValidator.validate_config(config)
        
        assert len(errors) > 0


class TestTrainingComponents:
    """Test training components"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_config = HAMNetConfig()
        self.training_config = TrainingConfig(
            epochs=2,  # Short training for testing
            batch_size=16,
            checkpoint_dir=self.temp_dir
        )
        
    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_trainer_creation(self):
        """Test trainer creation"""
        model = HAMNet(self.model_config)
        trainer = HAMNetTrainer(model, self.training_config)
        
        assert isinstance(trainer, HAMNetTrainer)
        
    def test_trainer_train_epoch(self):
        """Test trainer training epoch"""
        model = HAMNet(self.model_config)
        trainer = HAMNetTrainer(model, self.training_config)
        
        # Create dummy data
        dataset = HAMNetDataset(
            data={
                'clinical': np.random.randn(100, 100),
                'imaging': np.random.randn(100, 512)
            },
            targets=np.random.randn(100)
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.training_config.batch_size, shuffle=True
        )
        
        # Train one epoch
        metrics = trainer.train_epoch(train_loader)
        
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)
        
    def test_trainer_validate_epoch(self):
        """Test trainer validation epoch"""
        model = HAMNet(self.model_config)
        trainer = HAMNetTrainer(model, self.training_config)
        
        # Create dummy data
        dataset = HAMNetDataset(
            data={
                'clinical': np.random.randn(100, 100),
                'imaging': np.random.randn(100, 512)
            },
            targets=np.random.randn(100)
        )
        
        val_loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.training_config.batch_size, shuffle=False
        )
        
        # Validate one epoch
        metrics = trainer.validate_epoch(val_loader)
        
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)
        
    def test_model_ensemble(self):
        """Test model ensemble"""
        # Create multiple models
        models = []
        for _ in range(3):
            model = HAMNet(self.model_config)
            models.append(model)
        
        ensemble = ModelEnsemble(models)
        
        assert len(ensemble.models) == 3
        
        # Test ensemble prediction
        dataset = HAMNetDataset(
            data={
                'clinical': np.random.randn(100, 100),
                'imaging': np.random.randn(100, 512)
            },
            targets=np.random.randn(100)
        )
        
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False
        )
        
        predictions = ensemble.predict(data_loader)
        
        assert 'predictions' in predictions
        assert 'uncertainties' in predictions
        
    def test_compute_loss(self):
        """Test loss computation"""
        predictions = torch.randn(32)
        targets = torch.randn(32)
        
        # Test MSE loss
        loss_mse = compute_loss(predictions, targets, loss_type="mse")
        assert isinstance(loss_mse, torch.Tensor)
        
        # Test MAE loss
        loss_mae = compute_loss(predictions, targets, loss_type="mae")
        assert isinstance(loss_mae, torch.Tensor)
        
        # Test Huber loss
        loss_huber = compute_loss(predictions, targets, loss_type="huber")
        assert isinstance(loss_huber, torch.Tensor)
        
        # Test with uncertainty
        uncertainties = torch.rand(32)
        loss_uncertainty = compute_loss(predictions, targets, uncertainties)
        assert isinstance(loss_uncertainty, torch.Tensor)


class TestIntegration:
    """Integration tests"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Cleanup"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline"""
        # Create model and configs
        model_config = HAMNetConfig()
        training_config = TrainingConfig(
            epochs=2,  # Short training for testing
            batch_size=16,
            checkpoint_dir=self.temp_dir
        )
        
        # Create model
        model = HAMNet(model_config)
        
        # Create dummy data
        dataset = HAMNetDataset(
            data={
                'clinical': np.random.randn(100, 100),
                'imaging': np.random.randn(100, 512),
                'genetic': np.random.randn(100, 1000),
                'lifestyle': np.random.randn(100, 50)
            },
            targets=np.random.randn(100)
        )
        
        # Split data
        train_size = 80
        train_dataset = torch.utils.data.Subset(dataset, list(range(train_size)))
        val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, 100)))
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=training_config.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=training_config.batch_size, shuffle=False
        )
        
        # Create trainer
        trainer = HAMNetTrainer(model, training_config)
        
        # Train
        results = trainer.train(train_loader, val_loader)
        
        # Check results
        assert 'best_epoch' in results
        assert 'best_val_loss' in results
        assert 'final_train_metrics' in results
        assert 'final_val_metrics' in results
        assert 'history' in results
        
        # Check that training actually happened
        assert results['best_epoch'] >= 0
        assert results['best_val_loss'] > 0
        
    def test_cross_validation(self):
        """Test cross-validation"""
        model_config = HAMNetConfig()
        training_config = TrainingConfig(
            epochs=1,  # Very short for testing
            batch_size=16
        )
        
        # Create dummy data
        dataset = HAMNetDataset(
            data={
                'clinical': np.random.randn(100, 100),
                'imaging': np.random.randn(100, 512)
            },
            targets=np.random.randn(100)
        )
        
        # Create cross-validator
        cv = CrossValidator(model_config, training_config)
        
        # Run cross-validation
        results = cv.cross_validate(dataset, n_folds=3)
        
        # Check results
        assert 'mean_val_loss' in results
        assert 'std_val_loss' in results
        assert 'mean_val_mae' in results
        assert 'std_val_mae' in results
        assert 'fold_results' in results
        
        assert len(results['fold_results']) == 3
        
    def test_model_save_load(self):
        """Test model save and load"""
        # Create model
        model_config = HAMNetConfig()
        model = HAMNet(model_config)
        
        # Save model
        model_path = os.path.join(self.temp_dir, "model.pt")
        torch.save(model.state_dict(), model_path)
        
        # Load model
        loaded_model = HAMNet(model_config)
        loaded_model.load_state_dict(torch.load(model_path))
        
        # Test that models produce same output
        inputs = {
            'clinical': torch.randn(16, 100),
            'imaging': torch.randn(16, 512)
        }
        
        model.eval()
        loaded_model.eval()
        
        with torch.no_grad():
            output1 = model(inputs)
            output2 = loaded_model(inputs)
            
        assert torch.allclose(output1['predictions'], output2['predictions'])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])