"""
HAMNet: Hierarchical Attention-based Multimodal Network for Biological Age Prediction

A PyTorch implementation of HAMNet for predicting biological age using multimodal data
from UK Biobank. The architecture includes modality-specific encoders, cross-modal attention
fusion, temporal integration layers, and uncertainty quantification.

Main Components:
- HAMNet model class with hierarchical architecture
- Modality-specific encoders (Clinical, Imaging, Genetic, Lifestyle)
- Cross-modal attention fusion mechanisms
- Temporal integration layers for longitudinal data
- Uncertainty quantification through Bayesian neural networks
- Missing data handling capabilities

Example Usage:
    from hamnet import HAMNet, HAMNetConfig, TrainingConfig
    
    # Create model configuration
    config = HAMNetConfig(
        model_tier="standard",
        embedding_dim=256,
        hidden_dim=512,
        num_heads=8,
        enable_uncertainty=True
    )
    
    # Create model
    model = HAMNet(config)
    
    # Create training configuration
    training_config = TrainingConfig(
        epochs=100,
        batch_size=32,
        learning_rate=1e-4
    )
    
    # Train model
    from hamnet.training import HAMNetTrainer
    trainer = HAMNetTrainer(model, training_config)
    results = trainer.train(train_loader, val_loader)
"""

from .models.hamnet import (
    HAMNet,
    HAMNetConfig,
    ModalityEncoder,
    ClinicalEncoder,
    ImagingEncoder,
    GeneticEncoder,
    LifestyleEncoder,
    CrossModalAttention,
    TemporalIntegrationLayer,
    UncertaintyQuantification,
    create_hamnet_model,
    count_parameters,
    model_summary
)

from .utils.utils import (
    TrainingConfig,
    HAMNetDataset,
    CheckpointManager,
    EarlyStopping,
    MetricsTracker,
    create_optimizer,
    create_scheduler,
    compute_loss,
    collate_fn,
    setup_logging,
    save_config,
    load_config,
    create_data_loaders,
    visualize_attention_weights,
    plot_training_history,
    get_model_size,
    profile_model,
    DataValidator,
    create_experiment_dir,
    save_results
)

from .training import (
    HAMNetTrainer,
    CrossValidator,
    HyperparameterOptimizer,
    ModelEnsemble
)

__version__ = "1.0.0"
__author__ = "HAMNet Development Team"
__email__ = "hamnet@example.com"

__all__ = [
    # Model classes
    "HAMNet",
    "HAMNetConfig",
    "ModalityEncoder",
    "ClinicalEncoder",
    "ImagingEncoder",
    "GeneticEncoder",
    "LifestyleEncoder",
    "CrossModalAttention",
    "TemporalIntegrationLayer",
    "UncertaintyQuantification",
    
    # Utility functions
    "create_hamnet_model",
    "count_parameters",
    "model_summary",
    
    # Training utilities
    "TrainingConfig",
    "HAMNetDataset",
    "CheckpointManager",
    "EarlyStopping",
    "MetricsTracker",
    "create_optimizer",
    "create_scheduler",
    "compute_loss",
    "collate_fn",
    "setup_logging",
    "save_config",
    "load_config",
    "create_data_loaders",
    "visualize_attention_weights",
    "plot_training_history",
    "get_model_size",
    "profile_model",
    "DataValidator",
    "create_experiment_dir",
    "save_results",
    
    # Training classes
    "HAMNetTrainer",
    "CrossValidator",
    "HyperparameterOptimizer",
    "ModelEnsemble"
]