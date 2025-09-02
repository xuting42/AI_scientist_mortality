"""
Models module for HAMNet

This module contains the core model implementations:
- Main HAMNet model class
- Modality-specific encoders
- Attention and fusion mechanisms
- Temporal integration layers
- Uncertainty quantification modules
- Advanced missing data imputation methods
- GAN-based imputation
- Graph-based imputation
- Integrated missing data handling
"""

from .hamnet import (
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

from .gan_imputation import (
    GANImputationConfig,
    GANImputer,
    ConditionalGenerator,
    MultimodalDiscriminator,
    CycleConsistentGAN,
    WassersteinGAN
)

from .graph_imputation import (
    GraphImputationConfig,
    GraphImputer,
    PatientSimilarityGraph,
    GraphAttentionLayer,
    CommunityAwareGNN
)

from .advanced_imputation import (
    AdvancedImputationConfig,
    AdvancedImputer,
    VariationalAutoencoder,
    ProbabilisticMatrixFactorization,
    TemporalImputer,
    MultiTaskImputer
)

from .integrated_missing_data import (
    IntegratedMissingDataConfig,
    IntegratedMissingDataHandler,
    MissingDataAnalyzer,
    IntegratedImputationModule
)

from .uncertainty_quantification import (
    UncertaintyConfig,
    BayesianLinear,
    HeteroscedasticLoss,
    EvidentialOutput,
    DeepEnsemble,
    ComprehensiveUncertainty,
    UncertaintyMetrics,
    create_uncertainty_config,
    uncertainty_aware_training_step,
    monte_carlo_prediction
)

from .xai_module import (
    XAIConfig,
    SHAPExplainer,
    IntegratedGradients,
    AttentionVisualizer,
    LRPExplainer,
    LIMEExplainer,
    ClinicalInterpretability,
    ComprehensiveXAI,
    create_xai_config,
    explain_hamnet_prediction
)

from .integrated_uncertainty_xai import (
    IntegrationConfig,
    UncertaintyAwareAttention,
    ExplainableMultimodalFusion,
    ConfidenceWeightedPredictor,
    UncertaintyPropagation,
    InteractiveVisualization,
    IntegratedHAMNet,
    create_integration_config,
    create_integrated_hamnet
)

__all__ = [
    # Core HAMNet
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
    "create_hamnet_model",
    "count_parameters",
    "model_summary",
    
    # GAN-based Imputation
    "GANImputationConfig",
    "GANImputer",
    "ConditionalGenerator",
    "MultimodalDiscriminator",
    "CycleConsistentGAN",
    "WassersteinGAN",
    
    # Graph-based Imputation
    "GraphImputationConfig",
    "GraphImputer",
    "PatientSimilarityGraph",
    "GraphAttentionLayer",
    "CommunityAwareGNN",
    
    # Advanced Imputation
    "AdvancedImputationConfig",
    "AdvancedImputer",
    "VariationalAutoencoder",
    "ProbabilisticMatrixFactorization",
    "TemporalImputer",
    "MultiTaskImputer",
    
    # Integrated Missing Data
    "IntegratedMissingDataConfig",
    "IntegratedMissingDataHandler",
    "MissingDataAnalyzer",
    "IntegratedImputationModule",
    
    # Uncertainty Quantification
    "UncertaintyConfig",
    "BayesianLinear",
    "HeteroscedasticLoss",
    "EvidentialOutput",
    "DeepEnsemble",
    "ComprehensiveUncertainty",
    "UncertaintyMetrics",
    "create_uncertainty_config",
    "uncertainty_aware_training_step",
    "monte_carlo_prediction",
    
    # Explainable AI (XAI)
    "XAIConfig",
    "SHAPExplainer",
    "IntegratedGradients",
    "AttentionVisualizer",
    "LRPExplainer",
    "LIMEExplainer",
    "ClinicalInterpretability",
    "ComprehensiveXAI",
    "create_xai_config",
    "explain_hamnet_prediction",
    
    # Integrated Uncertainty-XAI System
    "IntegrationConfig",
    "UncertaintyAwareAttention",
    "ExplainableMultimodalFusion",
    "ConfidenceWeightedPredictor",
    "UncertaintyPropagation",
    "InteractiveVisualization",
    "IntegratedHAMNet",
    "create_integration_config",
    "create_integrated_hamnet"
]