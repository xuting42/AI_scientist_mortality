"""
Configuration management for HAMBAE algorithm system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import yaml
import os
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data processing and loading."""
    data_root: str = "data/ukbb"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # UKBB specific settings
    ukbb_fields: Dict[str, List[int]] = field(default_factory=lambda: {
        "blood_biomarkers": [30000, 30010, 30020, 30030, 30040, 30050, 30060, 30070, 30080, 30090, 30100, 30110, 30120],
        "metabolomics": [23400, 23401, 23402, 23403, 23404, 23405, 23406, 23407, 23408, 23409],
        "retinal": [22400, 22401, 22402],
        "genetic": [22000, 22001, 22002, 22003, 22004, 22005]
    })
    
    # Preprocessing settings
    normalization_method: str = "robust"  # robust, standard, minmax
    handle_missing: str = "median"  # median, mean, drop
    outlier_method: str = "iqr"  # iqr, zscore, none
    outlier_threshold: float = 3.0
    
    # Feature engineering
    create_ratios: bool = True
    create_interactions: bool = True
    temporal_features: bool = True
    
    # Quality control
    min_data_quality: float = 0.7
    max_missing_rate: float = 0.3
    correct_batch_effects: bool = True


@dataclass
class ModelConfig:
    """Base model configuration."""
    hidden_dim: int = 256
    num_layers: int = 3
    dropout_rate: float = 0.1
    activation: str = "gelu"  # relu, gelu, swish
    use_batch_norm: bool = True
    use_layer_norm: bool = True
    
    # Uncertainty quantification
    enable_uncertainty: bool = True
    mc_dropout: bool = True
    mc_samples: int = 50
    heteroscedastic: bool = True
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adam, adamw, sgd
    scheduler: str = "cosine"  # cosine, linear, exponential
    warmup_steps: int = 1000
    
    # Regularization
    l1_lambda: float = 0.0
    l2_lambda: float = 1e-4
    early_stopping_patience: int = 20
    
    # Training
    max_epochs: int = 100
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class Tier1Config(ModelConfig):
    """Tier 1: Clinical Biomarker Aging Network configuration."""
    # Input features
    blood_biomarker_count: int = 13
    
    # Epigenetic proxy
    enable_epigenetic_proxy: bool = True
    epigenetic_hidden_dim: int = 128
    epigenetic_layers: int = 2
    
    # Explainable components
    enable_explainability: bool = True
    feature_importance_method: str = "shap"  # shap, lime, integrated_gradients
    
    # Multi-task learning
    multi_task: bool = True
    mortality_weight: float = 0.3
    
    # Biological constraints
    enforce_monotonicity: bool = True
    monotonicity_penalty: float = 0.1
    
    # Expected performance targets
    target_mae: float = 5.5
    target_r2: float = 0.75


@dataclass
class Tier2Config(ModelConfig):
    """Tier 2: Metabolic Network Aging Integrator configuration."""
    # Metabolomics features
    metabolomic_count: int = 400
    
    # Graph neural network
    gnn_hidden_dim: int = 256
    gnn_layers: int = 3
    gnn_attention_heads: int = 8
    
    # Pathway analysis
    enable_pathway_analysis: bool = True
    pathway_database: str = "kegg"  # kegg, reactome
    pathway_hidden_dim: int = 128
    
    # Cross-modal attention
    cross_modal_attention: bool = True
    attention_dim: int = 256
    attention_heads: int = 8
    
    # Integration with Tier 1
    use_tier1_features: bool = True
    tier1_feature_weight: float = 0.5
    
    # Expected performance targets
    target_mae: float = 4.5
    target_r2: float = 0.82


@dataclass
class Tier3Config(ModelConfig):
    """Tier 3: Multi-Modal Biological Age Transformer configuration."""
    # Multi-modal features
    retinal_feature_dim: int = 768  # RETFound features
    genetic_feature_dim: int = 1000  # PRS + PCs
    
    # Transformer architecture
    transformer_layers: int = 6
    transformer_heads: int = 12
    transformer_dim: int = 768
    transformer_dropout: float = 0.1
    
    # Retinal processing
    retinal_model: str = "retfound"  # retfound, resnet, efficientnet
    retinal_pretrained: bool = True
    
    # Multi-modal attention
    modal_attention: str = "cross"  # cross, self, co
    modal_fusion: str = "attention"  # attention, concatenation, gating
    
    # Longitudinal modeling
    enable_longitudinal: bool = True
    longitudinal_window: int = 5  # years
    aging_velocity: bool = True
    
    # Organ-specific aging
    organ_specific: bool = True
    organs: List[str] = field(default_factory=lambda: ["brain", "heart", "liver", "kidney", "lung"])
    
    # Expected performance targets
    target_mae: float = 3.5
    target_r2: float = 0.88


@dataclass
class UncertaintyConfig:
    """Uncertainty quantification configuration."""
    # Bayesian methods
    enable_bayesian: bool = True
    prior_scale: float = 1.0
    posterior_samples: int = 100
    
    # Aleatoric uncertainty
    heteroscedastic_loss: bool = True
    uncertainty_weight: float = 0.1
    
    # Calibration
    enable_calibration: bool = True
    calibration_method: str = "temperature"  # temperature, isotonic, beta
    temperature_init: float = 1.0
    
    # Data quality uncertainty
    data_quality_weight: float = 0.05
    missing_data_penalty: float = 0.1
    
    # Ensemble methods
    ensemble_size: int = 5
    ensemble_method: str = "deep"  # deep, bagging, boosting


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Hardware
    device: str = "auto"  # auto, cpu, cuda
    num_gpus: int = 1
    distributed: bool = False
    backend: str = "nccl"  # nccl, gloo
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_freq: int = 10
    save_best_only: bool = True
    load_checkpoint: Optional[str] = None
    
    # Logging
    log_dir: str = "logs"
    log_freq: int = 10
    enable_wandb: bool = False
    wandb_project: str = "hambae"
    
    # Validation
    val_freq: int = 5
    val_split: float = 0.2
    test_split: float = 0.1
    stratified_split: bool = True
    
    # Cross-validation
    enable_cv: bool = False
    cv_folds: int = 5
    cv_method: str = "stratified"  # stratified, kfold, group
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    # Model serving
    serve_port: int = 8000
    serve_host: str = "0.0.0.0"
    max_batch_size: int = 32
    timeout: int = 30
    
    # API settings
    api_version: str = "v1"
    enable_docs: bool = True
    enable_cors: bool = True
    
    # Monitoring
    enable_monitoring: bool = True
    metrics_port: int = 8001
    health_check_interval: int = 30
    
    # Versioning
    model_version: str = "latest"
    enable_versioning: bool = True
    
    # Scaling
    max_workers: int = 4
    worker_timeout: int = 120


@dataclass
class HAMBAEConfig:
    """Main HAMBAE configuration."""
    # Sub-configurations
    data: DataConfig = field(default_factory=DataConfig)
    tier1: Tier1Config = field(default_factory=Tier1Config)
    tier2: Tier2Config = field(default_factory=Tier2Config)
    tier3: Tier3Config = field(default_factory=Tier3Config)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    
    # Global settings
    project_name: str = "hambae"
    experiment_name: str = "default"
    output_dir: str = "outputs"
    
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = False
    verbose: bool = True
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> 'HAMBAEConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    def to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = self.to_dict()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HAMBAEConfig':
        """Create configuration from dictionary."""
        # This is a simplified version - in practice, you'd need more sophisticated
        # deserialization logic for nested dataclasses
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'data': self.data.__dict__,
            'tier1': self.tier1.__dict__,
            'tier2': self.tier2.__dict__,
            'tier3': self.tier3.__dict__,
            'uncertainty': self.uncertainty.__dict__,
            'training': self.training.__dict__,
            'deployment': self.deployment.__dict__,
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'environment': self.environment,
            'debug': self.debug,
            'verbose': self.verbose
        }
    
    def update_from_env(self) -> None:
        """Update configuration from environment variables."""
        # This method would read environment variables and update the config
        # Implementation depends on specific environment variable naming convention
        pass
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        # Add validation logic here
        if self.tier1.blood_biomarker_count <= 0:
            raise ValueError("Blood biomarker count must be positive")
        
        if self.tier2.metabolomic_count <= 0:
            raise ValueError("Metabolomic count must be positive")
        
        if self.tier3.retinal_feature_dim <= 0 or self.tier3.genetic_feature_dim <= 0:
            raise ValueError("Retinal and genetic feature dimensions must be positive")
        
        if self.data.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        if self.training.num_gpus < 0:
            raise ValueError("Number of GPUs must be non-negative")


# Type aliases for backward compatibility
TierConfig = Union[Tier1Config, Tier2Config, Tier3Config]


def load_config(config_path: Union[str, Path]) -> HAMBAEConfig:
    """Load HAMBAE configuration from file."""
    return HAMBAEConfig.from_yaml(config_path)


def save_config(config: HAMBAEConfig, config_path: Union[str, Path]) -> None:
    """Save HAMBAE configuration to file."""
    config.to_yaml(config_path)


def get_default_config() -> HAMBAEConfig:
    """Get default HAMBAE configuration."""
    return HAMBAEConfig()