"""
Configuration management for biological age algorithms.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import yaml
import json
from pathlib import Path
import torch


@dataclass
class DataConfig:
    """Data configuration for UK Biobank processing."""
    
    # Data paths
    ukbb_data_path: str = "/mnt/data1/UKBB"
    retinal_img_path: str = "/mnt/data1/UKBB_retinal_img/UKB_new_2024"
    output_dir: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Processing parameters
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # Data splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify_by_age: bool = True
    
    # Preprocessing
    normalize_features: bool = True
    imputation_strategy: str = "median"  # "median", "mean", "knn", "mice"
    handle_outliers: bool = True
    outlier_threshold: float = 5.0  # Standard deviations
    
    # Feature selection
    feature_selection_method: Optional[str] = None  # "lasso", "mutual_info", "boruta"
    n_features_to_select: Optional[int] = None


@dataclass
class HENAWConfig:
    """Configuration for HENAW model."""
    
    # Model architecture
    input_features: List[str] = field(default_factory=lambda: [
        "albumin", "alkaline_phosphatase", "creatinine", "c_reactive_protein",
        "glucose", "red_cell_distribution_width", "white_blood_cell_count",
        "systolic_blood_pressure", "diastolic_blood_pressure", "bmi"
    ])
    
    # Hierarchical components
    rapid_window: Tuple[int, int] = (0, 2)  # years
    intermediate_window: Tuple[int, int] = (2, 5)  # years
    slow_window: Tuple[int, int] = (5, 10)  # years
    
    # Ensemble models
    use_ridge: bool = True
    use_random_forest: bool = True
    use_xgboost: bool = True
    
    # Ridge parameters
    ridge_alpha: float = 1.0
    ridge_max_iter: int = 1000
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = 10
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    
    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    
    # Attention mechanism
    attention_dim: int = 128
    attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Uncertainty quantification
    use_monte_carlo_dropout: bool = True
    mc_dropout_rate: float = 0.2
    mc_n_samples: int = 100
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine", "step", "exponential", "plateau"
    warmup_epochs: int = 5


@dataclass
class MODALConfig:
    """Configuration for MODAL model."""
    
    # Input modalities
    use_oct_imaging: bool = True
    use_blood_biomarkers: bool = True
    
    # OCT imaging parameters
    oct_image_size: Tuple[int, int] = (224, 224)
    oct_normalization: str = "imagenet"  # "imagenet", "custom"
    oct_augmentation: bool = True
    
    # Vision Transformer parameters
    vit_patch_size: int = 16
    vit_embed_dim: int = 768
    vit_depth: int = 12
    vit_num_heads: int = 12
    vit_mlp_ratio: float = 4.0
    vit_dropout: float = 0.1
    vit_attention_dropout: float = 0.1
    pretrained_vit: bool = True
    vit_checkpoint: Optional[str] = "vit_base_patch16_224"
    
    # Biomarker MLP parameters
    biomarker_features: List[str] = field(default_factory=lambda: [
        "hemoglobin", "hematocrit", "red_blood_cell_count", "white_blood_cell_count",
        "platelet_count", "lymphocyte_count", "monocyte_count", "neutrophil_count",
        "eosinophil_count", "basophil_count", "albumin", "alkaline_phosphatase",
        "alanine_aminotransferase", "aspartate_aminotransferase", "bilirubin",
        "calcium", "creatinine", "c_reactive_protein", "cystatin_c", "gamma_gt",
        "glucose", "hba1c", "hdl_cholesterol", "ldl_cholesterol", "phosphate",
        "total_protein", "testosterone", "triglycerides", "urate", "urea", "vitamin_d"
    ])
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    mlp_dropout: float = 0.2
    mlp_activation: str = "gelu"  # "relu", "gelu", "silu"
    
    # Contrastive learning
    use_contrastive_learning: bool = True
    contrastive_temperature: float = 0.07
    contrastive_projection_dim: int = 128
    
    # Multi-modal fusion
    fusion_method: str = "cross_attention"  # "concat", "add", "cross_attention", "film"
    fusion_dim: int = 256
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    
    # Organ-specific subscores
    compute_subscores: bool = True
    organ_systems: List[str] = field(default_factory=lambda: [
        "cardiovascular", "metabolic", "hepatic", "renal", "hematologic", "inflammatory"
    ])
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    warmup_steps: int = 1000
    max_epochs: int = 100


@dataclass
class METAGEConfig:
    """Configuration for METAGE model."""
    
    # NMR metabolomics features
    n_metabolomic_features: int = 250
    metabolomic_groups: List[str] = field(default_factory=lambda: [
        "lipids", "lipoproteins", "fatty_acids", "amino_acids",
        "glycolysis", "ketone_bodies", "inflammation", "fluid_balance"
    ])
    
    # LSTM architecture
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 3
    lstm_dropout: float = 0.2
    lstm_bidirectional: bool = True
    
    # Temporal modeling
    sequence_length: int = 5  # Number of time points
    time_encoding: str = "sinusoidal"  # "sinusoidal", "learnable", "relative"
    
    # Trajectory modeling
    use_ode_solver: bool = False  # Use Neural ODE for continuous trajectories
    ode_solver: str = "dopri5"  # "euler", "rk4", "dopri5"
    ode_rtol: float = 1e-3
    ode_atol: float = 1e-4
    
    # Personalized aging rate
    estimate_aging_rate: bool = True
    aging_rate_min: float = 0.5
    aging_rate_max: float = 2.0
    
    # Intervention response prediction
    predict_intervention_response: bool = True
    intervention_types: List[str] = field(default_factory=lambda: [
        "exercise", "diet", "medication", "lifestyle"
    ])
    
    # Attention mechanism
    use_attention: bool = True
    attention_type: str = "self"  # "self", "cross", "multi_head"
    attention_dim: int = 128
    attention_heads: int = 8
    
    # Regularization
    use_variational: bool = True
    kl_weight: float = 1e-3
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    scheduler: str = "plateau"


@dataclass
class TrainingConfig:
    """General training configuration."""
    
    # Training parameters
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer: str = "adamw"  # "adam", "adamw", "sgd", "rmsprop"
    scheduler: str = "cosine"  # "cosine", "step", "exponential", "plateau", "onecycle"
    
    # Mixed precision training
    use_amp: bool = True
    amp_backend: str = "native"  # "native", "apex"
    
    # Distributed training
    use_ddp: bool = False
    ddp_backend: str = "nccl"  # "nccl", "gloo"
    num_nodes: int = 1
    gpus_per_node: int = 1
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    save_top_k: int = 3
    checkpoint_metric: str = "val_loss"
    checkpoint_mode: str = "min"  # "min" or "max"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "ukbb-biological-age"
    use_tensorboard: bool = True
    tensorboard_dir: str = "./tensorboard"
    log_every_n_steps: int = 10
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    
    # Resource management
    num_sanity_val_steps: int = 2
    limit_train_batches: Optional[float] = None
    limit_val_batches: Optional[float] = None
    limit_test_batches: Optional[float] = None
    
    # Cross-validation
    use_cross_validation: bool = True
    n_folds: int = 5
    stratified: bool = True


@dataclass
class InferenceConfig:
    """Inference configuration."""
    
    # Model loading
    checkpoint_path: str = "./checkpoints/best_model.ckpt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Batch processing
    batch_size: int = 512
    num_workers: int = 4
    
    # Caching
    use_cache: bool = True
    cache_ttl: int = 3600  # seconds
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    max_request_size: int = 100  # MB
    
    # Monitoring
    enable_monitoring: bool = True
    prometheus_port: int = 9090
    
    # Output format
    output_format: str = "json"  # "json", "csv", "parquet"
    include_uncertainty: bool = True
    include_subscores: bool = True


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.data_config = DataConfig()
        self.henaw_config = HENAWConfig()
        self.modal_config = MODALConfig()
        self.metage_config = METAGEConfig()
        self.training_config = TrainingConfig()
        self.inference_config = InferenceConfig()
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        with open(self.config_path, 'r') as f:
            if self.config_path.suffix == '.yaml':
                config_dict = yaml.safe_load(f)
            elif self.config_path.suffix == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
        
        # Update configurations
        self._update_dataclass(self.data_config, config_dict.get('data', {}))
        self._update_dataclass(self.henaw_config, config_dict.get('henaw', {}))
        self._update_dataclass(self.modal_config, config_dict.get('modal', {}))
        self._update_dataclass(self.metage_config, config_dict.get('metage', {}))
        self._update_dataclass(self.training_config, config_dict.get('training', {}))
        self._update_dataclass(self.inference_config, config_dict.get('inference', {}))
    
    def _update_dataclass(self, obj: Any, updates: Dict) -> None:
        """Update dataclass fields from dictionary."""
        for key, value in updates.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def save_config(self, path: str) -> None:
        """Save configuration to file."""
        config_dict = {
            'data': self.data_config.__dict__,
            'henaw': self.henaw_config.__dict__,
            'modal': self.modal_config.__dict__,
            'metage': self.metage_config.__dict__,
            'training': self.training_config.__dict__,
            'inference': self.inference_config.__dict__
        }
        
        path = Path(path)
        with open(path, 'w') as f:
            if path.suffix == '.yaml':
                yaml.dump(config_dict, f, default_flow_style=False)
            elif path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def get_model_config(self, model_name: str) -> Any:
        """Get configuration for specific model."""
        model_configs = {
            'henaw': self.henaw_config,
            'modal': self.modal_config,
            'metage': self.metage_config
        }
        return model_configs.get(model_name.lower())
    
    def validate(self) -> bool:
        """Validate configuration consistency."""
        # Check data paths exist
        if not Path(self.data_config.ukbb_data_path).exists():
            print(f"Warning: UK Biobank data path does not exist: {self.data_config.ukbb_data_path}")
        
        # Check batch size is reasonable
        if self.data_config.batch_size > 1024:
            print(f"Warning: Large batch size ({self.data_config.batch_size}) may cause OOM")
        
        # Check GPU availability for distributed training
        if self.training_config.use_ddp and not torch.cuda.is_available():
            print("Warning: DDP enabled but no GPU available")
            return False
        
        return True