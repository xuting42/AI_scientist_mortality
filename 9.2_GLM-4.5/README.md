# HAMBAE: Hierarchical Adaptive Multi-modal Biological Age Estimation

A comprehensive PyTorch implementation of the HAMBAE algorithm system for biological age estimation using multi-modal data from UK Biobank.

## Overview

HAMBAE is a three-tiered algorithm system for biological age estimation:

- **Tier 1**: Clinical Biomarker Aging Network (CBAN) - Blood biomarker-based estimation
- **Tier 2**: Metabolic Network Aging Integrator (MNAI) - Metabolomics-enhanced estimation  
- **Tier 3**: Multi-Modal Biological Age Transformer (MM-BAT) - Full multi-modal integration

## Features

### Core Capabilities
- **Multi-modal data processing**: Blood biomarkers, metabolomics, retinal imaging, genetic data
- **Uncertainty quantification**: Bayesian neural networks, heteroscedastic uncertainty estimation
- **Explainable AI**: Feature importance, attention mechanisms, biological interpretability
- **Longitudinal modeling**: Aging velocity estimation, trajectory analysis
- **Epigenetic proxy**: Novel blood-based epigenetic age estimation
- **Performance targets**: Tier-specific accuracy benchmarks (MAE ≤ 3.5-5.5 years, R² ≥ 0.75-0.88)

### Technical Features
- **Modular architecture**: Clean separation of concerns with reusable components
- **Production-ready**: Comprehensive testing, logging, monitoring
- **Scalable**: Multi-GPU training, distributed computing support
- **Reproducible**: Deterministic training, version control integration
- **Deployable**: FastAPI serving, batch processing, model versioning

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (for GPU support)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Development Installation
```bash
git clone https://github.com/your-org/hambae.git
cd hambae
pip install -e .
```

## Quick Start

### Training a Model

#### Tier 1: Clinical Biomarker Aging Network
```bash
python hambae/scripts/train_tier1.py \
    --config hambae/config.yaml \
    --data_root data/ukbb \
    --experiment_dir experiments/tier1 \
    --batch_size 128 \
    --max_epochs 100
```

#### Tier 2: Metabolic Network Aging Integrator
```bash
python hambae/scripts/train_tier2.py \
    --config hambae/config.yaml \
    --data_root data/ukbb \
    --experiment_dir experiments/tier2 \
    --batch_size 64 \
    --max_epochs 150
```

#### Tier 3: Multi-Modal Biological Age Transformer
```bash
python hambae/scripts/train_tier3.py \
    --config hambae/config.yaml \
    --data_root data/ukbb \
    --experiment_dir experiments/tier3 \
    --batch_size 32 \
    --max_epochs 200
```

### Evaluating Models
```bash
python hambae/scripts/evaluate.py \
    --config hambae/config.yaml \
    --model_path experiments/tier1/best_model.pt \
    --model_tier 1 \
    --output_dir evaluation_results/tier1 \
    --compute_fairness \
    --save_predictions
```

## Architecture

### Data Processing Pipeline
```
hambae/data/
├── ukbb_loader.py          # UK Biobank data loading
├── preprocessing.py        # Data normalization, imputation
├── feature_engineering.py  # Feature creation, selection
└── quality_control.py     # Data quality assessment
```

### Model Architectures
```
hambae/models/
├── base_model.py           # Base model class
├── tier1_cban.py           # Clinical Biomarker Aging Network
├── tier2_mnai.py           # Metabolic Network Aging Integrator
├── tier3_mmbat.py          # Multi-Modal Biological Age Transformer
├── epigenetic_proxy.py     # Epigenetic proxy development
└── longitudinal_model.py   # Longitudinal aging velocity
```

### Training Infrastructure
```
hambae/training/
├── trainer.py              # Main training class
├── optimization.py        # Optimization strategies
├── validation.py          # Validation procedures
└── callbacks.py           # Training callbacks
```

### Uncertainty Quantification
```
hambae/uncertainty/
├── bayesian_layers.py     # Bayesian neural network layers
├── uncertainty_estimators.py # Uncertainty estimation methods
└── calibration.py         # Uncertainty calibration
```

## Configuration

### Configuration File Structure
```yaml
# Global settings
project_name: "hambae"
experiment_name: "default"
output_dir: "outputs"

# Data configuration
data:
  data_root: "data/ukbb"
  batch_size: 128
  num_workers: 4
  normalization_method: "robust"
  handle_missing: "median"

# Model configurations
tier1:
  hidden_dim: 256
  num_layers: 3
  dropout_rate: 0.1
  enable_epigenetic_proxy: true
  target_mae: 5.5
  target_r2: 0.75

tier2:
  hidden_dim: 256
  metabolomic_count: 400
  enable_pathway_analysis: true
  target_mae: 4.5
  target_r2: 0.82

tier3:
  hidden_dim: 256
  transformer_layers: 6
  retinal_feature_dim: 768
  genetic_feature_dim: 1000
  target_mae: 3.5
  target_r2: 0.88
```

### Customizing Configuration
```python
from hambae.config import HAMBAEConfig, Tier1Config

# Create custom configuration
config = HAMBAEConfig()
config.tier1.hidden_dim = 512
config.tier1.dropout_rate = 0.2
config.data.batch_size = 256
```

## Model Performance

### Expected Performance Targets
| Tier | Model | Input Modalities | Target MAE | Target R² |
|------|-------|-----------------|------------|-----------|
| 1    | CBAN  | Blood biomarkers | ≤ 5.5 years | ≥ 0.75 |
| 2    | MNAI  | Blood + Metabolomics | ≤ 4.5 years | ≥ 0.82 |
| 3    | MM-BAT | Blood + Metabolomics + Retinal + Genetic | ≤ 3.5 years | ≥ 0.88 |

### Uncertainty Quantification
- **Aleatoric uncertainty**: Heteroscedastic noise modeling
- **Epistemic uncertainty**: Bayesian neural networks with Monte Carlo dropout
- **Data quality uncertainty**: Input quality-based uncertainty estimation
- **Calibration**: Temperature scaling, isotonic regression

## Advanced Usage

### Custom Model Training
```python
from hambae.models.tier1_cban import create_cban_model
from hambae.training.trainer import HAMBATrainer
from hambae.data.ukbb_loader import create_data_loaders

# Create model
model = create_cban_model(config.tier1)

# Create data loaders
data_loaders = create_data_loaders(
    data_root="data/ukbb",
    config=config.data,
)

# Create trainer
trainer = HAMBATrainer(
    model=model,
    config=config,
    train_loader=data_loaders['train'],
    val_loader=data_loaders['val'],
    experiment_dir="experiments/custom",
)

# Train model
results = trainer.train(max_epochs=100)
```

### Model Evaluation
```python
from hambae.utils.metrics import compute_metrics, format_metrics_report

# Compute metrics
metrics = compute_metrics(
    predictions=predictions,
    targets=targets,
    uncertainty=uncertainties,
)

# Generate report
report = format_metrics_report(metrics)
print(report)
```

### Uncertainty Quantification
```python
from hambae.uncertainty.bayesian_layers import compute_bayesian_kl_divergence

# Enable uncertainty estimation
model.enable_uncertainty_estimation()

# Make predictions with uncertainty
predictions, uncertainties = model.predict(
    x, 
    return_uncertainty=True,
    n_samples=50
)

# Compute KL divergence for regularization
kl_divergence = compute_bayesian_kl_divergence(model)
```

## Deployment

### Model Serving
```python
from hambae.utils.deployment import create_serving_api, deploy_model

# Create FastAPI serving API
app = create_serving_model(
    model_path="experiments/tier1/best_model.pt",
    model_tier=1,
    config_path="hambae/config.yaml",
)

# Deploy model
deploy_model(
    app=app,
    host="0.0.0.0",
    port=8000,
    workers=4,
)
```

### Batch Processing
```python
from hambae.utils.deployment import BatchProcessor

# Create batch processor
processor = BatchProcessor(
    model_path="experiments/tier1/best_model.pt",
    model_tier=1,
    config_path="hambae/config.yaml",
    batch_size=1000,
)

# Process large dataset
results = processor.process_dataset(
    data_path="data/large_dataset.csv",
    output_path="results/predictions.csv",
)
```

## API Reference

### Core Classes
- `HAMBAEConfig`: Main configuration class
- `ClinicalBiomarkerAgingNetwork`: Tier 1 model
- `MetabolicNetworkAgingIntegrator`: Tier 2 model
- `MultiModalBiologicalAgeTransformer`: Tier 3 model
- `HAMBATrainer`: Main training class

### Key Functions
- `create_data_loaders()`: Create data loaders for training
- `compute_metrics()`: Compute evaluation metrics
- `compute_uncertainty_metrics()`: Compute uncertainty metrics
- `deploy_model()`: Deploy model for serving

## Testing

Run the test suite:
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Write comprehensive unit tests
- Update documentation for new features
- Use type hints for all functions
- Follow the established code patterns

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use HAMBAE in your research, please cite:

```bibtex
@article{hambae2024,
  title={HAMBAE: Hierarchical Adaptive Multi-modal Biological Age Estimation},
  author={Your Name and Others},
  journal={Journal of Computational Biology},
  year={2024},
  volume={1},
  number={1},
  pages={1--15}
}
```

## Acknowledgments

- UK Biobank for providing the multi-modal health data
- Contributors to the open-source libraries used in this project
- Research team members who provided valuable insights and feedback

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Contact the development team

## Changelog

### Version 1.0.0
- Initial release of HAMBAE algorithm system
- Complete implementation of all three tiers
- Comprehensive uncertainty quantification
- Production-ready training and deployment infrastructure