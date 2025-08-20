# HENAW Implementation Usage Guide

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd /mnt/data3/xuting/ai_scientist/claudeV2/henaw_implementation

# Create virtual environment (recommended)
python3 -m venv henaw_env
source henaw_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your UK Biobank Data

The implementation expects UK Biobank data in CSV format with the following columns (using field IDs):

```python
# Required biomarker columns (UK Biobank Field IDs)
- 30710: CRP (C-reactive protein)
- 30750: HbA1c (Glycated haemoglobin)
- 30700: Creatinine
- 30600: Albumin
- 30180: Lymphocyte percentage
- 30070: RDW (Red cell distribution width)
- 30730: GGT (Gamma glutamyltransferase)
- 30650: AST (Aspartate aminotransferase)
- 30620: ALT (Alanine aminotransferase)

# Additional required columns
- eid: Participant ID
- age: Chronological age
- sex: Sex (0=Female, 1=Male)
- death_date: Date of death (optional, for mortality)
- death_age: Age at death (optional)
```

Example data preparation script:
```python
import pandas as pd

# Load your UK Biobank data
ukbb_data = pd.read_csv('/path/to/your/ukbb_data.csv')

# Select required columns
required_fields = [
    'eid', 'age', 'sex',
    '30710', '30750', '30700', '30600', '30180',
    '30070', '30730', '30650', '30620'
]

# Optional mortality data
if 'death_date' in ukbb_data.columns:
    required_fields.extend(['death_date', 'death_age'])

data = ukbb_data[required_fields]
data.to_csv('henaw_input_data.csv', index=False)
```

## 📊 Training the Model

### Basic Training
```bash
python train_henaw.py \
    --data-path henaw_input_data.csv \
    --output-dir ./outputs \
    --max-epochs 100 \
    --batch-size 256
```

### Advanced Training with All Options
```bash
python train_henaw.py \
    --data-path henaw_input_data.csv \
    --output-dir ./outputs \
    --max-epochs 100 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --device cuda \
    --n-folds 5 \
    --use-mixed-precision \
    --early-stopping-patience 10 \
    --checkpoint-interval 10 \
    --log-interval 50 \
    --use-wandb \
    --wandb-project henaw-ukbb \
    --seed 42
```

### Using the Pipeline Script (Recommended)
```bash
# Full pipeline: train, evaluate, and generate reports
python run_pipeline.py \
    --mode all \
    --data-path henaw_input_data.csv \
    --output-dir ./results \
    --device cuda
```

## 🔍 Model Evaluation

### Evaluate a Trained Model
```bash
python evaluate.py \
    --model-path outputs/checkpoints/best_model.pt \
    --data-path henaw_input_data.csv \
    --output-dir ./evaluation_results \
    --device cuda
```

This will generate:
- Performance metrics (MAE, RMSE, C-statistic, ICC)
- SHAP feature importance plots
- Age gap distribution analysis
- Clinical reports

## 🎯 Making Predictions

### Batch Prediction
```bash
python predict.py \
    --model outputs/checkpoints/best_model.pt \
    --input new_participants.csv \
    --output predictions.csv \
    --device cuda
```

### Single Sample Prediction
```python
from predict import HENAWPredictor

# Load model
predictor = HENAWPredictor('outputs/checkpoints/best_model.pt')

# Prepare sample data
sample = {
    'age': 55,
    'sex': 1,
    '30710': 2.5,  # CRP
    '30750': 5.8,  # HbA1c
    '30700': 85,   # Creatinine
    '30600': 42,   # Albumin
    '30180': 30,   # Lymphocyte %
    '30070': 13.5, # RDW
    '30730': 35,   # GGT
    '30650': 25,   # AST
    '30620': 28    # ALT
}

# Get prediction
result = predictor.predict_single(sample)
print(f"Biological Age: {result['biological_age']:.1f}")
print(f"Age Gap: {result['age_gap']:.1f}")
print(f"95% CI: [{result['ci_lower']:.1f}, {result['ci_upper']:.1f}]")
```

### API Server for Real-time Inference
```bash
# Start the prediction server
python predict.py \
    --model outputs/checkpoints/best_model.pt \
    --server \
    --port 8080 \
    --device cuda
```

Then make requests:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 55,
    "sex": 1,
    "30710": 2.5,
    "30750": 5.8,
    "30700": 85,
    "30600": 42,
    "30180": 30,
    "30070": 13.5,
    "30730": 35,
    "30650": 25,
    "30620": 28
  }'
```

## 📈 Expected Performance

With proper training on UK Biobank data, you should achieve:
- **MAE**: < 5 years
- **C-statistic**: > 0.75
- **ICC**: > 0.85
- **Inference time**: < 100ms per individual

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture parameters
- Training hyperparameters
- Data preprocessing settings
- Output paths

## 💡 Tips

1. **Data Quality**: Ensure your UK Biobank data is properly cleaned. The model handles missing values but complete cases give best results.

2. **GPU Usage**: Training is ~10x faster on GPU. Use `--device cuda` if available.

3. **Checkpointing**: Models are automatically saved during training. Use `--checkpoint-interval` to control frequency.

4. **Cross-validation**: Use `--n-folds 5` for robust evaluation.

5. **Monitoring**: Use `--use-wandb` for real-time training monitoring.

## 🐛 Troubleshooting

### Out of Memory
- Reduce batch size: `--batch-size 128`
- Enable gradient checkpointing: Add to config.yaml
- Use mixed precision: `--use-mixed-precision`

### Slow Training
- Ensure you're using GPU: `--device cuda`
- Enable mixed precision training
- Reduce validation frequency

### Poor Performance
- Check data quality and missing values
- Increase training epochs
- Tune hyperparameters using cross-validation
- Ensure sufficient training data (>50,000 samples recommended)

## 📊 Output Files

After training, you'll find:
```
outputs/
├── checkpoints/
│   ├── best_model.pt          # Best model weights
│   ├── checkpoint_epoch_10.pt # Periodic checkpoints
│   └── ...
├── logs/
│   ├── training_log.txt       # Training progress
│   └── tensorboard/           # TensorBoard logs
├── results/
│   ├── metrics.json           # Performance metrics
│   ├── predictions.csv        # Validation predictions
│   ├── feature_importance.png # SHAP plots
│   └── age_gap_dist.png      # Age gap distribution
└── reports/
    └── clinical_report.pdf    # Clinical interpretation
```

## 🆘 Support

For issues or questions:
1. Check the logs in `outputs/logs/`
2. Verify data format matches requirements
3. Ensure all dependencies are installed
4. Try with a smaller subset of data first