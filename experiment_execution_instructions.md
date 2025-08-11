# Experiment Execution Instructions - Multimodal Biological Age Prediction Model Training

## Objective
Following the model design plan, implement and train multiple candidate models, conduct systematic experimental comparisons to find the optimal multimodal biological age prediction architecture.

## Experimental Environment Setup

### 1. Environment Preparation
**Hardware Requirements Confirmation**:
- GPU memory requirement assessment
- CPU core count and memory usage planning
- Storage space requirement calculation

**Software Environment**:
- Python 3.8+
- PyTorch 2.0+
- Scikit-learn, XGBoost, LightGBM
- Data processing: Pandas, Numpy, Dask (for big data processing)
- Visualization: Matplotlib, Seaborn, Plotly
- Experiment management: MLflow, Weights & Biases (optional)

### 2. Data Preparation Pipeline

#### 2.1 Data Loading and Preprocessing
**Implement Memory-Efficient Data Loader**:
```python
# Pseudocode framework
class UKBiobankDataLoader:
    def __init__(self, data_path, image_path):
        # Implement batch loading for large CSV files
        # Implement lazy loading for images
    
    def get_train_val_test_split(self, test_size=0.2, val_size=0.2):
        # Implement stratified sampling to ensure uniform age distribution
        
    def preprocess_tabular_data(self):
        # Missing value handling, standardization, feature engineering
        
    def preprocess_images(self):
        # Image standardization, data augmentation
```

#### 2.2 Feature Engineering Implementation
**Implement Feature Engineering Based on EDA Results**:
- Missing value imputation strategies
- Outlier handling
- Feature standardization and encoding
- Derived feature creation

#### 2.3 Data Quality Control
**Implement Data Quality Checks**:
- Sample completeness validation
- Data leakage detection
- Age distribution balance check

## Experimental Design Plan

### 3. Baseline Experiments

#### 3.1 Single-Modal Baseline Models
**Tabular Data Baselines**:
- **Linear Regression**: Simple baseline
- **Random Forest**: Traditional ensemble baseline
- **XGBoost**: Gradient boosting baseline
- **ElasticNet**: Regularized linear baseline

**Image Data Baselines**:
- **Pretrained CNN + Linear Regression**: ResNet-50 feature extraction
- **Pretrained ViT + Linear Regression**: ViT feature extraction
- **RetiZero Features + XGBoost**: Ophthalmology-specific pretrained model

#### 3.2 Baseline Experiment Protocol
**Training Configuration**:
- 5-fold cross-validation
- Hyperparameter search range definition
- Early stopping strategy setup
- Performance metric recording

### 4. Multimodal Fusion Experiments

#### 4.1 Early Fusion Experiments
**Experimental Variants**:
- **Simple Concatenation**: Direct feature concatenation
- **Weighted Concatenation**: Inter-modal weight adjustment
- **Post-Selection Concatenation**: Fusion after dimensionality reduction

**Implementation and Training**:
```python
# Experiment 1: Simple Early Fusion
tabular_features = process_tabular_data(data)
image_features = extract_image_features(images, model='resnet50')
fused_features = np.concatenate([tabular_features, image_features], axis=1)
model = XGBoostRegressor()
results = cross_validate(model, fused_features, targets)
```

#### 4.2 Late Fusion Experiments
**Fusion Strategy Testing**:
- **Simple Average**: (pred1 + pred2) / 2
- **Weighted Average**: α*pred1 + (1-α)*pred2
- **Stacking**: Meta-learner fusion
- **Voting Mechanism**: Soft and hard voting

#### 4.3 Deep Learning Multimodal Experiments
**Model Architecture Testing**:
- **MMA-Basic**: Basic multimodal architecture
- **MMA-Attention**: Attention mechanism version
- **MMA-Cross-Modal**: Cross-modal interaction
- **MMA-MTL**: Multi-task learning version

### 5. Systematic Experiment Execution

#### 5.1 Hyperparameter Optimization
**Optimization Strategies**:
- **Grid Search**: For low-parameter models
- **Random Search**: For multi-parameter models
- **Bayesian Optimization**: For deep learning models
- **Optuna**: Efficient hyperparameter search

**Key Hyperparameters**:
- Learning rate scheduling
- Regularization parameters
- Fusion weights
- Network architecture parameters

#### 5.2 Experiment Management
**Experiment Tracking**:
- Configuration file saving for each experiment
- Training log and metric recording
- Model checkpoint management
- Result visualization and analysis

### 6. Special Experimental Design

#### 6.1 Robustness Testing
**Missing Data Experiments**:
- Simulate different proportions of missing data (10%, 25%, 50%)
- Test effects of different imputation strategies
- Evaluate model performance under incomplete data

**Noise Robustness**:
- Add different levels of noise to input data
- Test model stability

#### 6.2 Interpretability Experiments
**Feature Importance Analysis**:
- SHAP value calculation and visualization
- Permutation importance analysis
- Feature ablation experiments

**Biomedical Consistency Validation**:
- Consistency check between model predictions and known biological mechanisms
- Case analysis of anomalous samples

#### 6.3 Generalization Ability Testing
**Cross-Validation Strategies**:
- Time-split validation (if temporal information available)
- Stratified cross-validation
- Hold-out validation

## Specific Execution Plan

### 7. Phase-wise Execution

#### Phase 1: Infrastructure Setup (1-2 days)
- [ ] Data loader implementation and testing
- [ ] Preprocessing pipeline implementation
- [ ] Experimental framework setup
- [ ] Basic evaluation metrics implementation

#### Phase 2: Single-Modal Baselines (2-3 days)
- [ ] Tabular data baseline model training
- [ ] Image data baseline model training
- [ ] Baseline results analysis and recording
- [ ] Best single-modal configuration determination

#### Phase 3: Multimodal Fusion (3-4 days)
- [ ] Early fusion experiments
- [ ] Late fusion experiments
- [ ] Deep learning multimodal experiments
- [ ] Fusion strategy comparison analysis

#### Phase 4: Model Optimization (2-3 days)
- [ ] Hyperparameter tuning for optimal architecture
- [ ] Robustness testing
- [ ] Interpretability analysis
- [ ] Final model training

### 8. Experiment Results Management

#### 8.1 Result Recording Format
**Each Experiment Record**:
```json
{
  "experiment_id": "exp_001_xgb_tabular_baseline",
  "model_type": "XGBoost",
  "data_modality": "tabular",
  "fusion_strategy": "none",
  "hyperparameters": {...},
  "cv_scores": {
    "mae": [2.1, 2.3, 2.0, 2.2, 2.1],
    "r2": [0.72, 0.69, 0.74, 0.71, 0.73],
    "rmse": [3.1, 3.4, 2.9, 3.2, 3.0]
  },
  "training_time": "45 minutes",
  "memory_usage": "2.3 GB",
  "feature_importance": {...},
  "notes": "Best baseline for tabular data"
}
```

#### 8.2 Experiment Comparison Table
**Automatic Result Comparison Generation**:
- Performance metric summary for all models
- Statistical significance test results
- Computational efficiency comparison
- Interpretability scoring

### 9. Code Implementation Requirements

#### 9.1 Core Training Scripts
**Main Script Files**:
- `train_baseline.py`: Baseline model training
- `train_multimodal.py`: Multimodal model training
- `hyperparameter_search.py`: Hyperparameter optimization
- `evaluate_models.py`: Model evaluation script

#### 9.2 Experiment Configuration Management
**Configuration File Structure**:
```yaml
# experiment_config.yaml
data:
  tabular_path: "/mnt/data1/UKBB/"
  image_path: "/mnt/data1/UKBB_retinal_img/"
  target_column: "biological_age"
  
preprocessing:
  missing_value_strategy: "iterative"
  normalization: "standard"
  feature_selection: "shap_based"
  
models:
  baseline:
    - name: "xgboost"
      params: {max_depth: 6, learning_rate: 0.1}
    - name: "random_forest"
      params: {n_estimators: 100, max_depth: 10}
      
  multimodal:
    - name: "early_fusion_xgb"
      fusion_type: "early"
      tabular_model: "xgboost"
      image_model: "resnet50"
```

#### 9.3 Error Handling and Logging
**Implement Comprehensive Error Handling**:
- Recovery mechanism for data loading failures
- Exception capture during training process
- Detailed training log recording
- Recovery capability after experiment interruption

### 10. Performance Monitoring

#### 10.1 Training Process Monitoring
**Real-time Monitoring Metrics**:
- Training and validation loss curves
- Learning rate changes
- Gradient norms (deep learning models)
- Memory usage

#### 10.2 Early Stopping and Checkpoints
**Training Control Strategy**:
- Early stopping based on validation set performance
- Regular model checkpoint saving
- Automatic best model saving
- Training process visualization

### 11. Batch Experiment Execution

#### 11.1 Experiment Queue Management
**Batch Experiment Script**:
```python
# batch_experiments.py
experiments = [
    {"model": "xgboost", "modality": "tabular"},
    {"model": "random_forest", "modality": "tabular"},
    {"model": "resnet50", "modality": "image"},
    {"model": "early_fusion", "modality": "multimodal"},
    # ... more experiment configurations
]

for exp in experiments:
    try:
        run_experiment(exp)
        save_results(exp)
    except Exception as e:
        log_error(exp, e)
        continue
```

#### 11.2 Resource Management
**Computational Resource Optimization**:
- GPU usage scheduling
- Memory usage monitoring
- Parallel experiment execution (if resources allow)
- Experiment priority management

## Output Requirements

### Primary Deliverables:
1. **experiment_results_summary.md**: Summary of all experiment results
2. **model_performance_comparison.html**: Interactive performance comparison report
3. **training_logs/**: Detailed training logs for all experiments
4. **saved_models/**: Trained model files
5. **hyperparameter_optimization_results.json**: Hyperparameter search results

### Code Deliverables:
1. **training_pipeline.py**: Complete training pipeline
2. **data_loader.py**: Data loader implementation
3. **models/**: All model implementation code
4. **utils/**: Utility functions and tools
5. **configs/**: Experiment configuration files

### Analysis Reports:
1. **best_model_analysis.md**: Detailed analysis of best model
2. **fusion_strategy_comparison.md**: Fusion strategy effectiveness comparison
3. **feature_importance_report.html**: Feature importance visualization
4. **robustness_analysis.md**: Robustness testing results

## Success Criteria

### Performance Goals:
- **Accuracy**: Best model R² > 0.7, MAE < 3 years
- **Robustness**: Performance degradation < 15% with 25% missing data
- **Efficiency**: Single training time < 8 hours
- **Reproducibility**: Consistent results under same configuration

### Technical Quality:
- All experiments fully reproducible
- Clear code structure with complete comments
- Experimental results with statistical significance validation
- Complete error handling mechanisms

### Scientific Value:
- Multimodal fusion shows significant improvement over single-modal
- Feature importance consistent with biomedical knowledge
- Model interpretability meets clinical application requirements
- Identify new biological age-related patterns

## Completion Time Estimate
- **Total**: 10-12 days
- **Infrastructure**: 2 days
- **Baseline experiments**: 3 days
- **Multimodal experiments**: 4 days
- **Optimization and analysis**: 3 days

## Risk Mitigation
- **Data issues**: Prepare backup data processing strategies
- **Computational resources**: Implement adaptive model scaling
- **Experiment failures**: Design experiment checkpoints and recovery mechanisms
- **Time overruns**: Prioritize core experiments, secondary experiments optional