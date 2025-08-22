# Model Design Instructions - Multimodal Biological Age Prediction Framework

## Objective
Based on literature research and data analysis results, design a multimodal machine learning framework that fuses tabular data and fundus images for biological age prediction.

## Design Principles
1. **Biomedical Interpretability First**: Models should provide biologically meaningful explanations
2. **Modular Design**: Each modality branch can be trained and evaluated independently
3. **Practicality**: Use mature Python toolstack, easy to deploy
4. **Robustness**: Handle missing data and noise effectively

## Core Architecture Design

### 1. Overall Framework Architecture
**Design Multimodal Fusion Architecture**:
- **Early Fusion**: Feature-level fusion
- **Late Fusion**: Decision-level fusion  
- **Hybrid Fusion**: Combining early and late fusion
- **Cross-Modal Attention**: Attention mechanism fusion

**Output**: `multimodal_architecture_diagram.png` and detailed architecture description

### 2. Tabular Data Branch Design

#### 2.1 Feature Selection Module
**Select Features Based on Literature and EDA Results**:
- **Core Aging Biomarkers** (Top 20-30 features):
  - Blood biochemistry: CRP, Albumin, HbA1c, Creatinine, etc.
  - Metabolomics: Specific lipid profiles, amino acids
  - Body composition: VAT, Lean mass, BMD
  - Functional indicators: Grip strength, Physical activity
  - Neuroimaging: Brain volume, WMH volume

**Feature Importance Ranking Methods**:
- Correlation-based analysis
- Literature evidence weighting
- SHAP value-based feature selection

#### 2.2 Tabular Data Model Candidates
**Traditional Machine Learning Models**:
- **Ensemble Methods**: XGBoost, Random Forest, LightGBM
- **Linear Models**: ElasticNet (for baseline and interpretability)
- **SVM**: Non-linear kernel functions

**Deep Learning Models**:
- **TabNet**: Deep learning specifically for tabular data
- **Neural Oblivious Decision Trees**: NODE
- **Deep Neural Networks**: Multi-layer perceptron variants

### 3. Fundus Image Branch Design

#### 3.1 Pretrained Model Selection
**Candidate Pretrained Models** (based on literature research):
- **RetiZero**: Specialized fundus image pretrained model
- **RETFound**: Large-scale fundus image pretraining
- **Vision Transformer (ViT)**: General vision pretrained model
- **ResNet-50/101**: Classic CNN architecture

#### 3.2 Feature Extraction Strategy
**Feature Extraction Methods**:
- **Global Features**: Use final layer features from pretrained models
- **Local Features**: Attention region feature extraction
- **Hybrid Features**: Combine global and local information

**Feature Dimensionality Reduction**:
- PCA reduction to appropriate dimensions (128-512 dim)
- Feature selection to remove redundant information

### 4. Multimodal Fusion Strategies

#### 4.1 Early Fusion (Feature-level Fusion)
**Implementation Approach**:
```python
# Pseudocode example
tabular_features = tabular_model.get_features(tabular_data)  # shape: (n, 100)
image_features = image_model.get_features(image_data)        # shape: (n, 256)
fused_features = concat([tabular_features, image_features])  # shape: (n, 356)
prediction = fusion_model(fused_features)
```

#### 4.2 Late Fusion (Decision-level Fusion)
**Fusion Methods**:
- **Weighted Average**: `α * tabular_pred + β * image_pred`
- **Voting Mechanism**: Soft voting of multiple models
- **Stack Ensemble**: Use meta-learner to combine predictions

#### 4.3 Cross-Modal Attention
**Attention Mechanism Design**:
- Attention weights from tabular features to image features
- Attention weights from image features to tabular features
- Adaptive weight adjustment mechanism

### 5. Loss Function Design

#### 5.1 Main Task Loss
**Biological Age Prediction Loss**:
- **MSE Loss**: Standard regression loss
- **MAE Loss**: Robust to outliers
- **Huber Loss**: Combines advantages of MSE and MAE

#### 5.2 Auxiliary Loss
**Multi-task Learning**:
- **Mortality Prediction**: Binary classification auxiliary training
- **Chronic Disease Prediction**: Multi-label classification task
- **Modality Reconstruction Loss**: Self-supervised learning enhancement

#### 5.3 Regularization Strategy
**Prevent Overfitting**:
- L1/L2 regularization
- Dropout mechanism
- Early stopping
- Cross-validation

## Specific Implementation Approach

### 6. Model Implementation Architecture

#### 6.1 PyTorch Implementation Framework
**Core Module Design**:
- `TabularEncoder`: Tabular data encoder
- `ImageEncoder`: Image feature extractor
- `FusionModule`: Multimodal fusion module
- `BiologicalAgePredictor`: Final predictor

#### 6.2 Scikit-learn Baseline
**Traditional Method Implementation**:
- Use sklearn ensemble methods
- Pipelined data preprocessing
- Cross-validation evaluation framework

### 7. Interpretability Design

#### 7.1 Feature Importance Analysis
**Method Selection**:
- **SHAP (SHapley Additive exPlanations)**: Unified feature importance
- **LIME**: Local interpretability
- **Permutation Importance**: Performance degradation-based importance

#### 7.2 Biomedical Interpretation
**Interpretable Output**:
- Each biomarker's contribution to biological age
- Visualization of aging-related regions in fundus images
- Individualized aging risk factor analysis

### 8. Model Variant Design

#### 8.1 Single-Modal Baseline Models
- Optimal model using only tabular data
- Optimal model using only image data
- For comparing multimodal fusion effects

#### 8.2 Multimodal Variants
- **MMA (Multi-Modal Aging)**: Basic fusion architecture
- **MMA-Attention**: Added attention mechanism
- **MMA-MTL**: Multi-task learning version
- **MMA-Robust**: Optimized for missing data

## Output Requirements

### Primary Deliverables:
1. **model_architecture_document.md**: Detailed architecture design document
2. **fusion_strategy_comparison.md**: Theoretical analysis of different fusion strategies
3. **implementation_plan.md**: Specific coding implementation plan
4. **baseline_models_spec.md**: Detailed baseline model specifications
5. **interpretability_framework.md**: Interpretability analysis framework

### Technical Specifications:
1. **pytorch_model_skeleton.py**: PyTorch model framework code
2. **sklearn_baseline.py**: Scikit-learn baseline implementation
3. **data_loader.py**: Multimodal data loader
4. **loss_functions.py**: Custom loss functions
5. **evaluation_metrics.py**: Evaluation metrics implementation

## Design Validation

### Theoretical Validation:
- Biomedical rationality analysis of architecture
- Computational complexity estimation
- Memory requirement assessment

### Expected Performance Goals:
- **Accuracy**: R² > 0.7, MAE < 3 years
- **Robustness**: Performance degradation < 10% with missing data
- **Efficiency**: Training time < 24 hours (single GPU)
- **Interpretability**: SHAP values consistent with known biological mechanisms

## Completion Criteria
- Provide 3-5 different fusion architecture options
- Each architecture has detailed implementation specifications
- Include complete interpretability analysis scheme
- Provide performance expectations and risk assessment
- Form complete technical implementation roadmap表格数据模型候选
**Traditional Machine Learning Models**:
- **Ensemble Methods**: XGBoost, Random Forest, LightGBM
- **Linear Models**: ElasticNet (for baseline and interpretability)
- **SVM**: Non-linear kernel functions

**Deep Learning Models**:
- **TabNet**: Deep learning specifically for tabular data
- **Neural Oblivious Decision Trees**: NODE
- **Deep Neural Networks**: Multi-layer perceptron variants

### 3. Fundus Image Branch Design

#### 3.1 Pretrained Model Selection
**Candidate Pretrained Models** (based on literature research results):
- **RetiZero**: Specialized fundus image pretrained model
- **RETFound**: Large-scale fundus image pretraining
- **Vision Transformer (ViT)**: General vision pretrained model
- **ResNet-50/101**: Classic CNN architecture

#### 3.2 Feature Extraction Strategy
**Feature Extraction Methods**:
- **Global Features**: Use final layer features from pretrained models
- **Local Features**: Attention region feature extraction
- **Hybrid Features**: Combine global and local information

**Feature Dimensionality Reduction**:
- PCA reduction to appropriate dimensions (128-512 dim)
- Feature selection to remove redundant information

### 4. Multimodal Fusion Strategies

#### 4.1 Early Fusion (Feature-level Fusion)
**Implementation Approach**:
```python
# Pseudocode example
tabular_features = tabular_model.get_features(tabular_data)  # shape: (n, 100)
image_features = image_model.get_features(image_data)        # shape: (n, 256)
fused_features = concat([tabular_features, image_features])  # shape: (n, 356)
prediction = fusion_model(fused_features)
```

#### 4.2 Late Fusion (Decision-level Fusion)
**Fusion Methods**:
- **Weighted Average**: `α * tabular_pred + β * image_pred`
- **Voting Mechanism**: Soft voting of multiple models
- **Stack Ensemble**: Use meta-learner to combine predictions

#### 4.3 Cross-Modal Attention
**Attention Mechanism Design**:
- Attention weights from tabular features to image features
- Attention weights from image features to tabular features
- Adaptive weight adjustment mechanism

### 5. Loss Function Design

#### 5.1 Main Task Loss
**Biological Age Prediction Loss**:
- **MSE Loss**: Standard regression loss
- **MAE Loss**: Robust to outliers
- **Huber Loss**: Combines advantages of MSE and MAE

#### 5.2 Auxiliary Loss
**Multi-task Learning**:
- **Mortality Prediction**: Binary classification auxiliary training
- **Chronic Disease Prediction**: Multi-label classification task
- **Modality Reconstruction Loss**: Self-supervised learning enhancement

#### 5.3 Regularization Strategy
**Prevent Overfitting**:
- L1/L2 regularization
- Dropout mechanism
- Early stopping
- Cross-validation

## Specific Implementation Approach

### 6. Model Implementation Architecture

#### 6.1 PyTorch Implementation Framework
**Core Module Design**:
- `TabularEncoder`: Tabular data encoder
- `ImageEncoder`: Image feature extractor
- `FusionModule`: Multimodal fusion module
- `BiologicalAgePredictor`: Final predictor

#### 6.2 Scikit-learn Baseline
**Traditional Method Implementation**:
- Use sklearn ensemble methods
- Pipelined data preprocessing
- Cross-validation evaluation framework

### 7. Interpretability Design

#### 7.1 Feature Importance Analysis
**Method Selection**:
- **SHAP (SHapley Additive exPlanations)**: Unified feature importance
- **LIME**: Local interpretability
- **Permutation Importance**: Performance degradation-based importance

#### 7.2 Biomedical Interpretation
**Interpretable Output**:
- Each biomarker's contribution to biological age
- Visualization of aging-related regions in fundus images
- Individualized aging risk factor analysis

### 8. Model Variant Design

#### 8.1 Single-Modal Baseline Models
- Optimal model using only tabular data
- Optimal model using only image data
- For comparing multimodal fusion effects

#### 8.2 Multimodal Variants
- **MMA (Multi-Modal Aging)**: Basic fusion architecture
- **MMA-Attention**: Added attention mechanism
- **MMA-MTL**: Multi-task learning version
- **MMA-Robust**: Optimized for missing data

## Output Requirements

### Primary Deliverables:
1. **model_architecture_document.md**: Detailed architecture design document
2. **fusion_strategy_comparison.md**: Theoretical analysis of different fusion strategies
3. **implementation_plan.md**: Specific coding implementation plan
4. **baseline_models_spec.md**: Detailed baseline model specifications
5. **interpretability_framework.md**: Interpretability analysis framework

### Technical Specifications:
1. **pytorch_model_skeleton.py**: PyTorch model framework code
2. **sklearn_baseline.py**: Scikit-learn baseline implementation
3. **data_loader.py**: Multimodal data loader
4. **loss_functions.py**: Custom loss functions
5. **evaluation_metrics.py**: Evaluation metrics implementation

## Design Validation

### Theoretical Validation:
- Biomedical rationality analysis of architecture
- Computational complexity estimation
- Memory requirement assessment

### Expected Performance Goals:
- **Accuracy**: R² > 0.7, MAE < 3 years
- **Robustness**: Performance degradation < 10% with missing data
- **Efficiency**: Training time < 24 hours (single GPU)
- **Interpretability**: SHAP values consistent with known biological mechanisms

## Completion Criteria
- Provide 3-5 different fusion architecture options
- Each architecture has detailed implementation specifications
- Include complete interpretability analysis scheme
- Provide performance expectations and risk assessment
- Form complete technical implementation roadmap