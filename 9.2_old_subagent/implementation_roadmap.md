# Implementation Roadmap for Biological Age Algorithms
## Technical Specification and Development Guide

**Version:** 1.0  
**Date:** September 2025  
**Purpose:** Step-by-step implementation guidance for HENAW, MODAL, and METAGE algorithms

---

## Overview

This document provides detailed implementation guidance for three biological age algorithms, including data preprocessing pipelines, model architectures, training procedures, and deployment strategies.

---

## Part 1: HENAW Implementation

### 1.1 Data Preprocessing Pipeline

#### Stage 1: Data Extraction
```python
# Required UK Biobank fields
blood_biomarkers = [
    # Essential Tier 1
    30000,  # White blood cell count
    30010,  # Red blood cell count  
    30020,  # Haemoglobin concentration
    30030,  # Haematocrit percentage
    30040,  # Mean corpuscular volume
    30050,  # Mean corpuscular haemoglobin
    30060,  # Mean corpuscular haemoglobin concentration
    30070,  # Red blood cell distribution width
    30080,  # Platelet count
    30090,  # Platelet crit
    30100,  # Mean platelet volume
    30110,  # Platelet distribution width
    30120,  # Lymphocyte count
    30130,  # Monocyte count
    30140,  # Neutrophil count
    30150,  # Eosinophil count
    30160,  # Basophil count
    30180,  # Lymphocyte percentage
    30190,  # Monocyte percentage
    30200,  # Neutrophil percentage
    30210,  # Eosinophil percentage
    30220,  # Basophil percentage
    30240,  # Reticulocyte percentage
    30250,  # Reticulocyte count
    30260,  # Mean reticulocyte volume
    30270,  # Mean sphered cell volume
    30280,  # Immature reticulocyte fraction
    30290,  # High light scatter reticulocyte percentage
    30300,  # High light scatter reticulocyte count
    
    # Biochemistry markers
    30600,  # Albumin
    30610,  # Alkaline phosphatase
    30620,  # Alanine aminotransferase
    30630,  # Apolipoprotein A
    30640,  # Apolipoprotein B
    30650,  # Aspartate aminotransferase
    30660,  # Direct bilirubin
    30670,  # Urea
    30680,  # Calcium
    30690,  # Cholesterol
    30700,  # Creatinine
    30710,  # C-reactive protein
    30720,  # Cystatin C
    30730,  # Gamma glutamyltransferase
    30740,  # Glucose
    30750,  # HbA1c
    30760,  # HDL cholesterol
    30770,  # IGF-1
    30780,  # LDL direct
    30790,  # Lipoprotein A
    30800,  # Oestradiol
    30810,  # Phosphate
    30820,  # Rheumatoid factor
    30830,  # SHBG
    30840,  # Total bilirubin
    30850,  # Testosterone
    30860,  # Total protein
    30870,  # Triglycerides
    30880,  # Urate
    30890,  # Vitamin D
]

body_measurements = [
    21001,  # Body mass index
    48,     # Waist circumference
    49,     # Hip circumference
    50,     # Standing height
    46,     # Hand grip strength (left)
    47,     # Hand grip strength (right)
    3062,   # Forced vital capacity (FVC)
    3063,   # Forced expiratory volume in 1-second (FEV1)
    4079,   # Diastolic blood pressure (automated)
    4080,   # Systolic blood pressure (automated)
    102,    # Pulse rate (automated)
]
```

#### Stage 2: Data Cleaning
```python
def clean_biomarker_data(df):
    """
    Clean and preprocess biomarker data
    """
    # Handle outliers using IQR method
    for col in biomarker_columns:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        df[col] = df[col].clip(Q1, Q3)
    
    # Log transform skewed variables
    skewed_vars = ['creatinine', 'CRP', 'triglycerides', 'GGT']
    for var in skewed_vars:
        df[var] = np.log1p(df[var])
    
    # Standardization by age and sex
    for age_group in age_groups:
        for sex in [0, 1]:
            mask = (df['age_group'] == age_group) & (df['sex'] == sex)
            df.loc[mask, biomarker_columns] = StandardScaler().fit_transform(
                df.loc[mask, biomarker_columns]
            )
    
    return df
```

#### Stage 3: Missing Data Imputation
```python
def impute_missing_data(df):
    """
    Multiple imputation for missing values
    """
    # Strategy 1: MICE imputation for random missingness
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10),
        max_iter=10,
        random_state=42
    )
    
    # Group by similar biomarker types
    groups = {
        'blood_counts': ['WBC', 'RBC', 'Hgb', 'Hct', 'MCV'],
        'liver': ['ALT', 'AST', 'GGT', 'ALP', 'bilirubin'],
        'kidney': ['creatinine', 'urea', 'cystatin_C'],
        'metabolic': ['glucose', 'HbA1c', 'cholesterol', 'triglycerides']
    }
    
    for group_name, group_vars in groups.items():
        df[group_vars] = imputer.fit_transform(df[group_vars])
    
    return df
```

### 1.2 Hierarchical Component Implementation

#### Component Architecture
```python
class HierarchicalAgeComponent:
    """
    Single hierarchical component for HENAW
    """
    def __init__(self, input_dim, hidden_dims, timescale):
        self.timescale = timescale
        self.layers = self._build_network(input_dim, hidden_dims)
        
    def _build_network(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append({
                'weight': np.random.randn(prev_dim, hidden_dim) * 0.01,
                'bias': np.zeros(hidden_dim),
                'activation': 'relu'
            })
            prev_dim = hidden_dim
            
        # Output layer
        layers.append({
            'weight': np.random.randn(prev_dim, 1) * 0.01,
            'bias': np.zeros(1),
            'activation': 'linear'
        })
        
        return layers
    
    def forward(self, X):
        """
        Forward pass through component
        """
        h = X
        for layer in self.layers[:-1]:
            h = np.dot(h, layer['weight']) + layer['bias']
            h = np.maximum(0, h)  # ReLU activation
            
        # Output layer
        output = np.dot(h, self.layers[-1]['weight']) + self.layers[-1]['bias']
        return output
```

#### Ensemble Integration
```python
class HENAW:
    """
    Hierarchical Ensemble Network for Aging Waves
    """
    def __init__(self):
        # Initialize three hierarchical components
        self.rapid_component = HierarchicalAgeComponent(
            input_dim=8,  # metabolic markers
            hidden_dims=[32, 16],
            timescale='weeks'
        )
        
        self.intermediate_component = HierarchicalAgeComponent(
            input_dim=10,  # organ function markers
            hidden_dims=[64, 32],
            timescale='months'
        )
        
        self.slow_component = HierarchicalAgeComponent(
            input_dim=12,  # structural markers
            hidden_dims=[64, 32, 16],
            timescale='years'
        )
        
        # Ensemble weights (learnable)
        self.ensemble_weights = np.array([0.3, 0.4, 0.3])
        
    def predict(self, X_rapid, X_intermediate, X_slow, chronological_age):
        """
        Generate biological age prediction
        """
        # Get component predictions
        h1 = self.rapid_component.forward(X_rapid)
        h2 = self.intermediate_component.forward(X_intermediate)
        h3 = self.slow_component.forward(X_slow)
        
        # Weighted ensemble
        bio_age = (self.ensemble_weights[0] * h1 + 
                  self.ensemble_weights[1] * h2 + 
                  self.ensemble_weights[2] * h3)
        
        # Add chronological age adjustment
        bio_age = bio_age + 0.85 * chronological_age
        
        return bio_age
    
    def calculate_uncertainty(self, X, n_bootstrap=100):
        """
        Calculate prediction uncertainty using bootstrap
        """
        predictions = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[idx]
            
            # Get prediction
            pred = self.predict(X_boot)
            predictions.append(pred)
        
        # Calculate confidence interval
        lower = np.percentile(predictions, 2.5)
        upper = np.percentile(predictions, 97.5)
        
        return lower, upper
```

### 1.3 Training Implementation

```python
class HENAWTrainer:
    """
    Training pipeline for HENAW
    """
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.lr = learning_rate
        self.loss_history = []
        
    def train_epoch(self, X_train, y_train, batch_size=256):
        """
        Train one epoch
        """
        n_samples = len(X_train)
        indices = np.random.permutation(n_samples)
        
        epoch_loss = 0
        n_batches = 0
        
        for start_idx in range(0, n_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            
            # Get batch data
            X_batch = {
                'rapid': X_train['rapid'][batch_indices],
                'intermediate': X_train['intermediate'][batch_indices],
                'slow': X_train['slow'][batch_indices],
                'chronological_age': X_train['chronological_age'][batch_indices]
            }
            y_batch = y_train[batch_indices]
            
            # Forward pass
            predictions = self.model.predict(**X_batch)
            
            # Calculate loss
            loss = np.mean((predictions - y_batch) ** 2)
            
            # Backward pass (gradient calculation)
            gradients = self.calculate_gradients(X_batch, y_batch, predictions)
            
            # Update weights
            self.update_weights(gradients)
            
            epoch_loss += loss
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        self.loss_history.append(avg_loss)
        
        return avg_loss
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """
        Full training loop
        """
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(X_train, y_train)
            
            # Validation
            val_predictions = self.model.predict(**X_val)
            val_loss = np.mean((val_predictions - y_val) ** 2)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                self.save_model('best_model.pkl')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
```

---

## Part 2: MODAL Implementation

### 2.1 OCT Processing Pipeline

```python
class OCTProcessor:
    """
    Process OCT images for MODAL
    """
    def __init__(self, img_size=(496, 512, 128)):
        self.img_size = img_size
        self.preprocessor = self._build_preprocessor()
        
    def _build_preprocessor(self):
        """
        Build preprocessing pipeline
        """
        transforms = [
            ('denoise', self.denoise_oct),
            ('normalize', self.normalize_intensity),
            ('segment_layers', self.segment_retinal_layers),
            ('extract_features', self.extract_morphological_features)
        ]
        return transforms
    
    def denoise_oct(self, volume):
        """
        Denoise OCT volume using BM3D
        """
        # Apply BM3D denoising to each B-scan
        denoised = np.zeros_like(volume)
        for i in range(volume.shape[2]):
            denoised[:, :, i] = denoise_bm3d(volume[:, :, i])
        return denoised
    
    def segment_retinal_layers(self, volume):
        """
        Segment retinal layers using deep learning
        """
        # Layer boundaries to detect
        layers = [
            'ILM',     # Inner limiting membrane
            'RNFL',    # Retinal nerve fiber layer
            'GCL',     # Ganglion cell layer
            'IPL',     # Inner plexiform layer
            'INL',     # Inner nuclear layer
            'OPL',     # Outer plexiform layer
            'ONL',     # Outer nuclear layer
            'ELM',     # External limiting membrane
            'IS_OS',   # Inner/outer segment junction
            'RPE',     # Retinal pigment epithelium
            'BM'       # Bruch's membrane
        ]
        
        segmentation = {}
        for layer in layers:
            # Placeholder for actual segmentation
            segmentation[layer] = self.segment_layer(volume, layer)
            
        return segmentation
    
    def extract_morphological_features(self, volume, segmentation):
        """
        Extract morphological features from segmented OCT
        """
        features = {}
        
        # Layer thickness measurements
        features['rnfl_thickness'] = self.calculate_thickness(
            segmentation['ILM'], segmentation['RNFL']
        )
        features['gcipl_thickness'] = self.calculate_thickness(
            segmentation['GCL'], segmentation['IPL']
        )
        
        # Texture features
        features['texture'] = self.extract_texture_features(volume)
        
        # Vascular features
        features['vascular'] = self.extract_vascular_features(volume)
        
        return features
```

### 2.2 Cross-Modal Alignment

```python
class CrossModalAligner:
    """
    Align OCT and blood biomarker features
    """
    def __init__(self, oct_dim=2048, blood_dim=256, aligned_dim=128):
        self.oct_encoder = self._build_oct_encoder(oct_dim, aligned_dim)
        self.blood_encoder = self._build_blood_encoder(blood_dim, aligned_dim)
        self.temperature = 0.07
        
    def _build_oct_encoder(self, input_dim, output_dim):
        """
        Build OCT feature encoder
        """
        return {
            'layers': [
                {'dim': [input_dim, 1024], 'activation': 'relu'},
                {'dim': [1024, 512], 'activation': 'relu'},
                {'dim': [512, 256], 'activation': 'relu'},
                {'dim': [256, output_dim], 'activation': 'linear'}
            ]
        }
    
    def _build_blood_encoder(self, input_dim, output_dim):
        """
        Build blood biomarker encoder
        """
        return {
            'layers': [
                {'dim': [input_dim, 128], 'activation': 'relu'},
                {'dim': [128, 128], 'activation': 'relu'},
                {'dim': [128, output_dim], 'activation': 'linear'}
            ]
        }
    
    def contrastive_loss(self, z_oct, z_blood):
        """
        Calculate contrastive loss for alignment
        """
        # Normalize embeddings
        z_oct = z_oct / np.linalg.norm(z_oct, axis=1, keepdims=True)
        z_blood = z_blood / np.linalg.norm(z_blood, axis=1, keepdims=True)
        
        # Calculate similarity matrix
        similarity = np.dot(z_oct, z_blood.T) / self.temperature
        
        # Contrastive loss
        batch_size = z_oct.shape[0]
        labels = np.arange(batch_size)
        
        loss = 0
        for i in range(batch_size):
            pos_sim = similarity[i, i]
            neg_sims = np.concatenate([similarity[i, :i], similarity[i, i+1:]])
            
            loss += -np.log(np.exp(pos_sim) / np.sum(np.exp(neg_sims)))
        
        return loss / batch_size
```

### 2.3 MODAL Architecture

```python
class MODAL:
    """
    Multi-Organ Deep Aging Learner
    """
    def __init__(self):
        # Initialize components
        self.oct_processor = OCTProcessor()
        self.cross_modal_aligner = CrossModalAligner()
        
        # Organ-specific predictors
        self.retinal_predictor = self._build_predictor(128, 'retinal')
        self.metabolic_predictor = self._build_predictor(256, 'metabolic')
        self.cardiovascular_predictor = self._build_predictor(384, 'cardiovascular')
        
        # Fusion weights
        self.fusion_weights = np.array([0.35, 0.35, 0.30])
        
    def _build_predictor(self, input_dim, organ_type):
        """
        Build organ-specific age predictor
        """
        return {
            'type': organ_type,
            'layers': [
                {'dim': [input_dim, 64], 'activation': 'relu'},
                {'dim': [64, 32], 'activation': 'relu'},
                {'dim': [32, 1], 'activation': 'linear'}
            ]
        }
    
    def predict_biological_age(self, oct_image, blood_biomarkers, body_measurements):
        """
        Predict biological age using all modalities
        """
        # Process OCT image
        oct_features = self.oct_processor.process(oct_image)
        
        # Encode features
        z_oct = self.cross_modal_aligner.encode_oct(oct_features)
        z_blood = self.cross_modal_aligner.encode_blood(blood_biomarkers)
        
        # Organ-specific predictions
        ba_retinal = self.retinal_predictor.predict(z_oct)
        ba_metabolic = self.metabolic_predictor.predict(z_blood)
        ba_cardiovascular = self.cardiovascular_predictor.predict(
            np.concatenate([z_blood, body_measurements])
        )
        
        # Fusion
        ba_modal = (self.fusion_weights[0] * ba_retinal +
                   self.fusion_weights[1] * ba_metabolic +
                   self.fusion_weights[2] * ba_cardiovascular)
        
        return {
            'biological_age': ba_modal,
            'retinal_age': ba_retinal,
            'metabolic_age': ba_metabolic,
            'cardiovascular_age': ba_cardiovascular
        }
```

---

## Part 3: METAGE Implementation

### 3.1 NMR Metabolomics Processing

```python
class NMRProcessor:
    """
    Process NMR metabolomics data for METAGE
    """
    def __init__(self):
        self.metabolite_groups = self._define_metabolite_groups()
        
    def _define_metabolite_groups(self):
        """
        Define metabolite groupings
        """
        return {
            'lipoproteins': {
                'VLDL': ['VLDL_L', 'VLDL_M', 'VLDL_S', 'VLDL_XS'],
                'LDL': ['LDL_L', 'LDL_M', 'LDL_S'],
                'HDL': ['HDL_L', 'HDL_M', 'HDL_S']
            },
            'fatty_acids': {
                'saturated': ['SFA', 'MUFA'],
                'unsaturated': ['PUFA', 'Omega3', 'Omega6']
            },
            'amino_acids': ['Val', 'Leu', 'Ile', 'Phe', 'Tyr', 'Ala', 'Gln', 'Gly', 'His'],
            'glycolysis': ['Glucose', 'Lactate', 'Pyruvate', 'Citrate'],
            'ketones': ['Acetoacetate', '3_Hydroxybutyrate', 'Acetone'],
            'inflammation': ['GlycA', 'GlycB']
        }
    
    def preprocess_metabolomics(self, nmr_data):
        """
        Preprocess NMR metabolomics data
        """
        # Log transformation for concentration values
        log_cols = [col for col in nmr_data.columns if '_C' in col]
        nmr_data[log_cols] = np.log1p(nmr_data[log_cols])
        
        # Calculate ratios
        nmr_data['PUFA_SFA_ratio'] = nmr_data['PUFA'] / (nmr_data['SFA'] + 1e-6)
        nmr_data['Omega3_Omega6_ratio'] = nmr_data['Omega3'] / (nmr_data['Omega6'] + 1e-6)
        nmr_data['ApoB_ApoA_ratio'] = nmr_data['ApoB'] / (nmr_data['ApoA'] + 1e-6)
        
        # Normalize by total concentration
        for group in ['VLDL', 'LDL', 'HDL']:
            group_cols = [col for col in nmr_data.columns if group in col]
            total = nmr_data[group_cols].sum(axis=1)
            for col in group_cols:
                nmr_data[f'{col}_normalized'] = nmr_data[col] / (total + 1e-6)
        
        return nmr_data
```

### 3.2 Trajectory Modeling

```python
class TrajectoryModel:
    """
    Model metabolic aging trajectories
    """
    def __init__(self, n_metabolites=168):
        self.n_metabolites = n_metabolites
        self.state_dim = 64
        
        # State transition matrices
        self.A = np.random.randn(self.state_dim, self.state_dim) * 0.01
        self.B = np.random.randn(self.state_dim, n_metabolites) * 0.01
        self.C = np.random.randn(1, self.state_dim) * 0.01
        
    def predict_trajectory(self, metabolite_sequence, time_points):
        """
        Predict aging trajectory from metabolite sequence
        """
        n_timepoints = len(time_points)
        states = np.zeros((n_timepoints, self.state_dim))
        predictions = np.zeros(n_timepoints)
        
        # Initialize state
        states[0] = np.dot(self.B, metabolite_sequence[0])
        
        for t in range(1, n_timepoints):
            # State transition
            states[t] = np.dot(self.A, states[t-1]) + np.dot(self.B, metabolite_sequence[t])
            
            # Add process noise
            states[t] += np.random.normal(0, 0.01, self.state_dim)
        
        # Generate age predictions
        for t in range(n_timepoints):
            predictions[t] = np.dot(self.C, states[t])
        
        return predictions, states
    
    def estimate_aging_rate(self, trajectory, time_window=1.0):
        """
        Estimate instantaneous aging rate
        """
        if len(trajectory) < 2:
            return 1.0
        
        # Calculate derivative
        dt = time_window
        aging_rate = (trajectory[-1] - trajectory[-2]) / dt
        
        return aging_rate
```

### 3.3 METAGE Architecture

```python
class METAGE:
    """
    Metabolomic Trajectory Aging Estimator
    """
    def __init__(self):
        self.nmr_processor = NMRProcessor()
        self.trajectory_model = TrajectoryModel()
        
        # Temporal convolution parameters
        self.temporal_conv = self._build_temporal_conv()
        
        # Intervention response model
        self.intervention_model = self._build_intervention_model()
        
    def _build_temporal_conv(self):
        """
        Build temporal convolutional network
        """
        return {
            'conv_layers': [
                {'kernel_size': 3, 'filters': 64, 'dilation': 1},
                {'kernel_size': 5, 'filters': 128, 'dilation': 2},
                {'kernel_size': 7, 'filters': 256, 'dilation': 4}
            ],
            'pooling': 'global_max'
        }
    
    def _build_intervention_model(self):
        """
        Build intervention response predictor
        """
        return {
            'input_dim': 168 + 10,  # metabolites + intervention features
            'hidden_dims': [128, 64, 32],
            'output_dim': 1  # response score
        }
    
    def compute_biological_age(self, metabolomics_data, time_points=None):
        """
        Compute biological age from metabolomics
        """
        # Preprocess data
        processed_data = self.nmr_processor.preprocess_metabolomics(metabolomics_data)
        
        if time_points is None:
            # Single timepoint prediction
            return self._predict_single_timepoint(processed_data)
        else:
            # Trajectory prediction
            return self._predict_trajectory(processed_data, time_points)
    
    def _predict_single_timepoint(self, data):
        """
        Predict biological age for single timepoint
        """
        # Extract features
        features = data.values.flatten()
        
        # Simple linear model for demonstration
        weights = np.random.randn(len(features)) * 0.01
        bias = 40.0  # average age
        
        biological_age = np.dot(weights, features) + bias
        
        return biological_age
    
    def _predict_trajectory(self, data, time_points):
        """
        Predict aging trajectory
        """
        trajectory, states = self.trajectory_model.predict_trajectory(
            data.values, time_points
        )
        
        # Calculate aging rate
        aging_rate = self.trajectory_model.estimate_aging_rate(trajectory)
        
        # Project future age
        current_age = trajectory[-1]
        projected_10yr = current_age + aging_rate * 10
        
        return {
            'current_biological_age': current_age,
            'aging_rate': aging_rate,
            'projected_10yr_age': projected_10yr,
            'trajectory': trajectory
        }
    
    def predict_intervention_response(self, baseline_metabolomics, intervention_type):
        """
        Predict response to intervention
        """
        # Encode intervention
        intervention_encoding = self._encode_intervention(intervention_type)
        
        # Combine with baseline
        features = np.concatenate([
            baseline_metabolomics.values.flatten(),
            intervention_encoding
        ])
        
        # Predict response score
        response_score = self._compute_response_score(features)
        
        return {
            'response_score': response_score,
            'expected_rate_reduction': response_score * 0.2,  # 20% max reduction
            'confidence': 0.85  # placeholder
        }
```

---

## Part 4: Validation Framework

### 4.1 Cross-Validation Setup

```python
class ValidationFramework:
    """
    Comprehensive validation for all algorithms
    """
    def __init__(self, algorithm_type):
        self.algorithm_type = algorithm_type
        self.metrics = {}
        
    def cross_validate(self, X, y, n_folds=5):
        """
        Perform k-fold cross-validation
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model = self._get_model()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Calculate metrics
            fold_metrics = {
                'mae': mean_absolute_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'correlation': np.corrcoef(y_val, y_pred)[0, 1],
                'r2': r2_score(y_val, y_pred)
            }
            
            fold_results.append(fold_metrics)
        
        # Aggregate results
        self.metrics = {
            metric: {
                'mean': np.mean([f[metric] for f in fold_results]),
                'std': np.std([f[metric] for f in fold_results])
            }
            for metric in fold_metrics.keys()
        }
        
        return self.metrics
    
    def validate_mortality_prediction(self, predictions, mortality_data):
        """
        Validate using mortality outcomes
        """
        # Calculate age acceleration
        age_acceleration = predictions - mortality_data['chronological_age']
        
        # Cox proportional hazards model
        cph = CoxPHFitter()
        survival_df = pd.DataFrame({
            'age_acceleration': age_acceleration,
            'duration': mortality_data['follow_up_time'],
            'event': mortality_data['mortality_event']
        })
        
        cph.fit(survival_df, duration_col='duration', event_col='event')
        
        # Calculate C-index
        c_index = cph.concordance_index_
        
        # Hazard ratio
        hr = np.exp(cph.summary['coef']['age_acceleration'])
        
        return {
            'c_index': c_index,
            'hazard_ratio': hr,
            'p_value': cph.summary['p']['age_acceleration']
        }
```

### 4.2 External Validation

```python
def external_validation(model, external_dataset):
    """
    Validate on external dataset
    """
    results = {}
    
    # Preprocess external data to match training format
    X_external = preprocess_external_data(external_dataset)
    y_external = external_dataset['age']
    
    # Predict
    predictions = model.predict(X_external)
    
    # Calculate metrics
    results['mae'] = mean_absolute_error(y_external, predictions)
    results['correlation'] = np.corrcoef(y_external, predictions)[0, 1]
    
    # Subgroup analysis
    for subgroup in ['sex', 'ethnicity', 'bmi_category']:
        for value in external_dataset[subgroup].unique():
            mask = external_dataset[subgroup] == value
            results[f'{subgroup}_{value}'] = {
                'mae': mean_absolute_error(y_external[mask], predictions[mask]),
                'n': mask.sum()
            }
    
    return results
```

---

## Part 5: Deployment Infrastructure

### 5.1 API Design

```python
class BiologicalAgeAPI:
    """
    REST API for biological age computation
    """
    def __init__(self):
        self.models = {
            'HENAW': load_model('henaw_model.pkl'),
            'MODAL': load_model('modal_model.pkl'),
            'METAGE': load_model('metage_model.pkl')
        }
    
    def predict(self, request):
        """
        Handle prediction request
        """
        # Parse request
        algorithm = request.get('algorithm', 'HENAW')
        data = request.get('data')
        
        # Validate input
        if not self.validate_input(data, algorithm):
            return {'error': 'Invalid input data'}
        
        # Preprocess
        processed_data = self.preprocess(data, algorithm)
        
        # Predict
        model = self.models[algorithm]
        prediction = model.predict(processed_data)
        
        # Generate report
        report = self.generate_report(prediction, algorithm)
        
        return {
            'biological_age': prediction['biological_age'],
            'confidence_interval': prediction['confidence_interval'],
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
```

### 5.2 Clinical Integration

```python
class ClinicalIntegration:
    """
    Integration with clinical systems
    """
    def __init__(self):
        self.hl7_converter = HL7Converter()
        self.fhir_client = FHIRClient()
    
    def process_patient(self, patient_id):
        """
        Process patient through biological age assessment
        """
        # Fetch patient data from EHR
        patient_data = self.fetch_patient_data(patient_id)
        
        # Determine available data and select algorithm
        algorithm = self.select_algorithm(patient_data)
        
        # Compute biological age
        result = self.compute_age(patient_data, algorithm)
        
        # Store in EHR
        self.store_result(patient_id, result)
        
        # Generate clinical report
        report = self.generate_clinical_report(result)
        
        return report
    
    def generate_clinical_report(self, result):
        """
        Generate interpretable clinical report
        """
        report = {
            'summary': self._generate_summary(result),
            'risk_factors': self._identify_risk_factors(result),
            'recommendations': self._generate_recommendations(result),
            'follow_up': self._schedule_follow_up(result)
        }
        
        return report
```

---

## Conclusion

This implementation roadmap provides detailed technical specifications for translating the HENAW, MODAL, and METAGE algorithm designs into working systems. The modular architecture allows for flexible deployment options while maintaining scientific rigor and clinical utility.

Key implementation priorities:
1. Start with HENAW Tier 1 for broad applicability
2. Develop robust data preprocessing pipelines
3. Implement comprehensive validation framework
4. Create clear clinical integration pathways
5. Establish continuous monitoring and updating protocols

The provided code structures serve as implementation blueprints that can be adapted to specific programming languages and deployment environments while maintaining the core algorithmic innovations.