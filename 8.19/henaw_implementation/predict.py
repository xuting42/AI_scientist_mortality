"""
Inference Pipeline for HENAW Model
Fast batch prediction with <100ms latency per individual
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import argparse
from dataclasses import dataclass
import warnings

from henaw_model import HENAWModel, HENAWOutput
from data_loader import FeatureEngineer

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results"""
    participant_id: str
    biological_age: float
    chronological_age: float
    age_gap: float
    mortality_risk: Optional[float] = None
    morbidity_risks: Optional[Dict[str, float]] = None
    biomarker_weights: Optional[np.ndarray] = None
    feature_importance: Optional[np.ndarray] = None
    inference_time_ms: float = 0.0


class HENAWPredictor:
    """
    Fast inference pipeline for HENAW model
    Optimized for <100ms latency per individual
    """
    
    def __init__(self,
                 model_path: str,
                 config_path: str = 'config.yaml',
                 device: str = 'cuda',
                 optimize_model: bool = True):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device for inference
            optimize_model: Whether to optimize model for inference
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device with proper fallback
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Optimize for inference
        if optimize_model:
            self._optimize_model()
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.config)
        self._load_normalizers(model_path)
        
        # Initialize report generator (if available)
        try:
            from evaluate import ClinicalReportGenerator
            self.report_generator = ClinicalReportGenerator(self.config)
        except (ImportError, AttributeError) as e:
            logger.warning(f"ClinicalReportGenerator not available: {e}. Reports will be disabled.")
            self.report_generator = None
        
        # Warm up model
        self._warmup()
        
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint with comprehensive validation"""
        model_path = Path(model_path)
        
        # Validate model file exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not model_path.is_file():
            raise ValueError(f"Model path is not a file: {model_path}")
            
        # Check file size (should be reasonable for a model)
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        if file_size_mb < 0.1:
            raise ValueError(f"Model file too small ({file_size_mb:.1f}MB), likely corrupted")
        
        logger.info(f"Loading model from {model_path} ({file_size_mb:.1f}MB)")
        
        try:
            # Initialize model
            model = HENAWModel(self.config)
            
            # Load checkpoint with validation
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
            except Exception as e:
                raise RuntimeError(f"Failed to load checkpoint file: {e}") from e
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint must be a dictionary")
            
            # Validate architecture compatibility
            if 'config' in checkpoint:
                saved_config = checkpoint['config']
                if 'model' in saved_config:
                    # Check key model parameters match
                    saved_model_config = saved_config['model']
                    current_model_config = self.config['model']
                    
                    critical_params = ['input_dim', 'hidden_dims', 'system_embedding_dim']
                    for param in critical_params:
                        if param in saved_model_config and param in current_model_config:
                            if saved_model_config[param] != current_model_config[param]:
                                logger.warning(f"Model architecture mismatch: {param} - "
                                             f"saved: {saved_model_config[param]}, "
                                             f"current: {current_model_config[param]}")
                                raise ValueError(f"Incompatible model architecture: {param} mismatch")
            
            # Load state dict with proper handling
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                logger.info("Loading from model_state_dict")
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                logger.info("Loading from state_dict")
            else:
                # Assume the entire checkpoint is the state dict
                state_dict = checkpoint
                logger.info("Loading checkpoint as state_dict directly")
            
            # Validate state dict
            if not isinstance(state_dict, dict) or len(state_dict) == 0:
                raise ValueError("Invalid or empty state dict in checkpoint")
            
            # Load with error handling for mismatched architectures
            try:
                # First check parameter counts match approximately
                checkpoint_param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                model_param_count_before = sum(p.numel() for p in model.parameters())
                
                # Allow some tolerance for minor architecture changes
                if abs(checkpoint_param_count - model_param_count_before) / model_param_count_before > 0.1:
                    logger.warning(f"Significant parameter count mismatch: "
                                 f"checkpoint has {checkpoint_param_count}, "
                                 f"model expects {model_param_count_before}")
                
                incompatible_keys = model.load_state_dict(state_dict, strict=False)
                
                if incompatible_keys.missing_keys:
                    # Check if missing keys are critical
                    critical_missing = [k for k in incompatible_keys.missing_keys 
                                      if not any(skip in k for skip in ['num_batches_tracked', 'running_'])]
                    if critical_missing:
                        logger.warning(f"Critical missing keys in checkpoint: {critical_missing}")
                        if len(critical_missing) > 10:  # Too many missing keys
                            raise ValueError("Too many missing keys - likely incompatible architecture")
                
                if incompatible_keys.unexpected_keys:
                    logger.warning(f"Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys[:10]}")
                    
                # Check if critical components loaded successfully
                model_param_count_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
                if model_param_count_after == 0:
                    raise RuntimeError("Model has no trainable parameters after loading checkpoint")
                    
            except Exception as e:
                raise RuntimeError(f"Failed to load model state dict: {e}") from e
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            # Log model info
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model loaded successfully ({param_count:,} total parameters, "
                       f"{trainable_params:,} trainable)")
            
            # Validate model can perform inference
            self._validate_model_inference(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Cannot load model from {model_path}") from e
    
    def _optimize_model(self) -> None:
        """Optimize model for inference"""
        logger.info("Optimizing model for inference...")
        
        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Try to compile model with torch.compile (PyTorch 2.0+)
        try:
            self.model = torch.compile(self.model, mode='max-autotune')
            logger.info("Model compiled with torch.compile")
        except:
            logger.info("torch.compile not available, using standard model")
        
        # Enable cudnn benchmarking for optimal performance
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
    
    def _validate_model_inference(self, model: nn.Module) -> None:
        """Validate that the model can perform inference correctly"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 9, device=self.device)
            dummy_age = torch.tensor([[55.0]], device=self.device)
            
            with torch.no_grad():
                output = model(dummy_input, age=dummy_age)
                
            # Validate output structure
            if not hasattr(output, 'biological_age'):
                raise RuntimeError("Model output missing biological_age attribute")
                
            if output.biological_age is None:
                raise RuntimeError("Model produced None biological_age")
                
            # Check for valid values
            bio_age = output.biological_age.cpu().item()
            if not isinstance(bio_age, (int, float)) or np.isnan(bio_age) or np.isinf(bio_age):
                raise RuntimeError(f"Model produced invalid biological age: {bio_age}")
                
            logger.info(f"Model validation successful - dummy prediction: {bio_age:.1f} years")
            
        except Exception as e:
            raise RuntimeError(f"Model validation failed: {e}") from e
    
    def _load_normalizers(self, model_path: str) -> None:
        """Load feature normalizers from training with error handling"""
        model_path = Path(model_path)
        
        # Try multiple locations for normalizers
        possible_paths = [
            model_path.parent / 'normalizers.pkl',
            model_path.parent / 'feature_normalizers.pkl',
            model_path.with_suffix('.normalizers.pkl'),
        ]
        
        normalizers_loaded = False
        
        for normalizer_path in possible_paths:
            if normalizer_path.exists():
                try:
                    import pickle
                    
                    # Validate file is readable
                    if normalizer_path.stat().st_size == 0:
                        logger.warning(f"Normalizer file is empty: {normalizer_path}")
                        continue
                    
                    with open(normalizer_path, 'rb') as f:
                        normalizers = pickle.load(f)
                    
                    # Validate normalizers structure
                    if not isinstance(normalizers, dict) or len(normalizers) == 0:
                        logger.warning(f"Invalid normalizers format in {normalizer_path}")
                        continue
                    
                    self.feature_engineer.set_normalizers(normalizers)
                    logger.info(f"Loaded feature normalizers from {normalizer_path} ({len(normalizers)} normalizers)")
                    normalizers_loaded = True
                    break
                    
                except Exception as e:
                    logger.warning(f"Failed to load normalizers from {normalizer_path}: {e}")
                    continue
        
        if not normalizers_loaded:
            logger.warning("No valid normalizers found, using default normalization. "
                         "This may affect prediction accuracy.")
    
    def _warmup(self) -> None:
        """Warm up model with dummy predictions"""
        logger.info("Warming up model...")
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 9).to(self.device)
            dummy_age = torch.tensor([[55.0]]).to(self.device)
            
            # Run a few forward passes
            with torch.no_grad():
                for _ in range(5):
                    _ = self.model(dummy_input, dummy_age)
            
            logger.info("Model warmed up")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}. Continuing without warmup.")
    
    def predict_single(self,
                      biomarkers: Union[np.ndarray, Dict[str, float]],
                      chronological_age: float,
                      sex: int,
                      participant_id: Optional[str] = None,
                      return_report: bool = False) -> PredictionResult:
        """
        Predict biological age for a single individual
        
        Args:
            biomarkers: Biomarker values (array or dict)
            chronological_age: Chronological age
            sex: Sex (0: Female, 1: Male)
            participant_id: Optional participant ID
            return_report: Whether to generate clinical report
        
        Returns:
            PredictionResult object
        """
        start_time = time.perf_counter()
        
        # Convert biomarkers to array if dict
        if isinstance(biomarkers, dict):
            biomarker_names = list(self.config['ukbb_fields']['biomarkers'].keys())
            biomarker_array = np.array([biomarkers.get(name, np.nan) for name in biomarker_names])
        else:
            biomarker_array = biomarkers
        
        # Check for missing values
        if np.any(np.isnan(biomarker_array)):
            warnings.warn("Missing biomarker values detected")
        
        # Apply feature engineering
        features = self.feature_engineer.transform(biomarker_array, chronological_age, sex)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        age_tensor = torch.FloatTensor([[chronological_age]]).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(features_tensor, age_tensor, return_intermediates=True)
        
        # Extract results
        biological_age = output.biological_age.squeeze().cpu().item()
        age_gap = biological_age - chronological_age
        
        # Extract additional outputs
        mortality_risk = None
        if output.mortality_risk is not None:
            mortality_risk = output.mortality_risk.squeeze().cpu().item()
        
        morbidity_risks = None
        if output.morbidity_risks is not None:
            morbidity_risks = {
                disease: risk.squeeze().cpu().item()
                for disease, risk in output.morbidity_risks.items()
            }
        
        biomarker_weights = None
        if output.biomarker_weights is not None:
            biomarker_weights = output.biomarker_weights.squeeze().cpu().numpy()
        
        feature_importance = None
        if output.feature_importance is not None:
            feature_importance = output.feature_importance.squeeze().cpu().numpy()
        
        # Calculate inference time
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Create result
        result = PredictionResult(
            participant_id=participant_id or "unknown",
            biological_age=biological_age,
            chronological_age=chronological_age,
            age_gap=age_gap,
            mortality_risk=mortality_risk,
            morbidity_risks=morbidity_risks,
            biomarker_weights=biomarker_weights,
            feature_importance=feature_importance,
            inference_time_ms=inference_time_ms
        )
        
        # Generate clinical report if requested and available
        if return_report and self.report_generator is not None:
            try:
                report = self.report_generator.generate_individual_report(
                    biomarkers=biomarker_array,
                    biological_age=biological_age,
                    chronological_age=chronological_age,
                    mortality_risk=mortality_risk,
                    morbidity_risks=morbidity_risks,
                    feature_importance=feature_importance
                )
                result.clinical_report = report
            except Exception as e:
                logger.warning(f"Failed to generate clinical report: {e}")
                result.clinical_report = None
        elif return_report:
            logger.warning("Clinical report requested but generator not available")
        
        # Check latency requirement
        if inference_time_ms > 100:
            logger.warning(f"Inference time {inference_time_ms:.2f}ms exceeds 100ms target")
        
        return result
    
    def predict_batch(self,
                     data: pd.DataFrame,
                     batch_size: Optional[int] = None,
                     show_progress: bool = True) -> List[PredictionResult]:
        """
        Predict biological age for multiple individuals
        
        Args:
            data: DataFrame with biomarkers, age, and sex columns
            batch_size: Batch size for inference
            show_progress: Whether to show progress bar
        
        Returns:
            List of PredictionResult objects
        """
        if batch_size is None:
            batch_size = self.config['output']['inference']['batch_size']
        
        n_samples = len(data)
        results = []
        
        # Prepare biomarker columns
        biomarker_names = list(self.config['ukbb_fields']['biomarkers'].keys())
        
        # Process in batches
        from tqdm import tqdm
        iterator = range(0, n_samples, batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting", total=len(iterator))
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, n_samples)
            batch_data = data.iloc[start_idx:end_idx]
            
            # Extract batch features
            batch_biomarkers = []
            batch_ages = []
            batch_sexes = []
            batch_ids = []
            
            for idx, row in batch_data.iterrows():
                # Extract biomarkers
                biomarkers = np.array([row.get(name, np.nan) for name in biomarker_names])
                
                # Apply feature engineering
                features = self.feature_engineer.transform(
                    biomarkers,
                    row['age'],
                    row['sex']
                )
                
                batch_biomarkers.append(features)
                batch_ages.append(row['age'])
                batch_sexes.append(row['sex'])
                batch_ids.append(row.get('participant_id', f'participant_{idx}'))
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(batch_biomarkers).to(self.device)
            ages_tensor = torch.FloatTensor(batch_ages).unsqueeze(1).to(self.device)
            
            # Batch prediction
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = self.model(features_tensor, ages_tensor, return_intermediates=True)
            
            batch_time_ms = (time.perf_counter() - start_time) * 1000
            avg_time_ms = batch_time_ms / len(batch_biomarkers)
            
            # Extract results for each individual
            for i in range(len(batch_biomarkers)):
                result = PredictionResult(
                    participant_id=batch_ids[i],
                    biological_age=output.biological_age[i].cpu().item(),
                    chronological_age=batch_ages[i],
                    age_gap=output.biological_age[i].cpu().item() - batch_ages[i],
                    mortality_risk=output.mortality_risk[i].cpu().item() if output.mortality_risk is not None else None,
                    morbidity_risks={
                        disease: risk[i].cpu().item()
                        for disease, risk in output.morbidity_risks.items()
                    } if output.morbidity_risks else None,
                    inference_time_ms=avg_time_ms
                )
                results.append(result)
        
        return results
    
    def predict_from_file(self,
                         input_path: str,
                         output_path: str,
                         file_format: str = 'csv') -> None:
        """
        Predict from file and save results
        
        Args:
            input_path: Path to input file
            output_path: Path to save results
            file_format: Output format ('csv', 'json', 'parquet')
        """
        logger.info(f"Loading data from {input_path}")
        
        # Load data
        if input_path.endswith('.csv'):
            data = pd.read_csv(input_path)
        elif input_path.endswith('.parquet'):
            data = pd.read_parquet(input_path)
        else:
            raise ValueError(f"Unsupported input format: {input_path}")
        
        logger.info(f"Loaded {len(data)} samples")
        
        # Predict
        results = self.predict_batch(data)
        
        # Convert to DataFrame
        results_data = []
        for result in results:
            row = {
                'participant_id': result.participant_id,
                'chronological_age': result.chronological_age,
                'biological_age': result.biological_age,
                'age_gap': result.age_gap,
                'mortality_risk': result.mortality_risk,
                'inference_time_ms': result.inference_time_ms
            }
            
            # Add morbidity risks
            if result.morbidity_risks:
                for disease, risk in result.morbidity_risks.items():
                    row[f'{disease}_risk'] = risk
            
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save results
        logger.info(f"Saving results to {output_path}")
        
        if file_format == 'csv':
            results_df.to_csv(output_path, index=False)
        elif file_format == 'json':
            results_df.to_json(output_path, orient='records', indent=2)
        elif file_format == 'parquet':
            results_df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {file_format}")
        
        # Print summary statistics
        logger.info("\nPrediction Summary:")
        logger.info(f"  Mean biological age: {results_df['biological_age'].mean():.2f}")
        logger.info(f"  Mean age gap: {results_df['age_gap'].mean():.2f}")
        logger.info(f"  Mean inference time: {results_df['inference_time_ms'].mean():.2f}ms")
        
        # Check if meeting latency requirements
        if results_df['inference_time_ms'].max() > 100:
            logger.warning(f"Maximum inference time {results_df['inference_time_ms'].max():.2f}ms exceeds 100ms target")
        else:
            logger.info("All predictions within 100ms latency target")


class RealTimePredictor:
    """
    Real-time prediction server for HENAW model
    Designed for integration with clinical systems
    """
    
    def __init__(self, predictor: HENAWPredictor):
        self.predictor = predictor
        self.prediction_cache = {}
        self.cache_size = 1000
        
    def predict_with_cache(self,
                          biomarkers: Dict[str, float],
                          chronological_age: float,
                          sex: int,
                          participant_id: str) -> PredictionResult:
        """
        Predict with caching for repeated queries
        
        Args:
            biomarkers: Biomarker dictionary
            chronological_age: Age
            sex: Sex
            participant_id: Participant ID
        
        Returns:
            PredictionResult
        """
        # Create cache key
        cache_key = f"{participant_id}_{chronological_age}_{sex}"
        
        # Check cache
        if cache_key in self.prediction_cache:
            logger.info(f"Using cached prediction for {participant_id}")
            return self.prediction_cache[cache_key]
        
        # Predict
        result = self.predictor.predict_single(
            biomarkers,
            chronological_age,
            sex,
            participant_id
        )
        
        # Update cache
        self.prediction_cache[cache_key] = result
        
        # Maintain cache size
        if len(self.prediction_cache) > self.cache_size:
            # Remove oldest entries
            oldest_keys = list(self.prediction_cache.keys())[:100]
            for key in oldest_keys:
                del self.prediction_cache[key]
        
        return result
    
    def start_server(self, host: str = '0.0.0.0', port: int = 8080) -> None:
        """
        Start REST API server for predictions
        
        Args:
            host: Host address
            port: Port number
        """
        from flask import Flask, request, jsonify
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address
        import re
        
        app = Flask(__name__)
        
        # Add rate limiting
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            default_limits=["1000 per hour", "100 per minute"]
        )
        
        @app.route('/predict', methods=['POST'])
        @limiter.limit("10 per minute")  # Rate limit for prediction endpoint
        def predict():
            try:
                # Validate request has JSON data
                if not request.is_json:
                    return jsonify({'error': 'Content-Type must be application/json'}), 400
                
                data = request.json
                
                # Validate required fields
                required_fields = ['biomarkers', 'chronological_age', 'sex']
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
                
                # Validate and sanitize inputs
                try:
                    biomarkers = data['biomarkers']
                    if not isinstance(biomarkers, list):
                        return jsonify({'error': 'biomarkers must be a list'}), 400
                    if len(biomarkers) != 9:  # Expected number of biomarkers
                        return jsonify({'error': f'Expected 9 biomarkers, got {len(biomarkers)}'}), 400
                    
                    # Validate each biomarker is numeric
                    for i, val in enumerate(biomarkers):
                        if not isinstance(val, (int, float)):
                            return jsonify({'error': f'Biomarker at index {i} is not numeric'}), 400
                        if not -1000 <= val <= 10000:  # Reasonable bounds
                            return jsonify({'error': f'Biomarker at index {i} is out of reasonable range'}), 400
                    
                    chronological_age = float(data['chronological_age'])
                    if not 18 <= chronological_age <= 120:
                        return jsonify({'error': f'Age must be between 18 and 120, got {chronological_age}'}), 400
                    
                    sex = int(data['sex'])
                    if sex not in [0, 1]:
                        return jsonify({'error': f'Sex must be 0 (female) or 1 (male), got {sex}'}), 400
                    
                    # Sanitize participant ID (alphanumeric only)
                    participant_id = str(data.get('participant_id', 'unknown'))
                    participant_id = re.sub(r'[^a-zA-Z0-9_-]', '', participant_id)[:50]  # Limit length
                    
                except (ValueError, TypeError) as e:
                    return jsonify({'error': f'Invalid data type: {str(e)}'}), 400
                
                # Predict
                result = self.predict_with_cache(
                    biomarkers,
                    chronological_age,
                    sex,
                    participant_id
                )
                
                # Format response
                response = {
                    'participant_id': result.participant_id,
                    'biological_age': result.biological_age,
                    'age_gap': result.age_gap,
                    'mortality_risk': result.mortality_risk,
                    'morbidity_risks': result.morbidity_risks,
                    'inference_time_ms': result.inference_time_ms
                }
                
                return jsonify(response)
            
            except KeyError as e:
                logger.error(f"Missing key in request: {e}")
                return jsonify({'error': f'Missing required field: {str(e)}'}), 400
            except ValueError as e:
                logger.error(f"Value error in prediction: {e}")
                return jsonify({'error': f'Invalid value: {str(e)}'}), 400
            except Exception as e:
                logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
                # Don't expose internal errors to client
                return jsonify({'error': 'Internal server error'}), 500
        
        @app.route('/health', methods=['GET'])
        @limiter.exempt  # Health check doesn't need rate limiting
        def health():
            # Perform actual health checks
            health_status = {
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'device': str(self.device),
                'cache_size': len(self.prediction_cache)
            }
            
            # Check if model is responsive
            try:
                test_input = torch.randn(1, 9).to(self.device)
                with torch.no_grad():
                    _ = self.model(test_input)
                health_status['model_responsive'] = True
            except Exception as e:
                health_status['model_responsive'] = False
                health_status['status'] = 'degraded'
                logger.warning(f"Model health check failed: {e}")
            
            status_code = 200 if health_status['status'] == 'healthy' else 503
            return jsonify(health_status), status_code
        
        logger.info(f"Starting prediction server on {host}:{port}")
        
        # Import os module for environment checking
        import os
        
        # Use production-ready server based on environment
        environment = os.environ.get('FLASK_ENV', 'development')
        
        if environment == 'production':
            # Try to use a production WSGI server
            try:
                from waitress import serve
                logger.info("Using Waitress WSGI server for production")
                serve(app, host=host, port=port, threads=4, connection_limit=100,
                     cleanup_interval=30, channel_timeout=120)
            except ImportError:
                try:
                    from gunicorn.app.base import BaseApplication
                    
                    class StandaloneApplication(BaseApplication):
                        def __init__(self, app, options=None):
                            self.options = options or {}
                            self.application = app
                            super().__init__()

                        def load_config(self):
                            for key, value in self.options.items():
                                self.cfg.set(key.lower(), value)

                        def load(self):
                            return self.application
                    
                    options = {
                        'bind': f'{host}:{port}',
                        'workers': 1,  # Single worker for model consistency
                        'threads': 4,
                        'timeout': 60,
                        'keepalive': 5,
                        'max_requests': 1000,
                        'max_requests_jitter': 50,
                        'worker_class': 'gthread'
                    }
                    
                    logger.info("Using Gunicorn WSGI server for production")
                    StandaloneApplication(app, options).run()
                    
                except ImportError:
                    logger.warning("No production WSGI server available (waitress, gunicorn). "
                                 "Falling back to Flask dev server - NOT recommended for production!")
                    app.run(host=host, port=port, debug=False, threaded=True, 
                           use_reloader=False, processes=1)
        else:
            # Development server
            logger.info("Using Flask development server")
            app.run(host=host, port=port, debug=True, threaded=True, 
                   use_reloader=False, processes=1)


def main():
    """Main entry point for inference"""
    parser = argparse.ArgumentParser(description='HENAW model inference')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--input', type=str, help='Input data file')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json', 'parquet'],
                       help='Output format')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference')
    parser.add_argument('--server', action='store_true', help='Start prediction server')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = HENAWPredictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    if args.server:
        # Start server
        rt_predictor = RealTimePredictor(predictor)
        rt_predictor.start_server(port=args.port)
    elif args.input and args.output:
        # Batch prediction from file
        predictor.predict_from_file(
            input_path=args.input,
            output_path=args.output,
            file_format=args.format
        )
    else:
        # Demo prediction
        logger.info("Running demo prediction...")
        
        # Create sample data
        sample_biomarkers = {
            'crp': 2.5,
            'hba1c': 38.0,
            'creatinine': 72.0,
            'albumin': 45.0,
            'lymphocyte_pct': 32.0,
            'rdw': 13.5,
            'ggt': 35.0,
            'ast': 25.0,
            'alt': 28.0
        }
        
        result = predictor.predict_single(
            biomarkers=sample_biomarkers,
            chronological_age=55.0,
            sex=0,
            participant_id='demo_001',
            return_report=True
        )
        
        print("\nPrediction Result:")
        print(f"  Participant ID: {result.participant_id}")
        print(f"  Chronological Age: {result.chronological_age:.1f}")
        print(f"  Biological Age: {result.biological_age:.1f}")
        print(f"  Age Gap: {result.age_gap:.2f}")
        print(f"  Mortality Risk: {result.mortality_risk:.3f}" if result.mortality_risk else "")
        print(f"  Inference Time: {result.inference_time_ms:.2f}ms")
        
        if hasattr(result, 'clinical_report'):
            print("\nClinical Report:")
            print(result.clinical_report)


if __name__ == "__main__":
    main()