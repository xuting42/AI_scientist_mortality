#!/usr/bin/env python3
"""
Production Readiness Test Suite for HENAW Implementation
Tests all critical fixes and ensures production deployment readiness
"""

import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
import traceback
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionReadinessTests:
    """Comprehensive test suite for production readiness"""
    
    def __init__(self):
        self.results = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        
        # Load config
        config_path = Path('config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            logger.warning("config.yaml not found, using defaults")
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for testing"""
        return {
            'ukbb_fields': {
                'biomarkers': {
                    'crp': 30710,
                    'hba1c': 30750,
                    'creatinine': 30700,
                    'albumin': 30600,
                    'lymphocyte_pct': 30180,
                    'rdw': 30070,
                    'ggt': 30730,
                    'ast': 30650,
                    'alt': 30620
                }
            },
            'model': {
                'n_biomarkers': 9,
                'hidden_dims': [128, 64, 32],
                'n_systems': 5,
                'system_embedding_dim': 16
            },
            'feature_engineering': {
                'transformations': {'type': 'log_distance', 'clip_outliers': 5},
                'age_window': 5
            },
            'output': {
                'inference': {'batch_size': 32}
            }
        }
    
    def run_all_tests(self) -> None:
        """Run all production readiness tests"""
        print("\n" + "="*70)
        print("HENAW PRODUCTION READINESS TEST SUITE")
        print("="*70 + "\n")
        
        # Test 1: Import and class availability
        self.test_imports_and_classes()
        
        # Test 2: Device selection and CUDA fallback
        self.test_device_selection()
        
        # Test 3: Data loading with error handling
        self.test_data_loading()
        
        # Test 4: Feature engineering with division by zero protection
        self.test_feature_engineering()
        
        # Test 5: Checkpoint loading with error handling
        self.test_checkpoint_loading()
        
        # Test 6: Model inference robustness
        self.test_model_inference()
        
        # Test 7: Real UK Biobank data compatibility
        self.test_ukbb_data_compatibility()
        
        # Test 8: Production server deployment
        self.test_production_server()
        
        # Print summary
        self.print_summary()
    
    def test_imports_and_classes(self) -> None:
        """Test that all required imports and classes are available"""
        test_name = "Import and Class Availability"
        logger.info(f"Testing: {test_name}")
        
        try:
            # Test core imports
            from henaw_model import HENAWModel, HENAWOutput
            from data_loader import FeatureEngineer, UKBBDataset
            from predict import HENAWPredictor
            from ukbb_data_loader import UKBBRealDataLoader
            
            # Test optional imports (should not fail)
            try:
                from evaluate import ClinicalReportGenerator, InterpretabilityAnalyzer
                self.log_result(test_name, "PASS", "All classes available including optional ones")
            except ImportError as e:
                self.log_result(test_name, "WARN", f"Optional classes not fully available: {e}")
                
        except ImportError as e:
            self.log_result(test_name, "FAIL", f"Required imports failed: {e}")
        except Exception as e:
            self.log_result(test_name, "FAIL", f"Unexpected error: {e}")
    
    def test_device_selection(self) -> None:
        """Test automatic CUDA fallback to CPU"""
        test_name = "Device Selection and Fallback"
        logger.info(f"Testing: {test_name}")
        
        try:
            from predict import HENAWPredictor
            
            # Test CUDA fallback
            cuda_available = torch.cuda.is_available()
            
            # Create test checkpoint if needed
            test_checkpoint = Path('test_checkpoint.pt')
            if not test_checkpoint.exists():
                from henaw_model import HENAWModel
                model = HENAWModel(self.config)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': 0
                }, test_checkpoint)
            
            # Test with CUDA request
            predictor = HENAWPredictor(
                model_path=str(test_checkpoint),
                config_path='config.yaml' if Path('config.yaml').exists() else None,
                device='cuda',
                optimize_model=False
            )
            
            # Check device assignment
            if cuda_available:
                assert predictor.device.type == 'cuda', "Should use CUDA when available"
            else:
                assert predictor.device.type == 'cpu', "Should fallback to CPU when CUDA unavailable"
            
            self.log_result(test_name, "PASS", f"Device correctly set to {predictor.device}")
            
        except Exception as e:
            self.log_result(test_name, "FAIL", f"Device selection failed: {e}")
    
    def test_data_loading(self) -> None:
        """Test data loading with comprehensive error handling"""
        test_name = "Data Loading with Error Handling"
        logger.info(f"Testing: {test_name}")
        
        try:
            from ukbb_data_loader import UKBBRealDataLoader
            
            loader = UKBBRealDataLoader(self.config)
            
            # Test loading with missing file (should generate synthetic)
            df = loader.load_ukbb_csv(csv_path='nonexistent.csv', max_samples=100)
            
            assert df is not None, "Should return synthetic data when file missing"
            assert len(df) > 0, "Should have samples"
            assert 'age' in df.columns, "Should have age column"
            assert 'sex' in df.columns, "Should have sex column"
            
            # Test with existing synthetic data file if available
            synthetic_file = Path('synthetic_ukbb_data.csv')
            if synthetic_file.exists():
                df2 = loader.load_ukbb_csv(csv_path=str(synthetic_file), max_samples=100)
                assert df2 is not None, "Should load existing file"
            
            self.log_result(test_name, "PASS", "Data loading handles missing files correctly")
            
        except Exception as e:
            self.log_result(test_name, "FAIL", f"Data loading error: {e}")
    
    def test_feature_engineering(self) -> None:
        """Test feature engineering with division by zero protection"""
        test_name = "Feature Engineering Safety"
        logger.info(f"Testing: {test_name}")
        
        try:
            from data_loader import FeatureEngineer
            
            engineer = FeatureEngineer(self.config)
            
            # Test with normal values
            normal_biomarkers = np.array([2.0, 36.0, 70.0, 45.0, 30.0, 13.0, 40.0, 25.0, 30.0])
            features1 = engineer.transform(normal_biomarkers, age=55, sex=0)
            assert not np.any(np.isnan(features1)), "Should handle normal values"
            
            # Test with all zeros (division by zero risk)
            zero_biomarkers = np.zeros(9)
            features2 = engineer.transform(zero_biomarkers, age=55, sex=0)
            assert not np.any(np.isnan(features2)), "Should handle zero variance"
            assert not np.any(np.isinf(features2)), "Should not produce infinities"
            
            # Test with NaN values
            nan_biomarkers = np.full(9, np.nan)
            features3 = engineer.transform(nan_biomarkers, age=55, sex=0)
            assert np.all(np.isfinite(features3)), "Should handle NaN inputs"
            
            # Test with extreme values
            extreme_biomarkers = np.array([1e10, -1e10, 1e-10, 0, np.inf, -np.inf, 100, 200, 300])
            features4 = engineer.transform(extreme_biomarkers, age=55, sex=0)
            assert np.all(np.isfinite(features4)), "Should handle extreme values"
            
            self.log_result(test_name, "PASS", "Feature engineering handles edge cases safely")
            
        except Exception as e:
            self.log_result(test_name, "FAIL", f"Feature engineering error: {e}")
    
    def test_checkpoint_loading(self) -> None:
        """Test checkpoint loading with error handling"""
        test_name = "Checkpoint Loading Robustness"
        logger.info(f"Testing: {test_name}")
        
        try:
            from predict import HENAWPredictor
            from henaw_model import HENAWModel
            
            # Create valid checkpoint
            valid_checkpoint = Path('test_valid_checkpoint.pt')
            model = HENAWModel(self.config)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': 10,
                'best_val_metric': 0.95
            }, valid_checkpoint)
            
            # Test loading valid checkpoint
            try:
                predictor = HENAWPredictor(
                    model_path=str(valid_checkpoint),
                    device='cpu',
                    optimize_model=False
                )
                self.log_result(test_name + " - Valid", "PASS", "Loaded valid checkpoint")
            except Exception as e:
                self.log_result(test_name + " - Valid", "FAIL", f"Failed on valid checkpoint: {e}")
            
            # Test loading non-existent checkpoint
            try:
                predictor = HENAWPredictor(
                    model_path='nonexistent_checkpoint.pt',
                    device='cpu',
                    optimize_model=False
                )
                self.log_result(test_name + " - Missing", "FAIL", "Should fail on missing checkpoint")
            except FileNotFoundError:
                self.log_result(test_name + " - Missing", "PASS", "Correctly handles missing checkpoint")
            except Exception as e:
                self.log_result(test_name + " - Missing", "WARN", f"Unexpected error type: {e}")
            
            # Clean up
            valid_checkpoint.unlink(missing_ok=True)
            
        except Exception as e:
            self.log_result(test_name, "FAIL", f"Checkpoint test error: {e}")
    
    def test_model_inference(self) -> None:
        """Test model inference robustness"""
        test_name = "Model Inference Robustness"
        logger.info(f"Testing: {test_name}")
        
        try:
            from henaw_model import HENAWModel
            
            model = HENAWModel(self.config)
            model.eval()
            
            # Test normal inference
            normal_input = torch.randn(1, 9)
            age = torch.tensor([[55.0]])
            
            with torch.no_grad():
                output = model(normal_input, age)
            
            assert output.biological_age is not None, "Should produce biological age"
            assert torch.isfinite(output.biological_age).all(), "Should produce finite values"
            
            # Test batch inference
            batch_input = torch.randn(32, 9)
            batch_ages = torch.randn(32, 1) * 10 + 55
            
            with torch.no_grad():
                batch_output = model(batch_input, batch_ages)
            
            assert batch_output.biological_age.shape[0] == 32, "Should handle batch inference"
            
            self.log_result(test_name, "PASS", "Model inference is robust")
            
        except Exception as e:
            self.log_result(test_name, "FAIL", f"Inference error: {e}")
    
    def test_ukbb_data_compatibility(self) -> None:
        """Test compatibility with real UK Biobank data formats"""
        test_name = "UK Biobank Data Compatibility"
        logger.info(f"Testing: {test_name}")
        
        try:
            from ukbb_data_loader import UKBBRealDataLoader
            
            loader = UKBBRealDataLoader(self.config)
            
            # Create test UK Biobank-like CSV
            test_data = {
                'eid': [1000001, 1000002, 1000003],
                '30710-0.0': [2.5, 3.0, 1.8],  # CRP
                '30750-0.0': [38, 42, 35],      # HbA1c
                '30700-0.0': [72, 85, 68],      # Creatinine
                '21022-0.0': [55, 62, 48],      # Age
                '31-0.0': [0, 1, 0]              # Sex
            }
            
            test_df = pd.DataFrame(test_data)
            test_csv = Path('test_ukbb_format.csv')
            test_df.to_csv(test_csv, index=False)
            
            # Load and process
            loaded_df = loader.load_ukbb_csv(test_csv, required_complete=False)
            
            assert 'eid' in loaded_df.columns, "Should preserve participant ID"
            assert 'age' in loaded_df.columns, "Should extract age"
            assert 'sex' in loaded_df.columns, "Should extract sex"
            assert 'crp' in loaded_df.columns, "Should map biomarker fields"
            
            # Clean up
            test_csv.unlink(missing_ok=True)
            
            self.log_result(test_name, "PASS", "Handles UK Biobank data format correctly")
            
        except Exception as e:
            self.log_result(test_name, "FAIL", f"UK Biobank compatibility error: {e}")
    
    def test_production_server(self) -> None:
        """Test production server deployment readiness"""
        test_name = "Production Server Deployment"
        logger.info(f"Testing: {test_name}")
        
        try:
            from predict import RealTimePredictor, HENAWPredictor
            
            # Check for production server dependencies
            servers_available = []
            
            try:
                import waitress
                servers_available.append('waitress')
            except ImportError:
                pass
            
            try:
                import gunicorn
                servers_available.append('gunicorn')
            except ImportError:
                pass
            
            try:
                import flask
                servers_available.append('flask')
            except ImportError:
                self.log_result(test_name, "FAIL", "Flask not available - required for API")
                return
            
            if not servers_available:
                self.log_result(test_name, "WARN", 
                              "No production WSGI server available (waitress/gunicorn). " +
                              "Install with: pip install waitress")
            else:
                self.log_result(test_name, "PASS", 
                              f"Production servers available: {', '.join(servers_available)}")
            
        except Exception as e:
            self.log_result(test_name, "FAIL", f"Server test error: {e}")
    
    def log_result(self, test_name: str, status: str, message: str) -> None:
        """Log test result"""
        self.results.append((test_name, status, message))
        
        if status == "PASS":
            self.passed += 1
            logger.info(f"✓ {test_name}: {message}")
        elif status == "FAIL":
            self.failed += 1
            logger.error(f"✗ {test_name}: {message}")
        else:  # WARN
            self.warnings += 1
            logger.warning(f"⚠ {test_name}: {message}")
    
    def print_summary(self) -> None:
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = self.passed + self.failed + self.warnings
        
        print(f"\nTotal Tests: {total}")
        print(f"  Passed:   {self.passed} ({self.passed/total*100:.1f}%)")
        print(f"  Failed:   {self.failed} ({self.failed/total*100:.1f}%)")
        print(f"  Warnings: {self.warnings} ({self.warnings/total*100:.1f}%)")
        
        if self.failed > 0:
            print("\nFailed Tests:")
            for test_name, status, message in self.results:
                if status == "FAIL":
                    print(f"  - {test_name}: {message}")
        
        if self.warnings > 0:
            print("\nWarnings:")
            for test_name, status, message in self.results:
                if status == "WARN":
                    print(f"  - {test_name}: {message}")
        
        print("\n" + "="*70)
        
        if self.failed == 0:
            print("✓ PRODUCTION READY: All critical tests passed!")
        else:
            print("✗ NOT PRODUCTION READY: Critical issues need to be resolved")
        
        print("="*70)


def main():
    """Run production readiness tests"""
    tester = ProductionReadinessTests()
    
    try:
        tester.run_all_tests()
        
        # Return appropriate exit code
        if tester.failed > 0:
            sys.exit(1)  # Failure
        else:
            sys.exit(0)  # Success
            
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error during testing: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()