"""
UK Biobank Data Loader for HENAW Algorithm
Handles biomarker data loading, preprocessing, and feature engineering
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import logging
from pathlib import Path
import h5py
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)


@dataclass
class UKBBSample:
    """Container for a single UK Biobank sample"""
    eid: int  # Participant ID
    biomarkers: np.ndarray  # Raw biomarker values
    age: float  # Chronological age
    sex: int  # 0: Female, 1: Male
    survival_time: Optional[float] = None
    event_indicator: Optional[int] = None
    disease_labels: Optional[Dict[str, int]] = None


class UKBBDataset(Dataset):
    """
    UK Biobank Dataset for HENAW model
    Handles complete cases (n=404,956) with all required biomarkers
    """
    
    def __init__(self,
                 data_path: str,
                 config: Dict[str, Any],
                 split: str = 'train',
                 transform: Optional[Any] = None,
                 cache_processed: bool = True):
        """
        Initialize UK Biobank dataset
        
        Args:
            data_path: Path to UK Biobank data files
            config: Configuration dictionary
            split: 'train', 'val', or 'test'
            transform: Optional data transformations
            cache_processed: Whether to cache processed data
        """
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.transform = transform
        self.cache_processed = cache_processed
        
        # Extract field IDs from config
        self.field_ids = config['ukbb_fields']['biomarkers']
        self.biomarker_names = list(self.field_ids.keys())
        self.n_biomarkers = len(self.biomarker_names)
        
        # Load and process data
        self.samples = self._load_data()
        
        # Initialize feature engineering
        self.feature_engineer = FeatureEngineer(config)
        
        # Fit normalizers on training data
        if split == 'train':
            self._fit_normalizers()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _load_data(self) -> List[UKBBSample]:
        """Load UK Biobank data from files with comprehensive error handling"""
        cache_file = self.data_path / f"processed_{self.split}.h5"
        
        # Try to load cached data first if it exists
        if self.cache_processed and cache_file.exists():
            try:
                logger.info(f"Loading cached data from {cache_file}")
                return self._load_cached_data(cache_file)
            except (IOError, OSError, KeyError, ValueError, h5py.Error) as e:
                logger.warning(f"Failed to load cache from {cache_file}: {e}. Loading raw data instead.")
            except Exception as e:
                logger.warning(f"Unexpected error loading cache: {e}. Loading raw data instead.")
        
        # Load raw data with error handling
        try:
            logger.info(f"Loading raw data for {self.split} split")
            samples = self._load_raw_data()
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise RuntimeError(f"Cannot load data for {self.split} split") from e
        
        # Apply split with error handling
        try:
            samples = self._apply_split(samples)
        except Exception as e:
            logger.error(f"Failed to apply data split: {e}")
            raise RuntimeError(f"Cannot apply {self.split} split") from e
        
        # Cache if requested, with error handling
        if self.cache_processed and len(samples) > 0:
            try:
                logger.info(f"Caching processed data to {cache_file}")
                self._cache_data(samples, cache_file)
            except Exception as e:
                logger.warning(f"Failed to cache data to {cache_file}: {e}. Continuing without cache.")
        
        return samples
    
    def _load_raw_data(self) -> List[UKBBSample]:
        """Load raw UK Biobank data"""
        samples = []
        
        # In production, this would load from actual UK Biobank files
        # For now, create synthetic data matching UK Biobank structure
        n_samples = 404956  # Complete cases from specification
        
        # Simulate loading biomarker data with realistic distributions
        np.random.seed(42)
        
        # Age distribution (40-70 years, centered around 55)
        ages = np.random.normal(55, 8, n_samples)
        ages = np.clip(ages, 40, 70)
        
        # Sex distribution (roughly 50/50)
        sexes = np.random.binomial(1, 0.48, n_samples)  # 48% male
        
        # Generate biomarkers with realistic ranges and correlations
        biomarker_data = self._generate_realistic_biomarkers(n_samples, ages, sexes)
        
        # Generate survival data
        survival_times, events = self._generate_survival_data(ages, biomarker_data)
        
        # Generate disease labels
        disease_labels = self._generate_disease_labels(ages, biomarker_data)
        
        # Create samples
        for i in range(n_samples):
            sample = UKBBSample(
                eid=1000000 + i,
                biomarkers=biomarker_data[i],
                age=ages[i],
                sex=sexes[i],
                survival_time=survival_times[i],
                event_indicator=events[i],
                disease_labels={
                    'cardiovascular': disease_labels['cardiovascular'][i],
                    'diabetes': disease_labels['diabetes'][i],
                    'cancer': disease_labels['cancer'][i],
                    'dementia': disease_labels['dementia'][i]
                }
            )
            samples.append(sample)
        
        return samples
    
    def _generate_realistic_biomarkers(self, 
                                      n_samples: int,
                                      ages: np.ndarray,
                                      sexes: np.ndarray) -> np.ndarray:
        """Generate biomarkers with realistic distributions and correlations"""
        biomarkers = np.zeros((n_samples, self.n_biomarkers))
        
        # Reference ranges for each biomarker (based on UK Biobank distributions)
        ranges = {
            'crp': (0.1, 10.0, 2.0, 3.0),  # (min, max, mean, std)
            'hba1c': (20, 120, 36, 6),
            'creatinine': (40, 150, 70, 15),
            'albumin': (35, 50, 45, 3),
            'lymphocyte_pct': (15, 45, 30, 7),
            'rdw': (11, 16, 13, 1),
            'ggt': (10, 300, 40, 30),
            'ast': (10, 100, 25, 10),
            'alt': (10, 100, 30, 12)
        }
        
        # Generate base values
        for i, name in enumerate(self.biomarker_names):
            min_val, max_val, mean_val, std_val = ranges[name]
            
            # Base values
            base = np.random.normal(mean_val, std_val, n_samples)
            
            # Add age effect (biomarkers change with age)
            age_effect = (ages - 55) / 10 * std_val * 0.2
            
            # Add sex effect for certain biomarkers
            sex_effect = np.zeros(n_samples)
            if name in ['creatinine', 'albumin', 'alt', 'ast']:
                sex_effect = sexes * std_val * 0.3
            
            # Combine effects
            values = base + age_effect + sex_effect
            
            # Clip to realistic ranges
            values = np.clip(values, min_val, max_val)
            
            # Add some log-normal noise for certain inflammatory markers
            if name in ['crp', 'ggt']:
                values = np.exp(np.log(values) + np.random.normal(0, 0.1, n_samples))
                values = np.clip(values, min_val, max_val)
            
            biomarkers[:, i] = values
        
        # Add correlations between related biomarkers
        # AST and ALT are correlated
        correlation = 0.7
        biomarkers[:, 7] = correlation * biomarkers[:, 8] + (1 - correlation) * biomarkers[:, 7]
        
        # CRP and lymphocyte % are inversely correlated
        biomarkers[:, 4] -= 0.3 * (biomarkers[:, 0] - ranges['crp'][2]) / ranges['crp'][3]
        
        return biomarkers
    
    def _generate_survival_data(self,
                               ages: np.ndarray,
                               biomarkers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic survival times and event indicators"""
        n_samples = len(ages)
        
        # Base hazard increases with age
        base_hazard = 0.001 * np.exp((ages - 40) / 20)
        
        # Biomarker effects on hazard
        # Higher CRP, HbA1c, creatinine increase risk
        biomarker_hazard = (
            0.1 * (biomarkers[:, 0] - 2) / 3 +  # CRP
            0.15 * (biomarkers[:, 1] - 36) / 6 +  # HbA1c
            0.1 * (biomarkers[:, 2] - 70) / 15  # Creatinine
        )
        
        # Total hazard
        hazard = base_hazard * np.exp(biomarker_hazard)
        
        # Generate survival times (exponential distribution)
        survival_times = np.random.exponential(1 / (hazard + 1e-8))
        
        # Censoring (follow-up limited to 15 years)
        max_followup = 15
        events = (survival_times <= max_followup).astype(int)
        survival_times = np.minimum(survival_times, max_followup)
        
        return survival_times, events
    
    def _generate_disease_labels(self,
                                ages: np.ndarray,
                                biomarkers: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate disease labels based on biomarkers"""
        n_samples = len(ages)
        
        # Base prevalence increases with age
        age_factor = (ages - 40) / 30
        
        disease_labels = {}
        
        # Cardiovascular disease (influenced by CRP, HbA1c)
        cvd_risk = 0.1 + 0.3 * age_factor + 0.1 * (biomarkers[:, 0] > 3) + 0.1 * (biomarkers[:, 1] > 42)
        disease_labels['cardiovascular'] = np.random.binomial(1, np.clip(cvd_risk, 0, 1), n_samples)
        
        # Diabetes (influenced by HbA1c, GGT)
        diabetes_risk = 0.05 + 0.2 * age_factor + 0.3 * (biomarkers[:, 1] > 42) + 0.1 * (biomarkers[:, 6] > 50)
        disease_labels['diabetes'] = np.random.binomial(1, np.clip(diabetes_risk, 0, 1), n_samples)
        
        # Cancer (influenced by age, inflammation)
        cancer_risk = 0.02 + 0.15 * age_factor + 0.05 * (biomarkers[:, 0] > 5)
        disease_labels['cancer'] = np.random.binomial(1, np.clip(cancer_risk, 0, 1), n_samples)
        
        # Dementia (influenced by age, albumin)
        dementia_risk = 0.01 + 0.1 * (age_factor ** 2) + 0.05 * (biomarkers[:, 3] < 40)
        disease_labels['dementia'] = np.random.binomial(1, np.clip(dementia_risk, 0, 1), n_samples)
        
        return disease_labels
    
    def _apply_split(self, samples: List[UKBBSample]) -> List[UKBBSample]:
        """Apply train/val/test split"""
        n_samples = len(samples)
        
        # Use temporal holdout if configured
        if self.config['training'].get('temporal_holdout', False):
            # Sort by some proxy for time (e.g., participant ID)
            samples.sort(key=lambda x: x.eid)
        
        # Calculate split indices
        train_ratio = self.config['training']['train_ratio']
        val_ratio = self.config['training']['val_ratio']
        
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        if self.split == 'train':
            return samples[:train_end]
        elif self.split == 'val':
            return samples[train_end:val_end]
        else:  # test
            return samples[val_end:]
    
    def _fit_normalizers(self) -> None:
        """Fit normalization parameters on training data"""
        # Collect all biomarker values
        all_biomarkers = np.array([s.biomarkers for s in self.samples])
        all_ages = np.array([s.age for s in self.samples])
        all_sexes = np.array([s.sex for s in self.samples])
        
        # Fit age-sex specific normalizers
        self.normalizers = {}
        
        # Create age bins
        age_bins = [40, 45, 50, 55, 60, 65, 70]
        
        for sex in [0, 1]:
            sex_mask = all_sexes == sex
            
            for i in range(len(age_bins) - 1):
                age_min, age_max = age_bins[i], age_bins[i + 1]
                age_mask = (all_ages >= age_min) & (all_ages < age_max)
                
                mask = sex_mask & age_mask
                
                if mask.sum() > 0:
                    subset_biomarkers = all_biomarkers[mask]
                    
                    # Fit scaler
                    scaler = StandardScaler()
                    scaler.fit(subset_biomarkers)
                    
                    key = (sex, age_min, age_max)
                    self.normalizers[key] = scaler
        
        # Save normalizers for later use
        self.feature_engineer.set_normalizers(self.normalizers)
    
    def _load_cached_data(self, cache_file: Path) -> List[UKBBSample]:
        """Load preprocessed data from cache with error handling"""
        samples = []
        
        try:
            with h5py.File(cache_file, 'r') as f:
                # Validate cache file structure
                if 'n_samples' not in f.attrs:
                    raise ValueError("Cache file missing 'n_samples' attribute")
                
                n_samples = f.attrs['n_samples']
                logger.debug(f"Loading {n_samples} samples from cache")
                
                for i in range(n_samples):
                    try:
                        sample_key = f'sample_{i}'
                        if sample_key not in f:
                            logger.warning(f"Missing sample {i} in cache, skipping")
                            continue
                        
                        grp = f[sample_key]
                        
                        # Validate required attributes
                        required_attrs = ['eid', 'age', 'sex']
                        for attr in required_attrs:
                            if attr not in grp.attrs:
                                raise ValueError(f"Missing required attribute '{attr}' for sample {i}")
                        
                        # Validate required datasets
                        if 'biomarkers' not in grp:
                            raise ValueError(f"Missing biomarkers data for sample {i}")
                        
                        # Load disease labels safely
                        disease_labels = {}
                        if 'disease_labels' in grp:
                            try:
                                for disease in grp['disease_labels'].keys():
                                    disease_labels[disease] = grp['disease_labels'][disease][()]
                            except Exception as e:
                                logger.warning(f"Failed to load disease labels for sample {i}: {e}")
                        
                        # Create sample with validation
                        biomarkers = grp['biomarkers'][()]
                        if not isinstance(biomarkers, np.ndarray) or biomarkers.size == 0:
                            logger.warning(f"Invalid biomarkers data for sample {i}, skipping")
                            continue
                        
                        sample = UKBBSample(
                            eid=int(grp.attrs['eid']),
                            biomarkers=biomarkers,
                            age=float(grp.attrs['age']),
                            sex=int(grp.attrs['sex']),
                            survival_time=grp.attrs.get('survival_time', None),
                            event_indicator=grp.attrs.get('event_indicator', None),
                            disease_labels=disease_labels if disease_labels else None
                        )
                        samples.append(sample)
                        
                    except Exception as e:
                        logger.warning(f"Failed to load sample {i} from cache: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error reading cache file {cache_file}: {e}")
            raise
        
        if len(samples) == 0:
            raise ValueError(f"No valid samples loaded from cache file {cache_file}")
            
        logger.info(f"Successfully loaded {len(samples)} samples from cache")
        return samples
    
    def _cache_data(self, samples: List[UKBBSample], cache_file: Path) -> None:
        """Cache processed data to disk with error handling"""
        try:
            # Ensure parent directory exists
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use temporary file to avoid corruption
            temp_file = cache_file.with_suffix('.tmp')
            
            with h5py.File(temp_file, 'w') as f:
                f.attrs['n_samples'] = len(samples)
                f.attrs['cache_version'] = '1.0'
                
                for i, sample in enumerate(samples):
                    try:
                        grp = f.create_group(f'sample_{i}')
                        
                        # Validate sample data before writing
                        if not isinstance(sample.biomarkers, np.ndarray):
                            raise ValueError(f"Invalid biomarkers type for sample {i}")
                        
                        if sample.biomarkers.size == 0:
                            raise ValueError(f"Empty biomarkers for sample {i}")
                        
                        # Write attributes
                        grp.attrs['eid'] = sample.eid
                        grp.attrs['age'] = sample.age
                        grp.attrs['sex'] = sample.sex
                        
                        # Write biomarkers data
                        grp.create_dataset('biomarkers', data=sample.biomarkers,
                                         compression='gzip', compression_opts=6)
                        
                        # Write optional data
                        if sample.survival_time is not None:
                            grp.attrs['survival_time'] = sample.survival_time
                        if sample.event_indicator is not None:
                            grp.attrs['event_indicator'] = sample.event_indicator
                        
                        # Write disease labels
                        if sample.disease_labels:
                            disease_grp = grp.create_group('disease_labels')
                            for disease, label in sample.disease_labels.items():
                                disease_grp.create_dataset(disease, data=label)
                                
                    except Exception as e:
                        logger.warning(f"Failed to cache sample {i}: {e}")
                        continue
            
            # Atomically replace cache file
            temp_file.replace(cache_file)
            logger.debug(f"Successfully cached {len(samples)} samples to {cache_file}")
            
        except Exception as e:
            logger.error(f"Failed to cache data to {cache_file}: {e}")
            # Clean up temp file if it exists
            if 'temp_file' in locals() and temp_file.exists():
                temp_file.unlink(missing_ok=True)
            raise
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Apply feature engineering
        features = self.feature_engineer.transform(
            sample.biomarkers,
            sample.age,
            sample.sex
        )
        
        # Create output dictionary
        item = {
            'biomarkers': torch.FloatTensor(features),
            'chronological_age': torch.FloatTensor([sample.age]),
            'sex': torch.LongTensor([sample.sex])
        }
        
        # Add survival data if available
        if sample.survival_time is not None:
            item['survival_time'] = torch.FloatTensor([sample.survival_time])
            item['event_indicator'] = torch.LongTensor([sample.event_indicator])
        
        # Add disease labels if available
        if sample.disease_labels:
            for disease, label in sample.disease_labels.items():
                item[f'{disease}_label'] = torch.LongTensor([label])
        
        # Apply additional transforms if specified
        if self.transform:
            item = self.transform(item)
        
        return item


class FeatureEngineer:
    """
    Feature engineering for HENAW model
    Implements age-sex specific normalization and feature transformations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.feature_config = config['feature_engineering']
        self.normalizers = None
        
    def set_normalizers(self, normalizers: Dict) -> None:
        """Set fitted normalizers from training data"""
        self.normalizers = normalizers
    
    def transform(self,
                 biomarkers: np.ndarray,
                 age: float,
                 sex: int) -> np.ndarray:
        """
        Apply feature engineering transformations
        
        Args:
            biomarkers: Raw biomarker values
            age: Chronological age
            sex: Sex (0: Female, 1: Male)
        
        Returns:
            Transformed features
        """
        # Age-sex specific normalization
        if self.normalizers:
            normalized = self._normalize_age_sex(biomarkers, age, sex)
        else:
            # Fallback to simple standardization with safe division
            normalized = self._safe_normalize(biomarkers)
        
        # Apply non-linear transformation: h(x) = log(1 + |x - μ|/σ)
        if self.feature_config['transformations']['type'] == 'log_distance':
            transformed = np.log1p(np.abs(normalized))
        else:
            transformed = normalized
        
        # Clip outliers
        clip_value = self.feature_config['transformations'].get('clip_outliers', 5)
        transformed = np.clip(transformed, -clip_value, clip_value)
        
        return transformed
    
    def _normalize_age_sex(self,
                          biomarkers: np.ndarray,
                          age: float,
                          sex: int) -> np.ndarray:
        """Apply age-sex specific normalization"""
        # Find appropriate normalizer
        age_window = self.feature_config['age_window']
        
        # Find matching age bin
        best_key = None
        for key in self.normalizers.keys():
            key_sex, age_min, age_max = key
            if key_sex == sex and age_min <= age < age_max:
                best_key = key
                break
        
        # Fallback to closest age group if exact match not found
        if best_key is None:
            min_distance = float('inf')
            for key in self.normalizers.keys():
                key_sex, age_min, age_max = key
                if key_sex == sex:
                    distance = min(abs(age - age_min), abs(age - age_max))
                    if distance < min_distance:
                        min_distance = distance
                        best_key = key
        
        if best_key and best_key in self.normalizers:
            try:
                scaler = self.normalizers[best_key]
                # Reshape for sklearn
                biomarkers_2d = biomarkers.reshape(1, -1)
                normalized = scaler.transform(biomarkers_2d).flatten()
            except Exception as e:
                logger.warning(f"Failed to apply scaler for age={age}, sex={sex}: {e}")
                normalized = self._safe_normalize(biomarkers)
        else:
            # Fallback to safe normalization
            normalized = self._safe_normalize(biomarkers)
            warnings.warn(f"No normalizer found for age={age}, sex={sex}")
        
        return normalized
    
    def _safe_normalize(self, biomarkers: np.ndarray) -> np.ndarray:
        """Safe normalization with zero-variance handling"""
        try:
            mean = np.mean(biomarkers)
            std = np.std(biomarkers)
            
            # Handle edge cases
            if np.isnan(mean) or np.isnan(std):
                logger.warning("NaN detected in biomarkers during normalization, returning zeros")
                return np.zeros_like(biomarkers)
            
            if np.isinf(mean) or np.isinf(std):
                logger.warning("Inf detected in biomarkers during normalization, returning zeros")
                return np.zeros_like(biomarkers)
            
            if std < 1e-8:
                logger.warning(f"Near-zero variance detected (std={std}), returning centered values")
                return biomarkers - mean
            
            return (biomarkers - mean) / std
            
        except Exception as e:
            logger.error(f"Error in safe normalization: {e}")
            return np.zeros_like(biomarkers)


class StratifiedBatchSampler:
    """
    Stratified batch sampler for balanced age groups in training
    """
    
    def __init__(self,
                 dataset: UKBBDataset,
                 batch_size: int,
                 age_bins: List[float]):
        self.dataset = dataset
        self.batch_size = batch_size
        self.age_bins = age_bins
        
        # Group samples by age bin
        self.age_groups = self._create_age_groups()
        
    def _create_age_groups(self) -> Dict[int, List[int]]:
        """Group sample indices by age bin"""
        groups = {i: [] for i in range(len(self.age_bins) - 1)}
        
        for idx, sample in enumerate(self.dataset.samples):
            age = sample.age
            
            # Find age bin
            for i in range(len(self.age_bins) - 1):
                if self.age_bins[i] <= age < self.age_bins[i + 1]:
                    groups[i].append(idx)
                    break
        
        return groups
    
    def __iter__(self):
        """Generate stratified batches with proper iterator handling"""
        # Create copies for shuffling to avoid modifying original data
        group_indices = {i: list(indices) for i, indices in self.age_groups.items() if indices}
        
        if not group_indices:
            logger.warning("No age groups available for stratified sampling")
            return
        
        # Shuffle within each group
        for indices in group_indices.values():
            np.random.shuffle(indices)
        
        # Use cycle for infinite iteration until all samples are exhausted
        from itertools import cycle
        group_iterators = {}
        active_groups = set()
        
        for i, indices in group_indices.items():
            if indices:  # Only create iterators for non-empty groups
                group_iterators[i] = iter(indices)
                active_groups.add(i)
        
        batch = []
        samples_per_group = max(1, self.batch_size // len(active_groups)) if active_groups else 1
        total_samples = sum(len(indices) for indices in group_indices.values())
        samples_yielded = 0
        
        while samples_yielded < total_samples and active_groups:
            # Try to get samples from each active group
            for group_id in list(active_groups):  # Create copy to allow modification
                group_samples = []
                for _ in range(samples_per_group):
                    try:
                        sample_idx = next(group_iterators[group_id])
                        group_samples.append(sample_idx)
                    except StopIteration:
                        # This group is exhausted, remove it
                        active_groups.discard(group_id)
                        break
                
                batch.extend(group_samples)
            
            # Yield batch if we have enough samples
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                samples_yielded += self.batch_size
                batch = batch[self.batch_size:]
            elif not active_groups and batch:
                # All groups exhausted, yield final partial batch
                yield batch
                samples_yielded += len(batch)
                break
    
    def __len__(self) -> int:
        return sum(len(indices) for indices in self.age_groups.values()) // self.batch_size


def create_data_loaders(config: Dict[str, Any],
                       data_path: str,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration dictionary
        data_path: Path to UK Biobank data
        num_workers: Number of data loading workers
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = UKBBDataset(data_path, config, split='train')
    val_dataset = UKBBDataset(data_path, config, split='val')
    test_dataset = UKBBDataset(data_path, config, split='test')
    
    # Share normalizers from training set
    val_dataset.feature_engineer.set_normalizers(train_dataset.feature_engineer.normalizers)
    test_dataset.feature_engineer.set_normalizers(train_dataset.feature_engineer.normalizers)
    
    # Create data loaders
    batch_size = config['training']['batch_size']
    
    # Use stratified sampling for training if configured
    if config['training'].get('stratify_by_age', False):
        age_bins = [40] + config['training']['age_bins'] + [75]
        train_sampler = StratifiedBatchSampler(train_dataset, batch_size, age_bins)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loader
    import yaml
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        config,
        data_path='./data',  # Mock data path
        num_workers=2
    )
    
    # Test loading a batch
    for batch in train_loader:
        print("Batch keys:", batch.keys())
        print("Biomarkers shape:", batch['biomarkers'].shape)
        print("Age shape:", batch['chronological_age'].shape)
        
        if 'survival_time' in batch:
            print("Survival time shape:", batch['survival_time'].shape)
        
        # Check for disease labels
        disease_labels = [k for k in batch.keys() if k.endswith('_label')]
        print(f"Disease labels: {disease_labels}")
        
        break
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")