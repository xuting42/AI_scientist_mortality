"""
UK Biobank data loader for HAMBAE algorithm system.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass
from ..config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class UKBBDataSample:
    """A single UK Biobank data sample."""
    participant_id: str
    age: float
    sex: int
    blood_biomarkers: Optional[np.ndarray] = None
    metabolomics: Optional[np.ndarray] = None
    retinal_features: Optional[np.ndarray] = None
    genetic_features: Optional[np.ndarray] = None
    mortality_status: Optional[int] = None
    follow_up_time: Optional[float] = None
    quality_score: Optional[float] = None
    timestamp: Optional[float] = None


class UKBBDataLoader(Dataset):
    """
    UK Biobank data loader for multi-modal biological age estimation.
    
    This class handles loading and preprocessing of UK Biobank data
    for the HAMBAE algorithm system.
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        config: DataConfig,
        split: str = "train",
        transform: Optional[Any] = None,
        cache_data: bool = True,
    ):
        """
        Initialize UKBB data loader.
        
        Args:
            data_root: Path to UK Biobank data directory
            config: Data configuration
            split: Data split (train, val, test)
            transform: Optional data transformation
            cache_data: Whether to cache data in memory
        """
        self.data_root = Path(data_root)
        self.config = config
        self.split = split
        self.transform = transform
        self.cache_data = cache_data
        
        # Load data
        self.samples = self._load_data()
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        
        # Precompute statistics
        self._compute_statistics()
        
        # Cache data if requested
        if cache_data:
            self._cache_samples()
    
    def _load_data(self) -> List[UKBBDataSample]:
        """Load UK Biobank data from files."""
        samples = []
        
        # This is a simplified implementation
        # In practice, you would load actual UK Biobank data files
        
        # Load participant data
        participant_file = self.data_root / f"participants_{self.split}.csv"
        if participant_file.exists():
            df_participants = pd.read_csv(participant_file)
        else:
            # Create dummy data for demonstration
            logger.warning(f"Participant file not found: {participant_file}")
            df_participants = self._create_dummy_data()
        
        # Load modality-specific data
        for _, row in df_participants.iterrows():
            sample = UKBBDataSample(
                participant_id=str(row.get('participant_id', f"pid_{len(samples)}")),
                age=float(row.get('age', 50.0)),
                sex=int(row.get('sex', 1)),
                mortality_status=int(row.get('mortality_status', 0)) if 'mortality_status' in row else None,
                follow_up_time=float(row.get('follow_up_time', 0.0)) if 'follow_up_time' in row else None,
                quality_score=float(row.get('quality_score', 1.0)) if 'quality_score' in row else None,
                timestamp=float(row.get('timestamp', 0.0)) if 'timestamp' in row else None,
            )
            
            # Load blood biomarkers
            if 'blood_biomarkers' in row:
                sample.blood_biomarkers = np.array(row['blood_biomarkers'], dtype=np.float32)
            
            # Load metabolomics
            if 'metabolomics' in row:
                sample.metabolomics = np.array(row['metabolomics'], dtype=np.float32)
            
            # Load retinal features
            if 'retinal_features' in row:
                sample.retinal_features = np.array(row['retinal_features'], dtype=np.float32)
            
            # Load genetic features
            if 'genetic_features' in row:
                sample.genetic_features = np.array(row['genetic_features'], dtype=np.float32)
            
            samples.append(sample)
        
        return samples
    
    def _create_dummy_data(self) -> pd.DataFrame:
        """Create dummy data for demonstration purposes."""
        n_samples = 1000
        
        # Generate synthetic data
        np.random.seed(42)
        
        data = {
            'participant_id': [f"pid_{i}" for i in range(n_samples)],
            'age': np.random.normal(50, 15, n_samples).clip(18, 80),
            'sex': np.random.binomial(1, 0.5, n_samples),
            'mortality_status': np.random.binomial(1, 0.1, n_samples),
            'follow_up_time': np.random.exponential(5, n_samples),
            'quality_score': np.random.beta(2, 5, n_samples),
            'timestamp': np.random.uniform(0, 10, n_samples),
        }
        
        # Blood biomarkers (13 features)
        blood_biomarkers = []
        for i in range(n_samples):
            age = data['age'][i]
            # Generate biomarkers correlated with age
            biomarkers = []
            for j in range(13):
                base_value = np.random.normal(50, 10)
                age_effect = (age - 50) * 0.1 * (1 + 0.1 * j)
                noise = np.random.normal(0, 5)
                biomarkers.append(base_value + age_effect + noise)
            blood_biomarkers.append(biomarkers)
        data['blood_biomarkers'] = blood_biomarkers
        
        # Metabolomics (400 features)
        metabolomics = []
        for i in range(n_samples):
            age = data['age'][i]
            # Generate metabolomic features correlated with age
            features = []
            for j in range(400):
                base_value = np.random.normal(100, 20)
                age_effect = (age - 50) * 0.05 * (1 + 0.05 * j)
                noise = np.random.normal(0, 10)
                features.append(base_value + age_effect + noise)
            metabolomics.append(features)
        data['metabolomics'] = metabolomics
        
        # Retinal features (768 features)
        retinal_features = []
        for i in range(n_samples):
            age = data['age'][i]
            # Generate retinal features correlated with age
            features = []
            for j in range(768):
                base_value = np.random.normal(0, 1)
                age_effect = (age - 50) * 0.01 * (1 + 0.01 * j)
                noise = np.random.normal(0, 0.5)
                features.append(base_value + age_effect + noise)
            retinal_features.append(features)
        data['retinal_features'] = retinal_features
        
        # Genetic features (1000 features)
        genetic_features = []
        for i in range(n_samples):
            # Generate genetic features (less correlated with age)
            features = []
            for j in range(1000):
                base_value = np.random.normal(0, 1)
                noise = np.random.normal(0, 0.8)
                features.append(base_value + noise)
            genetic_features.append(features)
        data['genetic_features'] = genetic_features
        
        return pd.DataFrame(data)
    
    def _compute_statistics(self) -> None:
        """Compute dataset statistics."""
        ages = [sample.age for sample in self.samples]
        self.mean_age = np.mean(ages)
        self.std_age = np.std(ages)
        
        # Compute modality-specific statistics
        blood_available = sum(1 for s in self.samples if s.blood_biomarkers is not None)
        metabolomics_available = sum(1 for s in self.samples if s.metabolomics is not None)
        retinal_available = sum(1 for s in self.samples if s.retinal_features is not None)
        genetic_available = sum(1 for s in self.samples if s.genetic_features is not None)
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total samples: {len(self.samples)}")
        logger.info(f"  Age range: {min(ages):.1f} - {max(ages):.1f}")
        logger.info(f"  Blood biomarkers available: {blood_available}/{len(self.samples)}")
        logger.info(f"  Metabolomics available: {metabolomics_available}/{len(self.samples)}")
        logger.info(f"  Retinal features available: {retinal_available}/{len(self.samples)}")
        logger.info(f"  Genetic features available: {genetic_available}/{len(self.samples)}")
    
    def _cache_samples(self) -> None:
        """Cache samples in memory for faster access."""
        self.cached_samples = []
        for sample in self.samples:
            cached_sample = {
                'age': torch.tensor(sample.age, dtype=torch.float32),
                'sex': torch.tensor(sample.sex, dtype=torch.long),
            }
            
            if sample.blood_biomarkers is not None:
                cached_sample['blood_biomarkers'] = torch.tensor(sample.blood_biomarkers, dtype=torch.float32)
            
            if sample.metabolomics is not None:
                cached_sample['metabolomics'] = torch.tensor(sample.metabolomics, dtype=torch.float32)
            
            if sample.retinal_features is not None:
                cached_sample['retinal_features'] = torch.tensor(sample.retinal_features, dtype=torch.float32)
            
            if sample.genetic_features is not None:
                cached_sample['genetic_features'] = torch.tensor(sample.genetic_features, dtype=torch.float32)
            
            if sample.mortality_status is not None:
                cached_sample['mortality_status'] = torch.tensor(sample.mortality_status, dtype=torch.long)
            
            if sample.follow_up_time is not None:
                cached_sample['follow_up_time'] = torch.tensor(sample.follow_up_time, dtype=torch.float32)
            
            if sample.quality_score is not None:
                cached_sample['quality_score'] = torch.tensor(sample.quality_score, dtype=torch.float32)
            
            self.cached_samples.append(cached_sample)
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from dataset."""
        if self.cache_data:
            return self.cached_samples[idx]
        
        sample = self.samples[idx]
        
        item = {
            'age': torch.tensor(sample.age, dtype=torch.float32),
            'sex': torch.tensor(sample.sex, dtype=torch.long),
            'participant_id': sample.participant_id,
        }
        
        if sample.blood_biomarkers is not None:
            item['blood_biomarkers'] = torch.tensor(sample.blood_biomarkers, dtype=torch.float32)
        
        if sample.metabolomics is not None:
            item['metabolomics'] = torch.tensor(sample.metabolomics, dtype=torch.float32)
        
        if sample.retinal_features is not None:
            item['retinal_features'] = torch.tensor(sample.retinal_features, dtype=torch.float32)
        
        if sample.genetic_features is not None:
            item['genetic_features'] = torch.tensor(sample.genetic_features, dtype=torch.float32)
        
        if sample.mortality_status is not None:
            item['mortality_status'] = torch.tensor(sample.mortality_status, dtype=torch.long)
        
        if sample.follow_up_time is not None:
            item['follow_up_time'] = torch.tensor(sample.follow_up_time, dtype=torch.float32)
        
        if sample.quality_score is not None:
            item['quality_score'] = torch.tensor(sample.quality_score, dtype=torch.float32)
        
        if sample.timestamp is not None:
            item['timestamp'] = torch.tensor(sample.timestamp, dtype=torch.float32)
        
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        stats = {
            'total_samples': len(self.samples),
            'age_stats': {
                'mean': self.mean_age,
                'std': self.std_age,
                'min': min(s.age for s in self.samples),
                'max': max(s.age for s in self.samples),
            },
            'sex_distribution': {
                'male': sum(1 for s in self.samples if s.sex == 1),
                'female': sum(1 for s in self.samples if s.sex == 0),
            },
            'modality_availability': {
                'blood_biomarkers': sum(1 for s in self.samples if s.blood_biomarkers is not None),
                'metabolomics': sum(1 for s in self.samples if s.metabolomics is not None),
                'retinal_features': sum(1 for s in self.samples if s.retinal_features is not None),
                'genetic_features': sum(1 for s in self.samples if s.genetic_features is not None),
            },
            'mortality_stats': {
                'mortality_rate': sum(1 for s in self.samples if s.mortality_status == 1) / len(self.samples),
                'mean_follow_up': np.mean([s.follow_up_time for s in self.samples if s.follow_up_time is not None]),
            }
        }
        
        return stats


def create_data_loaders(
    data_root: Union[str, Path],
    config: DataConfig,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test splits.
    
    Args:
        data_root: Path to UK Biobank data directory
        config: Data configuration
        batch_size: Batch size (overrides config)
        num_workers: Number of workers (overrides config)
    
    Returns:
        Dictionary of data loaders
    """
    batch_size = batch_size or config.batch_size
    num_workers = num_workers or config.num_workers
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        datasets[split] = UKBBDataLoader(
            data_root=data_root,
            config=config,
            split=split,
            cache_data=True,
        )
    
    # Create data loaders
    data_loaders = {}
    for split, dataset in datasets.items():
        is_train = split == 'train'
        
        data_loaders[split] = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=config.pin_memory,
            persistent_workers=config.persistent_workers,
            drop_last=is_train,
        )
    
    return data_loaders


def load_ukbb_fields(
    data_root: Union[str, Path],
    field_ids: List[int],
    participant_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load specific UK Biobank fields.
    
    Args:
        data_root: Path to UK Biobank data directory
        field_ids: List of UK Biobank field IDs to load
        participant_ids: Optional list of participant IDs to filter
    
    Returns:
        DataFrame with requested fields
    """
    # This is a placeholder implementation
    # In practice, you would implement proper UK Biobank field loading
    
    logger.info(f"Loading UK Biobank fields: {field_ids}")
    
    # Create dummy data for demonstration
    if participant_ids is None:
        participant_ids = [f"pid_{i}" for i in range(1000)]
    
    data = {'participant_id': participant_ids}
    
    for field_id in field_ids:
        # Generate synthetic data for each field
        if field_id in [30000, 30010, 30020, 30030, 30040, 30050, 30060, 30070, 30080, 30090, 30100, 30110, 30120]:
            # Blood biomarkers
            data[f'field_{field_id}'] = np.random.normal(50, 10, len(participant_ids))
        elif field_id >= 23400 and field_id <= 23409:
            # Metabolomics
            data[f'field_{field_id}'] = np.random.normal(100, 20, len(participant_ids))
        elif field_id >= 22400 and field_id <= 22402:
            # Retinal
            data[f'field_{field_id}'] = np.random.normal(0, 1, len(participant_ids))
        elif field_id >= 22000 and field_id <= 22005:
            # Genetic
            data[f'field_{field_id}'] = np.random.normal(0, 1, len(participant_ids))
        else:
            # Other fields
            data[f'field_{field_id}'] = np.random.normal(0, 1, len(participant_ids))
    
    return pd.DataFrame(data)