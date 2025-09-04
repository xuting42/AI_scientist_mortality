"""
UK Biobank data loader for multi-modal biological age estimation.

Handles loading and preprocessing of:
- Blood biomarkers
- OCT imaging data
- NMR metabolomics
- Clinical measurements
- Longitudinal data
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import h5py
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import logging
from PIL import Image
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UKBiobankDataset(Dataset):
    """
    UK Biobank dataset for biological age estimation.
    
    Supports multiple data modalities and temporal sequences.
    """
    
    def __init__(
        self,
        data_config: Any,
        model_type: str = 'henaw',
        split: str = 'train',
        transform: Optional[Any] = None,
        cache_data: bool = True
    ):
        """
        Initialize UK Biobank dataset.
        
        Args:
            data_config: Data configuration object
            model_type: Type of model ('henaw', 'modal', 'metage')
            split: Data split ('train', 'val', 'test')
            transform: Optional transforms for images
            cache_data: Whether to cache processed data
        """
        self.data_config = data_config
        self.model_type = model_type.lower()
        self.split = split
        self.transform = transform
        self.cache_data = cache_data
        
        # Set up paths
        self.ukbb_path = Path(data_config.ukbb_data_path)
        self.retinal_path = Path(data_config.retinal_img_path)
        self.cache_dir = Path(data_config.cache_dir) / model_type / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.biomarkers = None
        self.metabolomics = None
        self.oct_images = None
        self.participant_ids = None
        self.chronological_ages = None
        self.mortality_labels = None
        
        # Load or process data
        if self._check_cache():
            logger.info(f"Loading cached data for {model_type} - {split}")
            self._load_cached_data()
        else:
            logger.info(f"Processing UK Biobank data for {model_type} - {split}")
            self._load_and_process_data()
            if cache_data:
                self._save_cache()
        
        # Set up feature indices for HENAW temporal scales
        if self.model_type == 'henaw':
            self._setup_temporal_indices()
        
        logger.info(f"Dataset initialized: {len(self)} samples")
    
    def _check_cache(self) -> bool:
        """Check if cached data exists."""
        cache_file = self.cache_dir / 'processed_data.pkl'
        return cache_file.exists()
    
    def _load_cached_data(self):
        """Load cached processed data."""
        cache_file = self.cache_dir / 'processed_data.pkl'
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.biomarkers = cache_data['biomarkers']
        self.metabolomics = cache_data.get('metabolomics')
        self.oct_images = cache_data.get('oct_images')
        self.participant_ids = cache_data['participant_ids']
        self.chronological_ages = cache_data['chronological_ages']
        self.mortality_labels = cache_data.get('mortality_labels')
    
    def _save_cache(self):
        """Save processed data to cache."""
        cache_file = self.cache_dir / 'processed_data.pkl'
        cache_data = {
            'biomarkers': self.biomarkers,
            'metabolomics': self.metabolomics,
            'oct_images': self.oct_images,
            'participant_ids': self.participant_ids,
            'chronological_ages': self.chronological_ages,
            'mortality_labels': self.mortality_labels
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info(f"Cached data saved to {cache_file}")
    
    def _load_and_process_data(self):
        """Load and process UK Biobank data."""
        # Load participant data
        self._load_participant_data()
        
        # Load biomarkers
        self._load_biomarkers()
        
        # Load model-specific data
        if self.model_type == 'modal':
            self._load_oct_images()
        elif self.model_type == 'metage':
            self._load_metabolomics()
        
        # Apply data split
        self._apply_split()
        
        # Preprocess data
        self._preprocess_data()
    
    def _load_participant_data(self):
        """Load basic participant information."""
        # This is a placeholder - actual implementation would load from UK Biobank files
        # For now, create mock data for demonstration
        n_participants = 10000  # Mock data size
        
        np.random.seed(42)
        self.participant_ids = np.arange(n_participants)
        self.chronological_ages = np.random.normal(55, 10, n_participants)
        self.chronological_ages = np.clip(self.chronological_ages, 40, 80)
        
        # Mock mortality labels (10-year all-cause mortality)
        mortality_prob = 0.05 + (self.chronological_ages - 40) * 0.002
        self.mortality_labels = np.random.binomial(1, mortality_prob)
        
        logger.info(f"Loaded {n_participants} participants")
    
    def _load_biomarkers(self):
        """Load blood biomarker data."""
        n_participants = len(self.participant_ids)
        
        # Define biomarker features based on model configuration
        if self.model_type == 'henaw':
            n_features = 25  # Comprehensive tier
        elif self.model_type == 'modal':
            n_features = 31  # Extended biomarker panel
        else:
            n_features = 15  # Basic panel
        
        # Generate mock biomarker data
        # In practice, this would load actual UK Biobank field codes
        self.biomarkers = np.random.randn(n_participants, n_features)
        
        # Add realistic correlations with age
        age_effect = (self.chronological_ages - 55) / 10
        for i in range(n_features):
            correlation = np.random.uniform(-0.3, 0.3)
            self.biomarkers[:, i] += correlation * age_effect
        
        logger.info(f"Loaded {n_features} biomarkers")
    
    def _load_oct_images(self):
        """Load OCT imaging data for MODAL."""
        # In practice, this would load actual OCT images from UK Biobank
        # For now, create references to mock image paths
        n_participants = len(self.participant_ids)
        self.oct_images = [f"oct_{pid}.png" for pid in self.participant_ids]
        
        logger.info(f"Loaded OCT image references for {n_participants} participants")
    
    def _load_metabolomics(self):
        """Load NMR metabolomics data for METAGE."""
        n_participants = len(self.participant_ids)
        n_metabolites = self.data_config.metage_config.n_metabolomic_features if \
                       hasattr(self.data_config, 'metage_config') else 168
        
        # Generate mock metabolomics data with temporal sequences
        sequence_length = 5  # Number of time points
        
        # Create base metabolomic profiles
        base_metabolomics = np.random.randn(n_participants, n_metabolites)
        
        # Add age correlations
        age_effect = (self.chronological_ages - 55) / 10
        for i in range(n_metabolites):
            correlation = np.random.uniform(-0.2, 0.2)
            base_metabolomics[:, i] += correlation * age_effect[:, np.newaxis]
        
        # Create temporal sequences with realistic changes
        self.metabolomics = np.zeros((n_participants, sequence_length, n_metabolites))
        for t in range(sequence_length):
            # Add temporal drift
            temporal_effect = t * np.random.randn(n_metabolites) * 0.05
            self.metabolomics[:, t, :] = base_metabolomics + temporal_effect
        
        logger.info(f"Loaded {n_metabolites} metabolites with {sequence_length} time points")
    
    def _apply_split(self):
        """Apply train/val/test split."""
        n_total = len(self.participant_ids)
        
        # Calculate split indices
        train_end = int(n_total * self.data_config.train_ratio)
        val_end = train_end + int(n_total * self.data_config.val_ratio)
        
        # Apply split
        if self.split == 'train':
            indices = slice(0, train_end)
        elif self.split == 'val':
            indices = slice(train_end, val_end)
        else:  # test
            indices = slice(val_end, None)
        
        # Subset data
        self.participant_ids = self.participant_ids[indices]
        self.chronological_ages = self.chronological_ages[indices]
        self.mortality_labels = self.mortality_labels[indices]
        self.biomarkers = self.biomarkers[indices]
        
        if self.metabolomics is not None:
            self.metabolomics = self.metabolomics[indices]
        if self.oct_images is not None:
            self.oct_images = self.oct_images[indices]
        
        logger.info(f"Applied {self.split} split: {len(self.participant_ids)} samples")
    
    def _preprocess_data(self):
        """Preprocess data (normalization, imputation, etc.)."""
        # Impute missing values
        if self.data_config.imputation_strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=self.data_config.imputation_strategy)
        
        # Handle biomarkers
        mask = ~np.isnan(self.biomarkers).any(axis=1)
        if not mask.all():
            self.biomarkers = imputer.fit_transform(self.biomarkers)
        
        # Normalize features
        if self.data_config.normalize_features:
            if self.split == 'train':
                self.scaler = RobustScaler() if self.data_config.handle_outliers \
                            else StandardScaler()
                self.biomarkers = self.scaler.fit_transform(self.biomarkers)
                # Save scaler for val/test sets
                scaler_path = self.cache_dir.parent / 'scaler.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scaler, f)
            else:
                # Load scaler from train set
                scaler_path = self.cache_dir.parent / 'train' / '..' / 'scaler.pkl'
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                    self.biomarkers = self.scaler.transform(self.biomarkers)
        
        # Handle metabolomics
        if self.metabolomics is not None:
            n_participants, seq_len, n_features = self.metabolomics.shape
            metabolomics_flat = self.metabolomics.reshape(-1, n_features)
            
            # Impute and normalize
            mask = ~np.isnan(metabolomics_flat).any(axis=1)
            if not mask.all():
                metabolomics_flat = imputer.fit_transform(metabolomics_flat)
            
            if self.data_config.normalize_features:
                if self.split == 'train':
                    self.metabolomics_scaler = StandardScaler()
                    metabolomics_flat = self.metabolomics_scaler.fit_transform(metabolomics_flat)
                    # Save scaler
                    metab_scaler_path = self.cache_dir.parent / 'metabolomics_scaler.pkl'
                    with open(metab_scaler_path, 'wb') as f:
                        pickle.dump(self.metabolomics_scaler, f)
                else:
                    # Load scaler
                    metab_scaler_path = self.cache_dir.parent / 'train' / '..' / 'metabolomics_scaler.pkl'
                    if metab_scaler_path.exists():
                        with open(metab_scaler_path, 'rb') as f:
                            self.metabolomics_scaler = pickle.load(f)
                        metabolomics_flat = self.metabolomics_scaler.transform(metabolomics_flat)
            
            self.metabolomics = metabolomics_flat.reshape(n_participants, seq_len, n_features)
        
        logger.info("Data preprocessing completed")
    
    def _setup_temporal_indices(self):
        """Set up feature indices for HENAW temporal scales."""
        # Define which features belong to each temporal scale
        self.rapid_indices = [0, 1, 2, 3, 4, 5, 6, 7]  # Metabolic markers
        self.intermediate_indices = [8, 9, 10, 11, 12, 13, 14, 15]  # Organ function
        self.slow_indices = list(range(16, self.biomarkers.shape[1]))  # Structural
    
    def _load_oct_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess OCT image."""
        # In practice, load actual image
        # For now, create random tensor
        image = torch.randn(3, 224, 224)
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.participant_ids)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Get a data sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (inputs dict, target age)
        """
        # Prepare inputs based on model type
        inputs = {}
        
        if self.model_type == 'henaw':
            # Split features by temporal scale
            biomarker_tensor = torch.FloatTensor(self.biomarkers[idx])
            inputs['rapid'] = biomarker_tensor[self.rapid_indices]
            inputs['intermediate'] = biomarker_tensor[self.intermediate_indices]
            inputs['slow'] = biomarker_tensor[self.slow_indices]
            
        elif self.model_type == 'modal':
            # Include biomarkers and OCT image
            inputs['biomarkers'] = torch.FloatTensor(self.biomarkers[idx])
            if self.oct_images is not None:
                inputs['oct_image'] = self._load_oct_image(self.oct_images[idx])
            
        elif self.model_type == 'metage':
            # Include metabolomics sequences
            inputs['metabolomics'] = torch.FloatTensor(self.metabolomics[idx])
            # Add time points (normalized)
            seq_len = self.metabolomics.shape[1]
            inputs['time_points'] = torch.linspace(0, 1, seq_len)
        
        # Target is chronological age for training
        target = torch.FloatTensor([self.chronological_ages[idx]])
        
        return inputs, target
    
    def get_mortality_labels(self) -> np.ndarray:
        """Get mortality labels for survival analysis."""
        return self.mortality_labels


class StratifiedAgeSampler(Sampler):
    """
    Stratified sampler for age-balanced batches.
    
    Ensures each batch contains samples from different age groups.
    """
    
    def __init__(
        self,
        dataset: UKBiobankDataset,
        batch_size: int,
        n_bins: int = 10
    ):
        """
        Initialize stratified sampler.
        
        Args:
            dataset: UK Biobank dataset
            batch_size: Batch size
            n_bins: Number of age bins for stratification
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_bins = n_bins
        
        # Create age bins
        ages = dataset.chronological_ages
        self.bins = np.percentile(ages, np.linspace(0, 100, n_bins + 1))
        self.bin_indices = np.digitize(ages, self.bins[1:-1])
        
        # Group indices by bin
        self.stratified_indices = {}
        for bin_idx in range(n_bins):
            mask = self.bin_indices == bin_idx
            self.stratified_indices[bin_idx] = np.where(mask)[0].tolist()
    
    def __iter__(self):
        """Iterate through stratified batches."""
        # Shuffle indices within each bin
        for indices in self.stratified_indices.values():
            np.random.shuffle(indices)
        
        # Create batches with stratified sampling
        batch = []
        bin_counters = {i: 0 for i in range(self.n_bins)}
        
        n_samples = len(self.dataset)
        for _ in range(n_samples):
            # Sample from each bin in round-robin fashion
            for bin_idx in range(self.n_bins):
                if bin_counters[bin_idx] < len(self.stratified_indices[bin_idx]):
                    idx = self.stratified_indices[bin_idx][bin_counters[bin_idx]]
                    batch.append(idx)
                    bin_counters[bin_idx] += 1
                    
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
        
        # Yield remaining samples
        if batch:
            yield batch
    
    def __len__(self):
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def create_dataloaders(
    data_config: Any,
    model_config: Any,
    model_type: str,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    use_stratified: bool = True
) -> Dict[str, DataLoader]:
    """
    Create data loaders for train, validation, and test sets.
    
    Args:
        data_config: Data configuration
        model_config: Model-specific configuration
        model_type: Type of model
        batch_size: Optional batch size override
        num_workers: Optional num_workers override
        use_stratified: Whether to use stratified sampling
    
    Returns:
        Dictionary of DataLoaders
    """
    batch_size = batch_size or data_config.batch_size
    num_workers = num_workers or data_config.num_workers
    
    # Image transforms for MODAL
    if model_type == 'modal':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = val_transform = None
    
    # Create datasets
    train_dataset = UKBiobankDataset(
        data_config, model_type, 'train', train_transform
    )
    val_dataset = UKBiobankDataset(
        data_config, model_type, 'val', val_transform
    )
    test_dataset = UKBiobankDataset(
        data_config, model_type, 'test', val_transform
    )
    
    # Create samplers
    if use_stratified and data_config.stratify_by_age:
        train_sampler = StratifiedAgeSampler(train_dataset, batch_size)
        val_sampler = None
        test_sampler = None
    else:
        train_sampler = val_sampler = test_sampler = None
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.persistent_workers,
            prefetch_factor=data_config.prefetch_factor,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=data_config.pin_memory,
            persistent_workers=data_config.persistent_workers,
            prefetch_factor=data_config.prefetch_factor
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=data_config.pin_memory
        )
    }
    
    logger.info(f"Created dataloaders - Train: {len(dataloaders['train'])} batches, "
                f"Val: {len(dataloaders['val'])} batches, "
                f"Test: {len(dataloaders['test'])} batches")
    
    return dataloaders