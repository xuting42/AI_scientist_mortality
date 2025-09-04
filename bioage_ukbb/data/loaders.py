"""
UK Biobank data loaders for biological age algorithms.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import h5py
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
from concurrent.futures import ThreadPoolExecutor
import pickle
from functools import lru_cache


class UKBiobankDataset(Dataset):
    """Base dataset class for UK Biobank data."""
    
    def __init__(
        self,
        data_path: str,
        participant_ids: Optional[List[int]] = None,
        features: Optional[List[str]] = None,
        target: str = "age",
        transform: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize UK Biobank dataset.
        
        Args:
            data_path: Path to UK Biobank data
            participant_ids: List of participant IDs to include
            features: List of feature names to load
            target: Target variable name
            transform: Data transformations to apply
            cache_dir: Directory for caching processed data
            use_cache: Whether to use cached data if available
        """
        self.data_path = Path(data_path)
        self.participant_ids = participant_ids
        self.features = features
        self.target = target
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        # Load or process data
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and preprocess UK Biobank data."""
        cache_file = None
        if self.cache_dir and self.use_cache:
            cache_file = self.cache_dir / f"ukbb_cache_{hash(str(self.features))}.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.data = cache_data['data']
                    self.targets = cache_data['targets']
                    self.participant_ids = cache_data['participant_ids']
                    return
        
        # Load from raw data files
        self.data, self.targets = self._load_raw_data()
        
        # Save to cache
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': self.data,
                    'targets': self.targets,
                    'participant_ids': self.participant_ids
                }, f)
    
    def _load_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw data from UK Biobank files."""
        # This is a placeholder - actual implementation would read from UK Biobank format
        # For demonstration, create synthetic data
        n_samples = len(self.participant_ids) if self.participant_ids else 1000
        n_features = len(self.features) if self.features else 50
        
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        targets = np.random.uniform(40, 80, n_samples).astype(np.float32)
        
        return data, targets
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target


class HENAWDataset(UKBiobankDataset):
    """Dataset for HENAW algorithm with temporal features."""
    
    def __init__(
        self,
        data_path: str,
        temporal_windows: List[Tuple[int, int]],
        biomarkers: List[str],
        **kwargs
    ):
        """
        Initialize HENAW dataset.
        
        Args:
            data_path: Path to UK Biobank data
            temporal_windows: List of (start, end) year tuples for hierarchical modeling
            biomarkers: List of biomarker field IDs
            **kwargs: Additional arguments for base class
        """
        self.temporal_windows = temporal_windows
        self.biomarkers = biomarkers
        
        # Define HENAW-specific features
        henaw_features = [
            "30600-0.0",  # Albumin
            "30610-0.0",  # Alkaline phosphatase
            "30700-0.0",  # Creatinine
            "30710-0.0",  # C-reactive protein
            "30740-0.0",  # Glucose
            "30070-0.0",  # Red cell distribution width
            "30000-0.0",  # White blood cell count
            "4080-0.0",   # Systolic blood pressure
            "4079-0.0",   # Diastolic blood pressure
            "21001-0.0"   # BMI
        ]
        
        super().__init__(data_path, features=henaw_features, **kwargs)
        
        # Process temporal features
        self._process_temporal_features()
    
    def _process_temporal_features(self) -> None:
        """Extract temporal patterns at different scales."""
        temporal_features = []
        
        for start_year, end_year in self.temporal_windows:
            # Extract features for this temporal window
            window_features = self._extract_window_features(start_year, end_year)
            temporal_features.append(window_features)
        
        # Stack temporal features
        self.temporal_data = np.stack(temporal_features, axis=1)
    
    def _extract_window_features(self, start_year: int, end_year: int) -> np.ndarray:
        """Extract features for a specific temporal window."""
        # Placeholder for temporal feature extraction
        n_samples = len(self.data)
        n_features = self.data.shape[1]
        window_features = np.random.randn(n_samples, n_features).astype(np.float32)
        return window_features
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get sample with temporal features."""
        base_features = torch.tensor(self.data[idx], dtype=torch.float32)
        temporal_features = torch.tensor(self.temporal_data[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        sample = {
            'base_features': base_features,
            'temporal_features': temporal_features,
            'rapid': temporal_features[0],
            'intermediate': temporal_features[1] if len(temporal_features) > 1 else temporal_features[0],
            'slow': temporal_features[2] if len(temporal_features) > 2 else temporal_features[0]
        }
        
        return sample, target


class MODALDataset(UKBiobankDataset):
    """Dataset for MODAL algorithm with OCT imaging and biomarkers."""
    
    def __init__(
        self,
        data_path: str,
        oct_image_dir: str,
        biomarker_features: List[str],
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        **kwargs
    ):
        """
        Initialize MODAL dataset.
        
        Args:
            data_path: Path to UK Biobank data
            oct_image_dir: Directory containing OCT images
            biomarker_features: List of biomarker field IDs
            image_size: Target image size for OCT
            augment: Whether to apply data augmentation
            **kwargs: Additional arguments for base class
        """
        self.oct_image_dir = Path(oct_image_dir)
        self.biomarker_features = biomarker_features
        self.image_size = image_size
        
        # Define image transforms
        if augment:
            self.image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        super().__init__(data_path, features=biomarker_features, **kwargs)
        
        # Load OCT image paths
        self._load_image_paths()
    
    def _load_image_paths(self) -> None:
        """Load paths to OCT images for each participant."""
        self.image_paths = {}
        
        # Map participant IDs to image files
        if self.participant_ids:
            for pid in self.participant_ids:
                # Look for OCT images - format may vary
                image_path = self.oct_image_dir / f"{pid}_21016_0_0.png"
                if image_path.exists():
                    self.image_paths[pid] = image_path
                else:
                    # Use placeholder or skip
                    self.image_paths[pid] = None
    
    @lru_cache(maxsize=1000)
    def _load_oct_image(self, image_path: Path) -> torch.Tensor:
        """Load and preprocess OCT image."""
        if image_path and image_path.exists():
            image = Image.open(image_path).convert('RGB')
            return self.image_transform(image)
        else:
            # Return zero tensor if image not found
            return torch.zeros(3, *self.image_size)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get multi-modal sample."""
        # Get biomarker features
        biomarkers = torch.tensor(self.data[idx], dtype=torch.float32)
        
        # Get OCT image
        if self.participant_ids and idx < len(self.participant_ids):
            pid = self.participant_ids[idx]
            image_path = self.image_paths.get(pid)
            oct_image = self._load_oct_image(image_path) if image_path else torch.zeros(3, *self.image_size)
        else:
            oct_image = torch.zeros(3, *self.image_size)
        
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        sample = {
            'oct_image': oct_image,
            'biomarkers': biomarkers
        }
        
        return sample, target


class METAGEDataset(UKBiobankDataset):
    """Dataset for METAGE algorithm with NMR metabolomics."""
    
    def __init__(
        self,
        data_path: str,
        metabolomics_file: str,
        sequence_length: int = 5,
        stride: int = 1,
        **kwargs
    ):
        """
        Initialize METAGE dataset.
        
        Args:
            data_path: Path to UK Biobank data
            metabolomics_file: Path to NMR metabolomics data
            sequence_length: Length of temporal sequences
            stride: Stride for creating sequences
            **kwargs: Additional arguments for base class
        """
        self.metabolomics_file = Path(metabolomics_file)
        self.sequence_length = sequence_length
        self.stride = stride
        
        # NMR metabolomics field IDs (subset for demonstration)
        nmr_features = [f"23{i:03d}-0.0" for i in range(400, 650)]  # 250 features
        
        super().__init__(data_path, features=nmr_features, **kwargs)
        
        # Create temporal sequences
        self._create_sequences()
    
    def _create_sequences(self) -> None:
        """Create temporal sequences from metabolomics data."""
        sequences = []
        sequence_targets = []
        
        # For each participant, create overlapping sequences
        for i in range(0, len(self.data) - self.sequence_length + 1, self.stride):
            seq = self.data[i:i + self.sequence_length]
            target = self.targets[i + self.sequence_length - 1]
            
            sequences.append(seq)
            sequence_targets.append(target)
        
        self.sequences = np.array(sequences, dtype=np.float32)
        self.sequence_targets = np.array(sequence_targets, dtype=np.float32)
        
        # Add time encoding
        self.time_stamps = self._generate_time_encoding()
    
    def _generate_time_encoding(self) -> np.ndarray:
        """Generate time encodings for sequences."""
        # Sinusoidal encoding
        time_stamps = np.arange(self.sequence_length).reshape(-1, 1)
        dim = 32  # Time encoding dimension
        
        div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
        
        time_encoding = np.zeros((self.sequence_length, dim))
        time_encoding[:, 0::2] = np.sin(time_stamps * div_term)
        time_encoding[:, 1::2] = np.cos(time_stamps * div_term)
        
        return time_encoding.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get temporal sequence sample."""
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        time_encoding = torch.tensor(self.time_stamps, dtype=torch.float32)
        target = torch.tensor(self.sequence_targets[idx], dtype=torch.float32)
        
        sample = {
            'sequence': sequence,
            'time_encoding': time_encoding
        }
        
        return sample, target


class StratifiedAgeSampler(Sampler):
    """Stratified sampler for age-balanced batches."""
    
    def __init__(self, ages: np.ndarray, batch_size: int, n_bins: int = 10):
        """
        Initialize stratified sampler.
        
        Args:
            ages: Array of participant ages
            batch_size: Batch size
            n_bins: Number of age bins for stratification
        """
        self.ages = ages
        self.batch_size = batch_size
        self.n_bins = n_bins
        
        # Create age bins
        self.age_bins = pd.qcut(ages, n_bins, labels=False, duplicates='drop')
        self.bin_indices = {i: np.where(self.age_bins == i)[0] for i in range(n_bins)}
    
    def __iter__(self):
        """Generate stratified indices."""
        indices = []
        
        # Sample from each bin
        samples_per_bin = self.batch_size // self.n_bins
        
        while len(indices) < len(self.ages):
            batch_indices = []
            for bin_idx in range(self.n_bins):
                if len(self.bin_indices[bin_idx]) > 0:
                    bin_samples = np.random.choice(
                        self.bin_indices[bin_idx],
                        min(samples_per_bin, len(self.bin_indices[bin_idx])),
                        replace=True
                    )
                    batch_indices.extend(bin_samples)
            
            np.random.shuffle(batch_indices)
            indices.extend(batch_indices)
        
        return iter(indices[:len(self.ages)])
    
    def __len__(self) -> int:
        return len(self.ages)


def create_data_loaders(
    dataset_class: type,
    data_config: Any,
    training_config: Any,
    model_config: Any,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        dataset_class: Dataset class to use
        data_config: Data configuration
        training_config: Training configuration
        model_config: Model-specific configuration
        **dataset_kwargs: Additional arguments for dataset
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create full dataset
    full_dataset = dataset_class(
        data_path=data_config.ukbb_data_path,
        cache_dir=data_config.cache_dir,
        **dataset_kwargs
    )
    
    # Split into train, val, test
    n_samples = len(full_dataset)
    indices = np.arange(n_samples)
    
    # Get targets for stratification
    targets = full_dataset.targets if hasattr(full_dataset, 'targets') else None
    
    # Split indices
    train_idx, test_idx = train_test_split(
        indices,
        test_size=data_config.test_ratio,
        random_state=training_config.seed,
        stratify=targets if data_config.stratify_by_age and targets is not None else None
    )
    
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=data_config.val_ratio / (1 - data_config.test_ratio),
        random_state=training_config.seed,
        stratify=targets[train_idx] if data_config.stratify_by_age and targets is not None else None
    )
    
    # Create subset datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
    
    # Create samplers if needed
    train_sampler = None
    if data_config.stratify_by_age and targets is not None:
        train_sampler = StratifiedAgeSampler(
            targets[train_idx],
            data_config.batch_size
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers,
        prefetch_factor=data_config.prefetch_factor,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        persistent_workers=data_config.persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config.batch_size * 2,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory
    )
    
    return train_loader, val_loader, test_loader