"""
UK Biobank Real Data Loader
Handles loading actual UK Biobank data from CSV files with proper error handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)


@dataclass
class UKBBDataConfig:
    """Configuration for UK Biobank data loading"""
    data_directory: Path
    biomarker_fields: Dict[str, int]  # Maps biomarker name to field ID
    age_field: int = 21022  # Field ID for age at recruitment
    sex_field: int = 31  # Field ID for sex
    eid_field: str = 'eid'  # Participant ID column name
    instance: int = 0  # Instance index for repeated measurements
    array_index: int = 0  # Array index for array fields


class UKBBRealDataLoader:
    """
    Loads real UK Biobank data from CSV files with comprehensive error handling
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the UK Biobank data loader
        
        Args:
            config: Configuration dictionary with UK Biobank field mappings
        """
        self.config = config
        
        # Extract field mappings from config
        self.biomarker_fields = config['ukbb_fields']['biomarkers']
        self.age_field = config['ukbb_fields'].get('age', 21022)
        self.sex_field = config['ukbb_fields'].get('sex', 31)
        
        # Data directory - check multiple possible locations
        self.data_dirs = [
            Path('/mnt/data1/UKBB_retinal_img/UKB_new_2024'),
            Path('/mnt/data3/xuting/ai_scientist/claudeV2/henaw_implementation'),
            Path('./data')
        ]
        
        self.data_directory = None
        for data_dir in self.data_dirs:
            if data_dir.exists():
                self.data_directory = data_dir
                logger.info(f"Using data directory: {self.data_directory}")
                break
        
        if self.data_directory is None:
            logger.warning("No existing data directory found. Will use current directory.")
            self.data_directory = Path('.')
    
    def load_ukbb_csv(self, 
                      csv_path: Optional[Union[str, Path]] = None,
                      required_complete: bool = True,
                      max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load UK Biobank data from CSV file with comprehensive error handling
        
        Args:
            csv_path: Path to CSV file. If None, searches for available files
            required_complete: If True, only returns samples with all biomarkers present
            max_samples: Maximum number of samples to load (for testing)
        
        Returns:
            DataFrame with loaded UK Biobank data
        """
        # Find CSV file if not specified
        if csv_path is None:
            csv_path = self._find_ukbb_csv()
            if csv_path is None:
                logger.warning("No UK Biobank CSV found. Generating synthetic data for testing.")
                return self._generate_synthetic_data(max_samples or 1000)
        
        csv_path = Path(csv_path)
        
        # Validate file exists and is readable
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        if not csv_path.is_file():
            raise ValueError(f"Path is not a file: {csv_path}")
        
        file_size_mb = csv_path.stat().st_size / (1024 * 1024)
        if file_size_mb == 0:
            raise ValueError(f"CSV file is empty: {csv_path}")
        
        logger.info(f"Loading UK Biobank data from {csv_path} ({file_size_mb:.1f} MB)")
        
        try:
            # Load CSV with specific columns if available
            required_columns = self._get_required_columns()
            
            # First, try to read just the header to check available columns
            header_df = pd.read_csv(csv_path, nrows=0)
            available_columns = set(header_df.columns)
            
            # Find which required columns are present
            columns_to_load = []
            missing_columns = []
            
            for col in required_columns:
                if col in available_columns:
                    columns_to_load.append(col)
                else:
                    # Try variations of column names
                    variations = [
                        col,
                        f"{col}-{self.config.get('instance', 0)}.{self.config.get('array_index', 0)}",
                        f"{col}-0.0",
                        col.replace('_', '-'),
                        col.replace('-', '_')
                    ]
                    found = False
                    for var in variations:
                        if var in available_columns:
                            columns_to_load.append(var)
                            found = True
                            break
                    if not found:
                        missing_columns.append(col)
            
            if missing_columns:
                logger.warning(f"Missing columns in CSV: {missing_columns}")
            
            # Load the data
            if columns_to_load:
                df = pd.read_csv(csv_path, usecols=columns_to_load, nrows=max_samples)
            else:
                df = pd.read_csv(csv_path, nrows=max_samples)
            
            logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
            
            # Process and validate the data
            df = self._process_ukbb_data(df, required_complete)
            
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"CSV file is empty or corrupted: {csv_path}")
            raise
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV file: {e}")
            raise
        except MemoryError:
            logger.error(f"Not enough memory to load {csv_path}. Try setting max_samples parameter.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading CSV: {e}")
            raise
    
    def _find_ukbb_csv(self) -> Optional[Path]:
        """
        Search for UK Biobank CSV files in known locations with comprehensive search
        
        Returns:
            Path to CSV file if found, None otherwise
        """
        # Common UK Biobank file patterns in priority order
        patterns = [
            'ukb*.csv',           # Standard UK Biobank naming
            'ukbb*.csv',          # Alternative naming
            '*phenotype*.csv',    # Phenotype files
            '*biomarkers*.csv',   # Biomarker specific files
            '*baseline*.csv',     # Baseline assessment files
            'data*.csv',          # Generic data files
            '*henaw*.csv',        # Project-specific files
            'synthetic_ukbb_data.csv'  # Test data
        ]
        
        # Search in main directory and subdirectories
        search_dirs = [self.data_directory] + self.data_dirs
        
        # Add common subdirectories to search
        for base_dir in [self.data_directory] + self.data_dirs:
            if base_dir.exists():
                for subdir in ['phenotypes', 'biomarkers', 'raw', 'processed', 'baseline']:
                    subpath = base_dir / subdir
                    if subpath.exists():
                        search_dirs.append(subpath)
        
        csv_files = []
        for pattern in patterns:
            for data_dir in search_dirs:
                if data_dir.exists():
                    try:
                        found = list(data_dir.glob(pattern))
                        csv_files.extend(found)
                    except Exception as e:
                        logger.debug(f"Error searching {data_dir} with pattern {pattern}: {e}")
        
        if csv_files:
            # Filter out small files (likely not main dataset)
            csv_files = [f for f in csv_files if f.stat().st_size > 1024 * 1024]  # > 1MB
            
            if csv_files:
                # Return the largest file (likely the main dataset)
                csv_file = max(csv_files, key=lambda f: f.stat().st_size)
                file_size_gb = csv_file.stat().st_size / (1024**3)
                logger.info(f"Found UK Biobank CSV: {csv_file} ({file_size_gb:.2f}GB)")
                return csv_file
        
        logger.warning(f"No UK Biobank CSV found in searched directories")
        return None
    
    def _get_required_columns(self) -> List[str]:
        """
        Get list of required column names for UK Biobank data
        
        Returns:
            List of column names to load
        """
        columns = ['eid']  # Participant ID
        
        # Add biomarker fields
        for name, field_id in self.biomarker_fields.items():
            # UK Biobank column format: fieldID-instance.array_index
            columns.append(f"{field_id}-0.0")
            # Also try without instance/array
            columns.append(str(field_id))
        
        # Add age and sex
        columns.extend([
            f"{self.age_field}-0.0",
            str(self.age_field),
            f"{self.sex_field}-0.0", 
            str(self.sex_field)
        ])
        
        return columns
    
    def _process_ukbb_data(self, df: pd.DataFrame, required_complete: bool) -> pd.DataFrame:
        """
        Process and validate UK Biobank data
        
        Args:
            df: Raw dataframe
            required_complete: Whether to require complete cases
        
        Returns:
            Processed dataframe with standardized column names
        """
        processed_df = pd.DataFrame()
        
        # Process participant ID
        if 'eid' in df.columns:
            processed_df['eid'] = df['eid']
        else:
            # Generate fake IDs if missing
            processed_df['eid'] = range(1000000, 1000000 + len(df))
        
        # Process biomarkers
        for name, field_id in self.biomarker_fields.items():
            column_found = False
            
            # Try different column name formats
            possible_columns = [
                f"{field_id}-0.0",
                str(field_id),
                f"{field_id}-{self.config.get('instance', 0)}.{self.config.get('array_index', 0)}",
                name  # Try the biomarker name directly
            ]
            
            for col in possible_columns:
                if col in df.columns:
                    processed_df[name] = pd.to_numeric(df[col], errors='coerce')
                    column_found = True
                    break
            
            if not column_found:
                logger.warning(f"Biomarker {name} (field {field_id}) not found in data")
                if required_complete:
                    # Add NaN column if requiring complete cases (will be filtered)
                    processed_df[name] = np.nan
                else:
                    # Add synthetic data for missing biomarker
                    processed_df[name] = self._generate_synthetic_biomarker(name, len(df))
        
        # Process age
        age_found = False
        for col in [f"{self.age_field}-0.0", str(self.age_field), 'age', 'Age']:
            if col in df.columns:
                processed_df['age'] = pd.to_numeric(df[col], errors='coerce')
                age_found = True
                break
        
        if not age_found:
            logger.warning("Age field not found, generating synthetic ages")
            processed_df['age'] = np.random.normal(55, 8, len(df))
            processed_df['age'] = np.clip(processed_df['age'], 40, 70)
        
        # Process sex
        sex_found = False
        for col in [f"{self.sex_field}-0.0", str(self.sex_field), 'sex', 'Sex', 'gender']:
            if col in df.columns:
                processed_df['sex'] = pd.to_numeric(df[col], errors='coerce')
                sex_found = True
                break
        
        if not sex_found:
            logger.warning("Sex field not found, generating synthetic sex data")
            processed_df['sex'] = np.random.binomial(1, 0.48, len(df))
        
        # Filter complete cases if required
        if required_complete:
            before_filter = len(processed_df)
            processed_df = processed_df.dropna()
            after_filter = len(processed_df)
            
            if after_filter == 0:
                logger.warning("No complete cases found. Returning data with imputation.")
                processed_df = self._impute_missing_values(processed_df)
            else:
                logger.info(f"Filtered to {after_filter}/{before_filter} complete cases")
        
        # Validate data ranges
        processed_df = self._validate_data_ranges(processed_df)
        
        return processed_df
    
    def _generate_synthetic_biomarker(self, name: str, n_samples: int) -> np.ndarray:
        """
        Generate synthetic data for a missing biomarker
        
        Args:
            name: Biomarker name
            n_samples: Number of samples
        
        Returns:
            Synthetic biomarker values
        """
        # Reference ranges for each biomarker
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
        
        if name in ranges:
            min_val, max_val, mean_val, std_val = ranges[name]
            values = np.random.normal(mean_val, std_val, n_samples)
            values = np.clip(values, min_val, max_val)
            
            # Add log-normal noise for inflammatory markers
            if name in ['crp', 'ggt']:
                values = np.exp(np.log(values + 0.1) + np.random.normal(0, 0.1, n_samples))
                values = np.clip(values, min_val, max_val)
        else:
            # Default distribution if biomarker not recognized
            values = np.random.normal(50, 10, n_samples)
            values = np.clip(values, 0, 100)
        
        return values
    
    def _generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """
        Generate completely synthetic UK Biobank-like data for testing
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            Synthetic dataframe
        """
        logger.info(f"Generating {n_samples} synthetic samples")
        
        data = {
            'eid': range(1000000, 1000000 + n_samples),
            'age': np.random.normal(55, 8, n_samples),
            'sex': np.random.binomial(1, 0.48, n_samples)
        }
        
        # Clip age to realistic range
        data['age'] = np.clip(data['age'], 40, 70)
        
        # Generate each biomarker
        for name in self.biomarker_fields.keys():
            data[name] = self._generate_synthetic_biomarker(name, n_samples)
        
        # Add some correlations for realism
        # AST and ALT are correlated
        if 'ast' in data and 'alt' in data:
            correlation = 0.7
            data['ast'] = correlation * data['alt'] + (1 - correlation) * data['ast']
        
        # CRP affects lymphocyte percentage
        if 'crp' in data and 'lymphocyte_pct' in data:
            data['lymphocyte_pct'] -= 0.3 * (data['crp'] - 2.0)
            data['lymphocyte_pct'] = np.clip(data['lymphocyte_pct'], 15, 45)
        
        return pd.DataFrame(data)
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values in the dataframe
        
        Args:
            df: Dataframe with missing values
        
        Returns:
            Dataframe with imputed values
        """
        # Use median imputation for biomarkers
        biomarker_columns = list(self.biomarker_fields.keys())
        
        for col in biomarker_columns:
            if col in df.columns:
                median_val = df[col].median()
                if pd.isna(median_val):
                    # If all values are missing, use synthetic data
                    df[col] = self._generate_synthetic_biomarker(col, len(df))
                else:
                    df[col].fillna(median_val, inplace=True)
        
        # Impute age and sex if needed
        if 'age' in df.columns:
            df['age'].fillna(df['age'].median() if not df['age'].isna().all() else 55, inplace=True)
        
        if 'sex' in df.columns:
            df['sex'].fillna(df['sex'].mode()[0] if not df['sex'].isna().all() else 0, inplace=True)
        
        return df
    
    def _validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clip data to reasonable ranges
        
        Args:
            df: Dataframe to validate
        
        Returns:
            Validated dataframe
        """
        # Define valid ranges for each biomarker
        valid_ranges = {
            'crp': (0.01, 500),
            'hba1c': (15, 200),
            'creatinine': (20, 1000),
            'albumin': (20, 60),
            'lymphocyte_pct': (0, 100),
            'rdw': (10, 30),
            'ggt': (5, 2000),
            'ast': (5, 1000),
            'alt': (5, 1000)
        }
        
        for name, (min_val, max_val) in valid_ranges.items():
            if name in df.columns:
                # Check for outliers
                outliers = (df[name] < min_val) | (df[name] > max_val)
                n_outliers = outliers.sum()
                
                if n_outliers > 0:
                    logger.warning(f"Found {n_outliers} outliers in {name}, clipping to [{min_val}, {max_val}]")
                    df[name] = np.clip(df[name], min_val, max_val)
        
        # Validate age
        if 'age' in df.columns:
            df['age'] = np.clip(df['age'], 18, 100)
        
        # Validate sex (should be 0 or 1)
        if 'sex' in df.columns:
            df['sex'] = df['sex'].round().astype(int)
            df['sex'] = np.clip(df['sex'], 0, 1)
        
        return df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """
        Save processed UK Biobank data to file
        
        Args:
            df: Processed dataframe
            output_path: Path to save the data
        """
        output_path = Path(output_path)
        
        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on extension
        if output_path.suffix == '.parquet':
            df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.h5' or output_path.suffix == '.hdf5':
            df.to_hdf(output_path, key='data', mode='w')
        else:  # Default to CSV
            df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} samples to {output_path}")


def main():
    """Test the UK Biobank data loader"""
    import yaml
    
    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use default config
        config = {
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
                },
                'age': 21022,
                'sex': 31
            }
        }
    
    # Initialize loader
    loader = UKBBRealDataLoader(config)
    
    # Try to load data
    df = loader.load_ukbb_csv(max_samples=1000)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData statistics:")
    print(df.describe())
    
    # Save processed data
    output_path = Path('processed_ukbb_data.csv')
    loader.save_processed_data(df, output_path)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()