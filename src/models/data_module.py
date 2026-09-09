import os
import re
import glob
import random
import gc
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
import lightning.pytorch as pl
from typing import Dict, List, Tuple, Optional, Union, Callable
from tqdm import tqdm  # For progress bars


class TSFIterableDataset(IterableDataset):
    """Memory-efficient dataset that generates time series samples on-the-fly"""
    
    def __init__(
        self,
        series_data: List[Tuple[str, np.ndarray]],
        seq_length: int = 100,
        forecast_horizon: int = 30,
        stride: int = 1,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        shuffle: bool = True,
        seed: int = 42,
        verbose: bool = False,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            series_data: List of tuples (series_id, values)
            seq_length: Input sequence length
            forecast_horizon: Number of steps to forecast
            stride: Stride for sliding window
            transform: Transform to apply to input sequences
            target_transform: Transform to apply to target sequences
            shuffle: Whether to shuffle the series order
            seed: Random seed for shuffling
            verbose: Whether to show progress bars
            max_samples: Optional cap on number of samples to generate per epoch
        """
        self.series_data = series_data
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.transform = transform
        self.target_transform = target_transform
        self.shuffle = shuffle
        self.seed = seed
        self.verbose = verbose
        self.max_samples = max_samples
        
        # Calculate total number of samples without storing them
        self.total_samples = 0
        self.sample_indices = []  # Maps indices to (series_idx, window_idx)
        
        if verbose:
            series_iterator = tqdm(enumerate(series_data), total=len(series_data), desc="Indexing series")
        else:
            series_iterator = enumerate(series_data)
            
        for series_idx, (_, values) in series_iterator:
            if len(values) < seq_length + forecast_horizon:
                continue
                
            n_windows = max(0, (len(values) - seq_length - forecast_horizon + 1 + stride - 1) // stride)
            
            # Only store indices, not the actual data
            for window_idx in range(n_windows):
                self.sample_indices.append((series_idx, window_idx * stride))
                
            self.total_samples += n_windows
        
        # Cap the total samples if specified
        if max_samples and max_samples < self.total_samples:
            self.total_samples = max_samples
            self.sample_indices = self.sample_indices[:max_samples]
            
        if verbose:
            print(f"Dataset contains {self.total_samples} samples from {len(series_data)} series")
            print(f"Estimated memory savings: {self.total_samples * (seq_length + forecast_horizon) * 4 / (1024**2):.1f} MB")
    
    def __iter__(self):
        # Set worker seed for reproducibility
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        
        # Create a separate random generator for each worker
        rng = random.Random(self.seed + worker_id)
        
        # Get indices for this worker
        indices = list(self.sample_indices)
        if self.shuffle:
            rng.shuffle(indices)
            
        # Handle multi-worker splitting
        if worker_info is not None:
            per_worker = int(np.ceil(len(indices) / worker_info.num_workers))
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, len(indices))
            indices = indices[start_idx:end_idx]
            
        # Generate samples on-the-fly
        for series_idx, start_pos in indices:
            _, values = self.series_data[series_idx]
            
            # Extract input and target sequences
            x = values[start_pos:start_pos + self.seq_length]
            y = values[start_pos + self.seq_length:start_pos + self.seq_length + self.forecast_horizon]
            
            # Convert to tensors
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # [seq_len, 1]
            y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # [horizon, 1]
            
            # Apply transforms if available
            if self.transform:
                x = self.transform(x)
            if self.target_transform:
                y = self.target_transform(y)
                
            yield x, y
            
            # Add explicit cleanup
            del x, y
            # Every few batches, clean memory aggressively 
            if series_idx % 30 == 0:
                gc.collect()
    
    def __len__(self):
        return self.total_samples


class TSFDataModule(pl.LightningDataModule):
    """Memory-efficient PyTorch Lightning DataModule for TSF time series files"""
    
    def __init__(
        self,
        data_dir: str,
        seq_length: int = 100,
        forecast_horizon: int = 30,
        batch_size: int = 32,
        stride: int = 30,
        num_workers: int = 2,
        normalize: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        file_pattern: str = "*.tsf",
        split_by_series: bool = True,
        min_series_length: int = 0,
        exclude_placeholder_values: bool = True,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        verbose: bool = True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.stride = stride
        self.num_workers = num_workers
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.file_pattern = file_pattern
        self.split_by_series = split_by_series
        self.min_series_length = min_series_length
        self.exclude_placeholder_values = exclude_placeholder_values
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.verbose = verbose
        
        # These will be initialized in setup
        self.train_series = None
        self.val_series = None
        self.test_series = None
        
        print("Init Data Module (Memory-Efficient Version)")
    
    def prepare_data(self):
        """Check if TSF files exist"""
        file_list = glob.glob(os.path.join(self.data_dir, self.file_pattern))
        if not file_list:
            raise FileNotFoundError(f"No TSF files found in {self.data_dir} matching {self.file_pattern}")
        
        if self.verbose:
            print(f"Found {len(file_list)} TSF files in {self.data_dir}")
    
    def setup(self, stage=None):
        """Load and process TSF files with memory efficiency as top priority"""
        if self.train_series is not None and stage != 'test':
            # Already set up - use existing data splits
            return
            
        if self.verbose:
            print(f"Setting up data for stage: {stage if stage else 'all'}")
        
        # Process files one at a time instead of loading everything at once
        file_paths = glob.glob(os.path.join(self.data_dir, self.file_pattern))
        processed_series = []
        
        # Track series from the weather dataset separately
        weather_series = []
        
        # Process each file separately
        for file_path in tqdm(file_paths, desc="Processing files") if self.verbose else file_paths:
            filename = os.path.basename(file_path)
            is_weather = "weather" in filename
    
            
            # Load and process one file
            try:
                # Parse the file
                file_series = self._parse_tsf_file(file_path)
                
                # Process series from this file
                processed_from_file = []
                skipped_from_file = 0
                    
                for series_id, values in file_series:
                    # Skip short series
                    if len(values) < max(self.min_series_length, self.seq_length + self.forecast_horizon):
                        skipped_from_file += 1
                        continue
                        
                    # Handle placeholder values
                    if self.exclude_placeholder_values:
                        first_value = values[0]
                        repeated_mask = values == first_value
                        if np.sum(repeated_mask) > 0.5 * len(values):
                            values[repeated_mask] = np.nan
                            mask = np.isnan(values)
                            if mask.all():
                                skipped_from_file += 1
                                continue
                            indices = np.arange(len(values))
                            valid_indices = indices[~mask]
                            valid_values = values[~mask]
                            values = np.interp(indices, valid_indices, valid_values)
                    
                    # Normalize if required
                    if self.normalize:
                        mean = np.mean(values)
                        std = np.std(values)
                        if std > 0:
                            values = (values - mean) / std
                        else:
                            skipped_from_file += 1
                            continue
                    
                    # Add to the appropriate collection
                    if is_weather: 
                        weather_series.append((series_id, values))
                    else:
                        processed_from_file.append((series_id, values))
                
                # Add processed series from this file
                processed_series.extend(processed_from_file)
                
                if self.verbose:
                    print(f"  - Processed {len(processed_from_file)} series from {filename}")
                    print(f"  - Skipped {skipped_from_file} series")
                    
                # Force memory cleanup after each file
                del file_series
                gc.collect()
                    
            except Exception as e:
                if self.verbose:
                    print(f"Error processing {filename}: {e}")
        
        # Add the limited weather series to the processed set
        if self.verbose and weather_series:
            print(f"Adding {len(weather_series)} weather series to the dataset")
        
        processed_series.extend(weather_series)
        
        # Display statistics if verbose
        if self.verbose and processed_series:
            print(f"Total processed series: {len(processed_series)}")
            lengths = [len(values) for _, values in processed_series]
            print(f"Series length statistics:")
            print(f"  Min: {min(lengths)}")
            print(f"  Max: {max(lengths)}")
            print(f"  Mean: {np.mean(lengths):.1f}")
            print(f"  Median: {np.median(lengths):.1f}")
        
        # Split time series into train/val/test
        random.seed(42)  # For reproducibility
        random.shuffle(processed_series)
        
        train_size = int(self.train_ratio * len(processed_series))
        val_size = int(self.val_ratio * len(processed_series))
        
        self.train_series = processed_series[:train_size]
        self.val_series = processed_series[train_size:train_size + val_size]
        self.test_series = processed_series[train_size + val_size:]
        
        # Clean up to release memory
        del processed_series
        gc.collect()
        
        if self.verbose:
            print(f"Split data into {len(self.train_series)} training series, " 
                f"{len(self.val_series)} validation series, and {len(self.test_series)} test series")
    
    def train_dataloader(self):
        return DataLoader(
            TSFIterableDataset(
                self.train_series, 
                self.seq_length, 
                self.forecast_horizon, 
                self.stride,
                shuffle=True,
                verbose=self.verbose,
                max_samples=self.max_train_samples
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            TSFIterableDataset(
                self.val_series, 
                self.seq_length, 
                self.forecast_horizon, 
                self.stride,
                shuffle=False,
                verbose=self.verbose,
                max_samples=self.max_val_samples
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            TSFIterableDataset(
                self.test_series, 
                self.seq_length, 
                self.forecast_horizon, 
                self.stride,
                shuffle=False,
                verbose=self.verbose,
                max_samples=self.max_test_samples
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    # Keep the original file loading methods as they are
    def _load_all_tsf_files(self):
        """Load all time series from TSF files in directory"""
        file_list = glob.glob(os.path.join(self.data_dir, self.file_pattern))
        all_series = []
        
        if self.verbose:
            print(f"Loading {len(file_list)} TSF files...")
            file_iterator = tqdm(file_list, desc="Loading TSF files")
        else:
            file_iterator = file_list
        
        total_series = 0
        error_count = 0
        
        for file_path in file_iterator:
            try:
                series_from_file = self._parse_tsf_file(file_path)
                all_series.extend(series_from_file)
                total_series += len(series_from_file)
                
                if self.verbose:
                    file_iterator.set_postfix(total_series=total_series)
            except Exception as e:
                error_count += 1
                if self.verbose:
                    print(f"Error loading {os.path.basename(file_path)}: {e}")
        
        if self.verbose and error_count > 0:
            print(f"Encountered errors in {error_count} files")
            
        return all_series
    
    def _parse_tsf_file(self, file_path):
        """Parse a TSF file into a list of (series_id, values) tuples"""
        # Keep your original implementation for this method
        # It's handling the file parsing correctly
        series_list = []
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # Extract metadata
                metadata = {}
                for line in content.split('\n'):
                    if line.startswith('@'):
                        parts = line.strip().split(None, 1)
                        if len(parts) >= 2:
                            key = parts[0][1:]  # Remove the @ symbol
                            value = parts[1]
                            metadata[key] = value
                
                # Find the data section
                data_match = re.search(r'@data\s+(.*?)$', content, re.DOTALL)
                if not data_match:
                    if self.verbose:
                        print(f"No @data section found in {filename}")
                    return []
                
                data_section = data_match.group(1)
                
                # Parse each line in the data section
                lines = data_section.split('\n')
                
                if self.verbose and len(lines) > 200:
                    line_iterator = tqdm(lines, desc=f"Parsing {filename}", leave=False)
                else:
                    line_iterator = lines
                    
                for line in line_iterator:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    # Check for the common format: id:type:values
                    parts = line.split(':', 2)
                    if len(parts) < 3:
                        continue
                    
                    series_id = f"{parts[0]}:{parts[1]}"
                    values_str = parts[2]
                    
                    # Parse values
                    try:
                        values = [float(v) for v in values_str.split(',')]
                        series_list.append((series_id, np.array(values, dtype=np.float32)))
                    except ValueError:
                        continue
                
                if self.verbose:
                    total_points = sum(len(values) for _, values in series_list)
                    print(f"Loaded {len(series_list)} series with {total_points} data points from {filename}")
        
        except Exception as e:
            if self.verbose:
                print(f"Error parsing {filename}: {str(e)}")
            raise
            
        return series_list