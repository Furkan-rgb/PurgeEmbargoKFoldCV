import pandas as pd
import numpy as np


class PurgedKFoldCVWithEmbargos:
    def __init__(self, df: pd.DataFrame, label_end_times: pd.Series = None):
        """
        Initializes the PurgedKFoldCVWithEmbargos class implementing de Prado's Purged K-Fold
        Cross-Validation with Embargo for time-series data.

        This method prevents information leakage in financial machine learning by:
        1. Purging training samples whose label intervals overlap with test label intervals
        2. Embargoing training samples that start shortly after the test period ends

        Parameters:
        - df (pd.DataFrame): The dataset to perform cross-validation on. Index should be
                            the start time of each sample (can be DatetimeIndex or any ordered index).
        - label_end_times (pd.Series, optional): End times for each label/sample. Must be aligned
                                                 with df.index. If None, assumes labels end at the
                                                 same time they start (single-point labels).
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if len(df) == 0:
            raise ValueError("df must not be empty")
        
        self.df = df
        
        # Set up label intervals
        if label_end_times is None:
            # If no end times provided, assume labels are instantaneous (start = end)
            self.label_end_times = pd.Series(df.index, index=df.index)
        else:
            if not isinstance(label_end_times, pd.Series):
                raise TypeError("label_end_times must be a pandas Series")
            if len(label_end_times) != len(df):
                raise ValueError("label_end_times must have the same length as df")
            # Ensure label_end_times is aligned with df.index
            self.label_end_times = label_end_times.reindex(df.index)
        
        # Validate that end times are not before start times
        invalid_intervals = self.label_end_times < df.index
        if invalid_intervals.any():
            raise ValueError("label_end_times cannot be before df.index (sample start times)")

    def purged_k_fold_cv_with_embargos(self, n_splits: int, embargo_period_pct: float = 0.0):
        """
        Generate training and test splits using de Prado's Purged K-Fold CV with Embargo.

        The method:
        1. Divides the dataset into K contiguous test folds (covers entire dataset)
        2. For each test fold, uses all other samples as potential training data
        3. Purges training samples whose label intervals overlap with test label intervals
        4. Embargoes training samples that start within the embargo window after test fold ends

        Parameters:
        - n_splits (int): The number of splits for cross-validation. Must be >= 2.
        - embargo_period_pct (float): The embargo period as a percentage of dataset length.
                                     Training samples starting within this fraction of the
                                     dataset length after the test fold end are removed.
                                     Default is 0.0 (no embargo). Must be >= 0.

        Yields:
        - Tuple of (train_indices, test_indices) where both are pandas Index objects
          compatible with the DataFrame index.

        Raises:
        - ValueError: If n_splits < 2 or embargo_period_pct < 0
        """
        # Validation
        if not isinstance(n_splits, int) or n_splits < 2:
            raise ValueError(f"n_splits must be an integer >= 2, got {n_splits}")
        
        if embargo_period_pct < 0:
            raise ValueError(f"embargo_period_pct must be >= 0, got {embargo_period_pct}")
        
        n_samples = len(self.df)
        
        if n_splits > n_samples:
            raise ValueError(f"n_splits ({n_splits}) cannot be greater than number of samples ({n_samples})")
        
        # Calculate embargo size in number of samples
        embargo_size = int(n_samples * embargo_period_pct)
        
        # Create contiguous test folds
        test_fold_size = n_samples // n_splits
        indices = np.arange(n_samples)
        
        for fold_idx in range(n_splits):
            # Determine test fold boundaries
            test_start_idx = fold_idx * test_fold_size
            
            # Last fold gets any remaining samples
            if fold_idx == n_splits - 1:
                test_end_idx = n_samples
            else:
                test_end_idx = test_start_idx + test_fold_size
            
            # Get test indices (positional)
            test_pos_indices = indices[test_start_idx:test_end_idx]
            
            # Get all other indices as potential training indices
            train_pos_indices = np.concatenate([
                indices[:test_start_idx],
                indices[test_end_idx:]
            ])
            
            # Convert positional indices to DataFrame indices
            test_indices = self.df.index[test_pos_indices]
            train_indices = self.df.index[train_pos_indices]
            
            # Apply purge: remove training samples whose labels overlap with test labels
            train_indices = self._purge_overlapping_labels(train_indices, test_indices)
            
            # Apply embargo: remove training samples that start within embargo window after test fold
            if embargo_size > 0 and len(test_indices) > 0:
                train_indices = self._apply_embargo(train_indices, test_indices, embargo_size)
            
            yield train_indices, test_indices

    def _purge_overlapping_labels(self, train_indices, test_indices):
        """
        Purge training samples whose label intervals overlap with any test label interval.

        A training sample overlaps if its label interval [train_start, train_end] overlaps
        with any test interval [test_start, test_end]. Two intervals overlap if:
        train_start <= test_end AND train_end >= test_start

        Parameters:
        - train_indices (pd.Index): Index of potential training samples
        - test_indices (pd.Index): Index of test samples

        Returns:
        - pd.Index: Filtered training indices with overlapping samples removed
        """
        if len(train_indices) == 0 or len(test_indices) == 0:
            return train_indices
        
        # Get test label interval boundaries
        test_start_times = test_indices
        test_end_times = self.label_end_times[test_indices]
        
        # Find the earliest start and latest end in test set
        test_min_start = test_start_times.min()
        test_max_end = test_end_times.max()
        
        # Get training label intervals
        train_start_times = train_indices
        train_end_times = self.label_end_times[train_indices]
        
        # Keep training samples that don't overlap with any test sample
        # No overlap means: train_end < test_min_start OR train_start > test_max_end
        no_overlap = (train_end_times < test_min_start) | (train_start_times > test_max_end)
        
        return train_indices[no_overlap]

    def _apply_embargo(self, train_indices, test_indices, embargo_size):
        """
        Apply embargo by removing training samples that start within embargo_size samples
        after the test fold ends.

        Parameters:
        - train_indices (pd.Index): Index of training samples (after purging)
        - test_indices (pd.Index): Index of test samples
        - embargo_size (int): Number of samples in the embargo window

        Returns:
        - pd.Index: Training indices with embargoed samples removed
        """
        if len(train_indices) == 0 or len(test_indices) == 0:
            return train_indices
        
        # Find the last position of test samples in the original dataframe
        test_positions = self.df.index.get_indexer(test_indices)
        test_end_position = test_positions.max()
        
        # Define embargo window: samples at positions (test_end_position, test_end_position + embargo_size]
        embargo_end_position = min(test_end_position + embargo_size, len(self.df) - 1)
        
        # Get positions of training samples
        train_positions = self.df.index.get_indexer(train_indices)
        
        # Keep training samples that are NOT in the embargo window
        # Embargo window is: test_end_position < pos <= embargo_end_position
        not_in_embargo = (train_positions <= test_end_position) | (train_positions > embargo_end_position)
        
        return train_indices[not_in_embargo]
