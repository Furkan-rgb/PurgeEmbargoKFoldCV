import pandas as pd

class PurgedKFoldCVWithEmbargos:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the PurgedKFoldCVWithEmbargos class with a DataFrame. The DataFrame's
        index must be a pandas DatetimeIndex.
        
        Parameters:
        - df (pd.DataFrame): The dataset to perform cross-validation on, indexed by datetime.
        """
        self.df = df

    def purged_k_fold_cv_with_embargos(self, n_splits, train_size, embargo_period_pct, expanding_window=False):
        """
        Generate training and test splits with embargo periods from the DataFrame.
        
        Parameters:
        - n_splits (int): The number of splits for cross-validation.
        - train_size (float): The proportion of the dataset to include in the train split.
        - embargo_period_pct (float): The embargo period as a percentage of the dataset.
        - expanding_window (bool): Whether to use an expanding window approach (default: False).
        
        Returns:
        - List of tuples, where each tuple contains two ranges: training indices and test indices.
        """
        n_samples = len(self.df)
        embargo_size = int(n_samples * embargo_period_pct)
        adjusted_n_samples = n_samples - embargo_size * (n_splits - 1)
        split_size = adjusted_n_samples // n_splits
        splits = []

        for i in range(n_splits):
            split_start = i * (split_size + embargo_size)
            split_end = split_start + split_size

            train_start = 0 if expanding_window else split_start
            train_end = split_start + int(split_size * train_size)

            test_start = train_end + embargo_size
            test_end = split_end

            train_range = range(train_start, train_end)
            test_range = range(test_start, test_end)

            # Apply purging to remove overlapping samples
            train_indices = self.df.index[train_range]
            test_indices = self.df.index[test_range]
            purged_train_indices = self.purge_overlapping_samples(train_indices, test_indices, embargo_size)

            splits.append((purged_train_indices, list(test_range)))

        return splits

    def purge_overlapping_samples(self, train_indices, test_indices, embargo_size):
        """
        Purges training samples that are too close to the test set, based on timestamps and an embargo period.
        
        Parameters:
        - train_indices (pd.Index): Timestamp index for the training set.
        - test_indices (pd.Index): Timestamp index for the test set.
        - embargo_size (int): Number of samples to embargo before the test set begins.
        
        Returns:
        - List of int: Indices of training samples that do not overlap with the embargoed test period.
        """
        train_times = train_indices.to_series()
        test_times = test_indices.to_series()
        min_time_to_test = test_times.min() - train_times
        purged_train_times = min_time_to_test[min_time_to_test >= pd.Timedelta(seconds=embargo_size)].index
        return purged_train_times.tolist()
