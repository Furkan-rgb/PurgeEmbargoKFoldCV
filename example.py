import pandas as pd
import numpy as np
from purged_embargoed_kfold import PurgedKFoldCVWithEmbargos

# Create a simple DataFrame with DatetimeIndex
date_range = pd.date_range(start="2020-01-01", end="2020-01-30")
data = np.random.randn(len(date_range), 1)  # Random data
df = pd.DataFrame(data, index=date_range, columns=["Value"])

# Initialize the class with the DataFrame
purge_cv = PurgedKFoldCVWithEmbargos(df)

# Set parameters
n_splits = 3
train_size = 0.7
embargo_period_pct = 0.1

# Function to print fold information
def print_fold_info(fold_type, splits):
    print(f"{fold_type}:")
    for i, (train_indices, test_indices) in enumerate(splits, 1):
        print(f"Fold {i}:")
        print(f"  Train: {df.index[train_indices.start]} to {df.index[train_indices.stop - 1]}")
        print(f"    (Indices: {train_indices.start} to {train_indices.stop - 1}, Size: {len(train_indices)})")
        print(f"  Test:  {df.index[test_indices.start]} to {df.index[test_indices.stop - 1]}")
        print(f"    (Indices: {test_indices.start} to {test_indices.stop - 1}, Size: {len(test_indices)})")
        print()

# Test the purged_k_fold_cv_with_embargos method without expanding window
splits = purge_cv.purged_k_fold_cv_with_embargos(
    n_splits, train_size, embargo_period_pct, expanding_window=False
)
print_fold_info("Without expanding window", splits)

# Test the purged_k_fold_cv_with_embargos method with expanding window
splits_expanding = purge_cv.purged_k_fold_cv_with_embargos(
    n_splits, train_size, embargo_period_pct, expanding_window=True
)
print_fold_info("With expanding window", splits_expanding)
