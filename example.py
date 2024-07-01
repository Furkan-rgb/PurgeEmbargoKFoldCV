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

# Test the purged_k_fold_cv_with_embargos method without expanding window
print("Without expanding window:")
splits = purge_cv.purged_k_fold_cv_with_embargos(
    n_splits, train_size, embargo_period_pct, expanding_window=False
)

# Output the generated splits
for i, (train_indices, test_indices) in enumerate(splits, 1):
    print(f"Fold {i}:")
    print(f"  Train: {df.index[train_indices[0]]} to {df.index[train_indices[-1]]}")
    print(f"  Test:  {df.index[test_indices[0]]} to {df.index[test_indices[-1]]}")
    print()

# Test the purged_k_fold_cv_with_embargos method with expanding window
print("With expanding window:")
splits_expanding = purge_cv.purged_k_fold_cv_with_embargos(
    n_splits, train_size, embargo_period_pct, expanding_window=True
)

# Output the generated splits for expanding window
for i, (train_indices, test_indices) in enumerate(splits_expanding, 1):
    print(f"Fold {i}:")
    print(f"  Train: {df.index[train_indices[0]]} to {df.index[train_indices[-1]]}")
    print(f"  Test:  {df.index[test_indices[0]]} to {df.index[test_indices[-1]]}")
    print()
