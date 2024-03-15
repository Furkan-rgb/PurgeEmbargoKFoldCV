import pandas as pd
import numpy as np

from purged_embargoed_kfold import PurgedKFoldCVWithEmbargos

# Create a simple DataFrame with DatetimeIndex
date_range = pd.date_range(start="2020-01-01", end="2020-01-30")
data = np.random.randn(len(date_range), 1)  # Random data
df = pd.DataFrame(data, index=date_range, columns=["Value"])

# Initialize the class with the DataFrame
purge_cv = PurgedKFoldCVWithEmbargos(df)

# Test the purged_k_fold_cv_with_embargos method
n_splits = 3
train_size = 0.7
embargo_period_pct = 0.1
splits = purge_cv.purged_k_fold_cv_with_embargos(
    n_splits, train_size, embargo_period_pct
)

# Output the generated splits
for train_indices, test_indices in splits:
    print(
        f"Train start: {train_indices.start}, Train end: {train_indices.stop}, Test start: {test_indices.start}, Test end: {test_indices.stop}"
    )
