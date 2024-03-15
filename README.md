# PurgeEmbargoKFoldCV

A Python implementation of K-fold cross-validation tailored for time-series data, integrating purging and embargo techniques to effectively mitigate data leakage. This approach is particularly suitable for financial modeling and other scenarios where preserving the temporal order is crucial.

## Installation

To use PurgeEmbargoKFoldCV, first ensure you have Python and pandas installed in your environment. Then, follow these steps:

1. Clone this repository to your local machine or download the `purge_embargo_kfold.py` file directly.
   ```
   git clone https://github.com/Furkan-rgb/PurgeEmbargoKFoldCV.git
   ```
2. Install pandas, if you haven't already:
   ```
   pip install pandas
   ```

## Usage

Here's a simple example on how to use PurgeEmbargoKFoldCV in your project:

```python
import pandas as pd
from purge_embargo_kfold import PurgedKFoldCVWithEmbargos

# Example DataFrame
date_range = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
data = {'feature': range(len(date_range))}
df = pd.DataFrame(data, index=date_range)

# Initialize the class
purge_cv = PurgedKFoldCVWithEmbargos(df)

# Generate splits
n_splits = 5
train_size = 0.7
embargo_period_pct = 0.1
splits = purge_cv.purged_k_fold_cv_with_embargos(n_splits, train_size, embargo_period_pct)

for train_idx, test_idx in splits:
    print(f"Train indices: {train_idx}, Test indices: {test_idx}")
```
