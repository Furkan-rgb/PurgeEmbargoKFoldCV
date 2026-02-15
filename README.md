# PurgeEmbargoKFoldCV

A Python implementation of **de Prado's Purged K-Fold Cross-Validation with Embargo** for time-series data, as described in "Advances in Financial Machine Learning" by Marcos López de Prado.

This implementation prevents data leakage in machine learning models by:
1. **Purging**: Removing training samples whose label intervals overlap with test label intervals
2. **Embargo**: Removing training samples that start shortly after the test period ends

## Why Use This Method?

In time-series financial modeling and other domains where labels depend on future data, standard K-Fold cross-validation can cause severe data leakage:
- **Label Overlap**: A training sample's label might depend on data from the test period
- **Information Flow**: Market information flows forward in time, so samples immediately after the test set may contain information about test outcomes

De Prado's method addresses both issues by purging overlapping labels and embargoing samples close to the test period.

## Installation

To use PurgeEmbargoKFoldCV, ensure you have Python and pandas installed:

1. Clone this repository:
   ```bash
   git clone https://github.com/Furkan-rgb/PurgeEmbargoKFoldCV.git
   cd PurgeEmbargoKFoldCV
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy
   ```

## Usage

### Basic Example (Instantaneous Labels)

For labels that don't depend on future data (instantaneous):

```python
import pandas as pd
import numpy as np
from purged_embargoed_kfold import PurgedKFoldCVWithEmbargos

# Create dataset with datetime index
date_range = pd.date_range(start="2020-01-01", periods=100, freq='D')
df = pd.DataFrame({'feature': np.random.randn(100)}, index=date_range)

# Initialize cross-validator
cv = PurgedKFoldCVWithEmbargos(df)

# Generate 5-fold splits with 10% embargo
for train_idx, test_idx in cv.purged_k_fold_cv_with_embargos(
    n_splits=5, 
    embargo_period_pct=0.1
):
    # Use train_idx and test_idx to select data
    X_train, y_train = df.loc[train_idx], y.loc[train_idx]
    X_test, y_test = df.loc[test_idx], y.loc[test_idx]
    # Train and evaluate your model...
```

### Advanced Example (Overlapping Label Intervals)

For labels that depend on future data (e.g., 5-day forward returns):

```python
import pandas as pd
import numpy as np
from purged_embargoed_kfold import PurgedKFoldCVWithEmbargos

# Create dataset
date_range = pd.date_range(start="2020-01-01", periods=100, freq='D')
df = pd.DataFrame({'price': np.random.randn(100)}, index=date_range)

# Define label end times (e.g., labels use next 5 days)
label_end_times = pd.Series(
    date_range + pd.Timedelta(days=5),
    index=date_range
)

# Initialize with label intervals
cv = PurgedKFoldCVWithEmbargos(df, label_end_times=label_end_times)

# Generate splits - purging will remove overlapping labels
for train_idx, test_idx in cv.purged_k_fold_cv_with_embargos(
    n_splits=5,
    embargo_period_pct=0.05
):
    # Use the indices...
    pass
```

## Parameters

### `PurgedKFoldCVWithEmbargos.__init__`

- **df** (pd.DataFrame): Dataset indexed by sample start times
- **label_end_times** (pd.Series, optional): End time of each label interval, aligned with df.index. If None, assumes instantaneous labels.

### `purged_k_fold_cv_with_embargos`

- **n_splits** (int): Number of folds (must be ≥ 2)
- **embargo_period_pct** (float): Embargo window as fraction of dataset length (default: 0.0)

Returns: Generator yielding (train_indices, test_indices) tuples

## How It Works

1. **Split**: Dataset is divided into K contiguous test folds covering the entire dataset
2. **Initial Training Set**: For each test fold, all other samples are candidates for training
3. **Purge**: Remove training samples whose label intervals overlap with any test label interval
4. **Embargo**: Remove training samples that start within `embargo_period_pct * len(df)` samples after the test fold ends

## Example Output

Running `python example.py` demonstrates both basic and advanced usage:

```
Fold 1:
  Train size: 21, Test size: 6
  Test start: 2020-01-01, Test end: 2020-01-06
  Train date range: 2020-01-10 to 2020-01-30
```

Notice how the embargo removes training samples from Jan 7-9 (right after test fold ends).

## Testing

Run the test suite to verify the implementation:

```bash
pip install pytest
pytest test_purged_embargoed_kfold.py -v
```

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Chapter 7: Cross-Validation in Finance

## License

MIT License - see LICENSE file for details.
