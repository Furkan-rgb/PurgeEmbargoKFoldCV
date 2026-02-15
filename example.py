"""
Example usage of PurgedKFoldCVWithEmbargos implementing de Prado's
Purged K-Fold Cross-Validation with Embargo.
"""
import pandas as pd
import numpy as np

from purged_embargoed_kfold import PurgedKFoldCVWithEmbargos

# Example 1: Basic usage with datetime index and instantaneous labels
print("=" * 70)
print("Example 1: Basic K-Fold with Embargo (instantaneous labels)")
print("=" * 70)

date_range = pd.date_range(start="2020-01-01", periods=30, freq='D')
data = np.random.randn(len(date_range), 1)
df = pd.DataFrame(data, index=date_range, columns=["Value"])

# Initialize without label_end_times (assumes instantaneous labels)
cv = PurgedKFoldCVWithEmbargos(df)

# Generate splits with 5-fold CV and 10% embargo
n_splits = 5
embargo_period_pct = 0.1

for fold_idx, (train_idx, test_idx) in enumerate(
    cv.purged_k_fold_cv_with_embargos(n_splits=n_splits, embargo_period_pct=embargo_period_pct)
):
    print(f"\nFold {fold_idx + 1}:")
    print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    print(f"  Test start: {test_idx[0]}, Test end: {test_idx[-1]}")
    print(f"  Train date range: {train_idx[0] if len(train_idx) > 0 else 'N/A'} to "
          f"{train_idx[-1] if len(train_idx) > 0 else 'N/A'}")

# Example 2: Advanced usage with overlapping label intervals
print("\n" + "=" * 70)
print("Example 2: K-Fold with Overlapping Label Intervals")
print("=" * 70)

date_range = pd.date_range(start="2020-01-01", periods=20, freq='D')
df2 = pd.DataFrame({'feature1': np.random.randn(len(date_range)),
                    'feature2': np.random.randn(len(date_range))}, 
                   index=date_range)

# Simulate labels that span 3 days (e.g., for predicting 3-day forward returns)
label_end_times = pd.Series(date_range + pd.Timedelta(days=2), index=date_range)

cv2 = PurgedKFoldCVWithEmbargos(df2, label_end_times=label_end_times)

n_splits = 3
embargo_period_pct = 0.05

print(f"\nLabel intervals span {(label_end_times - date_range).iloc[0].days} days")

for fold_idx, (train_idx, test_idx) in enumerate(
    cv2.purged_k_fold_cv_with_embargos(n_splits=n_splits, embargo_period_pct=embargo_period_pct)
):
    print(f"\nFold {fold_idx + 1}:")
    print(f"  Train size: {len(train_idx)}, Test size: {len(test_idx)}")
    print(f"  Test period: {test_idx[0]} to {test_idx[-1]}")
    
    # Show purging effect
    if len(train_idx) > 0:
        train_label_ends = label_end_times[train_idx]
        test_label_starts = test_idx
        test_label_ends = label_end_times[test_idx]
        
        # Verify no overlap
        test_min = test_label_starts.min()
        test_max = test_label_ends.max()
        
        overlap_check = ((train_idx >= test_min) & (train_idx <= test_max)) | \
                       ((train_label_ends >= test_min) & (train_label_ends <= test_max))
        
        print(f"  Overlapping labels purged: {overlap_check.sum() == 0} (verified)")

print("\n" + "=" * 70)
print("Examples completed successfully!")
print("=" * 70)
