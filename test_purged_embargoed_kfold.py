"""
Test suite for PurgedKFoldCVWithEmbargos implementation.
"""
import pandas as pd
import numpy as np
import pytest
from purged_embargoed_kfold import PurgedKFoldCVWithEmbargos


def test_basic_initialization():
    """Test basic initialization with a simple DataFrame."""
    df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, 
                     index=pd.date_range('2020-01-01', periods=5, freq='D'))
    cv = PurgedKFoldCVWithEmbargos(df)
    assert len(cv.df) == 5
    assert len(cv.label_end_times) == 5


def test_initialization_with_label_end_times():
    """Test initialization with explicit label end times."""
    df = pd.DataFrame({'value': [1, 2, 3, 4, 5]}, 
                     index=pd.date_range('2020-01-01', periods=5, freq='D'))
    label_end_times = pd.Series(
        pd.date_range('2020-01-02', periods=5, freq='D'),
        index=df.index
    )
    cv = PurgedKFoldCVWithEmbargos(df, label_end_times)
    assert len(cv.label_end_times) == 5


def test_invalid_initialization():
    """Test that invalid inputs raise appropriate errors."""
    df = pd.DataFrame({'value': [1, 2, 3]})
    
    # Test invalid label_end_times type
    with pytest.raises(TypeError):
        PurgedKFoldCVWithEmbargos(df, label_end_times=[1, 2, 3])
    
    # Test empty DataFrame
    with pytest.raises(ValueError):
        PurgedKFoldCVWithEmbargos(pd.DataFrame())
    
    # Test non-DataFrame input
    with pytest.raises(TypeError):
        PurgedKFoldCVWithEmbargos([1, 2, 3])


def test_basic_k_fold_split():
    """Test that K-fold splitting covers entire dataset."""
    df = pd.DataFrame({'value': range(100)},
                     index=pd.date_range('2020-01-01', periods=100, freq='D'))
    cv = PurgedKFoldCVWithEmbargos(df)
    
    n_splits = 5
    splits = list(cv.purged_k_fold_cv_with_embargos(n_splits=n_splits, embargo_period_pct=0.0))
    
    # Should have n_splits folds
    assert len(splits) == n_splits
    
    # Collect all test indices
    all_test_indices = []
    for train_idx, test_idx in splits:
        all_test_indices.extend(test_idx.tolist())
        # Train and test should not overlap
        assert len(set(train_idx) & set(test_idx)) == 0
    
    # All samples should be in test set exactly once
    assert len(all_test_indices) == len(df)
    assert set(all_test_indices) == set(df.index)


def test_k_fold_with_embargo():
    """Test that embargo removes training samples after test fold."""
    df = pd.DataFrame({'value': range(100)},
                     index=pd.date_range('2020-01-01', periods=100, freq='D'))
    cv = PurgedKFoldCVWithEmbargos(df)
    
    n_splits = 5
    embargo_pct = 0.1  # 10% embargo = 10 samples
    
    splits_no_embargo = list(cv.purged_k_fold_cv_with_embargos(n_splits=n_splits, embargo_period_pct=0.0))
    splits_with_embargo = list(cv.purged_k_fold_cv_with_embargos(n_splits=n_splits, embargo_period_pct=embargo_pct))
    
    # With embargo, training set should be smaller (or equal for last fold)
    for i in range(n_splits):
        train_no_emb, test_no_emb = splits_no_embargo[i]
        train_with_emb, test_with_emb = splits_with_embargo[i]
        
        # Test sets should be the same
        assert list(test_no_emb) == list(test_with_emb)
        
        # Training set with embargo should be smaller or equal
        assert len(train_with_emb) <= len(train_no_emb)


def test_purge_with_overlapping_labels():
    """Test that purging removes training samples with overlapping label intervals."""
    # Create data where labels span multiple days
    df = pd.DataFrame({'value': range(10)},
                     index=pd.date_range('2020-01-01', periods=10, freq='D'))
    
    # Each label spans 3 days: day i to day i+2
    label_end_times = pd.Series(
        df.index + pd.Timedelta(days=2),
        index=df.index
    )
    
    cv = PurgedKFoldCVWithEmbargos(df, label_end_times)
    
    # With 2 splits, we expect significant purging due to overlapping labels
    splits = list(cv.purged_k_fold_cv_with_embargos(n_splits=2, embargo_period_pct=0.0))
    
    for train_idx, test_idx in splits:
        # Verify no overlap between training and test label intervals
        train_starts = train_idx
        train_ends = cv.label_end_times[train_idx]
        test_starts = test_idx
        test_ends = cv.label_end_times[test_idx]
        
        # Check that no training interval overlaps with any test interval
        for t_start, t_end in zip(train_starts, train_ends):
            for test_start, test_end in zip(test_starts, test_ends):
                # No overlap: t_end < test_start OR t_start > test_end
                assert t_end < test_start or t_start > test_end


def test_validation_errors():
    """Test that invalid parameters raise appropriate errors."""
    df = pd.DataFrame({'value': range(10)},
                     index=pd.date_range('2020-01-01', periods=10, freq='D'))
    cv = PurgedKFoldCVWithEmbargos(df)
    
    # Test n_splits < 2
    with pytest.raises(ValueError, match="n_splits must be an integer >= 2"):
        list(cv.purged_k_fold_cv_with_embargos(n_splits=1))
    
    # Test negative embargo
    with pytest.raises(ValueError, match="embargo_period_pct must be >= 0"):
        list(cv.purged_k_fold_cv_with_embargos(n_splits=2, embargo_period_pct=-0.1))
    
    # Test n_splits > n_samples
    with pytest.raises(ValueError, match="n_splits.*cannot be greater than number of samples"):
        list(cv.purged_k_fold_cv_with_embargos(n_splits=20))


def test_integer_index():
    """Test that the implementation works with integer index."""
    df = pd.DataFrame({'value': range(20)}, index=range(20))
    cv = PurgedKFoldCVWithEmbargos(df)
    
    splits = list(cv.purged_k_fold_cv_with_embargos(n_splits=4, embargo_period_pct=0.05))
    
    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert len(test_idx) > 0
        assert len(set(train_idx) & set(test_idx)) == 0


def test_uneven_splits():
    """Test that uneven splits work correctly (when n_samples % n_splits != 0)."""
    df = pd.DataFrame({'value': range(23)},  # 23 is not evenly divisible by 5
                     index=pd.date_range('2020-01-01', periods=23, freq='D'))
    cv = PurgedKFoldCVWithEmbargos(df)
    
    n_splits = 5
    splits = list(cv.purged_k_fold_cv_with_embargos(n_splits=n_splits, embargo_period_pct=0.0))
    
    assert len(splits) == n_splits
    
    # Verify all samples are used exactly once in test sets
    all_test_indices = []
    for train_idx, test_idx in splits:
        all_test_indices.extend(test_idx.tolist())
    
    assert len(all_test_indices) == len(df)
    assert set(all_test_indices) == set(df.index)


def test_edge_case_two_splits():
    """Test the minimum case of 2 splits."""
    df = pd.DataFrame({'value': range(10)},
                     index=pd.date_range('2020-01-01', periods=10, freq='D'))
    cv = PurgedKFoldCVWithEmbargos(df)
    
    splits = list(cv.purged_k_fold_cv_with_embargos(n_splits=2, embargo_period_pct=0.0))
    
    assert len(splits) == 2
    train1, test1 = splits[0]
    train2, test2 = splits[1]
    
    # Each should have approximately half the data as test
    assert 4 <= len(test1) <= 6
    assert 4 <= len(test2) <= 6
    
    # All indices should be covered
    all_test = set(test1.tolist() + test2.tolist())
    assert len(all_test) == len(df)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
