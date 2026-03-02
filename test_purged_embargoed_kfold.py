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


def test_embargo_anchored_to_purge_boundary():
    """
    Test that when label end times extend past the test fold end, the embargo starts
    from the purge boundary (test_max_end position), not from test_end_position.

    Without the fix the embargo window (200, 300] falls entirely inside the already-purged
    region (200, 350], so it has no effect. With the fix the embargo starts at position 350
    and removes samples in (350, 450].
    """
    n = 500
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    df = pd.DataFrame({'value': range(n)}, index=dates)

    # Labels in the test fold (positions 200-249) look 150 bars into the future,
    # so test_max_end is around position 350 in the index.
    label_end_times = pd.Series(df.index, index=df.index)  # default: same as start
    # Extend only the labels that fall inside the test fold (positions 200-249)
    test_fold_end_date = dates[249]
    test_max_end_date = dates[350]  # purge boundary
    for date in dates[200:250]:
        label_end_times[date] = test_max_end_date

    cv = PurgedKFoldCVWithEmbargos(df, label_end_times)

    # Use n_splits=5 so test fold 0 covers positions 0-99, fold 1 covers 100-199,
    # fold 2 covers 200-299 (our target fold), etc.
    # Use a small embargo that would be invisible without the fix:
    # embargo_size = 50 bars (10% of 500)
    embargo_pct = 50 / n  # 0.10

    splits = list(cv.purged_k_fold_cv_with_embargos(n_splits=5, embargo_period_pct=embargo_pct))

    # Fold 2: test covers positions 200-299
    train_idx, test_idx = splits[2]

    # Confirm the test fold is correct
    assert dates[200] in test_idx
    assert dates[299] in test_idx

    # Position 351 should be embargoed (just after purge boundary 350, within embargo window)
    assert dates[351] not in train_idx

    # Position 401 should NOT be embargoed (beyond embargo window end at 400)
    assert dates[401] in train_idx

    # Positions before the test fold (e.g. position 100) should still be in training
    assert dates[100] in train_idx


def test_embargo_invisible_without_fix():
    """
    Confirm that the bug scenario: embargo_size < gap between test_end_position and
    test_max_end_position means the embargo would have been entirely inside the purged
    region. Verify the fix makes the embargo effective.
    """
    n = 200
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    df = pd.DataFrame({'value': range(n)}, index=dates)

    # Labels in the first half (positions 0-99, the test fold) end at position 150.
    # embargo_size = 30 bars. Old code: embargo window (99, 129] — entirely inside
    # purged region (99, 150]. New code: embargo window (150, 180].
    label_end_times = pd.Series(df.index, index=df.index)
    purge_boundary_date = dates[150]
    for date in dates[0:100]:
        label_end_times[date] = purge_boundary_date

    cv = PurgedKFoldCVWithEmbargos(df, label_end_times)

    embargo_pct = 30 / n  # 15%

    splits = list(cv.purged_k_fold_cv_with_embargos(n_splits=2, embargo_period_pct=embargo_pct))
    train_idx, test_idx = splits[0]  # test = first half (positions 0-99)

    # Positions 151-180 should be embargoed with the fix
    for pos in range(151, 181):
        assert dates[pos] not in train_idx, (
            f"Position {pos} should be embargoed but was found in train_idx"
        )

    # Positions 181-199 should remain in training
    for pos in range(181, 200):
        assert dates[pos] in train_idx, (
            f"Position {pos} should be in training but was not found"
        )


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
