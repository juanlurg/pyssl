# ABOUTME: Tests for the SSL strategy classes
# ABOUTME: Validates label selection and integration strategy behaviors

import numpy as np
from ssl_framework.strategies import ConfidenceThreshold, AppendAndGrow, TopKFixedCount, FullReLabeling, ConfidenceWeighting


def test_confidence_threshold_selection():
    """Test that ConfidenceThreshold correctly selects samples based on threshold."""
    # Create test data
    X_unlabeled = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    # Probabilities where some samples are very confident, others are not
    y_proba = np.array([
        [0.9, 0.1],    # Very confident for class 0
        [0.6, 0.4],    # Not confident enough
        [0.05, 0.95],  # Very confident for class 1
        [0.55, 0.45]   # Not confident enough
    ])

    strategy = ConfidenceThreshold(threshold=0.8)
    X_new, y_new, indices = strategy.select_labels(X_unlabeled, y_proba)

    # Should select indices 0 and 2 (the confident ones)
    assert len(X_new) == 2, f"Expected 2 selected samples, got {len(X_new)}"
    assert len(y_new) == 2, f"Expected 2 labels, got {len(y_new)}"
    assert len(indices) == 2, f"Expected 2 indices, got {len(indices)}"

    # Check that correct samples were selected
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_array_equal(y_new, [0, 1])  # Predicted classes for confident samples


def test_confidence_threshold_no_confident_samples():
    """Test behavior when no samples meet the confidence threshold."""
    X_unlabeled = np.array([[1, 2], [3, 4]])
    y_proba = np.array([
        [0.6, 0.4],    # Not confident enough
        [0.55, 0.45]   # Not confident enough
    ])

    strategy = ConfidenceThreshold(threshold=0.8)
    X_new, y_new, indices = strategy.select_labels(X_unlabeled, y_proba)

    # Should return empty arrays
    assert len(X_new) == 0, "Should return empty array when no confident samples"
    assert len(y_new) == 0, "Should return empty array when no confident samples"
    assert len(indices) == 0, "Should return empty array when no confident samples"


def test_append_and_grow_integration():
    """Test that AppendAndGrow correctly concatenates arrays."""
    # Current labeled data
    X_labeled = np.array([[1, 2], [3, 4]])
    y_labeled = np.array([0, 1])

    # New pseudo-labeled data
    X_new = np.array([[5, 6], [7, 8]])
    y_new = np.array([1, 0])

    strategy = AppendAndGrow()
    X_updated, y_updated, weights = strategy.integrate_labels(
        X_labeled, y_labeled, X_new, y_new
    )

    # Check that arrays were correctly concatenated
    expected_X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    expected_y = np.array([0, 1, 1, 0])

    np.testing.assert_array_equal(X_updated, expected_X)
    np.testing.assert_array_equal(y_updated, expected_y)
    assert weights is None, "AppendAndGrow should return None for sample weights"


def test_append_and_grow_empty_new_data():
    """Test AppendAndGrow behavior when no new data is provided."""
    X_labeled = np.array([[1, 2], [3, 4]])
    y_labeled = np.array([0, 1])
    X_new = np.empty((0, 2))  # Empty new data
    y_new = np.empty(0, dtype=int)

    strategy = AppendAndGrow()
    X_updated, y_updated, weights = strategy.integrate_labels(
        X_labeled, y_labeled, X_new, y_new
    )

    # Should return unchanged data
    np.testing.assert_array_equal(X_updated, X_labeled)
    np.testing.assert_array_equal(y_updated, y_labeled)
    assert weights is None


def test_top_k_fixed_count():
    """Test that TopKFixedCount always returns exactly K samples."""
    X_unlabeled = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_proba = np.array([
        [0.8, 0.2],    # confidence: 0.8
        [0.6, 0.4],    # confidence: 0.6
        [0.1, 0.9],    # confidence: 0.9 (highest)
        [0.7, 0.3],    # confidence: 0.7
        [0.55, 0.45]   # confidence: 0.55 (lowest)
    ])

    strategy = TopKFixedCount(k=3)
    X_new, y_new, indices = strategy.select_labels(X_unlabeled, y_proba)

    # Should select exactly 3 samples
    assert len(X_new) == 3, f"Expected 3 selected samples, got {len(X_new)}"
    assert len(y_new) == 3, f"Expected 3 labels, got {len(y_new)}"
    assert len(indices) == 3, f"Expected 3 indices, got {len(indices)}"

    # Should select the top 3 most confident samples (indices 2, 0, 3)
    expected_indices = [2, 0, 3]  # Sorted by confidence (highest first)
    np.testing.assert_array_equal(indices, expected_indices)


def test_top_k_fewer_samples_than_k():
    """Test TopKFixedCount when fewer samples available than K."""
    X_unlabeled = np.array([[1, 2], [3, 4]])
    y_proba = np.array([[0.8, 0.2], [0.6, 0.4]])

    strategy = TopKFixedCount(k=5)  # Ask for more than available
    X_new, y_new, indices = strategy.select_labels(X_unlabeled, y_proba)

    # Should return all available samples
    assert len(X_new) == 2, "Should return all available samples"
    assert len(y_new) == 2, "Should return all available samples"
    assert len(indices) == 2, "Should return all available samples"


def test_full_relabeling():
    """Test that FullReLabeling returns original + new data, ignoring current."""
    # Original labeled data
    X_original = np.array([[1, 1], [2, 2]])
    y_original = np.array([0, 1])

    # Current labeled data (different from original)
    X_current = np.array([[1, 1], [2, 2], [3, 3]])
    y_current = np.array([0, 1, 0])

    # New pseudo-labeled data
    X_new = np.array([[4, 4], [5, 5]])
    y_new = np.array([1, 0])

    strategy = FullReLabeling(X_original, y_original)
    X_updated, y_updated, weights = strategy.integrate_labels(
        X_current, y_current, X_new, y_new
    )

    # Should return original + new (ignoring current)
    expected_X = np.array([[1, 1], [2, 2], [4, 4], [5, 5]])
    expected_y = np.array([0, 1, 1, 0])

    np.testing.assert_array_equal(X_updated, expected_X)
    np.testing.assert_array_equal(y_updated, expected_y)
    assert weights is None, "FullReLabeling should return None for weights"


def test_confidence_weighting():
    """Test that ConfidenceWeighting returns correct sample weights."""
    X_labeled = np.array([[1, 1], [2, 2]])
    y_labeled = np.array([0, 1])
    X_new = np.array([[3, 3], [4, 4]])
    y_new = np.array([1, 0])

    # Probabilities for all unlabeled samples
    y_proba = np.array([
        [0.2, 0.8],    # confidence: 0.8
        [0.9, 0.1],    # confidence: 0.9
        [0.6, 0.4]     # confidence: 0.6 (not selected)
    ])
    indices = np.array([0, 1])  # Selected the first two samples

    strategy = ConfidenceWeighting()
    X_updated, y_updated, weights = strategy.integrate_labels(
        X_labeled, y_labeled, X_new, y_new,
        y_proba=y_proba, indices=indices
    )

    # Check that data was concatenated correctly
    expected_X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    expected_y = np.array([0, 1, 1, 0])
    np.testing.assert_array_equal(X_updated, expected_X)
    np.testing.assert_array_equal(y_updated, expected_y)

    # Check weights: original samples get 1.0, new samples get their confidence
    expected_weights = np.array([1.0, 1.0, 0.8, 0.9])
    np.testing.assert_array_almost_equal(weights, expected_weights)


def test_confidence_weighting_no_proba():
    """Test ConfidenceWeighting fallback when no probabilities provided."""
    X_labeled = np.array([[1, 1]])
    y_labeled = np.array([0])
    X_new = np.array([[2, 2]])
    y_new = np.array([1])

    strategy = ConfidenceWeighting()
    X_updated, y_updated, weights = strategy.integrate_labels(
        X_labeled, y_labeled, X_new, y_new
    )

    # Should fallback to weight 1.0 for all samples
    expected_weights = np.array([1.0, 1.0])
    np.testing.assert_array_equal(weights, expected_weights)


if __name__ == "__main__":
    test_confidence_threshold_selection()
    test_confidence_threshold_no_confident_samples()
    test_append_and_grow_integration()
    test_append_and_grow_empty_new_data()
    test_top_k_fixed_count()
    test_top_k_fewer_samples_than_k()
    test_full_relabeling()
    test_confidence_weighting()
    test_confidence_weighting_no_proba()
    print("All strategy tests passed!")