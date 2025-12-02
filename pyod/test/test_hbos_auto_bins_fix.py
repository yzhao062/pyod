"""
Test case for HBOS IndexError fix when using n_bins='auto'

This test demonstrates the bug fix for issue #476 in the pyod library:
https://github.com/yzhao062/pyod/issues/476

The bug occurred when:
1. Using n_bins='auto' for automatic bin selection
2. Test data contains values outside the training data range
3. The function would incorrectly calculate optimal_n_bins on test data
   instead of using the training histogram size

The fix ensures that optimal_n_bins is derived from the training histogram
(hist[i].shape[0]) rather than recalculating on test data.
"""

import numpy as np

from pyod.models.hbos import HBOS


def test_hbos_auto_bins_with_out_of_range_values():
    """
    Test HBOS with n_bins='auto' when test data exceeds training range.
    
    This test reproduces the IndexError that occurred when test values
    exceeded the maximum training value for any feature.
    """
    print("Testing HBOS with n_bins='auto' and out-of-range test values...")

    # Create training data with limited range
    np.random.seed(42)
    n_train = 100
    n_features = 5

    # Training data ranges roughly from 0 to 10
    X_train = np.random.randn(n_train, n_features) * 2 + 5
    X_train = np.clip(X_train, 0, 10)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training data range per feature:")
    for i in range(n_features):
        print(
            f"  Feature {i}: [{X_train[:, i].min():.2f}, {X_train[:, i].max():.2f}]")

    # Initialize and fit HBOS with auto bins
    model = HBOS(n_bins='auto', contamination=0.1)
    model.fit(X_train)

    print(f"\nNumber of bins per feature after training:")
    for i in range(n_features):
        print(f"  Feature {i}: {model.hist_[i].shape[0]} bins")

    # Create test data with some values OUTSIDE the training range
    # This is the critical part that triggers the bug
    X_test = np.random.randn(10, n_features) * 3 + 5
    # Force some values to be higher than training max
    X_test[0, 2] = X_train[:,
                   2].max() + 5  # Way above training max for feature 2
    X_test[1, 3] = X_train[:, 3].max() + 3  # Above training max for feature 3
    X_test[2, 0] = X_train[:, 0].min() - 2  # Below training min for feature 0

    print(f"\nTest data shape: {X_test.shape}")
    print(f"Test data range per feature:")
    for i in range(n_features):
        print(
            f"  Feature {i}: [{X_test[:, i].min():.2f}, {X_test[:, i].max():.2f}]")

    # This would fail with IndexError before the fix
    try:
        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)

        print(f"\n✓ SUCCESS: Predictions completed without error!")
        print(f"  Predictions shape: {predictions.shape}")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Number of outliers detected: {predictions.sum()}")
        print(f"  Outlier indices: {np.where(predictions == 1)[0].tolist()}")

        return True

    except IndexError as e:
        print(f"\n✗ FAILED: IndexError occurred!")
        print(f"  Error: {e}")
        print(f"  This indicates the bug is NOT fixed.")
        return False


def test_hbos_auto_bins_edge_cases():
    """
    Additional edge case tests for HBOS with auto bins.
    """
    print("\n" + "=" * 70)
    print("Testing edge cases...")
    print("=" * 70)

    np.random.seed(123)

    # Test 1: All test values above training range
    print("\nTest 1: All test values above training range")
    X_train = np.random.randn(50, 3) * 1 + 5
    model = HBOS(n_bins='auto', contamination=0.1)
    model.fit(X_train)

    X_test = X_train.max() + np.random.rand(5, 3) * 3
    try:
        predictions = model.predict(X_test)
        print(f"  ✓ Success: {predictions.sum()} outliers detected")
    except IndexError as e:
        print(f"  ✗ Failed with IndexError: {e}")
        return False

    # Test 2: All test values below training range
    print("\nTest 2: All test values below training range")
    X_test = X_train.min() - np.random.rand(5, 3) * 3
    try:
        predictions = model.predict(X_test)
        print(f"  ✓ Success: {predictions.sum()} outliers detected")
    except IndexError as e:
        print(f"  ✗ Failed with IndexError: {e}")
        return False

    # Test 3: Mixed in-range and out-of-range values
    print("\nTest 3: Mixed in-range and out-of-range values")
    X_test = np.vstack([
        X_train[:2],  # In range
        X_train.max() + np.random.rand(2, 3),  # Above range
        X_train.min() - np.random.rand(2, 3),  # Below range
    ])
    try:
        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)
        print(f"  ✓ Success: {predictions.sum()} outliers detected")
        print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    except IndexError as e:
        print(f"  ✗ Failed with IndexError: {e}")
        return False

    return True


def test_hbos_static_bins_comparison():
    """
    Compare behavior between auto and static bins to ensure consistency.
    """
    print("\n" + "=" * 70)
    print("Comparing auto bins vs static bins behavior...")
    print("=" * 70)

    np.random.seed(456)
    X_train = np.random.randn(100, 4) * 2 + 5
    X_test = np.vstack([
        X_train[:5],
        X_train.max(axis=0) + 2,  # One sample above all training ranges
    ])

    # Test with auto bins
    model_auto = HBOS(n_bins='auto', contamination=0.1)
    model_auto.fit(X_train)
    try:
        pred_auto = model_auto.predict(X_test)
        score_auto = model_auto.decision_function(X_test)
        print(f"  Auto bins: ✓ Success")
        print(
            f"    Outliers: {pred_auto.sum()}, Score range: [{score_auto.min():.4f}, {score_auto.max():.4f}]")
    except Exception as e:
        print(f"  Auto bins: ✗ Failed - {e}")
        return False

    # Test with static bins
    model_static = HBOS(n_bins=10, contamination=0.1)
    model_static.fit(X_train)
    try:
        pred_static = model_static.predict(X_test)
        score_static = model_static.decision_function(X_test)
        print(f"  Static bins: ✓ Success")
        print(
            f"    Outliers: {pred_static.sum()}, Score range: [{score_static.min():.4f}, {score_static.max():.4f}]")
    except Exception as e:
        print(f"  Static bins: ✗ Failed - {e}")
        return False

    print(f"\n  Both methods handled out-of-range values correctly!")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("HBOS IndexError Fix Test Suite")
    print("Issue: https://github.com/yzhao062/pyod/issues/476")
    print("=" * 70)

    all_passed = True

    # Run main test
    all_passed &= test_hbos_auto_bins_with_out_of_range_values()

    # Run edge case tests
    all_passed &= test_hbos_auto_bins_edge_cases()

    # Run comparison test
    all_passed &= test_hbos_static_bins_comparison()

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("The fix correctly handles out-of-range test values.")
    else:
        print("✗ SOME TESTS FAILED!")
        print("The bug may not be fully fixed.")
    print("=" * 70)
