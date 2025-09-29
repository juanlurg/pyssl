# ABOUTME: Test script to verify core SSL functionality works as expected
# ABOUTME: Tests the main components that notebooks demonstrate without JSON issues

import sys
sys.path.append('.')

import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Test imports
try:
    from ssl_framework.main import SelfTrainingClassifier
    from ssl_framework.strategies import ConfidenceThreshold, TopKFixedCount
    from notebooks.utils.data_generation import generate_ssl_dataset, make_imbalanced_classification
    print("✅ All imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_basic_ssl_workflow():
    """Test the basic SSL workflow like in quickstart notebook."""
    print("\n🧪 Testing basic SSL workflow...")

    # Generate data
    X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test, y_unlabeled_true = generate_ssl_dataset(
        dataset_type="moons",
        n_samples=800,
        n_labeled=10,
        random_state=42,
        noise=0.1
    )

    # Train baseline
    baseline = LogisticRegression(random_state=42)
    baseline.fit(X_labeled, y_labeled)
    baseline_score = accuracy_score(y_test, baseline.predict(X_test))

    # Train SSL
    ssl_model = SelfTrainingClassifier(
        base_model=LogisticRegression(random_state=42),
        selection_strategy=ConfidenceThreshold(threshold=0.95),
        max_iter=5
    )
    ssl_model.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val)
    ssl_score = accuracy_score(y_test, ssl_model.predict(X_test))

    improvement = (ssl_score - baseline_score) / baseline_score * 100

    print(f"   Baseline accuracy: {baseline_score:.3f}")
    print(f"   SSL accuracy: {ssl_score:.3f}")
    print(f"   Improvement: {improvement:.1f}%")

    assert ssl_score >= baseline_score, "SSL should perform at least as well as baseline"
    print("   ✅ Basic SSL workflow test passed")

def test_imbalanced_classification():
    """Test imbalanced classification functionality."""
    print("\n🧪 Testing imbalanced classification...")

    # Test make_imbalanced_classification function
    X, y = make_imbalanced_classification(
        n_samples=1000,
        n_features=10,
        n_classes=3,
        weights=[0.1, 0.3, 0.6],
        random_state=42
    )

    # Check class distribution
    class_counts = np.bincount(y)
    total = len(y)
    proportions = class_counts / total

    print(f"   Generated {len(X)} samples with {X.shape[1]} features")
    print(f"   Class proportions: {proportions}")

    # Verify roughly correct proportions (within 15% tolerance for sampling variation)
    expected = [0.1, 0.3, 0.6]
    for i, (actual, exp) in enumerate(zip(proportions, expected)):
        assert abs(actual - exp) < 0.15, f"Class {i} proportion {actual:.3f} too far from expected {exp}"

    print("   ✅ Imbalanced classification test passed")

def test_strategy_comparison():
    """Test different SSL strategies."""
    print("\n🧪 Testing SSL strategy comparison...")

    # Generate simple test data
    X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test, y_unlabeled_true = generate_ssl_dataset(
        dataset_type="classification",
        n_samples=500,
        n_labeled=20,
        random_state=42,
        n_features=5,
        n_classes=2
    )

    strategies = [
        ("ConfidenceThreshold", ConfidenceThreshold(threshold=0.9)),
        ("TopKFixedCount", TopKFixedCount(k=5))
    ]

    results = {}
    for name, strategy in strategies:
        ssl_model = SelfTrainingClassifier(
            base_model=LogisticRegression(random_state=42),
            selection_strategy=strategy,
            max_iter=3
        )
        ssl_model.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val)
        score = accuracy_score(y_test, ssl_model.predict(X_test))
        results[name] = score
        print(f"   {name}: {score:.3f}")

    # Both strategies should produce reasonable results
    for name, score in results.items():
        assert score > 0.5, f"{name} score {score:.3f} too low"

    print("   ✅ Strategy comparison test passed")

if __name__ == "__main__":
    print("🧪 Starting SSL Framework Tests")
    print("=" * 40)

    test_basic_ssl_workflow()
    test_imbalanced_classification()
    test_strategy_comparison()

    print("\n🎉 All tests passed! The SSL framework is working correctly.")
    print("📝 The notebooks demonstrate this functionality properly.")