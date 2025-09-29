# ABOUTME: Comprehensive demonstration of the SSL framework capabilities
# ABOUTME: Shows baseline comparison, different strategies, and framework features

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Import our SSL framework
from ssl_framework.main import SelfTrainingClassifier
from ssl_framework.strategies import (
    ConfidenceThreshold, TopKFixedCount,
    AppendAndGrow, FullReLabeling, ConfidenceWeighting
)


def setup_ssl_data(random_state=42):
    """Create a semi-supervised learning scenario from synthetic data."""
    print("Setting up semi-supervised learning scenario...")

    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        n_clusters_per_class=2,
        class_sep=1.5,
        random_state=random_state
    )

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    # Create semi-supervised scenario: small labeled set, larger unlabeled set
    X_labeled, X_unlabeled_full, y_labeled, y_unlabeled_full = train_test_split(
        X_train, y_train, test_size=0.8, random_state=random_state, stratify=y_train
    )

    # Keep some data for validation (early stopping)
    X_unlabeled, X_val, y_unlabeled, y_val = train_test_split(
        X_unlabeled_full, y_unlabeled_full, test_size=0.2, random_state=random_state, stratify=y_unlabeled_full
    )

    print(f"Dataset sizes:")
    print(f"  Labeled training: {len(X_labeled)} samples")
    print(f"  Unlabeled training: {len(X_unlabeled)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    print()

    return X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test


def demonstration_1_baseline():
    """Demonstration 1: Baseline Supervised Model."""
    print("=" * 60)
    print("DEMONSTRATION 1: Baseline Supervised Model")
    print("=" * 60)

    X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test = setup_ssl_data()

    # Train standard supervised model on only the small labeled dataset
    baseline_model = LogisticRegression(random_state=42, max_iter=1000)
    baseline_model.fit(X_labeled, y_labeled)

    # Evaluate on test set
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

    print(f"Baseline Supervised Model Results:")
    print(f"  Training data: {len(X_labeled)} labeled samples only")
    print(f"  Test accuracy: {baseline_accuracy:.4f}")
    print(f"  Unused unlabeled data: {len(X_unlabeled)} samples")
    print()

    return baseline_accuracy


def demonstration_2_confidence_threshold():
    """Demonstration 2: Self-Training with Confidence Threshold."""
    print("=" * 60)
    print("DEMONSTRATION 2: Self-Training with Confidence Threshold")
    print("=" * 60)

    X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test = setup_ssl_data()

    # Create strategies
    selection_strategy = ConfidenceThreshold(threshold=0.8)
    integration_strategy = AppendAndGrow()

    # Create and train SSL classifier
    ssl_classifier = SelfTrainingClassifier(
        base_model=LogisticRegression(random_state=42, max_iter=1000),
        selection_strategy=selection_strategy,
        integration_strategy=integration_strategy,
        max_iter=10,
        patience=3,
        tol=0.01
    )

    print("Training SSL classifier with confidence threshold strategy...")
    ssl_classifier.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val)

    # Evaluate on test set
    y_pred_ssl = ssl_classifier.predict(X_test)
    ssl_accuracy = accuracy_score(y_test, y_pred_ssl)

    print(f"\nSelf-Training Results:")
    print(f"  Test accuracy: {ssl_accuracy:.4f}")
    print(f"  Stopping reason: {ssl_classifier.stopping_reason_}")
    print(f"  Training iterations: {len(ssl_classifier.history_)}")

    # Print iteration-by-iteration progress
    print(f"\nTraining Progress:")
    for log in ssl_classifier.history_:
        iteration = log['iteration']
        labeled_count = log['labeled_data_count']
        new_count = log['new_labels_count']
        confidence = log['average_confidence']
        val_score = log.get('validation_score', 'N/A')

        print(f"  Iteration {iteration}: {labeled_count} -> {labeled_count + new_count} samples "
              f"(+{new_count}), confidence: {confidence:.3f}, val_score: {val_score}")

    print()
    return ssl_accuracy


def demonstration_3_topk_weighting():
    """Demonstration 3: Self-Training with Top-K and Confidence Weighting."""
    print("=" * 60)
    print("DEMONSTRATION 3: Self-Training with Top-K and Weighting")
    print("=" * 60)

    X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test = setup_ssl_data()

    # Create different strategies
    selection_strategy = TopKFixedCount(k=20)
    integration_strategy = ConfidenceWeighting()

    # Create and train SSL classifier
    ssl_classifier = SelfTrainingClassifier(
        base_model=LogisticRegression(random_state=42, max_iter=1000),
        selection_strategy=selection_strategy,
        integration_strategy=integration_strategy,
        max_iter=8,
        patience=3,
        tol=0.01,
        labeling_convergence_threshold=5
    )

    print("Training SSL classifier with Top-K selection and confidence weighting...")
    ssl_classifier.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val)

    # Evaluate on test set
    y_pred_ssl = ssl_classifier.predict(X_test)
    ssl_accuracy = accuracy_score(y_test, y_pred_ssl)

    print(f"\nTop-K + Weighting Results:")
    print(f"  Test accuracy: {ssl_accuracy:.4f}")
    print(f"  Stopping reason: {ssl_classifier.stopping_reason_}")
    print(f"  Training iterations: {len(ssl_classifier.history_)}")

    # Print iteration-by-iteration progress
    print(f"\nTraining Progress:")
    for log in ssl_classifier.history_:
        iteration = log['iteration']
        labeled_count = log['labeled_data_count']
        new_count = log['new_labels_count']
        confidence = log['average_confidence']
        val_score = log.get('validation_score', 'N/A')

        print(f"  Iteration {iteration}: {labeled_count} -> {labeled_count + new_count} samples "
              f"(+{new_count}), confidence: {confidence:.3f}, val_score: {val_score}")

    print()
    return ssl_accuracy


def demonstration_4_pandas_compatibility():
    """Demonstration 4: Pandas DataFrame Compatibility."""
    print("=" * 60)
    print("DEMONSTRATION 4: Pandas DataFrame Compatibility")
    print("=" * 60)

    X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test = setup_ssl_data()

    # Convert to pandas DataFrames with feature names
    feature_names = [f"feature_{i}" for i in range(X_labeled.shape[1])]

    X_labeled_df = pd.DataFrame(X_labeled, columns=feature_names)
    X_unlabeled_df = pd.DataFrame(X_unlabeled, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    y_labeled_series = pd.Series(y_labeled, name='target')
    y_val_series = pd.Series(y_val, name='target')

    print("Training with pandas DataFrames...")

    # Train SSL classifier with DataFrames
    ssl_classifier = SelfTrainingClassifier(
        base_model=LogisticRegression(random_state=42, max_iter=1000),
        selection_strategy=ConfidenceThreshold(threshold=0.85),
        integration_strategy=AppendAndGrow(),
        max_iter=5
    )

    ssl_classifier.fit(X_labeled_df, y_labeled_series, X_unlabeled_df, X_val_df, y_val_series)

    # Make predictions
    y_pred = ssl_classifier.predict(X_test_df)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nPandas DataFrame Results:")
    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Feature names stored: {ssl_classifier.feature_names_[:5]}... (showing first 5)")
    print(f"  Input types handled: DataFrame -> NumPy conversion successful")
    print()


def print_final_summary(baseline_acc, ssl_acc1, ssl_acc2):
    """Print final comparison summary."""
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("Performance Comparison:")
    print(f"  Baseline (supervised only):     {baseline_acc:.4f}")
    print(f"  SSL + Confidence Threshold:     {ssl_acc1:.4f}  (+{ssl_acc1-baseline_acc:+.4f})")
    print(f"  SSL + Top-K + Weighting:        {ssl_acc2:.4f}  (+{ssl_acc2-baseline_acc:+.4f})")

    print(f"\nFramework Features Demonstrated:")
    print(f"  ✓ Scikit-learn API compatibility")
    print(f"  ✓ Modular strategy injection")
    print(f"  ✓ Multiple selection strategies (ConfidenceThreshold, TopKFixedCount)")
    print(f"  ✓ Multiple integration strategies (AppendAndGrow, ConfidenceWeighting)")
    print(f"  ✓ Advanced stopping criteria (early stopping, convergence)")
    print(f"  ✓ Comprehensive logging and diagnostics")
    print(f"  ✓ Pandas DataFrame support")
    print(f"  ✓ Type hints and comprehensive documentation")

    improvement1 = (ssl_acc1 - baseline_acc) / baseline_acc * 100
    improvement2 = (ssl_acc2 - baseline_acc) / baseline_acc * 100

    print(f"\nPerformance Improvements:")
    print(f"  Confidence Threshold: {improvement1:+.1f}% relative improvement")
    print(f"  Top-K + Weighting:    {improvement2:+.1f}% relative improvement")


if __name__ == "__main__":
    print("🚀 Semi-Supervised Learning Framework - Comprehensive Demo")
    print("=" * 60)
    print()

    # Run all demonstrations
    baseline_accuracy = demonstration_1_baseline()
    ssl_accuracy_1 = demonstration_2_confidence_threshold()
    ssl_accuracy_2 = demonstration_3_topk_weighting()
    demonstration_4_pandas_compatibility()

    # Print final summary
    print_final_summary(baseline_accuracy, ssl_accuracy_1, ssl_accuracy_2)

    print("\n🎉 Demo completed successfully!")
    print("\nTo explore further:")
    print("  • Try different strategy combinations")
    print("  • Experiment with hyperparameters (thresholds, K values)")
    print("  • Use different base models (RandomForest, SVM, etc.)")
    print("  • Implement custom selection/integration strategies")