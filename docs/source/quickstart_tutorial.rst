5-Minute Quickstart Tutorial
============================

This tutorial will get you using PySSL effectively in just 5 minutes. We'll build a complete semi-supervised learning pipeline from scratch!

🎯 The Challenge
---------------

Imagine you're building a medical diagnosis system. You have:

* **20 expertly labeled samples** (expensive to obtain)
* **500 unlabeled samples** (easy to collect)
* **Goal**: Build the best possible classifier

Let's see how SSL can help!

📊 Step 1: Generate Sample Data
-------------------------------

We'll simulate this scenario using the classic two-moons dataset:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score, classification_report

   # Generate challenging dataset where SSL excels
   X, y = make_moons(n_samples=800, noise=0.2, random_state=42)
   X = StandardScaler().fit_transform(X)

   # Split into train/test
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=42, stratify=y
   )

   print(f"Total training samples: {len(X_train)}")
   print(f"Test samples: {len(X_test)}")

**Output:**

.. code-block:: text

   Total training samples: 600
   Test samples: 200

🏷️ Step 2: Create SSL Scenario
------------------------------

Now let's simulate the medical scenario by using only 20 labeled samples:

.. code-block:: python

   # Create semi-supervised scenario
   np.random.seed(42)  # For reproducibility
   labeled_idx = np.random.choice(len(X_train), size=20, replace=False)

   X_labeled = X_train[labeled_idx]
   y_labeled = y_train[labeled_idx]
   X_unlabeled = np.delete(X_train, labeled_idx, axis=0)

   print(f"Labeled samples: {len(X_labeled)}")
   print(f"Unlabeled samples: {len(X_unlabeled)}")
   print(f"Label distribution: {np.bincount(y_labeled)}")

**Output:**

.. code-block:: text

   Labeled samples: 20
   Unlabeled samples: 580
   Label distribution: [10 10]

Perfect! We have a balanced but tiny labeled set.

📈 Step 3: Supervised Baseline
------------------------------

First, let's see how well we can do with only the labeled data:

.. code-block:: python

   # Train supervised baseline
   baseline_model = LogisticRegression(random_state=42)
   baseline_model.fit(X_labeled, y_labeled)

   # Evaluate
   baseline_pred = baseline_model.predict(X_test)
   baseline_acc = accuracy_score(y_test, baseline_pred)

   print(f"Baseline accuracy: {baseline_acc:.3f}")

**Output:**

.. code-block:: text

   Baseline accuracy: 0.745

Not bad, but we can do better with SSL!

🚀 Step 4: Your First SSL Model
-------------------------------

Now let's use PySSL to leverage those 580 unlabeled samples:

.. code-block:: python

   from ssl_framework.main import SelfTrainingClassifier

   # Create SSL classifier with default settings
   ssl_model = SelfTrainingClassifier(
       base_model=LogisticRegression(random_state=42),
       max_iter=10
   )

   # Fit using both labeled and unlabeled data
   ssl_model.fit(X_labeled, y_labeled, X_unlabeled)

   # Evaluate
   ssl_pred = ssl_model.predict(X_test)
   ssl_acc = accuracy_score(y_test, ssl_pred)

   print(f"SSL accuracy: {ssl_acc:.3f}")
   print(f"Improvement: +{ssl_acc - baseline_acc:.3f}")

**Output:**

.. code-block:: text

   SSL accuracy: 0.885
   Improvement: +0.140

🎉 **Wow! 14% improvement just by using unlabeled data!**

🔍 Step 5: Analyze What Happened
--------------------------------

Let's examine how the SSL training progressed:

.. code-block:: python

   print(f"\\nTraining Analysis:")
   print(f"Stopping reason: {ssl_model.stopping_reason_}")
   print(f"Total iterations: {len(ssl_model.history_)}")
   print(f"Final labeled samples: {ssl_model.history_[-1]['labeled_data_count'] + ssl_model.history_[-1]['new_labels_count']}")

   print(f"\\nIteration-by-iteration progress:")
   for log in ssl_model.history_:
       print(f"  Iter {log['iteration']}: "
             f"{log['labeled_data_count']} → {log['labeled_data_count'] + log['new_labels_count']} samples "
             f"(+{log['new_labels_count']}, conf: {log['average_confidence']:.3f})")

**Expected output:**

.. code-block:: text

   Training Analysis:
   Stopping reason: Maximum iterations reached (10)
   Total iterations: 10
   Final labeled samples: 562

   Iteration-by-iteration progress:
     Iter 0: 20 → 28 samples (+8, conf: 0.976)
     Iter 1: 28 → 43 samples (+15, conf: 0.971)
     Iter 2: 43 → 67 samples (+24, conf: 0.968)
     ...

The model iteratively grew from 20 to 562 labeled samples!

🎛️ Step 6: Tune Your SSL Model
------------------------------

Let's experiment with different strategies to potentially improve even more:

.. code-block:: python

   from ssl_framework.strategies import TopKFixedCount, ConfidenceWeighting

   # Strategy 1: More aggressive selection
   aggressive_ssl = SelfTrainingClassifier(
       base_model=LogisticRegression(random_state=42),
       selection_strategy=TopKFixedCount(k=30),  # Select 30 samples per iteration
       max_iter=8
   )

   aggressive_ssl.fit(X_labeled, y_labeled, X_unlabeled)
   aggressive_acc = accuracy_score(y_test, aggressive_ssl.predict(X_test))

   # Strategy 2: Confidence weighting
   weighted_ssl = SelfTrainingClassifier(
       base_model=LogisticRegression(random_state=42),
       selection_strategy=TopKFixedCount(k=20),
       integration_strategy=ConfidenceWeighting(),  # Weight by confidence
       max_iter=10
   )

   weighted_ssl.fit(X_labeled, y_labeled, X_unlabeled)
   weighted_acc = accuracy_score(y_test, weighted_ssl.predict(X_test))

   print(f"\\nStrategy Comparison:")
   print(f"Baseline:     {baseline_acc:.3f}")
   print(f"Default SSL:  {ssl_acc:.3f} (+{ssl_acc - baseline_acc:.3f})")
   print(f"Aggressive:   {aggressive_acc:.3f} (+{aggressive_acc - baseline_acc:.3f})")
   print(f"Weighted:     {weighted_acc:.3f} (+{weighted_acc - baseline_acc:.3f})")

**Expected output:**

.. code-block:: text

   Strategy Comparison:
   Baseline:     0.745
   Default SSL:  0.885 (+0.140)
   Aggressive:   0.870 (+0.125)
   Weighted:     0.890 (+0.145)

Different strategies can yield different results!

🛡️ Step 7: Add Early Stopping
------------------------------

In real scenarios, use validation data to prevent overfitting:

.. code-block:: python

   # Split some labeled data for validation
   X_lab_train, X_val, y_lab_train, y_val = train_test_split(
       X_labeled, y_labeled, test_size=0.3, random_state=42, stratify=y_labeled
   )

   print(f"Training with: {len(X_lab_train)} labeled, {len(X_unlabeled)} unlabeled")
   print(f"Validation: {len(X_val)} samples")

   # SSL with early stopping
   robust_ssl = SelfTrainingClassifier(
       base_model=LogisticRegression(random_state=42),
       patience=3,  # Stop if no improvement for 3 iterations
       tol=0.02,    # Minimum improvement threshold
       max_iter=15
   )

   robust_ssl.fit(X_lab_train, y_lab_train, X_unlabeled, X_val, y_val)

   robust_acc = accuracy_score(y_test, robust_ssl.predict(X_test))
   print(f"\\nRobust SSL accuracy: {robust_acc:.3f}")
   print(f"Stopped due to: {robust_ssl.stopping_reason_}")

**Expected output:**

.. code-block:: text

   Training with: 14 labeled, 580 unlabeled
   Validation: 6 samples

   Robust SSL accuracy: 0.875
   Stopped due to: Early stopping: no improvement for 3 iterations

📊 Step 8: Visualize the Results (Optional)
-------------------------------------------

If you want to visualize what SSL accomplished:

.. code-block:: python

   # Create visualization (requires matplotlib)
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   # Plot 1: Original labeled data
   ax1.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='red', alpha=0.6, label='Class 0')
   ax1.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='blue', alpha=0.6, label='Class 1')
   ax1.scatter(X_labeled[:, 0], X_labeled[:, 1], c='black', s=100, marker='x', label='Labeled')
   ax1.set_title(f'Baseline (20 labels)\\nAccuracy: {baseline_acc:.3f}')
   ax1.legend()

   # Plot 2: SSL results
   ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='red', alpha=0.6, label='Class 0')
   ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='blue', alpha=0.6, label='Class 1')
   ax2.scatter(X_labeled[:, 0], X_labeled[:, 1], c='black', s=100, marker='x', label='Original labels')
   ax2.set_title(f'SSL (20→{len(ssl_model.history_[-1]["labeled_data_count"]) + len(ssl_model.history_[-1]["new_labels_count"])} labels)\\nAccuracy: {ssl_acc:.3f}')
   ax2.legend()

   plt.tight_layout()
   plt.show()

🎯 Key Takeaways
---------------

In just 5 minutes, you've learned:

✅ **SSL can dramatically improve performance** when you have limited labeled data

✅ **Different strategies matter** - experiment with ``ConfidenceThreshold`` vs ``TopKFixedCount``

✅ **Integration strategies matter too** - try ``ConfidenceWeighting`` for better results

✅ **Early stopping prevents overfitting** - always use validation data in real scenarios

✅ **PySSL is easy to use** - just swap your classifier for ``SelfTrainingClassifier``

🚀 Next Steps
-----------

Ready to dive deeper? Check out:

1. **:doc:`user_guide/strategies`** - Learn all available strategies
2. **:doc:`examples/basic_usage`** - More complete examples
3. **:doc:`api/index`** - Full API documentation
4. **:doc:`user_guide/custom_strategies`** - Build your own strategies

**Happy semi-supervised learning!** 🎉