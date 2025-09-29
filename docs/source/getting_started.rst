Getting Started with PySSL
==========================

Welcome to PySSL! This guide will walk you through the fundamental concepts and show you how to get up and running with semi-supervised learning in just a few minutes.

🎯 What is Semi-Supervised Learning?
------------------------------------

Semi-supervised learning (SSL) sits between supervised and unsupervised learning. It uses both **labeled** and **unlabeled** data during training, making it perfect for scenarios where:

* Labeled data is expensive or time-consuming to obtain
* You have large amounts of unlabeled data available
* Manual annotation is a bottleneck in your ML pipeline

### The Self-Training Approach

PySSL implements **self-training**, a popular SSL technique that works by:

1. Training a model on the small labeled dataset
2. Using this model to predict labels for unlabeled data
3. Selecting the most confident predictions as "pseudo-labels"
4. Adding these pseudo-labeled samples to the training set
5. Repeating until stopping criteria are met

🚀 Your First PySSL Model
-------------------------

Let's start with a complete example that demonstrates PySSL's power:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from ssl_framework.main import SelfTrainingClassifier

   # Generate synthetic data where SSL excels
   X, y = make_moons(n_samples=1000, noise=0.15, random_state=42)
   X = StandardScaler().fit_transform(X)

   # Split into train/test
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42
   )

   # Create SSL scenario: only 20 labeled samples
   labeled_idx = np.random.choice(len(X_train), size=20, replace=False)
   X_labeled = X_train[labeled_idx]
   y_labeled = y_train[labeled_idx]
   X_unlabeled = np.delete(X_train, labeled_idx, axis=0)

   print(f"Labeled samples: {len(X_labeled)}")
   print(f"Unlabeled samples: {len(X_unlabeled)}")

   # Train SSL model
   ssl_model = SelfTrainingClassifier(
       base_model=LogisticRegression(random_state=42),
       max_iter=10
   )
   ssl_model.fit(X_labeled, y_labeled, X_unlabeled)

   # Compare with supervised baseline
   baseline = LogisticRegression(random_state=42)
   baseline.fit(X_labeled, y_labeled)

   print(f"\\nResults:")
   print(f"Baseline accuracy: {baseline.score(X_test, y_test):.3f}")
   print(f"SSL accuracy: {ssl_model.score(X_test, y_test):.3f}")
   print(f"Improvement: {ssl_model.score(X_test, y_test) - baseline.score(X_test, y_test):.3f}")

**Expected output:**

.. code-block:: text

   Labeled samples: 20
   Unlabeled samples: 680

   Results:
   Baseline accuracy: 0.767
   SSL accuracy: 0.887
   Improvement: 0.120

🔍 Understanding the Results
---------------------------

The SSL model significantly outperforms the baseline! Let's explore why by examining the training history:

.. code-block:: python

   # Examine training progress
   print(f"\\nTraining Progress:")
   print(f"Stopping reason: {ssl_model.stopping_reason_}")
   print(f"Total iterations: {len(ssl_model.history_)}")

   for i, log in enumerate(ssl_model.history_):
       print(f"Iteration {log['iteration']}: "
             f"{log['labeled_data_count']} → "
             f"{log['labeled_data_count'] + log['new_labels_count']} samples "
             f"(confidence: {log['average_confidence']:.3f})")

This shows how the model iteratively grows the labeled dataset by selecting confident predictions.

🧩 Understanding PySSL's Architecture
-------------------------------------

PySSL is built around two key concepts:

### 1. Selection Strategies

These determine **which** unlabeled samples to pseudo-label:

.. code-block:: python

   from ssl_framework.strategies import ConfidenceThreshold, TopKFixedCount

   # Select samples above 90% confidence
   confident_strategy = ConfidenceThreshold(threshold=0.9)

   # Always select exactly 10 most confident samples
   fixed_strategy = TopKFixedCount(k=10)

### 2. Integration Strategies

These determine **how** to integrate pseudo-labeled samples:

.. code-block:: python

   from ssl_framework.strategies import AppendAndGrow, ConfidenceWeighting

   # Simply add new samples to labeled set
   append_strategy = AppendAndGrow()

   # Weight samples by their confidence
   weighted_strategy = ConfidenceWeighting()

### Combining Strategies

Mix and match strategies for different behaviors:

.. code-block:: python

   # Conservative approach: high confidence + append
   conservative_ssl = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       selection_strategy=ConfidenceThreshold(threshold=0.95),
       integration_strategy=AppendAndGrow()
   )

   # Aggressive approach: fixed count + weighting
   aggressive_ssl = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       selection_strategy=TopKFixedCount(k=50),
       integration_strategy=ConfidenceWeighting()
   )

🛡️ Advanced Features
--------------------

### Early Stopping with Validation

Prevent overfitting using validation-based early stopping:

.. code-block:: python

   # Split some labeled data for validation
   X_lab_train, X_val, y_lab_train, y_val = train_test_split(
       X_labeled, y_labeled, test_size=0.3, random_state=42
   )

   ssl_model = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       patience=3,  # Stop if no improvement for 3 iterations
       tol=0.01    # Minimum improvement threshold
   )

   # Pass validation data
   ssl_model.fit(X_lab_train, y_lab_train, X_unlabeled, X_val, y_val)

### Labeling Convergence

Automatically stop when few new labels are added:

.. code-block:: python

   ssl_model = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       labeling_convergence_threshold=10  # Stop if <10 new labels
   )

### Pandas Integration

PySSL works seamlessly with DataFrames:

.. code-block:: python

   import pandas as pd

   # Convert to DataFrame
   feature_names = ['feature_1', 'feature_2']
   X_labeled_df = pd.DataFrame(X_labeled, columns=feature_names)
   X_unlabeled_df = pd.DataFrame(X_unlabeled, columns=feature_names)

   # Fit with DataFrames
   ssl_model.fit(X_labeled_df, y_labeled, X_unlabeled_df)

   # Feature names are preserved
   print(ssl_model.feature_names_)  # ['feature_1', 'feature_2']

🎯 When Does SSL Work Best?
--------------------------

SSL is most effective when your data satisfies the **cluster assumption**:

✅ **Good for SSL:**
   - Data forms distinct clusters
   - Similar samples have similar labels
   - Clear decision boundaries
   - Examples: image classification, text categorization

❌ **Challenging for SSL:**
   - Random/noisy data
   - No clear patterns
   - Complex decision boundaries
   - Very small datasets (< 50 samples)

🔄 Common Patterns
-----------------

### Pattern 1: Limited Labeled Data
Perfect for medical diagnosis, expert annotation scenarios:

.. code-block:: python

   # Medical diagnosis scenario
   ssl_medical = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       selection_strategy=ConfidenceThreshold(threshold=0.95),  # Be conservative
       max_iter=5  # Limited iterations
   )

### Pattern 2: Large Unlabeled Dataset
Ideal for web scraping, sensor data:

.. code-block:: python

   # Web scraping scenario
   ssl_web = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       selection_strategy=TopKFixedCount(k=100),  # Process in batches
       integration_strategy=ConfidenceWeighting()  # Weight by confidence
   )

### Pattern 3: Rapid Prototyping
Quick experiments and proof-of-concepts:

.. code-block:: python

   # Quick prototype
   ssl_prototype = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       max_iter=3  # Fast iteration
   )

📚 Next Steps
-----------

Now that you understand the basics:

1. **Try the quickstart tutorial**: :doc:`quickstart_tutorial`
2. **Explore strategy combinations**: :doc:`user_guide/strategies`
3. **See real-world examples**: :doc:`examples/basic_usage`
4. **Learn custom strategies**: :doc:`user_guide/custom_strategies`

🤔 Questions?
-----------

* Check the :doc:`api/index` for detailed API documentation
* Browse :doc:`examples/index` for more complex scenarios
* Read about :doc:`user_guide/stopping_criteria` for training control
* See :doc:`contributing` if you want to contribute to PySSL

Ready to leverage your unlabeled data? Let's dive deeper! 🚀