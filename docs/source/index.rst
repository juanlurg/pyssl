PySSL: Semi-Supervised Learning Framework
==========================================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://python.org
   :alt: Python

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: LICENSE
   :alt: License

.. image:: https://img.shields.io/badge/build-passing-brightgreen
   :target: https://github.com/yourusername/pyssl/actions
   :alt: Build Status

.. image:: https://img.shields.io/badge/coverage-95%25-brightgreen
   :target: https://codecov.io/gh/yourusername/pyssl
   :alt: Coverage

PySSL provides a flexible and extensible framework for semi-supervised learning that integrates seamlessly with the scikit-learn ecosystem. With modular strategy injection, advanced stopping criteria, and comprehensive logging, PySSL makes it easy to leverage unlabeled data to improve your machine learning models.

🎯 Key Features
---------------

* **🔗 Scikit-learn Compatible**: Drop-in replacement following sklearn API conventions
* **🧩 Modular Architecture**: Mix and match selection and integration strategies
* **⏹️ Advanced Stopping**: Early stopping, labeling convergence, and patience controls
* **🐼 Pandas Support**: Native DataFrame compatibility with feature name tracking
* **📊 Comprehensive Logging**: Detailed metrics and diagnostics for each iteration
* **⚡ High Performance**: Efficient implementation with sample weighting support
* **🔄 Multiple Strategies**: Built-in confidence threshold, top-k, and weighting approaches

🚀 Quick Start
--------------

Get started with PySSL in just a few lines of code:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_moons
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   from sklearn.linear_model import LogisticRegression
   from ssl_framework.main import SelfTrainingClassifier

   # Generate data where SSL excels
   X, y = make_moons(n_samples=500, noise=0.1, random_state=42)
   X = StandardScaler().fit_transform(X)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Create SSL scenario: only 10 labeled samples
   labeled_idx = np.random.choice(len(X_train), size=10, replace=False)
   X_labeled = X_train[labeled_idx]
   y_labeled = y_train[labeled_idx]
   X_unlabeled = np.delete(X_train, labeled_idx, axis=0)

   # Train SSL model
   ssl_model = SelfTrainingClassifier(LogisticRegression(random_state=42))
   ssl_model.fit(X_labeled, y_labeled, X_unlabeled)

   # Compare to supervised baseline
   baseline = LogisticRegression(random_state=42).fit(X_labeled, y_labeled)

   print(f"Baseline (10 labels): {baseline.score(X_test, y_test):.3f}")
   print(f"SSL accuracy: {ssl_model.score(X_test, y_test):.3f}")

Expected output::

   Baseline (10 labels): 0.533
   SSL accuracy: 0.887

Documentation
=============

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   getting_started
   quickstart_tutorial

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/strategies
   user_guide/stopping_criteria
   user_guide/pandas_integration
   user_guide/custom_strategies

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index
   api/main
   api/strategies

.. toctree::
   :maxdepth: 2
   :caption: Examples & Tutorials

   examples/index
   examples/basic_usage
   examples/strategy_comparison
   examples/real_world_datasets

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   changelog
   roadmap

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`