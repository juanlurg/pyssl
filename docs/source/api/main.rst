ssl_framework.main
==================

This module contains the core :class:`SelfTrainingClassifier` class, which implements the main semi-supervised learning functionality.

.. currentmodule:: ssl_framework.main

SelfTrainingClassifier
----------------------

.. autoclass:: SelfTrainingClassifier
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      SelfTrainingClassifier.fit
      SelfTrainingClassifier.predict
      SelfTrainingClassifier.predict_proba

   .. rubric:: Attributes

   After fitting, the following attributes are available:

   .. attribute:: classes_
      :type: numpy.ndarray

      The classes seen during :meth:`fit`.

   .. attribute:: history_
      :type: List[Dict[str, Any]]

      Training history containing metrics for each iteration.
      Each dictionary contains:

      * ``iteration`` (int): Iteration number
      * ``labeled_data_count`` (int): Number of labeled samples before adding new ones
      * ``new_labels_count`` (int): Number of new pseudo-labels added
      * ``average_confidence`` (float): Mean confidence of newly added samples
      * ``validation_score`` (float, optional): Validation score if validation data provided
      * ``stopping_reason`` (str, optional): Reason for stopping if applicable

   .. attribute:: stopping_reason_
      :type: str

      Reason why training stopped (e.g., "Maximum iterations reached",
      "Early stopping: no improvement", "Labeling convergence").

   .. attribute:: feature_names_
      :type: List[str] or None

      Feature names if input was DataFrame, None otherwise.

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from ssl_framework.main import SelfTrainingClassifier
   import numpy as np

   # Sample data
   X_labeled = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
   y_labeled = np.array([0, 0, 1, 1])
   X_unlabeled = np.array([[0.5, 0.5], [10.5, 10.5]])

   # Create and fit SSL classifier
   ssl_clf = SelfTrainingClassifier(LogisticRegression())
   ssl_clf.fit(X_labeled, y_labeled, X_unlabeled)

   # Make predictions
   predictions = ssl_clf.predict([[0.2, 0.2], [10.2, 10.2]])
   probabilities = ssl_clf.predict_proba([[0.2, 0.2], [10.2, 10.2]])

With Custom Strategies
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ssl_framework.strategies import TopKFixedCount, ConfidenceWeighting

   ssl_clf = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       selection_strategy=TopKFixedCount(k=10),
       integration_strategy=ConfidenceWeighting(),
       max_iter=5
   )

With Early Stopping
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Validation data for early stopping
   X_val = np.array([[0.3, 0.3], [10.3, 10.3]])
   y_val = np.array([0, 1])

   ssl_clf = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       patience=3,
       tol=0.01
   )

   ssl_clf.fit(X_labeled, y_labeled, X_unlabeled, X_val, y_val)
   print(f"Stopped due to: {ssl_clf.stopping_reason_}")

Pandas Integration
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd

   # DataFrame inputs
   X_labeled_df = pd.DataFrame([[0, 0], [1, 1]], columns=['x', 'y'])
   y_labeled_series = pd.Series([0, 1], name='target')
   X_unlabeled_df = pd.DataFrame([[0.5, 0.5]], columns=['x', 'y'])

   ssl_clf = SelfTrainingClassifier(LogisticRegression())
   ssl_clf.fit(X_labeled_df, y_labeled_series, X_unlabeled_df)

   print(ssl_clf.feature_names_)  # ['x', 'y']