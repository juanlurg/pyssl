ssl_framework.strategies
========================

This module contains strategy classes for label selection and integration in the SSL framework.
These provide modular, swappable components for customizing semi-supervised learning behavior.

.. currentmodule:: ssl_framework.strategies

Selection Strategies
-------------------

Selection strategies determine **which** unlabeled samples to pseudo-label.

ConfidenceThreshold
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConfidenceThreshold
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Example:

   .. code-block:: python

      from ssl_framework.strategies import ConfidenceThreshold

      # Select samples with >95% confidence
      strategy = ConfidenceThreshold(threshold=0.95)

      # Use with SelfTrainingClassifier
      from ssl_framework.main import SelfTrainingClassifier
      ssl_clf = SelfTrainingClassifier(
          base_model=LogisticRegression(),
          selection_strategy=strategy
      )

TopKFixedCount
~~~~~~~~~~~~~~

.. autoclass:: TopKFixedCount
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Example:

   .. code-block:: python

      from ssl_framework.strategies import TopKFixedCount

      # Always select top 20 most confident samples
      strategy = TopKFixedCount(k=20)

      # Use with SelfTrainingClassifier
      ssl_clf = SelfTrainingClassifier(
          base_model=LogisticRegression(),
          selection_strategy=strategy
      )

Integration Strategies
---------------------

Integration strategies determine **how** to integrate pseudo-labeled samples into the training set.

AppendAndGrow
~~~~~~~~~~~~~

.. autoclass:: AppendAndGrow
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Example:

   .. code-block:: python

      from ssl_framework.strategies import AppendAndGrow

      # Simply append new samples to labeled set
      strategy = AppendAndGrow()

      ssl_clf = SelfTrainingClassifier(
          base_model=LogisticRegression(),
          integration_strategy=strategy
      )

FullReLabeling
~~~~~~~~~~~~~~

.. autoclass:: FullReLabeling
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Example:

   .. code-block:: python

      from ssl_framework.strategies import FullReLabeling

      # Re-label from scratch each iteration
      strategy = FullReLabeling(X_original, y_original)

      ssl_clf = SelfTrainingClassifier(
          base_model=LogisticRegression(),
          integration_strategy=strategy
      )

ConfidenceWeighting
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ConfidenceWeighting
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   Example:

   .. code-block:: python

      from ssl_framework.strategies import ConfidenceWeighting

      # Weight samples by their confidence
      strategy = ConfidenceWeighting()

      ssl_clf = SelfTrainingClassifier(
          base_model=LogisticRegression(),
          integration_strategy=strategy
      )

Strategy Combinations
--------------------

Mix and match strategies for different behaviors:

Conservative SSL
~~~~~~~~~~~~~~~~

High confidence threshold with simple append strategy:

.. code-block:: python

   from ssl_framework.strategies import ConfidenceThreshold, AppendAndGrow

   ssl_conservative = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       selection_strategy=ConfidenceThreshold(threshold=0.98),
       integration_strategy=AppendAndGrow(),
       max_iter=10
   )

Aggressive SSL
~~~~~~~~~~~~~~

Fixed count selection with confidence weighting:

.. code-block:: python

   from ssl_framework.strategies import TopKFixedCount, ConfidenceWeighting

   ssl_aggressive = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       selection_strategy=TopKFixedCount(k=50),
       integration_strategy=ConfidenceWeighting(),
       max_iter=5
   )

Experimental SSL
~~~~~~~~~~~~~~~~

Full re-labeling approach (can be computationally expensive):

.. code-block:: python

   from ssl_framework.strategies import TopKFixedCount, FullReLabeling

   ssl_experimental = SelfTrainingClassifier(
       base_model=LogisticRegression(),
       selection_strategy=TopKFixedCount(k=10),
       integration_strategy=FullReLabeling(X_original, y_original),
       max_iter=3
   )

Custom Strategy Implementation
-----------------------------

To implement your own strategies, follow these interfaces:

Selection Strategy Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomSelectionStrategy:
       def select_labels(self, X_unlabeled, y_proba):
           \"\"\"Select samples for pseudo-labeling.

           Parameters
           ----------
           X_unlabeled : ndarray of shape (n_unlabeled_samples, n_features)
               Unlabeled feature data.
           y_proba : ndarray of shape (n_unlabeled_samples, n_classes)
               Predicted class probabilities for unlabeled samples.

           Returns
           -------
           X_new_labeled : ndarray
               Feature data for newly selected samples.
           y_new_labels : ndarray
               Predicted labels for newly selected samples.
           indices_to_remove : ndarray
               Indices of samples to remove from unlabeled set.
           \"\"\"
           # Your selection logic here
           pass

Integration Strategy Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class CustomIntegrationStrategy:
       def integrate_labels(self, X_labeled, y_labeled, X_new_labeled, y_new_labels, **kwargs):
           \"\"\"Integrate new pseudo-labeled samples.

           Parameters
           ----------
           X_labeled : ndarray
               Current labeled feature data.
           y_labeled : ndarray
               Current labeled targets.
           X_new_labeled : ndarray
               New pseudo-labeled feature data.
           y_new_labels : ndarray
               New pseudo-labels.
           **kwargs
               Additional parameters (y_proba, indices, etc.).

           Returns
           -------
           X_labeled_next : ndarray
               Updated labeled feature data.
           y_labeled_next : ndarray
               Updated labeled targets.
           sample_weights_next : ndarray or None
               Sample weights (None if not using weighting).
           \"\"\"
           # Your integration logic here
           pass