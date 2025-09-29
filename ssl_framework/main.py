# ABOUTME: Core SelfTrainingClassifier implementation for semi-supervised learning
# ABOUTME: Provides scikit-learn compatible SSL classifier with strategy injection support

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from .strategies import ConfidenceThreshold, AppendAndGrow


class SelfTrainingClassifier(BaseEstimator, ClassifierMixin):
    """Semi-supervised learning classifier using self-training approach.

    This classifier wraps a base supervised model and iteratively trains it
    on both labeled and pseudo-labeled data, following the scikit-learn API.
    """

    def __init__(self, base_model, max_iter=10, selection_strategy=None, integration_strategy=None):
        """Initialize the SelfTrainingClassifier.

        Parameters
        ----------
        base_model : estimator
            Base supervised model that implements fit, predict, and predict_proba.
        max_iter : int, default=10
            Maximum number of iterations for the self-training loop.
        selection_strategy : object, default=None
            Strategy for selecting which unlabeled samples to pseudo-label.
            If None, uses ConfidenceThreshold(0.95).
        integration_strategy : object, default=None
            Strategy for integrating pseudo-labeled samples into the labeled set.
            If None, uses AppendAndGrow().
        """
        self.base_model = base_model
        self.max_iter = max_iter
        self.selection_strategy = selection_strategy or ConfidenceThreshold(0.95)
        self.integration_strategy = integration_strategy or AppendAndGrow()

    def fit(self, X_labeled, y_labeled, X_unlabeled, X_val=None, y_val=None):
        """Fit the self-training classifier.

        For now, this simply trains the base model on labeled data only,
        but includes comprehensive input validation.

        Parameters
        ----------
        X_labeled : array-like of shape (n_labeled_samples, n_features)
            Labeled training data.
        y_labeled : array-like of shape (n_labeled_samples,)
            Target values for labeled data.
        X_unlabeled : array-like of shape (n_unlabeled_samples, n_features)
            Unlabeled training data (ignored in this basic version).
        X_val : array-like of shape (n_val_samples, n_features), optional
            Validation data for early stopping.
        y_val : array-like of shape (n_val_samples,), optional
            Validation targets for early stopping.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Data Conversion: Convert DataFrames to NumPy arrays
        if isinstance(X_labeled, pd.DataFrame):
            self.feature_names_ = X_labeled.columns.tolist()
            X_labeled = X_labeled.values
        else:
            X_labeled = np.asarray(X_labeled)
            self.feature_names_ = None

        if isinstance(X_unlabeled, pd.DataFrame):
            X_unlabeled = X_unlabeled.values
        else:
            X_unlabeled = np.asarray(X_unlabeled)

        if X_val is not None:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            else:
                X_val = np.asarray(X_val)

        y_labeled = np.asarray(y_labeled)
        if y_val is not None:
            y_val = np.asarray(y_val)

        # Base Estimator Check: Verify required methods exist
        required_methods = ['fit', 'predict', 'predict_proba']
        for method in required_methods:
            if not hasattr(self.base_model, method):
                raise TypeError(
                    f"Base estimator must implement {method} method. "
                    f"Got {type(self.base_model).__name__} which is missing {method}."
                )

        # Labeled Data Consistency Check
        if X_labeled.shape[0] != y_labeled.shape[0]:
            raise ValueError(
                f"X_labeled and y_labeled must have the same number of samples. "
                f"Got X_labeled: {X_labeled.shape[0]}, y_labeled: {y_labeled.shape[0]}"
            )

        # Feature Dimensionality Check
        if X_labeled.shape[1] != X_unlabeled.shape[1]:
            raise ValueError(
                f"X_labeled and X_unlabeled must have the same number of features. "
                f"Got X_labeled: {X_labeled.shape[1]}, X_unlabeled: {X_unlabeled.shape[1]}"
            )

        if X_val is not None and X_labeled.shape[1] != X_val.shape[1]:
            raise ValueError(
                f"X_labeled and X_val must have the same number of features. "
                f"Got X_labeled: {X_labeled.shape[1]}, X_val: {X_val.shape[1]}"
            )

        # Store the classes found in y_labeled
        self.classes_ = np.unique(y_labeled)

        # Initialize history for logging
        self.history_ = []

        # Make copies of input data to avoid modifying user's original data
        X_labeled_current = X_labeled.copy()
        y_labeled_current = y_labeled.copy()
        X_unlabeled_current = X_unlabeled.copy()

        # Initialize sample weights for the first iteration
        sample_weights = None

        # Iterative self-training loop
        for iteration in range(self.max_iter):
            # Train the base model on current labeled data
            if sample_weights is not None:
                self.base_model.fit(X_labeled_current, y_labeled_current, sample_weight=sample_weights)
            else:
                self.base_model.fit(X_labeled_current, y_labeled_current)

            # If no unlabeled data left, break
            if len(X_unlabeled_current) == 0:
                break

            # Predict probabilities on unlabeled data
            y_proba = self.base_model.predict_proba(X_unlabeled_current)

            # Label Selection: Use strategy to select samples for pseudo-labeling
            X_new_pseudo, y_new_pseudo, indices_to_remove = self.selection_strategy.select_labels(
                X_unlabeled_current, y_proba
            )

            # If no new samples selected, break
            if len(X_new_pseudo) == 0:
                break

            # Calculate confidences for logging
            if len(indices_to_remove) > 0:
                max_proba = np.max(y_proba, axis=1)
                new_confidences = max_proba[indices_to_remove]
            else:
                new_confidences = np.array([])

            # Logging: Calculate and store metrics for this iteration
            iteration_log = {
                'iteration': iteration,
                'labeled_data_count': len(X_labeled_current),
                'new_labels_count': len(X_new_pseudo),
                'average_confidence': np.mean(new_confidences) if len(new_confidences) > 0 else 0.0
            }
            self.history_.append(iteration_log)

            # Label Integration: Use strategy to integrate pseudo-labeled data
            X_labeled_current, y_labeled_current, sample_weights = self.integration_strategy.integrate_labels(
                X_labeled_current, y_labeled_current, X_new_pseudo, y_new_pseudo,
                y_proba=y_proba, indices=indices_to_remove
            )

            # Remove newly labeled samples from unlabeled set
            X_unlabeled_current = np.delete(X_unlabeled_current, indices_to_remove, axis=0)

        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        # Check if the model has been fitted
        check_is_fitted(self, 'classes_')

        # Delegate prediction to the fitted base model
        return self.base_model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict probabilities for.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Check if the model has been fitted
        check_is_fitted(self, 'classes_')

        # Delegate probability prediction to the fitted base model
        return self.base_model.predict_proba(X)