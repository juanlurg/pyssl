# ABOUTME: Core SelfTrainingClassifier implementation for semi-supervised learning
# ABOUTME: Provides scikit-learn compatible SSL classifier with strategy injection support

from typing import Optional, Union, Any
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

    def __init__(
        self,
        base_model: Any,
        max_iter: int = 10,
        selection_strategy: Optional[Any] = None,
        integration_strategy: Optional[Any] = None,
        patience: int = 3,
        tol: float = 0.01,
        labeling_convergence_threshold: int = 5
    ) -> None:
        """Initialize the SelfTrainingClassifier.

        Parameters
        ----------
        base_model : estimator
            Base supervised model that implements fit, predict, and predict_proba.
            Must be sklearn-compatible (e.g., LogisticRegression, RandomForestClassifier).
        max_iter : int, default=10
            Maximum number of iterations for the self-training loop.
        selection_strategy : object, default=None
            Strategy for selecting which unlabeled samples to pseudo-label.
            If None, uses ConfidenceThreshold(0.95).
            Available strategies: ConfidenceThreshold, TopKFixedCount.
        integration_strategy : object, default=None
            Strategy for integrating pseudo-labeled samples into the labeled set.
            If None, uses AppendAndGrow().
            Available strategies: AppendAndGrow, FullReLabeling, ConfidenceWeighting.
        patience : int, default=3
            Number of iterations with no improvement to wait before early stopping.
            Only used when validation data is provided.
        tol : float, default=0.01
            The minimum improvement in validation score to be considered an improvement.
            Only used when validation data is provided.
        labeling_convergence_threshold : int, default=5
            Stop if fewer than this many new labels are added in an iteration.
            Prevents infinite loops when no more confident samples can be found.

        Examples
        --------
        >>> from sklearn.linear_model import LogisticRegression
        >>> from ssl_framework.main import SelfTrainingClassifier
        >>> from ssl_framework.strategies import ConfidenceThreshold, AppendAndGrow
        >>>
        >>> base_model = LogisticRegression(random_state=42)
        >>> selection_strategy = ConfidenceThreshold(threshold=0.9)
        >>> integration_strategy = AppendAndGrow()
        >>>
        >>> ssl_clf = SelfTrainingClassifier(
        ...     base_model=base_model,
        ...     selection_strategy=selection_strategy,
        ...     integration_strategy=integration_strategy,
        ...     max_iter=10
        ... )
        """
        self.base_model = base_model
        self.max_iter = max_iter
        self.selection_strategy = selection_strategy or ConfidenceThreshold(0.95)
        self.integration_strategy = integration_strategy or AppendAndGrow()
        self.patience = patience
        self.tol = tol
        self.labeling_convergence_threshold = labeling_convergence_threshold

    def fit(
        self,
        X_labeled: Union[np.ndarray, pd.DataFrame],
        y_labeled: Union[np.ndarray, pd.Series],
        X_unlabeled: Union[np.ndarray, pd.DataFrame],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> 'SelfTrainingClassifier':
        """Fit the self-training classifier using semi-supervised learning.

        This method iteratively trains the base model by:
        1. Training on current labeled data
        2. Making predictions on unlabeled data
        3. Selecting confident predictions using the selection strategy
        4. Integrating new pseudo-labels using the integration strategy
        5. Repeating until stopping criteria are met

        Parameters
        ----------
        X_labeled : array-like of shape (n_labeled_samples, n_features)
            Initial labeled training data. Can be numpy array or pandas DataFrame.
        y_labeled : array-like of shape (n_labeled_samples,)
            Target values for labeled data. Can be numpy array or pandas Series.
        X_unlabeled : array-like of shape (n_unlabeled_samples, n_features)
            Unlabeled training data to iteratively pseudo-label.
            Can be numpy array or pandas DataFrame.
        X_val : array-like of shape (n_val_samples, n_features), optional
            Validation data for early stopping. If provided with y_val,
            enables early stopping based on validation score plateau.
        y_val : array-like of shape (n_val_samples,), optional
            Validation targets for early stopping.

        Returns
        -------
        self : SelfTrainingClassifier
            Returns the fitted instance.

        Attributes
        ----------
        classes_ : ndarray of shape (n_classes,)
            The classes seen during fit.
        history_ : list of dict
            Training history containing metrics for each iteration:
            - iteration: iteration number
            - labeled_data_count: number of labeled samples before adding new ones
            - new_labels_count: number of new pseudo-labels added
            - average_confidence: mean confidence of newly added samples
            - validation_score: validation score (if validation data provided)
            - stopping_reason: reason for stopping (if applicable)
        stopping_reason_ : str
            Reason why training stopped (e.g., "Maximum iterations reached",
            "Early stopping: no improvement", "Labeling convergence").
        feature_names_ : list or None
            Feature names if input was DataFrame, None otherwise.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.linear_model import LogisticRegression
        >>> from ssl_framework.main import SelfTrainingClassifier
        >>>
        >>> # Create sample data
        >>> X_labeled = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        >>> y_labeled = np.array([0, 0, 1, 1])
        >>> X_unlabeled = np.array([[0.5, 0.5], [10.5, 10.5], [5, 5]])
        >>>
        >>> # Fit SSL classifier
        >>> ssl_clf = SelfTrainingClassifier(LogisticRegression())
        >>> ssl_clf.fit(X_labeled, y_labeled, X_unlabeled)
        >>>
        >>> # Check training progress
        >>> print(f"Stopped due to: {ssl_clf.stopping_reason_}")
        >>> print(f"Training iterations: {len(ssl_clf.history_)}")
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

        # Initialize variables for early stopping
        best_score = -1
        patience_counter = 0
        stopping_reason = None

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
                stopping_reason = "No confident samples found"
                break

            # Labeling Convergence Check: Stop if too few new labels
            if len(X_new_pseudo) < self.labeling_convergence_threshold:
                stopping_reason = f"Labeling convergence: only {len(X_new_pseudo)} new labels (< {self.labeling_convergence_threshold})"

                # Still add these labels before stopping
                new_confidences = np.array([])
                if len(indices_to_remove) > 0:
                    max_proba = np.max(y_proba, axis=1)
                    new_confidences = max_proba[indices_to_remove]

                # Log this iteration
                iteration_log = {
                    'iteration': iteration,
                    'labeled_data_count': len(X_labeled_current),
                    'new_labels_count': len(X_new_pseudo),
                    'average_confidence': np.mean(new_confidences) if len(new_confidences) > 0 else 0.0,
                    'stopping_reason': stopping_reason
                }
                if X_val is not None and y_val is not None:
                    validation_score = self.base_model.score(X_val, y_val)
                    iteration_log['validation_score'] = validation_score

                self.history_.append(iteration_log)

                # Integrate the final labels and break
                X_labeled_current, y_labeled_current, sample_weights = self.integration_strategy.integrate_labels(
                    X_labeled_current, y_labeled_current, X_new_pseudo, y_new_pseudo,
                    y_proba=y_proba, indices=indices_to_remove
                )
                X_unlabeled_current = np.delete(X_unlabeled_current, indices_to_remove, axis=0)
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

            # Early Stopping Check: Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                validation_score = self.base_model.score(X_val, y_val)
                iteration_log['validation_score'] = validation_score

                # Check for improvement
                if validation_score > best_score + self.tol:
                    best_score = validation_score
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Stop if patience exceeded
                if patience_counter >= self.patience:
                    stopping_reason = f"Early stopping: no improvement for {self.patience} iterations"
                    iteration_log['stopping_reason'] = stopping_reason
                    self.history_.append(iteration_log)

                    # Integrate current labels and break
                    X_labeled_current, y_labeled_current, sample_weights = self.integration_strategy.integrate_labels(
                        X_labeled_current, y_labeled_current, X_new_pseudo, y_new_pseudo,
                        y_proba=y_proba, indices=indices_to_remove
                    )
                    break

            self.history_.append(iteration_log)

            # Label Integration: Use strategy to integrate pseudo-labeled data
            X_labeled_current, y_labeled_current, sample_weights = self.integration_strategy.integrate_labels(
                X_labeled_current, y_labeled_current, X_new_pseudo, y_new_pseudo,
                y_proba=y_proba, indices=indices_to_remove
            )

            # Remove newly labeled samples from unlabeled set
            X_unlabeled_current = np.delete(X_unlabeled_current, indices_to_remove, axis=0)

        # Store the stopping reason for inspection
        if stopping_reason is None:
            stopping_reason = f"Maximum iterations reached ({self.max_iter})"
        self.stopping_reason_ = stopping_reason

        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict. Can be numpy array or pandas DataFrame.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels for each sample.
        """
        # Check if the model has been fitted
        check_is_fitted(self, 'classes_')

        # Delegate prediction to the fitted base model
        return self.base_model.predict(X)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict probabilities for. Can be numpy array or pandas DataFrame.

        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities for each sample and class.
        """
        # Check if the model has been fitted
        check_is_fitted(self, 'classes_')

        # Delegate probability prediction to the fitted base model
        return self.base_model.predict_proba(X)