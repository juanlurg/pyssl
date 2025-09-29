# ABOUTME: Strategy classes for label selection and integration in SSL framework
# ABOUTME: Provides modular, swappable components for customizing semi-supervised learning behavior

from typing import Tuple, Optional
import numpy as np


class ConfidenceThreshold:
    """Label selection strategy based on confidence threshold.

    Selects unlabeled samples where the maximum predicted probability
    exceeds a specified threshold.
    """

    def __init__(self, threshold=0.95):
        """Initialize the confidence threshold strategy.

        Parameters
        ----------
        threshold : float, default=0.95
            Confidence threshold for selecting pseudo-labels.
            Samples with max probability > threshold will be selected.
        """
        self.threshold = threshold

    def select_labels(self, X_unlabeled, y_proba):
        """Select samples based on confidence threshold.

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
        """
        # Find samples with max probability > threshold
        max_proba = np.max(y_proba, axis=1)
        confident_indices = np.where(max_proba > self.threshold)[0]

        if len(confident_indices) == 0:
            # Return empty arrays if no confident samples
            return (
                np.empty((0, X_unlabeled.shape[1])),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int)
            )

        # Get pseudo-labels and corresponding features
        y_new_labels = np.argmax(y_proba[confident_indices], axis=1)
        X_new_labeled = X_unlabeled[confident_indices]

        return X_new_labeled, y_new_labels, confident_indices


class AppendAndGrow:
    """Label integration strategy that appends new labels to existing set.

    This strategy grows the labeled dataset monotonically by appending
    newly pseudo-labeled samples to the current labeled set.
    """

    def __init__(self):
        """Initialize the append-and-grow strategy."""
        pass

    def integrate_labels(self, X_labeled, y_labeled, X_new_labeled, y_new_labels, **kwargs):
        """Integrate new pseudo-labeled samples by appending them.

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
            Additional parameters (ignored).

        Returns
        -------
        X_labeled_next : ndarray
            Updated labeled feature data.
        y_labeled_next : ndarray
            Updated labeled targets.
        sample_weights_next : None
            Sample weights (None for this strategy).
        """
        if len(X_new_labeled) == 0:
            # No new samples to add
            return X_labeled, y_labeled, None

        # Concatenate new data with existing labeled data
        X_labeled_next = np.vstack([X_labeled, X_new_labeled])
        y_labeled_next = np.hstack([y_labeled, y_new_labels])

        return X_labeled_next, y_labeled_next, None


class TopKFixedCount:
    """Label selection strategy that selects top K most confident samples.

    This strategy always selects exactly K samples with the highest
    maximum predicted probabilities, regardless of confidence threshold.
    """

    def __init__(self, k=10):
        """Initialize the top-K strategy.

        Parameters
        ----------
        k : int, default=10
            Number of samples to select in each iteration.
        """
        self.k = k

    def select_labels(self, X_unlabeled, y_proba):
        """Select the K most confident samples.

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
        """
        if len(X_unlabeled) == 0:
            return (
                np.empty((0, X_unlabeled.shape[1])),
                np.empty(0, dtype=int),
                np.empty(0, dtype=int)
            )

        # Find the K samples with highest maximum probability
        max_proba = np.max(y_proba, axis=1)
        k_actual = min(self.k, len(X_unlabeled))  # Don't exceed available samples

        # Get indices of top-k most confident samples
        top_k_indices = np.argpartition(max_proba, -k_actual)[-k_actual:]
        # Sort them by confidence (highest first)
        top_k_indices = top_k_indices[np.argsort(max_proba[top_k_indices])[::-1]]

        # Get pseudo-labels and corresponding features
        y_new_labels = np.argmax(y_proba[top_k_indices], axis=1)
        X_new_labeled = X_unlabeled[top_k_indices]

        return X_new_labeled, y_new_labels, top_k_indices


class FullReLabeling:
    """Label integration strategy that re-labels the entire dataset each iteration.

    Instead of growing the labeled set monotonically, this strategy always
    uses the original labeled data plus all newly pseudo-labeled samples.
    """

    def __init__(self, X_original, y_original):
        """Initialize the full re-labeling strategy.

        Parameters
        ----------
        X_original : ndarray
            Original labeled feature data.
        y_original : ndarray
            Original labeled targets.
        """
        self.X_original = X_original.copy()
        self.y_original = y_original.copy()

    def integrate_labels(self, X_labeled, y_labeled, X_new_labeled, y_new_labels, **kwargs):
        """Integrate labels by concatenating with original data only.

        Parameters
        ----------
        X_labeled : ndarray
            Current labeled feature data (ignored).
        y_labeled : ndarray
            Current labeled targets (ignored).
        X_new_labeled : ndarray
            New pseudo-labeled feature data.
        y_new_labels : ndarray
            New pseudo-labels.
        **kwargs
            Additional parameters (ignored).

        Returns
        -------
        X_labeled_next : ndarray
            Original data concatenated with new pseudo-labeled data.
        y_labeled_next : ndarray
            Original labels concatenated with new pseudo-labels.
        sample_weights_next : None
            Sample weights (None for this strategy).
        """
        if len(X_new_labeled) == 0:
            # No new samples, return original data
            return self.X_original, self.y_original, None

        # Concatenate original data with new pseudo-labeled data
        X_labeled_next = np.vstack([self.X_original, X_new_labeled])
        y_labeled_next = np.hstack([self.y_original, y_new_labels])

        return X_labeled_next, y_labeled_next, None


class ConfidenceWeighting:
    """Label integration strategy that weights samples by their confidence.

    Newly pseudo-labeled samples are assigned weights proportional to their
    confidence, while original labeled samples maintain weight 1.0.
    """

    def __init__(self):
        """Initialize the confidence weighting strategy."""
        pass

    def integrate_labels(self, X_labeled, y_labeled, X_new_labeled, y_new_labels, y_proba=None, indices=None):
        """Integrate labels with confidence-based weighting.

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
        y_proba : ndarray, optional
            Predicted probabilities for all unlabeled samples.
        indices : ndarray, optional
            Indices of selected samples in y_proba.

        Returns
        -------
        X_labeled_next : ndarray
            Updated labeled feature data.
        y_labeled_next : ndarray
            Updated labeled targets.
        sample_weights_next : ndarray
            Sample weights with confidence-based weighting.
        """
        if len(X_new_labeled) == 0:
            # No new samples, return current data with unit weights
            sample_weights = np.ones(len(X_labeled))
            return X_labeled, y_labeled, sample_weights

        # Concatenate new data with existing labeled data
        X_labeled_next = np.vstack([X_labeled, X_new_labeled])
        y_labeled_next = np.hstack([y_labeled, y_new_labels])

        # Create sample weights
        original_weights = np.ones(len(X_labeled))  # Original data gets weight 1.0

        if y_proba is not None and indices is not None:
            # Calculate confidence weights from probabilities
            max_proba = np.max(y_proba, axis=1)
            new_weights = max_proba[indices]
        else:
            # Fallback: assign weight 1.0 to new samples
            new_weights = np.ones(len(X_new_labeled))

        sample_weights_next = np.hstack([original_weights, new_weights])

        return X_labeled_next, y_labeled_next, sample_weights_next