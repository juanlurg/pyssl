# ABOUTME: Utility functions for generating synthetic datasets for SSL demonstrations
# ABOUTME: Provides consistent data generation patterns across all notebook examples

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union
import warnings


def generate_ssl_dataset(
    dataset_type: str = "moons",
    n_samples: int = 1000,
    n_labeled: int = 50,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    **dataset_kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a complete SSL dataset with proper train/labeled/unlabeled/val/test splits.

    Parameters
    ----------
    dataset_type : str, default="moons"
        Type of dataset to generate: "moons", "classification", "blobs"
    n_samples : int, default=1000
        Total number of samples to generate
    n_labeled : int, default=50
        Number of labeled samples for SSL training
    test_size : float, default=0.2
        Fraction of data to hold out for final testing
    val_size : float, default=0.1
        Fraction of training data to use for validation
    random_state : int, default=42
        Random seed for reproducibility
    **dataset_kwargs
        Additional arguments passed to the sklearn dataset generator

    Returns
    -------
    X_labeled : ndarray
        Labeled training features
    y_labeled : ndarray
        Labeled training targets
    X_unlabeled : ndarray
        Unlabeled training features (targets hidden)
    X_val : ndarray
        Validation features
    y_val : ndarray
        Validation targets
    X_test : ndarray
        Test features
    y_test : ndarray
        Test targets
    y_unlabeled_true : ndarray
        True labels for unlabeled data (for evaluation purposes only)
    """
    np.random.seed(random_state)

    # Generate base dataset
    if dataset_type == "moons":
        X, y = make_moons(n_samples=n_samples, random_state=random_state, **dataset_kwargs)
    elif dataset_type == "classification":
        X, y = make_classification(n_samples=n_samples, random_state=random_state, **dataset_kwargs)
    elif dataset_type == "blobs":
        X, y = make_blobs(n_samples=n_samples, random_state=random_state, **dataset_kwargs)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # First split: separate test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: separate validation set from training
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
        )
    else:
        X_val, y_val = None, None

    # Third split: create labeled/unlabeled split from remaining training data
    # Ensure all classes are represented in labeled set
    n_classes = len(np.unique(y_train))
    min_per_class = max(1, n_labeled // n_classes)

    labeled_indices = []
    for class_label in np.unique(y_train):
        class_indices = np.where(y_train == class_label)[0]
        selected = np.random.choice(
            class_indices,
            min(min_per_class, len(class_indices)),
            replace=False
        )
        labeled_indices.extend(selected)

    # Fill remaining labeled samples randomly if needed
    remaining_needed = n_labeled - len(labeled_indices)
    if remaining_needed > 0:
        available_indices = [i for i in range(len(y_train)) if i not in labeled_indices]
        additional = np.random.choice(available_indices, remaining_needed, replace=False)
        labeled_indices.extend(additional)

    labeled_indices = np.array(labeled_indices[:n_labeled])
    unlabeled_indices = np.array([i for i in range(len(y_train)) if i not in labeled_indices])

    X_labeled = X_train[labeled_indices]
    y_labeled = y_train[labeled_indices]
    X_unlabeled = X_train[unlabeled_indices]
    y_unlabeled_true = y_train[unlabeled_indices]  # Hidden from SSL algorithm

    return X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test, y_unlabeled_true


def make_imbalanced_classification(
    n_samples: int = 2000,
    n_features: int = 10,
    n_classes: int = 3,
    weights: Optional[list] = None,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate an imbalanced classification dataset ideal for SSL demonstration.

    Parameters
    ----------
    n_samples : int, default=2000
        Total number of samples
    n_features : int, default=10
        Number of features
    n_classes : int, default=3
        Number of classes
    weights : list, optional
        Class distribution weights. If None, creates [0.1, 0.3, 0.6] imbalance
    random_state : int, default=42
        Random seed
    **kwargs
        Additional arguments for make_classification

    Returns
    -------
    X : ndarray
        Feature matrix
    y : ndarray
        Target vector
    """
    if weights is None:
        if n_classes == 2:
            weights = [0.2, 0.8]
        elif n_classes == 3:
            weights = [0.1, 0.3, 0.6]
        else:
            # Create exponential imbalance
            weights = [0.6 * (0.4 ** i) for i in range(n_classes)]
            weights = [w / sum(weights) for w in weights]  # Normalize

    default_kwargs = {
        'n_informative': min(n_features, max(2, n_features // 2)),
        'n_redundant': min(2, n_features // 4),
        'n_clusters_per_class': min(2, max(1, n_classes // 2)),
        'class_sep': 0.8,
        'flip_y': 0.02,
    }
    default_kwargs.update(kwargs)

    return make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        weights=weights,
        random_state=random_state,
        **default_kwargs
    )


def generate_text_data(
    n_samples: int = 1000,
    categories: list = None,
    subset: str = 'train',
    random_state: int = 42
) -> Tuple[list, np.ndarray]:
    """
    Generate text data for NLP SSL demonstration using 20 Newsgroups.

    Parameters
    ----------
    n_samples : int, default=1000
        Maximum number of samples to return
    categories : list, optional
        List of newsgroup categories. If None, uses related science categories
    subset : str, default='train'
        Which subset to fetch ('train', 'test', 'all')
    random_state : int, default=42
        Random seed for sampling

    Returns
    -------
    texts : list
        List of text documents
    labels : ndarray
        Corresponding labels
    """
    try:
        from sklearn.datasets import fetch_20newsgroups
    except ImportError:
        warnings.warn("sklearn not available for 20newsgroups. Generating synthetic text.")
        return _generate_synthetic_text(n_samples, random_state)

    if categories is None:
        categories = ['sci.med', 'sci.space', 'comp.graphics']

    np.random.seed(random_state)

    # Fetch newsgroups data
    newsgroups = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )

    # Sample if we have more data than requested
    if len(newsgroups.data) > n_samples:
        indices = np.random.choice(len(newsgroups.data), n_samples, replace=False)
        texts = [newsgroups.data[i] for i in indices]
        labels = newsgroups.target[indices]
    else:
        texts = newsgroups.data
        labels = newsgroups.target

    return texts, labels


def _generate_synthetic_text(n_samples: int, random_state: int) -> Tuple[list, np.ndarray]:
    """Generate synthetic text data when real datasets aren't available."""
    np.random.seed(random_state)

    # Simple synthetic text for 3 categories
    templates = [
        # Science/Medical
        ["medical research shows", "patients with condition", "clinical trial results",
         "doctors recommend", "medical study found", "health benefits include"],
        # Science/Space
        ["space exploration mission", "NASA announced today", "astronomical observations",
         "spacecraft launched", "galaxy formation", "planetary surface analysis"],
        # Computer Graphics
        ["graphics rendering algorithm", "3D modeling software", "image processing technique",
         "computer vision system", "digital animation", "pixel shader optimization"]
    ]

    texts = []
    labels = []

    for i in range(n_samples):
        category = i % 3
        template = np.random.choice(templates[category])
        # Add some random variation
        variation = np.random.choice([
            "demonstrates significant", "provides evidence for", "indicates potential",
            "reveals important", "suggests new", "confirms previous"
        ])
        text = f"{template} {variation} findings in recent research studies."
        texts.append(text)
        labels.append(category)

    return texts, np.array(labels)


def create_ssl_benchmark(
    dataset_name: str = "moons_small",
    random_state: int = 42
) -> dict:
    """
    Create standard SSL benchmark scenarios.

    Parameters
    ----------
    dataset_name : str, default="moons_small"
        Benchmark scenario: "moons_small", "imbalanced_medium", "high_dim"
    random_state : int, default=42
        Random seed

    Returns
    -------
    benchmark : dict
        Dictionary containing all data splits and metadata
    """
    if dataset_name == "moons_small":
        # Classic small labeled set scenario
        data = generate_ssl_dataset(
            dataset_type="moons",
            n_samples=800,
            n_labeled=10,
            test_size=0.2,
            val_size=0.1,
            random_state=random_state,
            noise=0.1
        )
        metadata = {
            'name': 'Moons Small Label',
            'description': 'Non-linear 2D data with only 10 labeled samples',
            'expected_improvement': '40-60%',
            'difficulty': 'Easy'
        }

    elif dataset_name == "imbalanced_medium":
        # Imbalanced multi-class
        X, y = make_imbalanced_classification(
            n_samples=1500,
            n_classes=3,
            weights=[0.1, 0.2, 0.7],
            random_state=random_state
        )
        # Split the data
        splits = list(generate_ssl_dataset(
            dataset_type="custom",  # Won't be used since we pass X, y
            n_samples=len(X),
            n_labeled=60,
            random_state=random_state
        ))
        # Replace X, y in the generation
        data = _apply_custom_splits(X, y, n_labeled=60, random_state=random_state)
        metadata = {
            'name': 'Imbalanced Medium',
            'description': 'Imbalanced 3-class problem with moderate labeled data',
            'expected_improvement': '15-25%',
            'difficulty': 'Medium'
        }

    elif dataset_name == "high_dim":
        # High-dimensional challenging case
        X, y = make_classification(
            n_samples=2000,
            n_features=50,
            n_informative=20,
            n_classes=4,
            class_sep=0.6,
            random_state=random_state
        )
        data = _apply_custom_splits(X, y, n_labeled=100, random_state=random_state)
        metadata = {
            'name': 'High Dimensional',
            'description': '50D feature space with 4 classes',
            'expected_improvement': '10-20%',
            'difficulty': 'Hard'
        }
    else:
        raise ValueError(f"Unknown benchmark: {dataset_name}")

    X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test, y_unlabeled_true = data

    return {
        'X_labeled': X_labeled,
        'y_labeled': y_labeled,
        'X_unlabeled': X_unlabeled,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'y_unlabeled_true': y_unlabeled_true,
        'metadata': metadata
    }


def _apply_custom_splits(X, y, n_labeled=50, test_size=0.2, val_size=0.1, random_state=42):
    """Apply the standard SSL splitting logic to custom X, y data."""
    np.random.seed(random_state)

    # First split: separate test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: separate validation set
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, stratify=y_train, random_state=random_state
        )
    else:
        X_val, y_val = None, None

    # Create labeled/unlabeled split
    n_classes = len(np.unique(y_train))
    min_per_class = max(1, n_labeled // n_classes)

    labeled_indices = []
    for class_label in np.unique(y_train):
        class_indices = np.where(y_train == class_label)[0]
        selected = np.random.choice(
            class_indices,
            min(min_per_class, len(class_indices)),
            replace=False
        )
        labeled_indices.extend(selected)

    # Fill remaining if needed
    remaining_needed = n_labeled - len(labeled_indices)
    if remaining_needed > 0:
        available_indices = [i for i in range(len(y_train)) if i not in labeled_indices]
        additional = np.random.choice(available_indices, remaining_needed, replace=False)
        labeled_indices.extend(additional)

    labeled_indices = np.array(labeled_indices[:n_labeled])
    unlabeled_indices = np.array([i for i in range(len(y_train)) if i not in labeled_indices])

    X_labeled = X_train[labeled_indices]
    y_labeled = y_train[labeled_indices]
    X_unlabeled = X_train[unlabeled_indices]
    y_unlabeled_true = y_train[unlabeled_indices]

    return X_labeled, y_labeled, X_unlabeled, X_val, y_val, X_test, y_test, y_unlabeled_true


def make_imbalanced_classification(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    weights: list = None,
    n_informative: int = None,
    n_redundant: int = 2,
    class_sep: float = 0.7,
    random_state: int = 42,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an imbalanced classification dataset.

    This function creates a synthetic classification dataset with imbalanced
    class distributions using sklearn's make_classification.

    Parameters
    ----------
    n_samples : int, default=1000
        Total number of samples
    n_features : int, default=10
        Number of features
    n_classes : int, default=3
        Number of classes
    weights : list, default=None
        List of class weights (should sum to 1.0)
    n_informative : int, default=None
        Number of informative features (defaults to n_features-2)
    n_redundant : int, default=2
        Number of redundant features
    class_sep : float, default=0.7
        Class separation factor
    random_state : int, default=42
        Random seed for reproducibility
    **kwargs
        Additional arguments passed to make_classification

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target labels
    """
    if weights is None:
        weights = [1.0 / n_classes] * n_classes

    if n_informative is None:
        n_informative = max(1, n_features - n_redundant)

    # Generate balanced dataset first
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_informative,
        n_redundant=n_redundant,
        class_sep=class_sep,
        random_state=random_state,
        **kwargs
    )

    # Apply class imbalancing by sampling
    if weights != [1.0 / n_classes] * n_classes:
        np.random.seed(random_state)

        # Calculate target samples per class
        target_samples = [int(w * n_samples) for w in weights]
        # Ensure we don't exceed total samples
        while sum(target_samples) > n_samples:
            max_idx = np.argmax(target_samples)
            target_samples[max_idx] -= 1

        # Sample from each class
        indices_to_keep = []
        for class_label in range(n_classes):
            class_indices = np.where(y == class_label)[0]
            n_to_sample = min(target_samples[class_label], len(class_indices))
            if n_to_sample > 0:
                sampled_indices = np.random.choice(class_indices, n_to_sample, replace=False)
                indices_to_keep.extend(sampled_indices)

        # Keep only selected indices
        indices_to_keep = np.array(indices_to_keep)
        X = X[indices_to_keep]
        y = y[indices_to_keep]

    return X, y