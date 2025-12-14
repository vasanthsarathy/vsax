"""Dataset helper utilities for VSAX tutorials.

This module provides helper functions for loading and preprocessing datasets
used in the tutorials.
"""

import jax.numpy as jnp
import numpy as np


def load_mnist_digits() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the sklearn digits dataset (8x8 MNIST subset).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        - X_train: Training images (n_samples, 64) - flattened 8x8 images
        - X_test: Test images (n_samples, 64)
        - y_train: Training labels (n_samples,)
        - y_test: Test labels (n_samples,)
    """
    try:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        raise ImportError(
            "scikit-learn is required for this tutorial. "
            "Install it with: pip install scikit-learn"
        ) from e

    # Load digits dataset (8x8 images of digits 0-9)
    digits = load_digits()
    X = digits.data  # Already flattened to (n_samples, 64)
    y = digits.target

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def normalize_images(images: np.ndarray) -> np.ndarray:
    """Normalize image pixel values to [0, 1] range.

    Args:
        images: Array of images with shape (n_samples, n_pixels)

    Returns:
        Normalized images in [0, 1] range
    """
    return images / 16.0  # Digits dataset has values 0-16


def images_to_jax(images: np.ndarray) -> jnp.ndarray:
    """Convert numpy images to JAX arrays.

    Args:
        images: Numpy array of images

    Returns:
        JAX array of images
    """
    return jnp.array(images)
