"""Tests for quaternion similarity function."""

import jax
import jax.numpy as jnp
import pytest

from vsax.representations.quaternion_hv import QuaternionHypervector
from vsax.sampling import sample_quaternion_random
from vsax.similarity.quaternion import quaternion_similarity


class TestQuaternionSimilarity:
    """Test suite for quaternion_similarity."""

    def test_identical_vectors_similarity_one(self):
        """Test that identical vectors have similarity 1.0."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(512, 1, key)
        x = vectors[0]

        similarity = quaternion_similarity(x, x)

        assert abs(similarity - 1.0) < 1e-6

    def test_random_vectors_low_similarity(self):
        """Test that random vectors have low similarity."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(1024, 2, key)
        x, y = vectors[0], vectors[1]

        similarity = quaternion_similarity(x, y)

        # Random unit quaternions should have low similarity
        assert abs(similarity) < 0.3

    def test_symmetric(self):
        """Test that similarity is symmetric."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(512, 2, key)
        x, y = vectors[0], vectors[1]

        sim_xy = quaternion_similarity(x, y)
        sim_yx = quaternion_similarity(y, x)

        assert abs(sim_xy - sim_yx) < 1e-6

    def test_range_minus_one_to_one(self):
        """Test that similarity is in range [-1, 1]."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(512, 10, key)

        for i in range(10):
            for j in range(i, 10):
                sim = quaternion_similarity(vectors[i], vectors[j])
                assert -1.0 <= sim <= 1.0

    def test_opposite_vectors_similarity_negative_one(self):
        """Test that opposite vectors have similarity -1.0."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(512, 1, key)
        x = vectors[0]

        similarity = quaternion_similarity(x, -x)

        assert abs(similarity + 1.0) < 1e-6

    def test_works_with_hypervector_class(self):
        """Test that similarity works with QuaternionHypervector."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(256, 2, key)

        hv1 = QuaternionHypervector(vectors[0])
        hv2 = QuaternionHypervector(vectors[1])

        similarity = quaternion_similarity(hv1, hv2)

        assert isinstance(similarity, float)

    def test_shape_mismatch_raises(self):
        """Test that shape mismatch raises ValueError."""
        key = jax.random.PRNGKey(42)
        x = sample_quaternion_random(256, 1, key)[0]
        y = sample_quaternion_random(512, 1, key)[0]

        with pytest.raises(ValueError, match="Shape mismatch"):
            quaternion_similarity(x, y)

    def test_non_quaternion_shape_raises(self):
        """Test that non-quaternion shape raises ValueError."""
        x = jnp.ones((256, 3))  # Not quaternions
        y = jnp.ones((256, 3))

        with pytest.raises(ValueError, match="last dimension to be 4"):
            quaternion_similarity(x, y)

    def test_normalized_vs_unnormalized(self):
        """Test similarity with normalized vs unnormalized vectors."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(256, 1, key)
        x = vectors[0]

        # Scale x by factor (no longer unit)
        x_scaled = x * 2.0

        # Similarity with self should still be high
        sim = quaternion_similarity(x, x_scaled)
        # Not exactly 1.0 because scaling affects dot product
        assert sim > 0.9

    def test_average_across_coordinates(self):
        """Test that similarity averages across quaternion coordinates."""
        # Create vectors where first half matches, second half is orthogonal
        dim = 100
        x = jnp.zeros((dim, 4))
        y = jnp.zeros((dim, 4))

        # First 50 coordinates: same quaternion
        x = x.at[:50, 0].set(1.0)  # (1, 0, 0, 0)
        y = y.at[:50, 0].set(1.0)  # (1, 0, 0, 0)

        # Last 50 coordinates: orthogonal quaternions
        x = x.at[50:, 0].set(1.0)  # (1, 0, 0, 0)
        y = y.at[50:, 1].set(1.0)  # (0, 1, 0, 0)

        similarity = quaternion_similarity(x, y)

        # Expected: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        assert abs(similarity - 0.5) < 1e-6

    def test_returns_float(self):
        """Test that similarity returns a Python float."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(256, 2, key)
        x, y = vectors[0], vectors[1]

        similarity = quaternion_similarity(x, y)

        assert isinstance(similarity, float)
