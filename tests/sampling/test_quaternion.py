"""Tests for quaternion sampling functions."""

import jax
import jax.numpy as jnp

from vsax.sampling import sample_quaternion_random


class TestSampleQuaternionRandom:
    """Test suite for sample_quaternion_random."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(512, 10, key)

        assert vectors.shape == (10, 512, 4)

    def test_unit_quaternions(self):
        """Test that all quaternions have unit length."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(256, 20, key)

        norms = jnp.linalg.norm(vectors, axis=-1)

        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_different_keys_different_vectors(self):
        """Test that different keys produce different vectors."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)

        vectors1 = sample_quaternion_random(256, 5, key1)
        vectors2 = sample_quaternion_random(256, 5, key2)

        assert not jnp.allclose(vectors1, vectors2)

    def test_same_key_same_vectors(self):
        """Test that same key produces same vectors (reproducibility)."""
        key = jax.random.PRNGKey(42)

        vectors1 = sample_quaternion_random(256, 5, key)
        vectors2 = sample_quaternion_random(256, 5, key)

        assert jnp.allclose(vectors1, vectors2)

    def test_default_key(self):
        """Test that default key (None) works."""
        vectors = sample_quaternion_random(256, 5, None)

        assert vectors.shape == (256, 5, 4) or vectors.shape == (5, 256, 4)
        # Just verify it doesn't raise

    def test_quasi_orthogonal(self):
        """Test that different samples are quasi-orthogonal."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(1024, 100, key)

        # Compute average pairwise similarity
        total_sim = 0.0
        count = 0
        for i in range(10):  # Sample pairs
            for j in range(i + 1, 10):
                # Similarity: average dot product of quaternions
                sim = jnp.mean(jnp.sum(vectors[i] * vectors[j], axis=-1))
                total_sim += jnp.abs(sim)
                count += 1

        avg_sim = total_sim / count

        # Should be close to zero for random unit quaternions
        assert avg_sim < 0.2, f"Average similarity {avg_sim} too high"

    def test_single_vector(self):
        """Test sampling a single vector."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(512, 1, key)

        assert vectors.shape == (1, 512, 4)

        norms = jnp.linalg.norm(vectors, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_small_dimension(self):
        """Test with small dimension."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(4, 3, key)

        assert vectors.shape == (3, 4, 4)

        norms = jnp.linalg.norm(vectors, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_large_dimension(self):
        """Test with large dimension."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(4096, 2, key)

        assert vectors.shape == (2, 4096, 4)

        norms = jnp.linalg.norm(vectors, axis=-1)
        assert jnp.allclose(norms, 1.0, atol=1e-6)

    def test_no_nan_or_inf(self):
        """Test that output has no NaN or Inf values."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(512, 10, key)

        assert not jnp.any(jnp.isnan(vectors))
        assert not jnp.any(jnp.isinf(vectors))

    def test_all_components_used(self):
        """Test that all quaternion components (a, b, c, d) have variation."""
        key = jax.random.PRNGKey(42)
        vectors = sample_quaternion_random(1024, 100, key)

        # Check variance in each component
        for i in range(4):
            component_var = jnp.var(vectors[..., i])
            assert component_var > 0.1, f"Component {i} has low variance: {component_var}"
