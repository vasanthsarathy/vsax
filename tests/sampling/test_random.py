"""Tests for random sampling functions."""

import jax
import jax.numpy as jnp

from vsax.sampling import sample_binary_random, sample_complex_random, sample_random


class TestSampleRandom:
    """Test suite for sample_random function."""

    def test_sample_random_shape(self):
        """Test that sampled vectors have correct shape."""
        key = jax.random.PRNGKey(42)
        vectors = sample_random(dim=512, n=10, key=key)
        assert vectors.shape == (10, 512)

    def test_sample_random_is_real(self):
        """Test that sampled vectors are real-valued."""
        key = jax.random.PRNGKey(42)
        vectors = sample_random(dim=100, n=5, key=key)
        assert not jnp.iscomplexobj(vectors)

    def test_sample_random_distribution(self):
        """Test that samples approximate normal distribution."""
        key = jax.random.PRNGKey(42)
        vectors = sample_random(dim=10000, n=1, key=key)

        # Mean should be close to 0
        mean = jnp.mean(vectors)
        assert jnp.abs(mean) < 0.1

        # Std should be close to 1
        std = jnp.std(vectors)
        assert 0.9 < std < 1.1

    def test_sample_random_default_key(self):
        """Test sampling with default key (None)."""
        vectors = sample_random(dim=100, n=5, key=None)
        assert vectors.shape == (5, 100)

    def test_sample_random_reproducible(self):
        """Test that same key produces same samples."""
        key = jax.random.PRNGKey(42)
        vectors1 = sample_random(dim=100, n=5, key=key)
        vectors2 = sample_random(dim=100, n=5, key=key)
        assert jnp.array_equal(vectors1, vectors2)

    def test_sample_random_different_keys(self):
        """Test that different keys produce different samples."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)
        vectors1 = sample_random(dim=100, n=5, key=key1)
        vectors2 = sample_random(dim=100, n=5, key=key2)
        assert not jnp.array_equal(vectors1, vectors2)

    def test_sample_random_single_vector(self):
        """Test sampling a single vector."""
        key = jax.random.PRNGKey(42)
        vectors = sample_random(dim=100, n=1, key=key)
        assert vectors.shape == (1, 100)


class TestSampleComplexRandom:
    """Test suite for sample_complex_random function."""

    def test_sample_complex_random_shape(self):
        """Test that sampled vectors have correct shape."""
        key = jax.random.PRNGKey(42)
        vectors = sample_complex_random(dim=512, n=10, key=key)
        assert vectors.shape == (10, 512)

    def test_sample_complex_random_is_complex(self):
        """Test that sampled vectors are complex-valued."""
        key = jax.random.PRNGKey(42)
        vectors = sample_complex_random(dim=100, n=5, key=key)
        assert jnp.iscomplexobj(vectors)

    def test_sample_complex_random_unit_magnitude(self):
        """Test that all sampled vectors have unit magnitude."""
        key = jax.random.PRNGKey(42)
        vectors = sample_complex_random(dim=100, n=10, key=key)

        magnitudes = jnp.abs(vectors)
        assert jnp.allclose(magnitudes, 1.0)

    def test_sample_complex_random_phase_distribution(self):
        """Test that phases are uniformly distributed."""
        key = jax.random.PRNGKey(42)
        vectors = sample_complex_random(dim=10000, n=1, key=key)

        phases = jnp.angle(vectors)

        # Phases should be roughly uniform in [0, 2π)
        # Mean phase should be around π
        mean_phase = jnp.mean(phases)
        assert -0.5 < mean_phase < 0.5  # Could be near 0 or 2π

        # Check range
        assert jnp.min(phases) >= -jnp.pi
        assert jnp.max(phases) <= jnp.pi

    def test_sample_complex_random_default_key(self):
        """Test sampling with default key (None)."""
        vectors = sample_complex_random(dim=100, n=5, key=None)
        assert vectors.shape == (5, 100)
        assert jnp.iscomplexobj(vectors)

    def test_sample_complex_random_reproducible(self):
        """Test that same key produces same samples."""
        key = jax.random.PRNGKey(42)
        vectors1 = sample_complex_random(dim=100, n=5, key=key)
        vectors2 = sample_complex_random(dim=100, n=5, key=key)
        assert jnp.array_equal(vectors1, vectors2)

    def test_sample_complex_random_different_keys(self):
        """Test that different keys produce different samples."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)
        vectors1 = sample_complex_random(dim=100, n=5, key=key1)
        vectors2 = sample_complex_random(dim=100, n=5, key=key2)
        assert not jnp.array_equal(vectors1, vectors2)

    def test_sample_complex_random_single_vector(self):
        """Test sampling a single vector."""
        key = jax.random.PRNGKey(42)
        vectors = sample_complex_random(dim=100, n=1, key=key)
        assert vectors.shape == (1, 100)


class TestSampleBinaryRandom:
    """Test suite for sample_binary_random function."""

    def test_sample_binary_random_bipolar_shape(self):
        """Test that sampled bipolar vectors have correct shape."""
        key = jax.random.PRNGKey(42)
        vectors = sample_binary_random(dim=512, n=10, key=key, bipolar=True)
        assert vectors.shape == (10, 512)

    def test_sample_binary_random_binary_shape(self):
        """Test that sampled binary vectors have correct shape."""
        key = jax.random.PRNGKey(42)
        vectors = sample_binary_random(dim=512, n=10, key=key, bipolar=False)
        assert vectors.shape == (10, 512)

    def test_sample_binary_random_bipolar_values(self):
        """Test that bipolar sampling produces {-1, +1} values."""
        key = jax.random.PRNGKey(42)
        vectors = sample_binary_random(dim=100, n=10, key=key, bipolar=True)
        assert jnp.all(jnp.isin(vectors, jnp.array([-1, 1])))

    def test_sample_binary_random_binary_values(self):
        """Test that binary sampling produces {0, 1} values."""
        key = jax.random.PRNGKey(42)
        vectors = sample_binary_random(dim=100, n=10, key=key, bipolar=False)
        assert jnp.all(jnp.isin(vectors, jnp.array([0, 1])))

    def test_sample_binary_random_bipolar_distribution(self):
        """Test that bipolar values are roughly balanced."""
        key = jax.random.PRNGKey(42)
        vectors = sample_binary_random(dim=10000, n=1, key=key, bipolar=True)

        # Count of +1 and -1 should be roughly equal
        ones = jnp.sum(vectors == 1)
        minus_ones = jnp.sum(vectors == -1)

        # Allow 10% deviation
        assert 0.4 < (ones / 10000) < 0.6
        assert 0.4 < (minus_ones / 10000) < 0.6

    def test_sample_binary_random_binary_distribution(self):
        """Test that binary values are roughly balanced."""
        key = jax.random.PRNGKey(42)
        vectors = sample_binary_random(dim=10000, n=1, key=key, bipolar=False)

        # Count of 0 and 1 should be roughly equal
        zeros = jnp.sum(vectors == 0)
        ones = jnp.sum(vectors == 1)

        # Allow 10% deviation
        assert 0.4 < (zeros / 10000) < 0.6
        assert 0.4 < (ones / 10000) < 0.6

    def test_sample_binary_random_default_key(self):
        """Test sampling with default key (None)."""
        vectors = sample_binary_random(dim=100, n=5, key=None, bipolar=True)
        assert vectors.shape == (5, 100)

    def test_sample_binary_random_default_bipolar(self):
        """Test that bipolar=True is the default."""
        key = jax.random.PRNGKey(42)
        vectors = sample_binary_random(dim=100, n=5, key=key)
        assert jnp.all(jnp.isin(vectors, jnp.array([-1, 1])))

    def test_sample_binary_random_reproducible(self):
        """Test that same key produces same samples."""
        key = jax.random.PRNGKey(42)
        vectors1 = sample_binary_random(dim=100, n=5, key=key, bipolar=True)
        vectors2 = sample_binary_random(dim=100, n=5, key=key, bipolar=True)
        assert jnp.array_equal(vectors1, vectors2)

    def test_sample_binary_random_different_keys(self):
        """Test that different keys produce different samples."""
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)
        vectors1 = sample_binary_random(dim=100, n=5, key=key1, bipolar=True)
        vectors2 = sample_binary_random(dim=100, n=5, key=key2, bipolar=True)
        assert not jnp.array_equal(vectors1, vectors2)

    def test_sample_binary_random_single_vector(self):
        """Test sampling a single vector."""
        key = jax.random.PRNGKey(42)
        vectors = sample_binary_random(dim=100, n=1, key=key, bipolar=True)
        assert vectors.shape == (1, 100)
