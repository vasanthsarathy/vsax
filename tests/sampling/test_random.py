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


class TestSampleFHRRRandom:
    """Test suite for sample_fhrr_random function.

    These tests validate the critical conjugate symmetry property
    that enables real-valued IFFT results for FHRR operations.
    """

    def test_sample_fhrr_random_shape(self):
        """Test that sampled vectors have correct shape."""
        from vsax.sampling import sample_fhrr_random

        key = jax.random.PRNGKey(42)
        vectors = sample_fhrr_random(dim=512, n=10, key=key)
        assert vectors.shape == (10, 512)

    def test_sample_fhrr_random_is_real(self):
        """Test that sampled vectors are real-valued.

        CRITICAL: This validates that conjugate symmetry works.
        """
        from vsax.sampling import sample_fhrr_random

        key = jax.random.PRNGKey(42)
        vectors = sample_fhrr_random(dim=128, n=5, key=key)
        assert not jnp.iscomplexobj(vectors), "FHRR vectors must be real-valued"

    def test_sample_fhrr_random_conjugate_symmetry(self):
        """Test that frequency-domain representation has conjugate symmetry.

        CRITICAL: This is the test that would have caught the sampling bug.
        Validates: F[k] = conj(F[D-k])
        """
        from vsax.sampling import sample_fhrr_random

        key = jax.random.PRNGKey(42)
        vectors = sample_fhrr_random(dim=128, n=1, key=key)
        vec = vectors[0]

        # Transform to frequency domain
        freq_vec = jnp.fft.fft(vec)

        # Check conjugate symmetry: F[k] = conj(F[D-k])
        dim = len(freq_vec)
        for k in range(1, dim // 2):
            assert jnp.allclose(freq_vec[k], jnp.conj(freq_vec[dim - k]), atol=1e-5), (
                f"Conjugate symmetry violated at k={k}: "
                f"F[{k}]={freq_vec[k]:.6f}, conj(F[{dim - k}])={jnp.conj(freq_vec[dim - k]):.6f}"
            )

        # DC component should be real
        assert jnp.allclose(freq_vec[0].imag, 0, atol=1e-5), "DC component must be real"

        # Nyquist (for even dim) should be real
        if dim % 2 == 0:
            assert jnp.allclose(freq_vec[dim // 2].imag, 0, atol=1e-5), (
                "Nyquist component must be real"
            )

    def test_sample_fhrr_random_unbind_accuracy(self):
        """Test that FHRR sampling enables accurate unbinding.

        CRITICAL: This validates that the conjugate-symmetric sampling
        produces vectors suitable for high-accuracy FHRR unbinding.
        """
        from vsax.ops import FHRROperations
        from vsax.sampling import sample_fhrr_random
        from vsax.similarity import cosine_similarity

        key = jax.random.PRNGKey(42)
        vectors = sample_fhrr_random(dim=1024, n=2, key=key)

        ops = FHRROperations()
        a, b = vectors[0], vectors[1]

        # Bind and unbind
        bound = ops.bind(a, b)
        recovered = ops.unbind(bound, b)

        # Should have very high similarity
        similarity = cosine_similarity(a, recovered)
        assert similarity > 0.95, (
            f"Unbinding similarity {similarity:.4f} too low. "
            "sample_fhrr_random() vectors should enable >95% unbinding accuracy."
        )

    def test_sample_fhrr_random_reproducible(self):
        """Test that same key produces same samples."""
        from vsax.sampling import sample_fhrr_random

        key = jax.random.PRNGKey(42)
        vectors1 = sample_fhrr_random(dim=100, n=5, key=key)
        vectors2 = sample_fhrr_random(dim=100, n=5, key=key)
        assert jnp.allclose(vectors1, vectors2), "Same key must produce identical samples"

    def test_sample_fhrr_random_different_keys(self):
        """Test that different keys produce different samples."""
        from vsax.sampling import sample_fhrr_random

        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(43)
        vectors1 = sample_fhrr_random(dim=100, n=5, key=key1)
        vectors2 = sample_fhrr_random(dim=100, n=5, key=key2)
        assert not jnp.allclose(vectors1, vectors2), "Different keys must produce different samples"

    def test_sample_fhrr_random_default_key(self):
        """Test sampling with default key (None)."""
        from vsax.sampling import sample_fhrr_random

        vectors = sample_fhrr_random(dim=100, n=5, key=None)
        assert vectors.shape == (5, 100)
        assert not jnp.iscomplexobj(vectors)

    def test_sample_fhrr_random_minimum_dimension(self):
        """Test that dim < 2 raises error."""
        import pytest

        from vsax.sampling import sample_fhrr_random

        key = jax.random.PRNGKey(42)
        with pytest.raises(ValueError, match="at least 2"):
            sample_fhrr_random(dim=1, n=5, key=key)

    def test_sample_fhrr_random_odd_dimension(self):
        """Test sampling with odd dimension (no Nyquist frequency)."""
        from vsax.sampling import sample_fhrr_random

        key = jax.random.PRNGKey(42)
        vectors = sample_fhrr_random(dim=127, n=5, key=key)
        assert vectors.shape == (5, 127)
        assert not jnp.iscomplexobj(vectors)

        # Verify conjugate symmetry for odd dimension
        vec = vectors[0]
        freq_vec = jnp.fft.fft(vec)
        dim = 127
        for k in range(1, (dim - 1) // 2 + 1):
            assert jnp.allclose(freq_vec[k], jnp.conj(freq_vec[dim - k]), atol=1e-5)

    def test_sample_fhrr_random_even_dimension(self):
        """Test sampling with even dimension (has Nyquist frequency)."""
        from vsax.sampling import sample_fhrr_random

        key = jax.random.PRNGKey(42)
        vectors = sample_fhrr_random(dim=128, n=5, key=key)
        assert vectors.shape == (5, 128)
        assert not jnp.iscomplexobj(vectors)

        # Verify Nyquist is real
        vec = vectors[0]
        freq_vec = jnp.fft.fft(vec)
        assert jnp.allclose(freq_vec[64].imag, 0, atol=1e-5)
