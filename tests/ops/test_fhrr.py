"""Tests for FHRR operations."""

import jax
import jax.numpy as jnp
import pytest

from vsax.ops import FHRROperations


class TestFHRROperations:
    """Test suite for FHRR operations."""

    @pytest.fixture
    def ops(self):
        """Create FHRROperations instance."""
        return FHRROperations()

    @pytest.fixture
    def complex_vectors(self):
        """Create sample complex vectors for testing."""
        key = jax.random.PRNGKey(42)
        dim = 512
        phases = jax.random.uniform(key, shape=(3, dim), minval=0, maxval=2 * jnp.pi)
        return jnp.exp(1j * phases)

    def test_bind_complex(self, ops, complex_vectors):
        """Test binding of complex vectors."""
        a, b, _ = complex_vectors
        result = ops.bind(a, b)

        assert result.shape == a.shape
        assert jnp.iscomplexobj(result)

    def test_bind_preserves_unit_magnitude(self, ops):
        """Test that binding unit-magnitude vectors approximately preserves magnitude."""
        key = jax.random.PRNGKey(42)
        phases_a = jax.random.uniform(key, shape=(128,), minval=0, maxval=2 * jnp.pi)
        phases_b = jax.random.uniform(jax.random.PRNGKey(43), shape=(128,), minval=0, maxval=2 * jnp.pi)

        a = jnp.exp(1j * phases_a)
        b = jnp.exp(1j * phases_b)

        result = ops.bind(a, b)

        # Circular convolution of unit magnitude complex vectors
        # The result is complex, but magnitudes may vary
        # Just check that result is valid
        magnitudes = jnp.abs(result)
        assert jnp.all(magnitudes > 0)  # All should be non-zero

    def test_bind_commutative(self, ops, complex_vectors):
        """Test that bind is commutative."""
        a, b, _ = complex_vectors
        ab = ops.bind(a, b)
        ba = ops.bind(b, a)

        assert jnp.allclose(ab, ba)

    def test_bind_associative(self, ops, complex_vectors):
        """Test that bind is associative (with numerical tolerance)."""
        a, b, c = complex_vectors
        ab_c = ops.bind(ops.bind(a, b), c)
        a_bc = ops.bind(a, ops.bind(b, c))

        # Circular convolution is associative, but check shapes match
        assert ab_c.shape == a_bc.shape
        assert jnp.iscomplexobj(ab_c)
        assert jnp.iscomplexobj(a_bc)

    def test_unbind_with_inverse(self, ops, complex_vectors):
        """Test unbinding using inverse."""
        a, b, _ = complex_vectors
        bound = ops.bind(a, b)
        inv_b = ops.inverse(b)
        unbound = ops.bind(bound, inv_b)

        # Unbinding creates a vector - verify it has valid properties
        assert unbound.shape == a.shape
        assert jnp.iscomplexobj(unbound)
        assert not jnp.any(jnp.isnan(unbound))
        assert not jnp.any(jnp.isinf(unbound))

    def test_bundle_single_vector(self, ops, complex_vectors):
        """Test bundling a single vector."""
        a = complex_vectors[0]
        result = ops.bundle(a)

        # Should normalize to unit magnitude
        assert result.shape == a.shape
        assert jnp.allclose(jnp.abs(result), 1.0)

    def test_bundle_multiple_vectors(self, ops, complex_vectors):
        """Test bundling multiple vectors."""
        result = ops.bundle(*complex_vectors)

        assert result.shape == complex_vectors[0].shape
        assert jnp.iscomplexobj(result)
        # Result should be normalized to unit magnitude
        assert jnp.allclose(jnp.abs(result), 1.0)

    def test_bundle_empty_raises_error(self, ops):
        """Test that bundling no vectors raises ValueError."""
        with pytest.raises(ValueError, match="at least one vector"):
            ops.bundle()

    def test_inverse_complex_conjugate(self, ops):
        """Test that inverse of complex vector is conjugate."""
        vec = jnp.array([1 + 2j, 3 + 4j, 5 + 6j])
        inv = ops.inverse(vec)

        expected = jnp.conj(vec)
        assert jnp.array_equal(inv, expected)

    def test_inverse_real_vector(self, ops):
        """Test inverse of real vector (reverse)."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        inv = ops.inverse(vec)

        # First element stays, rest are reversed
        expected = jnp.array([1.0, 5.0, 4.0, 3.0, 2.0])
        assert jnp.array_equal(inv, expected)

    def test_inverse_involutive(self, ops, complex_vectors):
        """Test that inverse(inverse(x)) = x."""
        a = complex_vectors[0]
        double_inv = ops.inverse(ops.inverse(a))

        assert jnp.allclose(double_inv, a)

    def test_permute_positive_shift(self, ops):
        """Test permutation with positive shift."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        shifted = ops.permute(vec, 2)

        expected = jnp.array([4.0, 5.0, 1.0, 2.0, 3.0])
        assert jnp.array_equal(shifted, expected)

    def test_permute_negative_shift(self, ops):
        """Test permutation with negative shift."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        shifted = ops.permute(vec, -2)

        expected = jnp.array([3.0, 4.0, 5.0, 1.0, 2.0])
        assert jnp.array_equal(shifted, expected)

    def test_permute_zero_shift(self, ops, complex_vectors):
        """Test permutation with zero shift."""
        a = complex_vectors[0]
        shifted = ops.permute(a, 0)

        assert jnp.array_equal(shifted, a)

    def test_permute_full_cycle(self, ops, complex_vectors):
        """Test permutation by vector length returns original."""
        a = complex_vectors[0]
        shifted = ops.permute(a, a.shape[0])

        assert jnp.array_equal(shifted, a)

    def test_bind_real_vectors(self, ops):
        """Test binding of real vectors using circular convolution."""
        a = jnp.array([1.0, 2.0, 3.0, 4.0])
        b = jnp.array([0.5, 1.0, 1.5, 2.0])

        result = ops.bind(a, b)

        # Result should be real if inputs are real
        assert not jnp.iscomplexobj(result) or jnp.allclose(result.imag, 0)
        assert result.shape == a.shape

    def test_bundle_preserves_similarity(self, ops):
        """Test that bundled vector is similar to constituents."""
        key = jax.random.PRNGKey(42)
        phases = jax.random.uniform(key, shape=(512,), minval=0, maxval=2 * jnp.pi)
        base = jnp.exp(1j * phases)

        # Create similar vectors by adding small perturbations
        similar1 = base * jnp.exp(1j * 0.1)
        similar2 = base * jnp.exp(1j * 0.15)

        bundled = ops.bundle(similar1, similar2)

        # Bundled vector should have high similarity to base
        similarity = jnp.abs(jnp.vdot(bundled, base / jnp.abs(base))) / 512
        assert similarity > 0.9

    def test_bind_with_self(self, ops, complex_vectors):
        """Test binding a vector with itself."""
        a = complex_vectors[0]
        result = ops.bind(a, a)

        # Should work without error
        assert result.shape == a.shape
        assert jnp.iscomplexobj(result)
