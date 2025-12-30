"""Tests for MAP operations."""

import jax
import jax.numpy as jnp
import pytest

from vsax.ops import MAPOperations


class TestMAPOperations:
    """Test suite for MAP operations."""

    @pytest.fixture
    def ops(self):
        """Create MAPOperations instance."""
        return MAPOperations()

    @pytest.fixture
    def real_vectors(self):
        """Create sample real vectors for testing."""
        key = jax.random.PRNGKey(42)
        dim = 512
        vectors = jax.random.normal(key, shape=(3, dim))
        # Normalize to unit length
        return vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

    def test_bind_elementwise_multiply(self, ops, real_vectors):
        """Test binding via element-wise multiplication."""
        a, b, _ = real_vectors
        result = ops.bind(a, b)

        expected = a * b
        assert jnp.array_equal(result, expected)
        assert result.shape == a.shape

    def test_bind_commutative(self, ops, real_vectors):
        """Test that bind is commutative."""
        a, b, _ = real_vectors
        ab = ops.bind(a, b)
        ba = ops.bind(b, a)

        assert jnp.array_equal(ab, ba)

    def test_bind_associative(self, ops, real_vectors):
        """Test that bind is associative."""
        a, b, c = real_vectors
        ab_c = ops.bind(ops.bind(a, b), c)
        a_bc = ops.bind(a, ops.bind(b, c))

        assert jnp.allclose(ab_c, a_bc)

    def test_unbind_approximate(self, ops, real_vectors):
        """Test approximate unbinding using inverse."""
        a, b, _ = real_vectors
        bound = ops.bind(a, b)
        inv_b = ops.inverse(b)
        unbound = ops.bind(bound, inv_b)

        # MAP unbinding is approximate (not exact)
        # Normalize both vectors for fair comparison
        a_norm = a / (jnp.linalg.norm(a) + 1e-8)
        unbound_norm = unbound / (jnp.linalg.norm(unbound) + 1e-8)

        # Check cosine similarity
        similarity = jnp.dot(a_norm, unbound_norm)
        # Should have some similarity but MAP is very approximate
        assert similarity > 0.3  # Lowered threshold for MAP's approximate nature

    def test_bundle_single_vector(self, ops, real_vectors):
        """Test bundling a single vector."""
        a = real_vectors[0]
        result = ops.bundle(a)

        # Mean of single vector is itself
        assert jnp.array_equal(result, a)

    def test_bundle_two_vectors(self, ops, real_vectors):
        """Test bundling two vectors."""
        a, b, _ = real_vectors
        result = ops.bundle(a, b)

        expected = (a + b) / 2
        assert jnp.allclose(result, expected)

    def test_bundle_multiple_vectors(self, ops, real_vectors):
        """Test bundling multiple vectors."""
        result = ops.bundle(*real_vectors)

        expected = jnp.mean(jnp.stack(real_vectors), axis=0)
        assert jnp.allclose(result, expected)

    def test_bundle_empty_raises_error(self, ops):
        """Test that bundling no vectors raises ValueError."""
        with pytest.raises(ValueError, match="at least one vector"):
            ops.bundle()

    def test_inverse_approximate(self, ops):
        """Test approximate inverse calculation."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0])
        inv = ops.inverse(vec)

        # Inverse should be proportional to vec
        # inv ≈ vec / ||vec||²
        expected = vec / (jnp.sum(vec**2) + 1e-8)
        assert jnp.allclose(inv, expected)

    def test_inverse_with_zero_norm(self, ops):
        """Test inverse with zero-norm vector (division protection)."""
        vec = jnp.array([0.0, 0.0, 0.0])
        inv = ops.inverse(vec)

        # Should not raise error due to 1e-8 protection
        assert not jnp.any(jnp.isnan(inv))
        assert not jnp.any(jnp.isinf(inv))

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

    def test_permute_zero_shift(self, ops, real_vectors):
        """Test permutation with zero shift."""
        a = real_vectors[0]
        shifted = ops.permute(a, 0)

        assert jnp.array_equal(shifted, a)

    def test_bind_with_negative_values(self, ops):
        """Test binding with negative values."""
        a = jnp.array([1.0, -2.0, 3.0, -4.0])
        b = jnp.array([-1.0, 2.0, -3.0, 4.0])

        result = ops.bind(a, b)
        expected = a * b
        assert jnp.array_equal(result, expected)

    def test_bundle_preserves_similarity(self, ops):
        """Test that bundled vector is similar to constituents."""
        key = jax.random.PRNGKey(42)
        base = jax.random.normal(key, shape=(512,))
        base = base / (jnp.linalg.norm(base) + 1e-8)

        # Create very similar vectors (smaller perturbation)
        similar1 = base + jax.random.normal(jax.random.PRNGKey(43), shape=(512,)) * 0.05
        similar2 = base + jax.random.normal(jax.random.PRNGKey(44), shape=(512,)) * 0.05

        # Normalize
        similar1 = similar1 / (jnp.linalg.norm(similar1) + 1e-8)
        similar2 = similar2 / (jnp.linalg.norm(similar2) + 1e-8)

        bundled = ops.bundle(similar1, similar2)
        bundled_norm = bundled / (jnp.linalg.norm(bundled) + 1e-8)

        # Bundled vector should be similar to base
        similarity = jnp.dot(bundled_norm, base)
        assert similarity > 0.6  # Reasonable threshold for averaged vectors

    def test_bind_with_ones(self, ops):
        """Test binding with vector of ones (identity-like)."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0])
        ones = jnp.ones_like(vec)

        result = ops.bind(vec, ones)
        assert jnp.array_equal(result, vec)

    def test_bind_with_zeros(self, ops):
        """Test binding with vector of zeros (zero-like)."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0])
        zeros = jnp.zeros_like(vec)

        result = ops.bind(vec, zeros)
        assert jnp.array_equal(result, zeros)

    def test_bundle_weighted_average(self, ops):
        """Test that bundling creates weighted average."""
        a = jnp.array([1.0, 0.0, 0.0])
        b = jnp.array([0.0, 1.0, 0.0])
        c = jnp.array([0.0, 0.0, 1.0])

        result = ops.bundle(a, b, c)
        expected = jnp.array([1 / 3, 1 / 3, 1 / 3])
        assert jnp.allclose(result, expected)

    def test_bind_distributive_over_bundle(self, ops, real_vectors):
        """Test that bind distributes over bundle (approximately)."""
        a, b, c = real_vectors

        # bind(a, bundle(b, c)) ≈ bundle(bind(a, b), bind(a, c))
        left = ops.bind(a, ops.bundle(b, c))
        right = ops.bundle(ops.bind(a, b), ops.bind(a, c))

        # Should be approximately equal
        assert jnp.allclose(left, right, atol=1e-5)

    # ===== Unbinding Tests =====

    def test_unbind_method_exists(self, ops):
        """Test that unbind method is available in MAPOperations."""
        assert hasattr(ops, "unbind"), "MAPOperations must have unbind method"

    def test_unbind_approximate_recovery(self, ops, real_vectors):
        """Test that unbinding provides approximate recovery for MAP."""
        from vsax.similarity import cosine_similarity

        a, b, _ = real_vectors
        bound = ops.bind(a, b)
        recovered = ops.unbind(bound, b)

        # Normalize for comparison
        a_norm = a / (jnp.linalg.norm(a) + 1e-8)
        recovered_norm = recovered / (jnp.linalg.norm(recovered) + 1e-8)

        # MAP is approximate, so lower threshold than FHRR
        similarity = cosine_similarity(a_norm, recovered_norm)
        assert similarity > 0.3, f"MAP unbinding similarity {similarity:.4f} < 0.3"

    def test_unbind_equivalence(self, ops, real_vectors):
        """Test unbind(a, b) equals bind(a, inverse(b))."""
        a, b, _ = real_vectors

        method1 = ops.unbind(a, b)
        method2 = ops.bind(a, ops.inverse(b))

        assert jnp.allclose(method1, method2)
