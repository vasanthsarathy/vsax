"""Tests for RealHypervector representation."""

import jax.numpy as jnp
import numpy as np
import pytest

from vsax.representations import RealHypervector


class TestRealHypervector:
    """Test suite for RealHypervector."""

    def test_init_with_real_array(self):
        """Test initialization with real array."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0])
        hv = RealHypervector(vec)
        assert jnp.array_equal(hv.vec, vec)
        assert not jnp.iscomplexobj(hv.vec)

    def test_init_with_complex_array_raises_error(self):
        """Test that initialization with complex array raises TypeError."""
        vec = jnp.array([1 + 2j, 3 + 4j, 5 + 6j])
        with pytest.raises(TypeError, match="requires real array"):
            RealHypervector(vec)

    def test_normalize_l2(self):
        """Test L2 normalization produces unit length vectors."""
        vec = jnp.array([3.0, 4.0, 0.0])
        hv = RealHypervector(vec)
        normalized = hv.normalize()

        # Check L2 norm is 1.0
        assert isinstance(normalized, RealHypervector)
        norm = jnp.linalg.norm(normalized.vec)
        assert jnp.allclose(norm, 1.0)

    def test_normalize_preserves_direction(self):
        """Test normalization preserves vector direction."""
        vec = jnp.array([3.0, 4.0, 0.0])
        hv = RealHypervector(vec)
        normalized = hv.normalize()

        # Normalized vector should be parallel to original
        expected = vec / jnp.linalg.norm(vec)
        assert jnp.allclose(normalized.vec, expected)

    def test_normalize_idempotent(self):
        """Test that normalizing twice gives same result."""
        vec = jnp.array([1.0, 2.0, 3.0, 4.0])
        hv = RealHypervector(vec)
        normalized_once = hv.normalize()
        normalized_twice = normalized_once.normalize()

        assert jnp.allclose(normalized_once.vec, normalized_twice.vec)

    def test_shape_property(self):
        """Test shape property."""
        vec = jnp.array([1.0, 2.0, 3.0])
        hv = RealHypervector(vec)
        assert hv.shape == (3,)

    def test_dtype_property(self):
        """Test dtype property."""
        vec = jnp.array([1.0, 2.0], dtype=jnp.float32)
        hv = RealHypervector(vec)
        assert hv.dtype == jnp.float32

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        vec = jnp.array([1.0, 2.0, 3.0])
        hv = RealHypervector(vec)
        np_array = hv.to_numpy()

        assert isinstance(np_array, np.ndarray)
        assert np.array_equal(np_array, np.array(vec))

    def test_repr(self):
        """Test string representation."""
        vec = jnp.array([1.0, 2.0, 3.0])
        hv = RealHypervector(vec)
        repr_str = repr(hv)

        assert "RealHypervector" in repr_str
        assert "shape=(3,)" in repr_str

    def test_with_zero_vector(self):
        """Test behavior with zero vector."""
        vec = jnp.array([0.0, 0.0, 0.0])
        hv = RealHypervector(vec)
        normalized = hv.normalize()

        # Division by zero protection (1e-8 in normalize)
        # Should not raise error, but result will be close to zero
        assert not jnp.any(jnp.isnan(normalized.vec))

    def test_with_negative_values(self):
        """Test with negative values."""
        vec = jnp.array([-3.0, 4.0, -5.0])
        hv = RealHypervector(vec)
        normalized = hv.normalize()

        # Check L2 norm is 1.0
        norm = jnp.linalg.norm(normalized.vec)
        assert jnp.allclose(norm, 1.0)

        # Check direction preserved
        expected = vec / (jnp.linalg.norm(vec) + 1e-8)
        assert jnp.allclose(normalized.vec, expected)

    def test_multidimensional_array(self):
        """Test with multidimensional arrays."""
        vec = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        hv = RealHypervector(vec)
        assert hv.shape == (2, 2)

        normalized = hv.normalize()
        # L2 norm of flattened array should be 1.0
        norm = jnp.linalg.norm(normalized.vec)
        assert jnp.allclose(norm, 1.0)

    def test_init_with_integer_array(self):
        """Test initialization with integer array (should work, gets converted to float)."""
        vec = jnp.array([1, 2, 3, 4])
        hv = RealHypervector(vec)
        # JAX will handle the type conversion
        assert not jnp.iscomplexobj(hv.vec)
