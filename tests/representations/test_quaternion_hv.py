"""Tests for QuaternionHypervector representation."""

import jax.numpy as jnp
import pytest

from vsax.representations.quaternion_hv import QuaternionHypervector


class TestQuaternionHypervector:
    """Test suite for QuaternionHypervector."""

    def test_init_valid_shape(self):
        """Test initialization with valid shape."""
        vec = jnp.ones((512, 4))
        hv = QuaternionHypervector(vec)

        assert hv.shape == (512, 4)

    def test_init_invalid_shape_raises(self):
        """Test that invalid shape raises ValueError."""
        vec = jnp.ones((512, 3))  # Last dim should be 4

        with pytest.raises(ValueError, match="last dimension to be 4"):
            QuaternionHypervector(vec)

    def test_init_batch_shape(self):
        """Test initialization with batch dimensions."""
        vec = jnp.ones((10, 512, 4))  # Batch of 10
        hv = QuaternionHypervector(vec)

        assert hv.shape == (10, 512, 4)

    def test_normalize_produces_unit(self):
        """Test that normalize produces unit quaternions."""
        vec = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.float32)
        hv = QuaternionHypervector(vec)

        normalized = hv.normalize()

        assert normalized.is_unit()

    def test_normalize_preserves_shape(self):
        """Test that normalize preserves shape."""
        vec = jnp.ones((512, 4))
        hv = QuaternionHypervector(vec)

        normalized = hv.normalize()

        assert normalized.shape == hv.shape

    def test_normalize_returns_quaternion_hypervector(self):
        """Test that normalize returns QuaternionHypervector."""
        vec = jnp.ones((512, 4))
        hv = QuaternionHypervector(vec)

        normalized = hv.normalize()

        assert isinstance(normalized, QuaternionHypervector)

    def test_dim_property(self):
        """Test dim property returns quaternion count."""
        vec = jnp.ones((512, 4))
        hv = QuaternionHypervector(vec)

        assert hv.dim == 512

    def test_dim_property_batch(self):
        """Test dim property with batch dimensions."""
        vec = jnp.ones((10, 256, 4))
        hv = QuaternionHypervector(vec)

        assert hv.dim == 256

    def test_quaternion_norms(self):
        """Test quaternion_norms property."""
        # Create quaternions with known norms
        vec = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [3, 4, 0, 0]], dtype=jnp.float32)
        hv = QuaternionHypervector(vec)

        norms = hv.quaternion_norms

        assert norms.shape == (3,)
        assert jnp.allclose(norms, jnp.array([1.0, 1.0, 5.0]))

    def test_scalar_part(self):
        """Test scalar_part property."""
        vec = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.float32)
        hv = QuaternionHypervector(vec)

        scalar = hv.scalar_part

        assert scalar.shape == (2,)
        assert jnp.allclose(scalar, jnp.array([1.0, 5.0]))

    def test_vector_part(self):
        """Test vector_part property."""
        vec = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.float32)
        hv = QuaternionHypervector(vec)

        vector = hv.vector_part

        assert vector.shape == (2, 3)
        assert jnp.allclose(vector, jnp.array([[2, 3, 4], [6, 7, 8]]))

    def test_is_unit_true(self):
        """Test is_unit returns True for unit quaternions."""
        # Unit quaternion: (0.5, 0.5, 0.5, 0.5) has norm 1
        vec = jnp.array([[0.5, 0.5, 0.5, 0.5], [1, 0, 0, 0]], dtype=jnp.float32)
        hv = QuaternionHypervector(vec)

        assert hv.is_unit()

    def test_is_unit_false(self):
        """Test is_unit returns False for non-unit quaternions."""
        vec = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.float32)
        hv = QuaternionHypervector(vec)

        assert not hv.is_unit()

    def test_vec_property(self):
        """Test vec property returns underlying array."""
        vec = jnp.ones((512, 4))
        hv = QuaternionHypervector(vec)

        assert jnp.array_equal(hv.vec, vec)

    def test_dtype_property(self):
        """Test dtype property."""
        vec = jnp.ones((512, 4), dtype=jnp.float32)
        hv = QuaternionHypervector(vec)

        assert hv.dtype == jnp.float32

    def test_to_numpy(self):
        """Test to_numpy conversion."""
        import numpy as np

        vec = jnp.ones((512, 4))
        hv = QuaternionHypervector(vec)

        np_array = hv.to_numpy()

        assert isinstance(np_array, np.ndarray)
        assert np_array.shape == (512, 4)

    def test_repr(self):
        """Test string representation."""
        vec = jnp.ones((512, 4))
        hv = QuaternionHypervector(vec)

        repr_str = repr(hv)

        assert "QuaternionHypervector" in repr_str
        assert "(512, 4)" in repr_str
