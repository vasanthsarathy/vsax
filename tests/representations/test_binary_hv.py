"""Tests for BinaryHypervector representation."""

import jax.numpy as jnp
import numpy as np
import pytest

from vsax.representations import BinaryHypervector


class TestBinaryHypervector:
    """Test suite for BinaryHypervector."""

    def test_init_bipolar(self):
        """Test initialization with bipolar {-1, +1} values."""
        vec = jnp.array([1, -1, 1, -1, 1])
        hv = BinaryHypervector(vec, bipolar=True)
        assert jnp.array_equal(hv.vec, vec)
        assert hv.bipolar

    def test_init_binary(self):
        """Test initialization with binary {0, 1} values."""
        vec = jnp.array([1, 0, 1, 0, 1])
        hv = BinaryHypervector(vec, bipolar=False)
        assert jnp.array_equal(hv.vec, vec)
        assert not hv.bipolar

    def test_init_bipolar_invalid_values(self):
        """Test that invalid bipolar values raise ValueError."""
        vec = jnp.array([1, -1, 0, 1])  # 0 is invalid for bipolar
        with pytest.raises(ValueError, match="Bipolar"):
            BinaryHypervector(vec, bipolar=True)

    def test_init_binary_invalid_values(self):
        """Test that invalid binary values raise ValueError."""
        vec = jnp.array([1, 0, 2, 1])  # 2 is invalid for binary
        with pytest.raises(ValueError, match="binary"):
            BinaryHypervector(vec, bipolar=False)

    def test_normalize_no_op(self):
        """Test that normalization is a no-op for binary vectors."""
        vec = jnp.array([1, -1, 1, -1])
        hv = BinaryHypervector(vec, bipolar=True)
        normalized = hv.normalize()

        assert isinstance(normalized, BinaryHypervector)
        assert jnp.array_equal(normalized.vec, vec)
        assert normalized.bipolar == hv.bipolar

    def test_to_bipolar_already_bipolar(self):
        """Test to_bipolar when already bipolar."""
        vec = jnp.array([1, -1, 1, -1])
        hv = BinaryHypervector(vec, bipolar=True)
        bipolar_hv = hv.to_bipolar()

        assert bipolar_hv is hv  # Should return self
        assert jnp.array_equal(bipolar_hv.vec, vec)

    def test_to_bipolar_from_binary(self):
        """Test conversion from binary {0, 1} to bipolar {-1, +1}."""
        vec = jnp.array([1, 0, 1, 0])
        hv = BinaryHypervector(vec, bipolar=False)
        bipolar_hv = hv.to_bipolar()

        expected = jnp.array([1, -1, 1, -1])
        assert bipolar_hv.bipolar
        assert jnp.array_equal(bipolar_hv.vec, expected)

    def test_to_binary_already_binary(self):
        """Test to_binary when already binary."""
        vec = jnp.array([1, 0, 1, 0])
        hv = BinaryHypervector(vec, bipolar=False)
        binary_hv = hv.to_binary()

        assert binary_hv is hv  # Should return self
        assert jnp.array_equal(binary_hv.vec, vec)

    def test_to_binary_from_bipolar(self):
        """Test conversion from bipolar {-1, +1} to binary {0, 1}."""
        vec = jnp.array([1, -1, 1, -1])
        hv = BinaryHypervector(vec, bipolar=True)
        binary_hv = hv.to_binary()

        expected = jnp.array([1, 0, 1, 0])
        assert not binary_hv.bipolar
        assert jnp.array_equal(binary_hv.vec, expected)

    def test_round_trip_conversion(self):
        """Test that bipolar->binary->bipolar preserves values."""
        vec = jnp.array([1, -1, 1, -1, 1])
        hv = BinaryHypervector(vec, bipolar=True)

        # Convert to binary and back
        binary_hv = hv.to_binary()
        back_to_bipolar = binary_hv.to_bipolar()

        assert jnp.array_equal(back_to_bipolar.vec, vec)
        assert back_to_bipolar.bipolar

    def test_shape_property(self):
        """Test shape property."""
        vec = jnp.array([1, -1, 1])
        hv = BinaryHypervector(vec, bipolar=True)
        assert hv.shape == (3,)

    def test_dtype_property(self):
        """Test dtype property."""
        vec = jnp.array([1, -1, 1, -1], dtype=jnp.int32)
        hv = BinaryHypervector(vec, bipolar=True)
        assert hv.dtype == jnp.int32

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        vec = jnp.array([1, -1, 1, -1])
        hv = BinaryHypervector(vec, bipolar=True)
        np_array = hv.to_numpy()

        assert isinstance(np_array, np.ndarray)
        assert np.array_equal(np_array, np.array(vec))

    def test_repr_bipolar(self):
        """Test string representation for bipolar."""
        vec = jnp.array([1, -1, 1])
        hv = BinaryHypervector(vec, bipolar=True)
        repr_str = repr(hv)

        assert "BinaryHypervector" in repr_str
        assert "shape=(3,)" in repr_str

    def test_repr_binary(self):
        """Test string representation for binary."""
        vec = jnp.array([1, 0, 1])
        hv = BinaryHypervector(vec, bipolar=False)
        repr_str = repr(hv)

        assert "BinaryHypervector" in repr_str
        assert "shape=(3,)" in repr_str

    def test_multidimensional_bipolar(self):
        """Test with multidimensional bipolar arrays."""
        vec = jnp.array([[1, -1], [1, -1]])
        hv = BinaryHypervector(vec, bipolar=True)
        assert hv.shape == (2, 2)
        assert hv.bipolar

    def test_multidimensional_binary(self):
        """Test with multidimensional binary arrays."""
        vec = jnp.array([[1, 0], [0, 1]])
        hv = BinaryHypervector(vec, bipolar=False)
        assert hv.shape == (2, 2)
        assert not hv.bipolar

    def test_all_ones_bipolar(self):
        """Test with all +1 values."""
        vec = jnp.array([1, 1, 1, 1])
        hv = BinaryHypervector(vec, bipolar=True)
        assert jnp.array_equal(hv.vec, vec)

    def test_all_minus_ones_bipolar(self):
        """Test with all -1 values."""
        vec = jnp.array([-1, -1, -1, -1])
        hv = BinaryHypervector(vec, bipolar=True)
        assert jnp.array_equal(hv.vec, vec)

    def test_all_zeros_binary(self):
        """Test with all 0 values."""
        vec = jnp.array([0, 0, 0, 0])
        hv = BinaryHypervector(vec, bipolar=False)
        assert jnp.array_equal(hv.vec, vec)

    def test_all_ones_binary(self):
        """Test with all 1 values."""
        vec = jnp.array([1, 1, 1, 1])
        hv = BinaryHypervector(vec, bipolar=False)
        assert jnp.array_equal(hv.vec, vec)
