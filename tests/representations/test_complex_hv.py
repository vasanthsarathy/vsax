"""Tests for ComplexHypervector representation."""

import jax.numpy as jnp
import numpy as np
import pytest

from vsax.representations import ComplexHypervector


class TestComplexHypervector:
    """Test suite for ComplexHypervector."""

    def test_init_with_complex_array(self):
        """Test initialization with complex array."""
        vec = jnp.array([1 + 2j, 3 + 4j, 5 + 6j])
        hv = ComplexHypervector(vec)
        assert jnp.array_equal(hv.vec, vec)
        assert hv.vec.dtype == jnp.complex64 or hv.vec.dtype == jnp.complex128

    def test_init_with_real_array_raises_error(self):
        """Test that initialization with real array raises TypeError."""
        vec = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises(TypeError, match="requires complex array"):
            ComplexHypervector(vec)

    def test_normalize_unit_magnitude(self):
        """Test normalization produces unit magnitude vectors."""
        vec = jnp.array([1 + 2j, 3 + 4j, 5 + 6j])
        hv = ComplexHypervector(vec)
        normalized = hv.normalize()

        # Check all magnitudes are 1.0
        assert isinstance(normalized, ComplexHypervector)
        assert jnp.allclose(jnp.abs(normalized.vec), 1.0)

    def test_normalize_preserves_phase(self):
        """Test normalization preserves phase information."""
        vec = jnp.array([1 + 2j, 3 + 4j, 5 + 6j])
        hv = ComplexHypervector(vec)
        normalized = hv.normalize()

        # Phases should be preserved
        original_phases = jnp.angle(hv.vec)
        normalized_phases = jnp.angle(normalized.vec)
        assert jnp.allclose(original_phases, normalized_phases)

    def test_phase_property(self):
        """Test phase property extraction."""
        # Create vector with known phases
        phases = jnp.array([0.0, jnp.pi / 2, jnp.pi])
        vec = jnp.exp(1j * phases)
        hv = ComplexHypervector(vec)

        extracted_phases = hv.phase
        # Note: angle returns values in [-π, π], so π and -π are equivalent
        # Also check that the phase extraction works correctly
        assert jnp.allclose(jnp.abs(extracted_phases[0]), 0, atol=1e-6)
        assert jnp.allclose(jnp.abs(extracted_phases[1] - jnp.pi / 2), 0, atol=1e-6)
        # For π, could be ±π
        assert jnp.allclose(jnp.abs(jnp.abs(extracted_phases[2]) - jnp.pi), 0, atol=1e-6)

    def test_magnitude_property(self):
        """Test magnitude property extraction."""
        vec = jnp.array([3 + 4j, 5 + 12j, 8 + 15j])
        hv = ComplexHypervector(vec)

        expected_magnitudes = jnp.array([5.0, 13.0, 17.0])
        assert jnp.allclose(hv.magnitude, expected_magnitudes)

    def test_shape_property(self):
        """Test shape property."""
        vec = jnp.array([1 + 2j, 3 + 4j, 5 + 6j])
        hv = ComplexHypervector(vec)
        assert hv.shape == (3,)

    def test_dtype_property(self):
        """Test dtype property."""
        vec = jnp.array([1 + 2j, 3 + 4j], dtype=jnp.complex64)
        hv = ComplexHypervector(vec)
        assert hv.dtype == jnp.complex64

    def test_to_numpy(self):
        """Test conversion to numpy array."""
        vec = jnp.array([1 + 2j, 3 + 4j, 5 + 6j])
        hv = ComplexHypervector(vec)
        np_array = hv.to_numpy()

        assert isinstance(np_array, np.ndarray)
        assert np.array_equal(np_array, np.array(vec))

    def test_repr(self):
        """Test string representation."""
        vec = jnp.array([1 + 2j, 3 + 4j])
        hv = ComplexHypervector(vec)
        repr_str = repr(hv)

        assert "ComplexHypervector" in repr_str
        assert "shape=(2,)" in repr_str

    def test_normalize_idempotent(self):
        """Test that normalizing twice gives same result."""
        vec = jnp.array([1 + 2j, 3 + 4j, 5 + 6j])
        hv = ComplexHypervector(vec)
        normalized_once = hv.normalize()
        normalized_twice = normalized_once.normalize()

        assert jnp.allclose(normalized_once.vec, normalized_twice.vec)

    def test_with_zero_magnitude_element(self):
        """Test behavior with zero magnitude elements."""
        vec = jnp.array([0 + 0j, 1 + 1j, 2 + 2j])
        hv = ComplexHypervector(vec)
        normalized = hv.normalize()

        # Zero element should become nan or inf when normalized
        # This is expected behavior for division by zero
        assert jnp.isnan(normalized.vec[0]) or jnp.isinf(normalized.vec[0])
        assert jnp.allclose(jnp.abs(normalized.vec[1:]), 1.0)

    def test_multidimensional_array(self):
        """Test with multidimensional arrays."""
        vec = jnp.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]])
        hv = ComplexHypervector(vec)
        assert hv.shape == (2, 2)

        normalized = hv.normalize()
        assert jnp.allclose(jnp.abs(normalized.vec), 1.0)
