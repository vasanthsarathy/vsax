"""Tests for abstract base classes."""

import jax.numpy as jnp
import numpy as np

from vsax.core.base import AbstractHypervector, AbstractOpSet


class ConcreteHypervector(AbstractHypervector):
    """Concrete implementation for testing."""

    def normalize(self) -> "ConcreteHypervector":
        """L2 normalization."""
        norm = jnp.linalg.norm(self._vec)
        return ConcreteHypervector(self._vec / (norm + 1e-8))


class ConcreteOpSet(AbstractOpSet):
    """Concrete implementation for testing."""

    def bind(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Element-wise multiplication."""
        return a * b

    def bundle(self, *vecs: jnp.ndarray) -> jnp.ndarray:
        """Element-wise mean."""
        return jnp.mean(jnp.stack(vecs), axis=0)

    def inverse(self, a: jnp.ndarray) -> jnp.ndarray:
        """Identity for testing."""
        return a


class TestAbstractHypervector:
    """Test AbstractHypervector base class."""

    def test_vec_property(self) -> None:
        """Test that vec property returns underlying array."""
        arr = jnp.array([1.0, 2.0, 3.0])
        hv = ConcreteHypervector(arr)
        assert jnp.allclose(hv.vec, arr)

    def test_shape_property(self) -> None:
        """Test that shape property returns correct shape."""
        arr = jnp.array([1.0, 2.0, 3.0])
        hv = ConcreteHypervector(arr)
        assert hv.shape == (3,)

    def test_dtype_property(self) -> None:
        """Test that dtype property returns correct dtype."""
        arr = jnp.array([1.0, 2.0, 3.0])
        hv = ConcreteHypervector(arr)
        assert hv.dtype == jnp.float32 or hv.dtype == jnp.float64

    def test_normalize(self) -> None:
        """Test normalization."""
        arr = jnp.array([3.0, 4.0])
        hv = ConcreteHypervector(arr)
        normalized = hv.normalize()

        # Check that it's normalized (L2 norm = 1)
        assert jnp.allclose(jnp.linalg.norm(normalized.vec), 1.0)

    def test_to_numpy(self) -> None:
        """Test conversion to NumPy array."""
        arr = jnp.array([1.0, 2.0, 3.0])
        hv = ConcreteHypervector(arr)
        np_arr = hv.to_numpy()

        assert isinstance(np_arr, np.ndarray)
        assert np.allclose(np_arr, np.array([1.0, 2.0, 3.0]))

    def test_repr(self) -> None:
        """Test string representation."""
        arr = jnp.array([1.0, 2.0, 3.0])
        hv = ConcreteHypervector(arr)
        repr_str = repr(hv)

        assert "ConcreteHypervector" in repr_str
        assert "shape" in repr_str
        assert "dtype" in repr_str

    def test_2d_array(self) -> None:
        """Test with 2D array."""
        arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        hv = ConcreteHypervector(arr)
        assert hv.shape == (2, 2)


class TestAbstractOpSet:
    """Test AbstractOpSet base class."""

    def test_bind(self) -> None:
        """Test bind operation."""
        opset = ConcreteOpSet()
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([2.0, 3.0, 4.0])
        result = opset.bind(a, b)

        expected = jnp.array([2.0, 6.0, 12.0])
        assert jnp.allclose(result, expected)

    def test_bundle(self) -> None:
        """Test bundle operation."""
        opset = ConcreteOpSet()
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([3.0, 4.0, 5.0])
        result = opset.bundle(a, b)

        expected = jnp.array([2.0, 3.0, 4.0])
        assert jnp.allclose(result, expected)

    def test_bundle_multiple(self) -> None:
        """Test bundle with multiple vectors."""
        opset = ConcreteOpSet()
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([3.0, 4.0, 5.0])
        c = jnp.array([5.0, 6.0, 7.0])
        result = opset.bundle(a, b, c)

        expected = jnp.array([3.0, 4.0, 5.0])
        assert jnp.allclose(result, expected)

    def test_inverse(self) -> None:
        """Test inverse operation."""
        opset = ConcreteOpSet()
        a = jnp.array([1.0, 2.0, 3.0])
        result = opset.inverse(a)

        assert jnp.allclose(result, a)  # Identity for this implementation

    def test_permute_default(self) -> None:
        """Test default permute implementation (circular shift)."""
        opset = ConcreteOpSet()
        a = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Shift right by 1
        result = opset.permute(a, 1)
        expected = jnp.array([4.0, 1.0, 2.0, 3.0])
        assert jnp.allclose(result, expected)

        # Shift left by 1 (same as shift right by -1)
        result = opset.permute(a, -1)
        expected = jnp.array([2.0, 3.0, 4.0, 1.0])
        assert jnp.allclose(result, expected)

    def test_permute_zero(self) -> None:
        """Test permute with zero shift."""
        opset = ConcreteOpSet()
        a = jnp.array([1.0, 2.0, 3.0])
        result = opset.permute(a, 0)

        assert jnp.allclose(result, a)
