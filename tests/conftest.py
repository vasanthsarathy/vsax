"""Pytest configuration and fixtures."""

import jax
import jax.numpy as jnp
import pytest

from vsax.core import AbstractHypervector, AbstractOpSet


class MockHypervector(AbstractHypervector):
    """Mock hypervector for testing."""

    def normalize(self) -> "MockHypervector":
        """L2 normalization."""
        norm = jnp.linalg.norm(self._vec)
        return MockHypervector(self._vec / (norm + 1e-8))


class MockOpSet(AbstractOpSet):
    """Mock operation set for testing."""

    def bind(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Element-wise multiplication."""
        return a * b

    def bundle(self, *vecs: jnp.ndarray) -> jnp.ndarray:
        """Element-wise mean."""
        return jnp.mean(jnp.stack(vecs), axis=0)

    def inverse(self, a: jnp.ndarray) -> jnp.ndarray:
        """Identity (for testing)."""
        return a


@pytest.fixture
def mock_hypervector_class():
    """Fixture providing mock hypervector class."""
    return MockHypervector


@pytest.fixture
def mock_opset():
    """Fixture providing mock operation set instance."""
    return MockOpSet()


@pytest.fixture
def mock_sampler():
    """Fixture providing mock sampling function."""

    def sampler(dim: int, n: int, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        if key is None:
            key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape=(n, dim))

    return sampler


@pytest.fixture
def random_key():
    """Fixture providing a random JAX PRNGKey."""
    return jax.random.PRNGKey(42)
