"""Tests for VSAModel dataclass."""

import jax
import jax.numpy as jnp
import pytest

from vsax.core import AbstractHypervector, AbstractOpSet, VSAModel


class SimpleHypervector(AbstractHypervector):
    """Simple hypervector for testing."""

    def normalize(self) -> "SimpleHypervector":
        """L2 normalization."""
        norm = jnp.linalg.norm(self._vec)
        return SimpleHypervector(self._vec / (norm + 1e-8))


class SimpleOpSet(AbstractOpSet):
    """Simple operation set for testing."""

    def bind(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Element-wise multiplication."""
        return a * b

    def bundle(self, *vecs: jnp.ndarray) -> jnp.ndarray:
        """Element-wise mean."""
        return jnp.mean(jnp.stack(vecs), axis=0)

    def inverse(self, a: jnp.ndarray) -> jnp.ndarray:
        """Identity."""
        return a


def simple_sampler(dim: int, n: int, key: jax.random.PRNGKey = None) -> jnp.ndarray:
    """Simple random sampler."""
    if key is None:
        key = jax.random.PRNGKey(0)
    return jax.random.normal(key, shape=(n, dim))


class TestVSAModel:
    """Test VSAModel dataclass."""

    def test_creation(self) -> None:
        """Test basic model creation."""
        model = VSAModel(
            dim=512, rep_cls=SimpleHypervector, opset=SimpleOpSet(), sampler=simple_sampler
        )

        assert model.dim == 512
        assert model.rep_cls == SimpleHypervector
        assert isinstance(model.opset, SimpleOpSet)
        assert callable(model.sampler)

    def test_immutable(self) -> None:
        """Test that model is immutable."""
        model = VSAModel(
            dim=512, rep_cls=SimpleHypervector, opset=SimpleOpSet(), sampler=simple_sampler
        )

        with pytest.raises(Exception):  # FrozenInstanceError in dataclasses
            model.dim = 1024

    def test_validation_zero_dim(self) -> None:
        """Test validation rejects zero dimension."""
        with pytest.raises(ValueError, match="dim must be positive"):
            VSAModel(dim=0, rep_cls=SimpleHypervector, opset=SimpleOpSet(), sampler=simple_sampler)

    def test_validation_negative_dim(self) -> None:
        """Test validation rejects negative dimension."""
        with pytest.raises(ValueError, match="dim must be positive"):
            VSAModel(
                dim=-100, rep_cls=SimpleHypervector, opset=SimpleOpSet(), sampler=simple_sampler
            )

    def test_different_dimensions(self) -> None:
        """Test models with different dimensions."""
        model_512 = VSAModel(
            dim=512, rep_cls=SimpleHypervector, opset=SimpleOpSet(), sampler=simple_sampler
        )

        model_1024 = VSAModel(
            dim=1024, rep_cls=SimpleHypervector, opset=SimpleOpSet(), sampler=simple_sampler
        )

        assert model_512.dim == 512
        assert model_1024.dim == 1024
        assert model_512 != model_1024

    def test_large_dimension(self) -> None:
        """Test model with large dimension."""
        model = VSAModel(
            dim=10000, rep_cls=SimpleHypervector, opset=SimpleOpSet(), sampler=simple_sampler
        )

        assert model.dim == 10000

    def test_sampler_callable(self) -> None:
        """Test that sampler is callable and works."""
        model = VSAModel(
            dim=512, rep_cls=SimpleHypervector, opset=SimpleOpSet(), sampler=simple_sampler
        )

        key = jax.random.PRNGKey(42)
        samples = model.sampler(512, 3, key)

        assert samples.shape == (3, 512)

    def test_different_opsets(self) -> None:
        """Test models with different operation sets."""

        class OpSet1(SimpleOpSet):
            pass

        class OpSet2(SimpleOpSet):
            pass

        model1 = VSAModel(
            dim=512, rep_cls=SimpleHypervector, opset=OpSet1(), sampler=simple_sampler
        )

        model2 = VSAModel(
            dim=512, rep_cls=SimpleHypervector, opset=OpSet2(), sampler=simple_sampler
        )

        assert not isinstance(model1.opset, type(model2.opset))

    def test_different_rep_classes(self) -> None:
        """Test models with different representation classes."""

        class HV1(SimpleHypervector):
            pass

        class HV2(SimpleHypervector):
            pass

        model1 = VSAModel(dim=512, rep_cls=HV1, opset=SimpleOpSet(), sampler=simple_sampler)

        model2 = VSAModel(dim=512, rep_cls=HV2, opset=SimpleOpSet(), sampler=simple_sampler)

        assert model1.rep_cls != model2.rep_cls
