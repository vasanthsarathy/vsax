"""Tests for factory functions."""

import jax
import jax.numpy as jnp

from vsax import (
    BinaryHypervector,
    BinaryOperations,
    ComplexHypervector,
    FHRROperations,
    MAPOperations,
    RealHypervector,
    create_binary_model,
    create_fhrr_model,
    create_map_model,
)


class TestCreateFHRRModel:
    """Tests for create_fhrr_model factory function."""

    def test_creates_fhrr_model(self):
        """Test that create_fhrr_model creates correct model."""
        model = create_fhrr_model(dim=512)

        assert model.dim == 512
        assert model.rep_cls == ComplexHypervector
        assert isinstance(model.opset, FHRROperations)

    def test_default_dimension(self):
        """Test default dimension is 512."""
        model = create_fhrr_model()
        assert model.dim == 512

    def test_custom_dimension(self):
        """Test custom dimension."""
        model = create_fhrr_model(dim=1024)
        assert model.dim == 1024

    def test_sampler_produces_complex_vectors(self):
        """Test that sampler produces complex vectors."""
        model = create_fhrr_model(dim=256)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(dim=model.dim, n=5, key=key)

        assert vectors.shape == (5, 256)
        assert jnp.iscomplexobj(vectors)

    def test_can_create_hypervectors(self):
        """Test can create and use hypervectors with the model."""
        model = create_fhrr_model(dim=128)
        key = jax.random.PRNGKey(42)

        vec = model.sampler(dim=model.dim, n=1, key=key)[0]
        hv = model.rep_cls(vec)

        assert isinstance(hv, ComplexHypervector)
        assert hv.shape == (128,)

    def test_operations_work(self):
        """Test that operations work with sampled vectors."""
        model = create_fhrr_model(dim=128)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(dim=model.dim, n=2, key=key)
        a, b = vectors[0], vectors[1]

        # Bind should work
        bound = model.opset.bind(a, b)
        assert bound.shape == a.shape

        # Bundle should work
        bundled = model.opset.bundle(a, b)
        assert bundled.shape == a.shape


class TestCreateMAPModel:
    """Tests for create_map_model factory function."""

    def test_creates_map_model(self):
        """Test that create_map_model creates correct model."""
        model = create_map_model(dim=512)

        assert model.dim == 512
        assert model.rep_cls == RealHypervector
        assert isinstance(model.opset, MAPOperations)

    def test_default_dimension(self):
        """Test default dimension is 512."""
        model = create_map_model()
        assert model.dim == 512

    def test_custom_dimension(self):
        """Test custom dimension."""
        model = create_map_model(dim=2048)
        assert model.dim == 2048

    def test_sampler_produces_real_vectors(self):
        """Test that sampler produces real vectors."""
        model = create_map_model(dim=256)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(dim=model.dim, n=5, key=key)

        assert vectors.shape == (5, 256)
        assert not jnp.iscomplexobj(vectors)

    def test_can_create_hypervectors(self):
        """Test can create and use hypervectors with the model."""
        model = create_map_model(dim=128)
        key = jax.random.PRNGKey(42)

        vec = model.sampler(dim=model.dim, n=1, key=key)[0]
        hv = model.rep_cls(vec)

        assert isinstance(hv, RealHypervector)
        assert hv.shape == (128,)

    def test_operations_work(self):
        """Test that operations work with sampled vectors."""
        model = create_map_model(dim=128)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(dim=model.dim, n=2, key=key)
        a, b = vectors[0], vectors[1]

        # Bind should work (element-wise multiplication)
        bound = model.opset.bind(a, b)
        assert bound.shape == a.shape

        # Bundle should work (mean)
        bundled = model.opset.bundle(a, b)
        assert bundled.shape == a.shape


class TestCreateBinaryModel:
    """Tests for create_binary_model factory function."""

    def test_creates_binary_model(self):
        """Test that create_binary_model creates correct model."""
        model = create_binary_model(dim=1000)

        assert model.dim == 1000
        assert model.rep_cls == BinaryHypervector
        assert isinstance(model.opset, BinaryOperations)

    def test_default_dimension(self):
        """Test default dimension is 10000."""
        model = create_binary_model()
        assert model.dim == 10000

    def test_custom_dimension(self):
        """Test custom dimension."""
        model = create_binary_model(dim=5000)
        assert model.dim == 5000

    def test_bipolar_true_by_default(self):
        """Test bipolar=True is the default."""
        model = create_binary_model(dim=1000)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(dim=model.dim, n=5, key=key)

        assert vectors.shape == (5, 1000)
        assert jnp.all(jnp.isin(vectors, jnp.array([-1, 1])))

    def test_bipolar_false(self):
        """Test bipolar=False produces {0, 1} vectors."""
        model = create_binary_model(dim=1000, bipolar=False)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(dim=model.dim, n=5, key=key)

        assert vectors.shape == (5, 1000)
        assert jnp.all(jnp.isin(vectors, jnp.array([0, 1])))

    def test_can_create_hypervectors(self):
        """Test can create and use hypervectors with the model."""
        model = create_binary_model(dim=500, bipolar=True)
        key = jax.random.PRNGKey(42)

        vec = model.sampler(dim=model.dim, n=1, key=key)[0]
        hv = model.rep_cls(vec, bipolar=True)

        assert isinstance(hv, BinaryHypervector)
        assert hv.shape == (500,)

    def test_operations_work(self):
        """Test that operations work with sampled vectors."""
        model = create_binary_model(dim=500, bipolar=True)
        key = jax.random.PRNGKey(42)

        vectors = model.sampler(dim=model.dim, n=2, key=key)
        a, b = vectors[0], vectors[1]

        # Bind should work (XOR)
        bound = model.opset.bind(a, b)
        assert bound.shape == a.shape

        # Bundle should work (majority)
        bundled = model.opset.bundle(a, b)
        assert bundled.shape == a.shape


class TestFactoryFunctionsConsistency:
    """Tests for consistency across factory functions."""

    def test_all_factories_return_vsamodel(self):
        """Test that all factories return VSAModel instances."""
        from vsax.core.model import VSAModel

        fhrr = create_fhrr_model()
        map_model = create_map_model()
        binary = create_binary_model()

        assert isinstance(fhrr, VSAModel)
        assert isinstance(map_model, VSAModel)
        assert isinstance(binary, VSAModel)

    def test_all_models_have_required_attributes(self):
        """Test that all models have required attributes."""
        models = [
            create_fhrr_model(dim=512),
            create_map_model(dim=512),
            create_binary_model(dim=1000),
        ]

        for model in models:
            assert hasattr(model, "dim")
            assert hasattr(model, "rep_cls")
            assert hasattr(model, "opset")
            assert hasattr(model, "sampler")

    def test_all_models_are_immutable(self):
        """Test that all factory-created models are immutable."""
        import pytest

        models = [
            create_fhrr_model(),
            create_map_model(),
            create_binary_model(),
        ]

        for model in models:
            with pytest.raises(Exception):
                model.dim = 9999
