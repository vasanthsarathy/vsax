"""Tests for VSAMemory class."""

import jax
import jax.numpy as jnp
import pytest

from vsax import (
    VSAMemory,
    create_binary_model,
    create_fhrr_model,
    create_map_model,
)


class TestVSAMemoryFHRR:
    """Tests for VSAMemory with FHRR model."""

    @pytest.fixture
    def fhrr_memory(self):
        """Create VSAMemory with FHRR model."""
        model = create_fhrr_model(dim=512)
        return VSAMemory(model, key=jax.random.PRNGKey(42))

    def test_memory_creation(self, fhrr_memory):
        """Test VSAMemory can be created with FHRR model."""
        assert len(fhrr_memory) == 0
        assert fhrr_memory.keys() == []

    def test_add_symbol(self, fhrr_memory):
        """Test adding a single symbol."""
        hv = fhrr_memory.add("dog")

        assert "dog" in fhrr_memory
        assert len(fhrr_memory) == 1
        assert fhrr_memory.keys() == ["dog"]
        assert jnp.iscomplexobj(hv.vec)
        assert hv.shape == (512,)

    def test_add_duplicate_symbol(self, fhrr_memory):
        """Test adding duplicate symbol returns same hypervector."""
        hv1 = fhrr_memory.add("dog")
        hv2 = fhrr_memory.add("dog")

        assert jnp.array_equal(hv1.vec, hv2.vec)
        assert len(fhrr_memory) == 1

    def test_add_many(self, fhrr_memory):
        """Test adding multiple symbols at once."""
        hvs = fhrr_memory.add_many(["dog", "cat", "bird"])

        assert len(hvs) == 3
        assert len(fhrr_memory) == 3
        assert set(fhrr_memory.keys()) == {"dog", "cat", "bird"}

    def test_get_symbol(self, fhrr_memory):
        """Test retrieving symbol with get()."""
        fhrr_memory.add("dog")
        hv = fhrr_memory.get("dog")

        assert jnp.iscomplexobj(hv.vec)
        assert hv.shape == (512,)

    def test_get_missing_symbol_raises(self, fhrr_memory):
        """Test getting missing symbol raises KeyError."""
        with pytest.raises(KeyError):
            fhrr_memory.get("nonexistent")

    def test_getitem(self, fhrr_memory):
        """Test dictionary-style access."""
        fhrr_memory.add("dog")
        hv = fhrr_memory["dog"]

        assert jnp.iscomplexobj(hv.vec)

    def test_getitem_missing_raises(self, fhrr_memory):
        """Test __getitem__ with missing symbol raises KeyError."""
        with pytest.raises(KeyError):
            _ = fhrr_memory["nonexistent"]

    def test_contains(self, fhrr_memory):
        """Test __contains__ operator."""
        fhrr_memory.add("dog")

        assert "dog" in fhrr_memory
        assert "cat" not in fhrr_memory

    def test_clear(self, fhrr_memory):
        """Test clearing all symbols."""
        fhrr_memory.add_many(["dog", "cat", "bird"])
        assert len(fhrr_memory) == 3

        fhrr_memory.clear()
        assert len(fhrr_memory) == 0
        assert fhrr_memory.keys() == []

    def test_repr(self, fhrr_memory):
        """Test string representation."""
        fhrr_memory.add_many(["a", "b"])
        repr_str = repr(fhrr_memory)

        assert "VSAMemory" in repr_str
        assert "ComplexHypervector" in repr_str
        assert "symbols=2" in repr_str

    def test_reproducible_with_key(self):
        """Test that same key produces same hypervectors."""
        key = jax.random.PRNGKey(42)

        memory1 = VSAMemory(create_fhrr_model(dim=512), key=key)
        hv1 = memory1.add("dog")

        memory2 = VSAMemory(create_fhrr_model(dim=512), key=key)
        hv2 = memory2.add("dog")

        assert jnp.array_equal(hv1.vec, hv2.vec)

    def test_different_keys_produce_different_vectors(self):
        """Test that different keys produce different hypervectors."""
        memory1 = VSAMemory(create_fhrr_model(dim=512), key=jax.random.PRNGKey(42))
        hv1 = memory1.add("dog")

        memory2 = VSAMemory(create_fhrr_model(dim=512), key=jax.random.PRNGKey(43))
        hv2 = memory2.add("dog")

        assert not jnp.array_equal(hv1.vec, hv2.vec)


class TestVSAMemoryMAP:
    """Tests for VSAMemory with MAP model."""

    @pytest.fixture
    def map_memory(self):
        """Create VSAMemory with MAP model."""
        model = create_map_model(dim=512)
        return VSAMemory(model, key=jax.random.PRNGKey(42))

    def test_memory_creation(self, map_memory):
        """Test VSAMemory can be created with MAP model."""
        assert len(map_memory) == 0

    def test_add_symbol(self, map_memory):
        """Test adding a symbol with MAP model."""
        hv = map_memory.add("feature")

        assert "feature" in map_memory
        assert not jnp.iscomplexobj(hv.vec)
        assert hv.shape == (512,)

    def test_add_many(self, map_memory):
        """Test adding multiple symbols."""
        colors = map_memory.add_many(["red", "green", "blue"])

        assert len(colors) == 3
        assert len(map_memory) == 3
        for hv in colors:
            assert not jnp.iscomplexobj(hv.vec)

    def test_repr(self, map_memory):
        """Test string representation."""
        map_memory.add("feature")
        repr_str = repr(map_memory)

        assert "RealHypervector" in repr_str


class TestVSAMemoryBinary:
    """Tests for VSAMemory with Binary model."""

    @pytest.fixture
    def binary_memory(self):
        """Create VSAMemory with Binary model."""
        model = create_binary_model(dim=1000, bipolar=True)
        return VSAMemory(model, key=jax.random.PRNGKey(42))

    def test_memory_creation(self, binary_memory):
        """Test VSAMemory can be created with Binary model."""
        assert len(binary_memory) == 0

    def test_add_symbol(self, binary_memory):
        """Test adding a symbol with Binary model."""
        hv = binary_memory.add("concept")

        assert "concept" in binary_memory
        assert hv.shape == (1000,)
        assert jnp.all(jnp.isin(hv.vec, jnp.array([-1, 1])))

    def test_add_many(self, binary_memory):
        """Test adding multiple symbols."""
        concepts = binary_memory.add_many(["a", "b", "c"])

        assert len(concepts) == 3
        for hv in concepts:
            assert jnp.all(jnp.isin(hv.vec, jnp.array([-1, 1])))

    def test_repr(self, binary_memory):
        """Test string representation."""
        binary_memory.add("concept")
        repr_str = repr(binary_memory)

        assert "BinaryHypervector" in repr_str


class TestVSAMemoryCrossModel:
    """Tests comparing behavior across different models."""

    def test_all_models_support_same_interface(self):
        """Test that all models support the same VSAMemory interface."""
        models = [
            create_fhrr_model(dim=512),
            create_map_model(dim=512),
            create_binary_model(dim=1000, bipolar=True),
        ]

        for model in models:
            memory = VSAMemory(model, key=jax.random.PRNGKey(42))

            # All should support same operations
            memory.add("test")
            assert "test" in memory
            assert len(memory) == 1
            _ = memory["test"]
            assert memory.keys() == ["test"]
            memory.clear()
            assert len(memory) == 0

    def test_different_models_produce_different_types(self):
        """Test that different models produce different hypervector types."""
        fhrr_memory = VSAMemory(create_fhrr_model(dim=512), key=jax.random.PRNGKey(42))
        map_memory = VSAMemory(create_map_model(dim=512), key=jax.random.PRNGKey(42))
        binary_memory = VSAMemory(
            create_binary_model(dim=1000, bipolar=True), key=jax.random.PRNGKey(42)
        )

        fhrr_hv = fhrr_memory.add("symbol")
        map_hv = map_memory.add("symbol")
        binary_hv = binary_memory.add("symbol")

        # FHRR should be complex
        assert jnp.iscomplexobj(fhrr_hv.vec)

        # MAP should be real
        assert not jnp.iscomplexobj(map_hv.vec)

        # Binary should be bipolar
        assert jnp.all(jnp.isin(binary_hv.vec, jnp.array([-1, 1])))
