"""Tests for SetEncoder."""

import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_fhrr_model
from vsax.encoders import SetEncoder


def test_set_encoder_basic():
    """Test basic set encoding."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    encoder = SetEncoder(model, memory)
    set_hv = encoder.encode({"a", "b", "c"})

    assert set_hv.vec.shape == (128,)


def test_set_encoder_order_invariant():
    """Test that sets are order-invariant."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["x", "y", "z"])

    encoder = SetEncoder(model, memory)

    # Same elements in different order (as list)
    set1 = encoder.encode(["x", "y", "z"])
    set2 = encoder.encode(["z", "y", "x"])
    set3 = encoder.encode(["y", "x", "z"])

    # Should produce the same result (bundling is commutative)
    assert jnp.allclose(set1.vec, set2.vec)
    assert jnp.allclose(set1.vec, set3.vec)


def test_set_encoder_with_set_type():
    """Test encoding with actual Python set."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["dog", "cat", "bird"])

    encoder = SetEncoder(model, memory)
    set_hv = encoder.encode({"dog", "cat", "bird"})

    assert set_hv.vec.shape == (128,)


def test_set_encoder_empty_raises_error():
    """Test that empty set raises ValueError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    encoder = SetEncoder(model, memory)

    with pytest.raises(ValueError, match="Cannot encode empty set"):
        encoder.encode(set())

    with pytest.raises(ValueError, match="Cannot encode empty set"):
        encoder.encode([])


def test_set_encoder_missing_symbol_raises_error():
    """Test that missing symbol raises KeyError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("a")

    encoder = SetEncoder(model, memory)

    with pytest.raises(KeyError):
        encoder.encode({"a", "nonexistent"})


def test_set_encoder_deterministic():
    """Test deterministic encoding for sets."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    encoder = SetEncoder(model, memory)

    set1 = encoder.encode({"a", "b", "c"})
    set2 = encoder.encode({"a", "b", "c"})

    assert jnp.allclose(set1.vec, set2.vec)
