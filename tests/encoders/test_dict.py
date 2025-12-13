"""Tests for DictEncoder."""

import pytest

import jax.numpy as jnp
from vsax import VSAMemory, create_fhrr_model
from vsax.encoders import DictEncoder


def test_dict_encoder_basic():
    """Test basic dictionary encoding."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["subject", "action", "dog", "run"])

    encoder = DictEncoder(model, memory)
    dict_hv = encoder.encode({"subject": "dog", "action": "run"})

    assert dict_hv.vec.shape == (128,)


def test_dict_encoder_deterministic():
    """Test that encoding same dict twice gives same result."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["key1", "key2", "val1", "val2"])

    encoder = DictEncoder(model, memory)

    dict1 = encoder.encode({"key1": "val1", "key2": "val2"})
    dict2 = encoder.encode({"key1": "val1", "key2": "val2"})

    assert jnp.allclose(dict1.vec, dict2.vec)


def test_dict_encoder_order_invariant():
    """Test that dict encoding is order-invariant."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c", "d"])

    encoder = DictEncoder(model, memory)

    # Python dicts maintain insertion order, but bundling is commutative
    # so different iteration orders should give same result
    dict1 = encoder.encode({"a": "b", "c": "d"})
    dict2 = encoder.encode({"c": "d", "a": "b"})

    assert jnp.allclose(dict1.vec, dict2.vec)


def test_dict_encoder_empty_raises_error():
    """Test that empty dict raises ValueError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    encoder = DictEncoder(model, memory)

    with pytest.raises(ValueError, match="Cannot encode empty dictionary"):
        encoder.encode({})


def test_dict_encoder_missing_key_raises_error():
    """Test that missing key raises KeyError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("value")

    encoder = DictEncoder(model, memory)

    with pytest.raises(KeyError):
        encoder.encode({"nonexistent_key": "value"})


def test_dict_encoder_missing_value_raises_error():
    """Test that missing value raises KeyError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("key")

    encoder = DictEncoder(model, memory)

    with pytest.raises(KeyError):
        encoder.encode({"key": "nonexistent_value"})


def test_dict_encoder_different_dicts():
    """Test that different dicts produce different hypervectors."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["k1", "k2", "v1", "v2", "v3"])

    encoder = DictEncoder(model, memory)

    dict1 = encoder.encode({"k1": "v1", "k2": "v2"})
    dict2 = encoder.encode({"k1": "v2", "k2": "v1"})
    dict3 = encoder.encode({"k1": "v1", "k2": "v3"})

    # Different dictionaries should produce different results
    assert not jnp.allclose(dict1.vec, dict2.vec)
    assert not jnp.allclose(dict1.vec, dict3.vec)
    assert not jnp.allclose(dict2.vec, dict3.vec)
