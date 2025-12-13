"""Tests for ScalarEncoder."""

import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_binary_model, create_fhrr_model, create_map_model
from vsax.encoders import ScalarEncoder


def test_scalar_encoder_initialization():
    """Test ScalarEncoder initialization."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("temperature")

    encoder = ScalarEncoder(model, memory)

    assert encoder.model is model
    assert encoder.memory is memory
    assert encoder.min_val is None
    assert encoder.max_val is None


def test_scalar_encoder_with_range():
    """Test ScalarEncoder with specified range."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("temperature")

    encoder = ScalarEncoder(model, memory, min_val=0, max_val=100)

    assert encoder.min_val == 0
    assert encoder.max_val == 100


def test_scalar_encoder_complex_power_encoding():
    """Test power encoding for complex hypervectors (FHRR)."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("value")

    encoder = ScalarEncoder(model, memory)

    # Encode a value
    hv1 = encoder.encode("value", 1.0)
    hv2 = encoder.encode("value", 2.0)
    hv3 = encoder.encode("value", 0.5)

    # Check that results are complex hypervectors
    assert jnp.iscomplexobj(hv1.vec)
    assert jnp.iscomplexobj(hv2.vec)
    assert jnp.iscomplexobj(hv3.vec)

    # Different values should produce different hypervectors
    assert not jnp.allclose(hv1.vec, hv2.vec)
    assert not jnp.allclose(hv1.vec, hv3.vec)


def test_scalar_encoder_real_iterated_binding():
    """Test iterated binding for real hypervectors (MAP)."""
    model = create_map_model(dim=128)
    memory = VSAMemory(model)
    memory.add("value")

    encoder = ScalarEncoder(model, memory)

    # Encode values
    hv1 = encoder.encode("value", 0.0)
    hv2 = encoder.encode("value", 0.5)
    hv3 = encoder.encode("value", 1.0)

    # Check that results are real hypervectors
    assert not jnp.iscomplexobj(hv1.vec)
    assert not jnp.iscomplexobj(hv2.vec)
    assert not jnp.iscomplexobj(hv3.vec)

    # Different values should produce different hypervectors
    assert not jnp.allclose(hv2.vec, hv3.vec)


def test_scalar_encoder_binary():
    """Test encoding for binary hypervectors."""
    model = create_binary_model(dim=1000, bipolar=True)
    memory = VSAMemory(model)
    memory.add("value")

    encoder = ScalarEncoder(model, memory)

    # Encode values
    hv1 = encoder.encode("value", 0.5)
    hv2 = encoder.encode("value", 1.0)

    # Check that results are binary (bipolar: -1 or +1)
    assert jnp.all(jnp.isin(hv1.vec, jnp.array([-1, 1])))
    assert jnp.all(jnp.isin(hv2.vec, jnp.array([-1, 1])))


def test_scalar_encoder_with_normalization():
    """Test encoding with value normalization."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("temperature")

    encoder = ScalarEncoder(model, memory, min_val=0, max_val=100)

    # Encode values within range
    hv_low = encoder.encode("temperature", 0.0)
    hv_mid = encoder.encode("temperature", 50.0)
    hv_high = encoder.encode("temperature", 100.0)

    # All should be valid hypervectors
    assert hv_low.vec.shape == (128,)
    assert hv_mid.vec.shape == (128,)
    assert hv_high.vec.shape == (128,)

    # Different values should produce different results
    assert not jnp.allclose(hv_low.vec, hv_mid.vec)
    assert not jnp.allclose(hv_mid.vec, hv_high.vec)


def test_scalar_encoder_out_of_range_raises_error():
    """Test that encoding out-of-range values raises ValueError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("temperature")

    encoder = ScalarEncoder(model, memory, min_val=0, max_val=100)

    # Values outside range should raise error
    with pytest.raises(ValueError, match="outside range"):
        encoder.encode("temperature", -10.0)

    with pytest.raises(ValueError, match="outside range"):
        encoder.encode("temperature", 150.0)


def test_scalar_encoder_missing_symbol_raises_error():
    """Test that encoding with missing symbol raises KeyError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    encoder = ScalarEncoder(model, memory)

    # Symbol not in memory
    with pytest.raises(KeyError):
        encoder.encode("nonexistent", 1.0)


def test_scalar_encoder_deterministic():
    """Test that encoding the same value twice gives the same result."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("value")

    encoder = ScalarEncoder(model, memory)

    hv1 = encoder.encode("value", 42.5)
    hv2 = encoder.encode("value", 42.5)

    # Should be identical (basis vector doesn't change)
    assert jnp.allclose(hv1.vec, hv2.vec)
