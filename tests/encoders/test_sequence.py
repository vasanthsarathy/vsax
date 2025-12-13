"""Tests for SequenceEncoder."""

import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_binary_model, create_fhrr_model, create_map_model
from vsax.encoders import SequenceEncoder


def test_sequence_encoder_initialization():
    """Test SequenceEncoder initialization."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    encoder = SequenceEncoder(model, memory)

    assert encoder.model is model
    assert encoder.memory is memory


def test_sequence_encoder_basic():
    """Test basic sequence encoding."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["red", "green", "blue"])

    encoder = SequenceEncoder(model, memory)

    # Encode a sequence
    seq_hv = encoder.encode(["red", "green", "blue"])

    assert seq_hv.vec.shape == (128,)


def test_sequence_encoder_order_matters():
    """Test that different orders produce different hypervectors."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    encoder = SequenceEncoder(model, memory)

    # Same elements, different order
    seq1 = encoder.encode(["a", "b", "c"])
    seq2 = encoder.encode(["c", "b", "a"])
    seq3 = encoder.encode(["b", "a", "c"])

    # Different orders should produce different hypervectors
    assert not jnp.allclose(seq1.vec, seq2.vec)
    assert not jnp.allclose(seq1.vec, seq3.vec)
    assert not jnp.allclose(seq2.vec, seq3.vec)


def test_sequence_encoder_deterministic():
    """Test that encoding the same sequence twice gives the same result."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["x", "y", "z"])

    encoder = SequenceEncoder(model, memory)

    seq1 = encoder.encode(["x", "y", "z"])
    seq2 = encoder.encode(["x", "y", "z"])

    assert jnp.allclose(seq1.vec, seq2.vec)


def test_sequence_encoder_positions_auto_added():
    """Test that position hypervectors are automatically added to memory."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    assert "pos_0" not in memory
    assert "pos_1" not in memory
    assert "pos_2" not in memory

    encoder = SequenceEncoder(model, memory)
    encoder.encode(["a", "b", "c"])

    # Position hypervectors should now exist
    assert "pos_0" in memory
    assert "pos_1" in memory
    assert "pos_2" in memory


def test_sequence_encoder_different_lengths():
    """Test encoding sequences of different lengths."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c", "d"])

    encoder = SequenceEncoder(model, memory)

    seq_short = encoder.encode(["a", "b"])
    seq_long = encoder.encode(["a", "b", "c", "d"])

    # Both should be valid
    assert seq_short.vec.shape == (128,)
    assert seq_long.vec.shape == (128,)

    # Different lengths should produce different results
    assert not jnp.allclose(seq_short.vec, seq_long.vec)


def test_sequence_encoder_empty_raises_error():
    """Test that encoding an empty sequence raises ValueError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    encoder = SequenceEncoder(model, memory)

    with pytest.raises(ValueError, match="Cannot encode empty sequence"):
        encoder.encode([])


def test_sequence_encoder_missing_symbol_raises_error():
    """Test that encoding with missing symbol raises KeyError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("a")

    encoder = SequenceEncoder(model, memory)

    with pytest.raises(KeyError):
        encoder.encode(["a", "nonexistent"])


def test_sequence_encoder_with_map_model():
    """Test SequenceEncoder with MAP model."""
    model = create_map_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["x", "y", "z"])

    encoder = SequenceEncoder(model, memory)

    seq_hv = encoder.encode(["x", "y", "z"])

    # Should work with real hypervectors
    assert not jnp.iscomplexobj(seq_hv.vec)
    assert seq_hv.vec.shape == (128,)


def test_sequence_encoder_with_binary_model():
    """Test SequenceEncoder with Binary model."""
    model = create_binary_model(dim=1000, bipolar=True)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    encoder = SequenceEncoder(model, memory)

    seq_hv = encoder.encode(["a", "b", "c"])

    # Should work with binary hypervectors
    assert jnp.all(jnp.isin(seq_hv.vec, jnp.array([-1, 1])))
    assert seq_hv.vec.shape == (1000,)


def test_sequence_encoder_with_tuple():
    """Test that tuples work as well as lists."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    encoder = SequenceEncoder(model, memory)

    seq_list = encoder.encode(["a", "b", "c"])
    seq_tuple = encoder.encode(("a", "b", "c"))

    # Should produce the same result
    assert jnp.allclose(seq_list.vec, seq_tuple.vec)
