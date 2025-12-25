"""Tests for AbstractEncoder base class."""

import pytest

from vsax import AbstractEncoder, VSAMemory, create_fhrr_model
from vsax.core.base import AbstractHypervector


class ConcreteEncoder(AbstractEncoder):
    """Concrete implementation for testing."""

    def encode(self, data):
        """Simple test implementation."""
        # Just return the first symbol from memory
        return list(self.memory._symbols.values())[0]


def test_abstract_encoder_cannot_instantiate():
    """Test that AbstractEncoder cannot be instantiated directly."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        AbstractEncoder(model, memory)


def test_concrete_encoder_initialization():
    """Test that concrete encoders can be initialized."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    encoder = ConcreteEncoder(model, memory)

    assert encoder.model is model
    assert encoder.memory is memory


def test_concrete_encoder_encode():
    """Test that concrete encoders can encode data."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add("test_symbol")

    encoder = ConcreteEncoder(model, memory)
    result = encoder.encode("dummy_data")

    assert isinstance(result, AbstractHypervector)


def test_default_fit_does_nothing():
    """Test that the default fit() method does nothing."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    encoder = ConcreteEncoder(model, memory)

    # Should not raise any errors
    encoder.fit([1, 2, 3])
    encoder.fit({"key": "value"})
    encoder.fit(None)


def test_encoder_must_implement_encode():
    """Test that subclasses must implement encode()."""

    class IncompleteEncoder(AbstractEncoder):
        """Encoder that doesn't implement encode()."""

        pass

    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteEncoder(model, memory)
