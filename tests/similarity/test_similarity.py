"""Tests for similarity metrics."""

import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_binary_model, create_fhrr_model, create_map_model
from vsax.similarity import cosine_similarity, dot_similarity, hamming_similarity


def test_cosine_similarity_identical_fhrr() -> None:
    """Test that identical FHRR vectors have similarity 1.0."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add("test")

    vec = memory["test"]
    similarity = cosine_similarity(vec, vec)

    assert jnp.isclose(similarity, 1.0, atol=1e-5)


def test_cosine_similarity_identical_map() -> None:
    """Test that identical MAP vectors have similarity 1.0."""
    model = create_map_model(dim=512)
    memory = VSAMemory(model)
    memory.add("test")

    vec = memory["test"]
    similarity = cosine_similarity(vec, vec)

    assert jnp.isclose(similarity, 1.0, atol=1e-5)


def test_cosine_similarity_identical_binary() -> None:
    """Test that identical binary vectors have similarity 1.0."""
    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add("test")

    vec = memory["test"]
    similarity = cosine_similarity(vec, vec)

    assert jnp.isclose(similarity, 1.0, atol=1e-5)


def test_cosine_similarity_different_vectors() -> None:
    """Test that different vectors have similarity < 1.0."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["a", "b"])

    similarity = cosine_similarity(memory["a"], memory["b"])

    # Random vectors should have low similarity
    assert similarity < 0.9


def test_cosine_similarity_with_arrays() -> None:
    """Test cosine similarity works with raw arrays."""
    model = create_map_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["x", "y"])

    # Test with raw arrays
    similarity = cosine_similarity(memory["x"].vec, memory["y"].vec)

    assert isinstance(similarity, float)
    assert -1.0 <= similarity <= 1.0


def test_cosine_similarity_shape_mismatch_raises_error() -> None:
    """Test that shape mismatch raises ValueError."""
    vec_a = jnp.ones(512)
    vec_b = jnp.ones(256)

    with pytest.raises(ValueError, match="Shape mismatch"):
        cosine_similarity(vec_a, vec_b)


def test_dot_similarity_identical_vectors() -> None:
    """Test dot product of identical vectors."""
    model = create_map_model(dim=512)
    memory = VSAMemory(model)
    memory.add("test")

    vec = memory["test"]
    dot_sim = dot_similarity(vec, vec)

    # For normalized vectors, dot product should be close to 1
    assert isinstance(dot_sim, float)
    assert dot_sim > 0


def test_dot_similarity_complex_vectors() -> None:
    """Test dot similarity with complex (FHRR) vectors."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["a", "b"])

    dot_sim = dot_similarity(memory["a"], memory["b"])

    assert isinstance(dot_sim, float)
    # Should be real number (not complex)
    assert jnp.isreal(dot_sim)


def test_dot_similarity_with_arrays() -> None:
    """Test dot similarity works with raw arrays."""
    vec_a = jnp.array([1.0, 2.0, 3.0])
    vec_b = jnp.array([4.0, 5.0, 6.0])

    dot_sim = dot_similarity(vec_a, vec_b)

    # Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert jnp.isclose(dot_sim, 32.0)


def test_dot_similarity_shape_mismatch_raises_error() -> None:
    """Test that shape mismatch raises ValueError."""
    vec_a = jnp.ones(512)
    vec_b = jnp.ones(256)

    with pytest.raises(ValueError, match="Shape mismatch"):
        dot_similarity(vec_a, vec_b)


def test_hamming_similarity_identical_vectors() -> None:
    """Test that identical binary vectors have Hamming similarity 1.0."""
    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add("test")

    vec = memory["test"]
    similarity = hamming_similarity(vec, vec)

    assert jnp.isclose(similarity, 1.0)


def test_hamming_similarity_different_vectors() -> None:
    """Test Hamming similarity between different binary vectors."""
    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add_many(["a", "b"])

    similarity = hamming_similarity(memory["a"], memory["b"])

    # Random binary vectors should have ~50% match
    assert 0.4 < similarity < 0.6


def test_hamming_similarity_with_arrays() -> None:
    """Test Hamming similarity with raw binary arrays."""
    vec_a = jnp.array([1, -1, 1, -1, 1])
    vec_b = jnp.array([1, -1, 1, 1, 1])

    similarity = hamming_similarity(vec_a, vec_b)

    # 4 out of 5 match = 0.8
    assert jnp.isclose(similarity, 0.8)


def test_hamming_similarity_completely_different() -> None:
    """Test Hamming similarity for completely different vectors."""
    vec_a = jnp.array([1, 1, 1, 1])
    vec_b = jnp.array([-1, -1, -1, -1])

    similarity = hamming_similarity(vec_a, vec_b)

    # No matches = 0.0
    assert jnp.isclose(similarity, 0.0)


def test_hamming_similarity_shape_mismatch_raises_error() -> None:
    """Test that shape mismatch raises ValueError."""
    vec_a = jnp.ones(512)
    vec_b = jnp.ones(256)

    with pytest.raises(ValueError, match="Shape mismatch"):
        hamming_similarity(vec_a, vec_b)


def test_similarity_metrics_consistency() -> None:
    """Test that all metrics work consistently across models."""
    models = [
        create_fhrr_model(dim=512),
        create_map_model(dim=512),
        create_binary_model(dim=10000, bipolar=True),
    ]

    for model in models:
        memory = VSAMemory(model)
        memory.add_many(["a", "b"])

        # All metrics should work
        cos_sim = cosine_similarity(memory["a"], memory["b"])
        dot_sim = dot_similarity(memory["a"], memory["b"])

        assert isinstance(cos_sim, float)
        assert isinstance(dot_sim, float)
        assert -1.0 <= cos_sim <= 1.0


def test_cosine_similarity_range() -> None:
    """Test that cosine similarity is always in valid range."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["x", "y", "z"])

    for name1 in ["x", "y", "z"]:
        for name2 in ["x", "y", "z"]:
            sim = cosine_similarity(memory[name1], memory[name2])
            # Allow small numerical errors
            assert -1.01 <= sim <= 1.01


def test_hamming_similarity_range() -> None:
    """Test that Hamming similarity is always in [0, 1]."""
    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    for name1 in ["a", "b", "c"]:
        for name2 in ["a", "b", "c"]:
            sim = hamming_similarity(memory[name1], memory[name2])
            assert 0.0 <= sim <= 1.0
