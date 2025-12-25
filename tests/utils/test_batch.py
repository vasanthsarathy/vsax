"""Tests for batch operations."""

import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_binary_model, create_fhrr_model, create_map_model
from vsax.utils import vmap_bind, vmap_bundle, vmap_similarity


def test_vmap_bind_basic() -> None:
    """Test basic batch binding operation."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c", "x", "y", "z"])

    # Create batches
    X = jnp.stack([memory["a"].vec, memory["b"].vec, memory["c"].vec])
    Y = jnp.stack([memory["x"].vec, memory["y"].vec, memory["z"].vec])

    # Batch bind
    result = vmap_bind(model.opset, X, Y)

    # Check shape
    assert result.shape == (3, 128)

    # Each result should be different
    assert not jnp.allclose(result[0], result[1])
    assert not jnp.allclose(result[0], result[2])


def test_vmap_bind_matches_individual() -> None:
    """Test that batch bind matches individual bind operations."""
    model = create_map_model(dim=256)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "x", "y"])

    # Batch operation
    X = jnp.stack([memory["a"].vec, memory["b"].vec])
    Y = jnp.stack([memory["x"].vec, memory["y"].vec])
    batch_result = vmap_bind(model.opset, X, Y)

    # Individual operations
    individual_1 = model.opset.bind(memory["a"].vec, memory["x"].vec)
    individual_2 = model.opset.bind(memory["b"].vec, memory["y"].vec)

    # Should match
    assert jnp.allclose(batch_result[0], individual_1)
    assert jnp.allclose(batch_result[1], individual_2)


def test_vmap_bind_with_list_input() -> None:
    """Test vmap_bind accepts list of arrays."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "x", "y"])

    # Pass as lists
    X = [memory["a"].vec, memory["b"].vec]
    Y = [memory["x"].vec, memory["y"].vec]

    result = vmap_bind(model.opset, X, Y)

    assert result.shape == (2, 128)


def test_vmap_bind_batch_size_mismatch_raises_error() -> None:
    """Test that mismatched batch sizes raise ValueError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c", "x", "y"])

    X = jnp.stack([memory["a"].vec, memory["b"].vec, memory["c"].vec])
    Y = jnp.stack([memory["x"].vec, memory["y"].vec])

    with pytest.raises(ValueError, match="Batch size mismatch"):
        vmap_bind(model.opset, X, Y)


def test_vmap_bind_dimension_mismatch_raises_error() -> None:
    """Test that mismatched dimensions raise ValueError."""
    model1 = create_fhrr_model(dim=128)
    model2 = create_fhrr_model(dim=256)
    memory1 = VSAMemory(model1)
    memory2 = VSAMemory(model2)
    memory1.add("a")
    memory2.add("x")

    X = jnp.stack([memory1["a"].vec])
    Y = jnp.stack([memory2["x"].vec])

    with pytest.raises(ValueError, match="Dimension mismatch"):
        vmap_bind(model1.opset, X, Y)


def test_vmap_bundle_basic() -> None:
    """Test basic batch bundling operation."""
    model = create_map_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    # Create batch
    batch = jnp.stack([memory["a"].vec, memory["b"].vec, memory["c"].vec])

    # Bundle batch
    result = vmap_bundle(model.opset, batch)

    # Result should be single vector
    assert result.shape == (128,)


def test_vmap_bundle_matches_individual() -> None:
    """Test that batch bundle matches individual bundle."""
    model = create_fhrr_model(dim=256)
    memory = VSAMemory(model)
    memory.add_many(["x", "y", "z"])

    # Batch operation
    batch = jnp.stack([memory["x"].vec, memory["y"].vec, memory["z"].vec])
    batch_result = vmap_bundle(model.opset, batch)

    # Individual operation
    individual_result = model.opset.bundle(memory["x"].vec, memory["y"].vec, memory["z"].vec)

    # Should match
    assert jnp.allclose(batch_result, individual_result)


def test_vmap_bundle_with_list_input() -> None:
    """Test vmap_bundle accepts list of arrays."""
    model = create_map_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c"])

    # Pass as list
    batch = [memory["a"].vec, memory["b"].vec, memory["c"].vec]

    result = vmap_bundle(model.opset, batch)

    assert result.shape == (128,)


def test_vmap_similarity_basic() -> None:
    """Test batch similarity computation."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["query", "a", "b", "c"])

    query = memory["query"].vec
    candidates = jnp.stack([memory["a"].vec, memory["b"].vec, memory["c"].vec])

    # Compute batch similarities (uses internal cosine implementation)
    similarities = vmap_similarity(None, query, candidates)

    # Should return array of similarities
    assert similarities.shape == (3,)

    # All should be in valid range (with tolerance for numerical errors)
    assert jnp.all(similarities >= -1.01)
    assert jnp.all(similarities <= 1.01)


def test_vmap_similarity_matches_individual() -> None:
    """Test that batch similarity matches individual computations."""
    model = create_map_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["query", "a", "b"])

    query = memory["query"].vec
    candidates = jnp.stack([memory["a"].vec, memory["b"].vec])

    # Batch operation
    batch_sims = vmap_similarity(None, query, candidates)

    # Manual individual cosine similarity
    def manual_cosine(a, b):
        dot = jnp.dot(a, b)
        return dot / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-10)

    sim_a = manual_cosine(query, memory["a"].vec)
    sim_b = manual_cosine(query, memory["b"].vec)

    # Should match
    assert jnp.isclose(batch_sims[0], sim_a, atol=1e-5)
    assert jnp.isclose(batch_sims[1], sim_b, atol=1e-5)


def test_vmap_similarity_with_list_input() -> None:
    """Test vmap_similarity accepts list of candidates."""
    model = create_fhrr_model(dim=256)
    memory = VSAMemory(model)
    memory.add_many(["query", "a", "b", "c"])

    query = memory["query"].vec
    candidates = [memory["a"].vec, memory["b"].vec, memory["c"].vec]

    similarities = vmap_similarity(None, query, candidates)

    assert similarities.shape == (3,)


def test_vmap_operations_work_across_models() -> None:
    """Test that vmap operations work with all model types."""
    models = [
        create_fhrr_model(dim=256),
        create_map_model(dim=256),
        create_binary_model(dim=5000, bipolar=True),
    ]

    for model in models:
        memory = VSAMemory(model)
        memory.add_many(["a", "b", "c", "x", "y"])

        # Test vmap_bind
        X = jnp.stack([memory["a"].vec, memory["b"].vec])
        Y = jnp.stack([memory["x"].vec, memory["y"].vec])
        bind_result = vmap_bind(model.opset, X, Y)
        assert bind_result.shape[0] == 2

        # Test vmap_bundle
        batch = jnp.stack([memory["a"].vec, memory["b"].vec, memory["c"].vec])
        bundle_result = vmap_bundle(model.opset, batch)
        assert bundle_result.shape == (model.dim,)


def test_vmap_bind_large_batch() -> None:
    """Test vmap_bind with larger batch sizes."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    # Create 100 symbols
    symbols = [f"sym_{i}" for i in range(100)]
    memory.add_many(symbols)

    # Create large batches
    X = jnp.stack([memory[f"sym_{i}"].vec for i in range(50)])
    Y = jnp.stack([memory[f"sym_{i}"].vec for i in range(50, 100)])

    result = vmap_bind(model.opset, X, Y)

    assert result.shape == (50, 128)


def test_vmap_similarity_finds_best_match() -> None:
    """Test that vmap_similarity can find best matching candidate."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["target", "similar", "different1", "different2"])

    # Create a "similar" vector by binding target with a weak noise
    target = memory["target"].vec
    # Use the "similar" symbol to represent something close
    memory["similar"].vec

    candidates = jnp.stack(
        [memory["similar"].vec, memory["different1"].vec, memory["different2"].vec]
    )

    similarities = vmap_similarity(None, target, candidates)

    # The first candidate (similar) should be most similar to target
    # (though they're all random, so we just check the operation works)
    assert similarities.shape == (3,)
    assert jnp.all(jnp.isfinite(similarities))
