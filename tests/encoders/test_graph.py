"""Tests for GraphEncoder."""

import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_fhrr_model
from vsax.encoders import GraphEncoder


def test_graph_encoder_basic():
    """Test basic graph encoding."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["Alice", "Bob", "knows", "likes"])

    encoder = GraphEncoder(model, memory)
    graph_hv = encoder.encode([
        ("Alice", "knows", "Bob"),
        ("Alice", "likes", "Bob")
    ])

    assert graph_hv.vec.shape == (128,)


def test_graph_encoder_deterministic():
    """Test that encoding same graph twice gives same result."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["A", "B", "rel"])

    encoder = GraphEncoder(model, memory)

    graph1 = encoder.encode([("A", "rel", "B")])
    graph2 = encoder.encode([("A", "rel", "B")])

    assert jnp.allclose(graph1.vec, graph2.vec)


def test_graph_encoder_edge_order_invariant():
    """Test that edge order doesn't matter (bundling is commutative)."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["A", "B", "C", "r1", "r2"])

    encoder = GraphEncoder(model, memory)

    graph1 = encoder.encode([
        ("A", "r1", "B"),
        ("B", "r2", "C")
    ])
    graph2 = encoder.encode([
        ("B", "r2", "C"),
        ("A", "r1", "B")
    ])

    # Should be the same (bundling is commutative)
    assert jnp.allclose(graph1.vec, graph2.vec)


def test_graph_encoder_different_graphs():
    """Test that different graphs produce different hypervectors."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add_many(["A", "B", "C", "rel1", "rel2"])

    encoder = GraphEncoder(model, memory)

    graph1 = encoder.encode([("A", "rel1", "B")])
    graph2 = encoder.encode([("A", "rel1", "C")])
    graph3 = encoder.encode([("A", "rel2", "B")])

    # Different graphs should produce different results
    assert not jnp.allclose(graph1.vec, graph2.vec)
    assert not jnp.allclose(graph1.vec, graph3.vec)
    assert not jnp.allclose(graph2.vec, graph3.vec)


def test_graph_encoder_empty_raises_error():
    """Test that empty graph raises ValueError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    encoder = GraphEncoder(model, memory)

    with pytest.raises(ValueError, match="Cannot encode empty graph"):
        encoder.encode([])


def test_graph_encoder_missing_node_raises_error():
    """Test that missing node raises KeyError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["A", "rel"])

    encoder = GraphEncoder(model, memory)

    with pytest.raises(KeyError):
        encoder.encode([("A", "rel", "nonexistent")])


def test_graph_encoder_missing_relation_raises_error():
    """Test that missing relation raises KeyError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["A", "B"])

    encoder = GraphEncoder(model, memory)

    with pytest.raises(KeyError):
        encoder.encode([("A", "nonexistent_rel", "B")])


def test_graph_encoder_multi_edge_graph():
    """Test encoding a graph with multiple edges."""
    model = create_fhrr_model(dim=256)
    memory = VSAMemory(model)
    memory.add_many(["Alice", "Bob", "Charlie", "knows", "likes", "follows"])

    encoder = GraphEncoder(model, memory)

    graph_hv = encoder.encode([
        ("Alice", "knows", "Bob"),
        ("Alice", "likes", "Charlie"),
        ("Bob", "follows", "Charlie"),
        ("Charlie", "knows", "Alice")
    ])

    assert graph_hv.vec.shape == (256,)
