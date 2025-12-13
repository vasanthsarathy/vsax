"""Tests for pretty representation utilities."""

import jax.numpy as jnp

from vsax import VSAMemory, create_binary_model, create_fhrr_model, create_map_model
from vsax.utils import format_similarity_results, pretty_repr


def test_pretty_repr_complex_hypervector() -> None:
    """Test pretty_repr with complex hypervector."""
    model = create_fhrr_model(dim=512)
    memory = VSAMemory(model)
    memory.add("test")

    result = pretty_repr(memory["test"])

    # Should contain type name
    assert "ComplexHypervector" in result
    # Should contain dimension
    assert "dim=512" in result
    # Should contain dtype
    assert "complex" in result.lower()
    # Should contain sample values
    assert "Sample:" in result


def test_pretty_repr_real_hypervector() -> None:
    """Test pretty_repr with real hypervector."""
    model = create_map_model(dim=256)
    memory = VSAMemory(model)
    memory.add("test")

    result = pretty_repr(memory["test"])

    assert "RealHypervector" in result
    assert "dim=256" in result
    assert "float" in result.lower()
    assert "Mean:" in result
    assert "Std:" in result


def test_pretty_repr_binary_hypervector() -> None:
    """Test pretty_repr with binary hypervector."""
    model = create_binary_model(dim=10000, bipolar=True)
    memory = VSAMemory(model)
    memory.add("test")

    result = pretty_repr(memory["test"])

    assert "BinaryHypervector" in result
    assert "dim=10000" in result
    assert "Sample:" in result


def test_pretty_repr_with_array() -> None:
    """Test pretty_repr with raw array."""
    vec = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = pretty_repr(vec)

    assert "dim=5" in result
    assert "Sample:" in result
    assert "Mean:" in result


def test_pretty_repr_small_vector() -> None:
    """Test pretty_repr with small vector (shows all elements)."""
    vec = jnp.array([1.0, 2.0, 3.0])

    result = pretty_repr(vec, max_elements=5)

    # Should show all 3 elements
    assert "1.000" in result
    assert "2.000" in result
    assert "3.000" in result
    # Should not have ellipsis
    assert "..." not in result


def test_pretty_repr_large_vector() -> None:
    """Test pretty_repr with large vector (truncates with ellipsis)."""
    vec = jnp.arange(100, dtype=jnp.float32)

    result = pretty_repr(vec, max_elements=6)

    # Should have ellipsis
    assert "..." in result
    # Should show first few elements
    assert "0.000" in result
    # Should show last few elements
    assert "99.000" in result


def test_pretty_repr_complex_array() -> None:
    """Test pretty_repr with complex array."""
    vec = jnp.array([1 + 2j, 3 - 4j, 5 + 0j])

    result = pretty_repr(vec)

    # Should format complex numbers properly
    assert "j" in result
    # Should show magnitude
    assert "Mean magnitude:" in result


def test_format_similarity_results_basic() -> None:
    """Test formatting similarity search results."""
    similarities = jnp.array([0.85, 0.92, 0.23, 0.95])
    candidates = ["cat", "wolf", "bird", "puppy"]

    result = format_similarity_results("dog", candidates, similarities, top_k=3)

    # Should show query
    assert "Query: dog" in result
    # Should show top 3
    assert "Top 3 matches:" in result
    # Puppy should be #1 (0.95)
    assert "puppy" in result
    # Wolf should be #2 (0.92)
    assert "wolf" in result
    # Cat should be #3 (0.85)
    assert "cat" in result
    # Bird should NOT be in top 3
    # (it would be 4th with 0.23)


def test_format_similarity_results_top_k() -> None:
    """Test that top_k parameter works correctly."""
    similarities = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    candidates = ["a", "b", "c", "d", "e"]

    result = format_similarity_results("query", candidates, similarities, top_k=2)

    assert "Top 2 matches:" in result
    # Should show top 2: e (0.5) and d (0.4)
    assert "e" in result
    assert "d" in result


def test_format_similarity_results_shows_scores() -> None:
    """Test that similarity scores are displayed."""
    similarities = jnp.array([0.123, 0.456, 0.789])
    candidates = ["a", "b", "c"]

    result = format_similarity_results("query", candidates, similarities, top_k=3)

    # Should show scores with 3 decimal places
    assert "0.789" in result  # Highest
    assert "0.456" in result  # Middle
    assert "0.123" in result  # Lowest


def test_format_similarity_results_ranking() -> None:
    """Test that results are ranked correctly."""
    similarities = jnp.array([0.5, 0.9, 0.3, 0.7])
    candidates = ["low", "high", "lowest", "medium"]

    result = format_similarity_results("query", candidates, similarities, top_k=4)

    lines = result.split("\n")

    # Find ranking lines (skip header lines)
    rank_lines = [line for line in lines if line.strip().startswith(("1.", "2.", "3.", "4."))]

    # Check order: high (0.9), medium (0.7), low (0.5), lowest (0.3)
    assert "high" in rank_lines[0]
    assert "medium" in rank_lines[1]
    assert "low" in rank_lines[2]
    assert "lowest" in rank_lines[3]
