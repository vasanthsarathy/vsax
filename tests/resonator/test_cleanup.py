"""Tests for CleanupMemory."""

import pytest

from vsax import VSAMemory, create_binary_model, create_fhrr_model, create_map_model
from vsax.resonator import CleanupMemory


class TestCleanupMemoryInitialization:
    """Test CleanupMemory initialization."""

    def test_init_binary(self) -> None:
        """Test initialization with binary model."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue", "green"])

        cleanup = CleanupMemory(["red", "blue", "green"], memory)

        assert len(cleanup) == 3
        assert cleanup.codebook == ["red", "blue", "green"]
        assert cleanup.threshold == 0.0

    def test_init_fhrr(self) -> None:
        """Test initialization with FHRR model."""
        model = create_fhrr_model(dim=512)
        memory = VSAMemory(model)
        memory.add_many(["x", "y", "z"])

        cleanup = CleanupMemory(["x", "y", "z"], memory)

        assert len(cleanup) == 3

    def test_init_map(self) -> None:
        """Test initialization with MAP model."""
        model = create_map_model(dim=512)
        memory = VSAMemory(model)
        memory.add_many(["a", "b", "c"])

        cleanup = CleanupMemory(["a", "b", "c"], memory)

        assert len(cleanup) == 3

    def test_init_with_threshold(self) -> None:
        """Test initialization with custom threshold."""
        model = create_binary_model(dim=10000)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue"])

        cleanup = CleanupMemory(["red", "blue"], memory, threshold=0.5)

        assert cleanup.threshold == 0.5

    def test_init_missing_symbol(self) -> None:
        """Test that initialization fails with missing symbol."""
        model = create_binary_model(dim=10000)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue"])

        with pytest.raises(ValueError, match="Symbol 'green' not found"):
            CleanupMemory(["red", "blue", "green"], memory)


class TestCleanupMemoryQuery:
    """Test CleanupMemory query functionality."""

    def test_query_exact_match(self) -> None:
        """Test querying with exact codebook vector."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue", "green"])

        cleanup = CleanupMemory(["red", "blue", "green"], memory)

        # Query with exact vector should return same symbol
        result = cleanup.query(memory["red"].vec)
        assert result == "red"

        result = cleanup.query(memory["blue"].vec)
        assert result == "blue"

    def test_query_with_hypervector(self) -> None:
        """Test querying with hypervector object."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue", "green"])

        cleanup = CleanupMemory(["red", "blue", "green"], memory)

        # Query with hypervector object
        result = cleanup.query(memory["red"])
        assert result == "red"

    def test_query_noisy_vector(self) -> None:
        """Test querying with noisy vector."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue", "green"])

        cleanup = CleanupMemory(["red", "blue", "green"], memory)

        # Create noisy version of red (bundle with small amount of blue)
        noisy = model.opset.bundle(
            memory["red"].vec,
            memory["blue"].vec * 0.1,
        )

        result = cleanup.query(noisy)
        # Should still recover "red" as it's the dominant component
        assert result == "red"

    def test_query_with_return_similarity(self) -> None:
        """Test querying with similarity score."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue", "green"])

        cleanup = CleanupMemory(["red", "blue", "green"], memory)

        result, similarity = cleanup.query(memory["red"].vec, return_similarity=True)
        assert result == "red"
        assert isinstance(similarity, float)
        # Exact match should have very high similarity
        assert similarity > 9000  # Dot product for 10000-dim bipolar vectors

    def test_query_below_threshold(self) -> None:
        """Test querying with threshold filtering."""
        import jax
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue", "green"])

        # Set very high threshold
        cleanup = CleanupMemory(["red", "blue", "green"], memory, threshold=20000.0)

        # Create random vector unlikely to match well
        key = jax.random.PRNGKey(42)
        random_vec = model.sampler(10000, 1, key=key)[0]

        result = cleanup.query(random_vec)
        # Should return None because similarity is below threshold
        assert result is None

    def test_query_below_threshold_with_similarity(self) -> None:
        """Test querying below threshold returns None and similarity."""
        import jax
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue"])

        cleanup = CleanupMemory(["red", "blue"], memory, threshold=20000.0)

        key = jax.random.PRNGKey(42)
        random_vec = model.sampler(10000, 1, key=key)[0]
        result, similarity = cleanup.query(random_vec, return_similarity=True)

        assert result is None
        assert isinstance(similarity, float)


class TestCleanupMemoryTopK:
    """Test CleanupMemory top-k query functionality."""

    def test_query_top_k_basic(self) -> None:
        """Test basic top-k query."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue", "green"])

        cleanup = CleanupMemory(["red", "blue", "green"], memory)

        # Query for top 2
        results = cleanup.query_top_k(memory["red"].vec, k=2)

        assert len(results) == 2
        # First result should be exact match
        assert results[0][0] == "red"
        # All results should be tuples of (symbol, similarity)
        for symbol, sim in results:
            assert isinstance(symbol, str)
            assert isinstance(sim, float)

    def test_query_top_k_ordering(self) -> None:
        """Test that top-k results are ordered by similarity."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["red", "blue", "green", "yellow"])

        cleanup = CleanupMemory(["red", "blue", "green", "yellow"], memory)

        results = cleanup.query_top_k(memory["red"].vec, k=4)

        # Similarities should be in descending order
        similarities = [sim for _, sim in results]
        assert similarities == sorted(similarities, reverse=True)

        # First should be exact match
        assert results[0][0] == "red"

    def test_query_top_k_all(self) -> None:
        """Test querying all codebook vectors."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)
        memory.add_many(["a", "b", "c"])

        cleanup = CleanupMemory(["a", "b", "c"], memory)

        results = cleanup.query_top_k(memory["a"].vec, k=3)

        assert len(results) == 3
        # All symbols should be present
        symbols = [s for s, _ in results]
        assert "a" in symbols
        assert "b" in symbols
        assert "c" in symbols


class TestCleanupMemoryRepr:
    """Test CleanupMemory string representation."""

    def test_repr(self) -> None:
        """Test __repr__ method."""
        model = create_binary_model(dim=10000)
        memory = VSAMemory(model)
        memory.add_many(["x", "y", "z"])

        cleanup = CleanupMemory(["x", "y", "z"], memory, threshold=0.5)

        repr_str = repr(cleanup)
        assert "CleanupMemory" in repr_str
        assert "codebook_size=3" in repr_str
        assert "threshold=0.5" in repr_str


class TestCleanupMemoryEdgeCases:
    """Test edge cases for CleanupMemory."""

    def test_single_vector_codebook(self) -> None:
        """Test codebook with single vector."""
        model = create_binary_model(dim=10000)
        memory = VSAMemory(model)
        memory.add_many(["only"])

        cleanup = CleanupMemory(["only"], memory)

        assert len(cleanup) == 1
        result = cleanup.query(memory["only"].vec)
        assert result == "only"

    def test_large_codebook(self) -> None:
        """Test with large codebook."""
        model = create_binary_model(dim=10000, bipolar=True)
        memory = VSAMemory(model)

        # Create 100 symbols
        symbols = [f"symbol_{i}" for i in range(100)]
        memory.add_many(symbols)

        cleanup = CleanupMemory(symbols, memory)

        assert len(cleanup) == 100

        # Query should still work
        result = cleanup.query(memory["symbol_42"].vec)
        assert result == "symbol_42"

    def test_fhrr_complex_vectors(self) -> None:
        """Test cleanup with complex-valued FHRR vectors."""
        model = create_fhrr_model(dim=512)
        memory = VSAMemory(model)
        memory.add_many(["alpha", "beta", "gamma"])

        cleanup = CleanupMemory(["alpha", "beta", "gamma"], memory)

        # Should work with complex vectors
        result = cleanup.query(memory["alpha"].vec)
        assert result == "alpha"

    def test_map_real_vectors(self) -> None:
        """Test cleanup with real-valued MAP vectors."""
        model = create_map_model(dim=512)
        memory = VSAMemory(model)
        memory.add_many(["one", "two", "three"])

        cleanup = CleanupMemory(["one", "two", "three"], memory)

        # Should work with real vectors
        result = cleanup.query(memory["two"].vec)
        assert result == "two"
