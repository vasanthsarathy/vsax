"""Tests for save_basis and load_basis functions."""

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp
import pytest

from vsax import (
    VSAMemory,
    create_binary_model,
    create_fhrr_model,
    create_map_model,
    load_basis,
    save_basis,
)


def test_save_fhrr_creates_file() -> None:
    """Test that save_basis creates a JSON file for FHRR model."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)
    memory.add_many(["apple", "orange", "banana"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_basis.json"
        save_basis(memory, path)

        assert path.exists()
        assert path.suffix == ".json"


def test_save_fhrr_correct_format() -> None:
    """Test that saved JSON has correct structure for FHRR."""
    model = create_fhrr_model(dim=64)
    memory = VSAMemory(model)
    memory.add_many(["x", "y"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory, path)

        with open(path) as f:
            data = json.load(f)

        # Check metadata
        assert "metadata" in data
        assert data["metadata"]["dim"] == 64
        assert data["metadata"]["rep_type"] == "complex"
        assert data["metadata"]["num_vectors"] == 2

        # Check vectors
        assert "vectors" in data
        assert "x" in data["vectors"]
        assert "y" in data["vectors"]

        # Complex vectors should have real and imag parts
        assert "real" in data["vectors"]["x"]
        assert "imag" in data["vectors"]["x"]
        assert len(data["vectors"]["x"]["real"]) == 64
        assert len(data["vectors"]["x"]["imag"]) == 64


def test_save_map_correct_format() -> None:
    """Test that saved JSON has correct structure for MAP."""
    model = create_map_model(dim=128)
    memory = VSAMemory(model)
    memory.add("dog")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory, path)

        with open(path) as f:
            data = json.load(f)

        assert data["metadata"]["rep_type"] == "real"
        assert data["metadata"]["dim"] == 128

        # Real vectors should be simple lists
        assert isinstance(data["vectors"]["dog"], list)
        assert len(data["vectors"]["dog"]) == 128


def test_save_binary_correct_format() -> None:
    """Test that saved JSON has correct structure for Binary."""
    model = create_binary_model(dim=256, bipolar=True)
    memory = VSAMemory(model)
    memory.add("cat")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory, path)

        with open(path) as f:
            data = json.load(f)

        assert data["metadata"]["rep_type"] == "binary"
        assert data["metadata"]["dim"] == 256

        # Binary vectors should be integer lists
        vec = data["vectors"]["cat"]
        assert all(isinstance(x, int) for x in vec)
        assert len(vec) == 256


def test_load_fhrr_populates_memory() -> None:
    """Test that load_basis correctly populates FHRR memory."""
    # Create and save
    model = create_fhrr_model(dim=128)
    memory1 = VSAMemory(model)
    memory1.add_many(["apple", "orange", "banana"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        # Load into new memory
        memory2 = VSAMemory(model)
        load_basis(memory2, path)

        # Check vectors loaded
        assert len(memory2._symbols) == 3
        assert "apple" in memory2
        assert "orange" in memory2
        assert "banana" in memory2


def test_load_map_populates_memory() -> None:
    """Test that load_basis correctly populates MAP memory."""
    model = create_map_model(dim=256)
    memory1 = VSAMemory(model)
    memory1.add_many(["dog", "cat"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model)
        load_basis(memory2, path)

        assert len(memory2._symbols) == 2
        assert "dog" in memory2
        assert "cat" in memory2


def test_load_binary_populates_memory() -> None:
    """Test that load_basis correctly populates Binary memory."""
    model = create_binary_model(dim=512, bipolar=True)
    memory1 = VSAMemory(model)
    memory1.add("bird")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model)
        load_basis(memory2, path)

        assert len(memory2._symbols) == 1
        assert "bird" in memory2


def test_round_trip_fhrr_preserves_vectors() -> None:
    """Test that save/load round-trip preserves FHRR vectors."""
    model = create_fhrr_model(dim=128)
    memory1 = VSAMemory(model)
    memory1.add_many(["x", "y", "z"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model)
        load_basis(memory2, path)

        # Vectors should be identical (within floating point tolerance)
        for name in ["x", "y", "z"]:
            vec1 = memory1[name].vec
            vec2 = memory2[name].vec
            assert jnp.allclose(vec1, vec2, atol=1e-6)


def test_round_trip_map_preserves_vectors() -> None:
    """Test that save/load round-trip preserves MAP vectors."""
    model = create_map_model(dim=256)
    memory1 = VSAMemory(model)
    memory1.add_many(["a", "b"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model)
        load_basis(memory2, path)

        for name in ["a", "b"]:
            vec1 = memory1[name].vec
            vec2 = memory2[name].vec
            assert jnp.allclose(vec1, vec2, atol=1e-6)


def test_round_trip_binary_preserves_vectors() -> None:
    """Test that save/load round-trip preserves Binary vectors."""
    model = create_binary_model(dim=512, bipolar=True)
    memory1 = VSAMemory(model)
    memory1.add_many(["p", "q"])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model)
        load_basis(memory2, path)

        for name in ["p", "q"]:
            vec1 = memory1[name].vec
            vec2 = memory2[name].vec
            assert jnp.allclose(vec1, vec2)


def test_load_dimension_mismatch_raises_error() -> None:
    """Test that loading with mismatched dimension raises ValueError."""
    model128 = create_fhrr_model(dim=128)
    model256 = create_fhrr_model(dim=256)

    memory1 = VSAMemory(model128)
    memory1.add("x")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model256)
        with pytest.raises(ValueError, match="Dimension mismatch"):
            load_basis(memory2, path)


def test_load_rep_type_mismatch_raises_error() -> None:
    """Test that loading with mismatched rep type raises ValueError."""
    fhrr_model = create_fhrr_model(dim=128)
    map_model = create_map_model(dim=128)

    memory1 = VSAMemory(fhrr_model)
    memory1.add("x")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(map_model)
        with pytest.raises(ValueError, match="Representation type mismatch"):
            load_basis(memory2, path)


def test_load_into_non_empty_memory_raises_error() -> None:
    """Test that loading into non-empty memory raises ValueError."""
    model = create_fhrr_model(dim=128)
    memory1 = VSAMemory(model)
    memory1.add("x")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model)
        memory2.add("y")  # Pre-populate memory

        with pytest.raises(ValueError, match="Memory must be empty"):
            load_basis(memory2, path)


def test_load_nonexistent_file_raises_error() -> None:
    """Test that loading from non-existent file raises FileNotFoundError."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    with pytest.raises(FileNotFoundError):
        load_basis(memory, "nonexistent_file.json")


def test_save_with_string_path() -> None:
    """Test that save_basis accepts string paths."""
    model = create_map_model(dim=64)
    memory = VSAMemory(model)
    memory.add("test")

    with tempfile.TemporaryDirectory() as tmpdir:
        path_str = str(Path(tmpdir) / "test.json")
        save_basis(memory, path_str)

        assert Path(path_str).exists()


def test_load_with_string_path() -> None:
    """Test that load_basis accepts string paths."""
    model = create_map_model(dim=64)
    memory1 = VSAMemory(model)
    memory1.add("test")

    with tempfile.TemporaryDirectory() as tmpdir:
        path_str = str(Path(tmpdir) / "test.json")
        save_basis(memory1, path_str)

        memory2 = VSAMemory(model)
        load_basis(memory2, path_str)

        assert "test" in memory2


def test_save_empty_memory() -> None:
    """Test that saving empty memory works."""
    model = create_fhrr_model(dim=128)
    memory = VSAMemory(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.json"
        save_basis(memory, path)

        with open(path) as f:
            data = json.load(f)

        assert data["metadata"]["num_vectors"] == 0
        assert len(data["vectors"]) == 0


def test_load_empty_memory() -> None:
    """Test that loading empty memory works."""
    model = create_fhrr_model(dim=128)
    memory1 = VSAMemory(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "empty.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model)
        load_basis(memory2, path)

        assert len(memory2._symbols) == 0


def test_save_large_memory() -> None:
    """Test saving memory with many vectors."""
    model = create_map_model(dim=128)
    memory = VSAMemory(model)

    # Add 100 vectors
    names = [f"vec_{i}" for i in range(100)]
    memory.add_many(names)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "large.json"
        save_basis(memory, path)

        with open(path) as f:
            data = json.load(f)

        assert data["metadata"]["num_vectors"] == 100
        assert len(data["vectors"]) == 100


def test_round_trip_large_memory() -> None:
    """Test round-trip with many vectors."""
    model = create_fhrr_model(dim=256)
    memory1 = VSAMemory(model)

    names = [f"sym_{i}" for i in range(50)]
    memory1.add_many(names)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "large.json"
        save_basis(memory1, path)

        memory2 = VSAMemory(model)
        load_basis(memory2, path)

        assert len(memory2._symbols) == 50
        for name in names:
            assert name in memory2
            assert jnp.allclose(memory1[name].vec, memory2[name].vec, atol=1e-6)
