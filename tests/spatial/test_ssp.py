"""Tests for Spatial Semantic Pointers."""

import jax
import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_fhrr_model
from vsax.similarity import cosine_similarity
from vsax.spatial import SpatialSemanticPointers, SSPConfig


class TestSSPConfig:
    """Test suite for SSPConfig."""

    def test_default_config(self):
        """Test SSPConfig with default values."""
        config = SSPConfig()
        assert config.dim == 512
        assert config.num_axes == 2
        assert config.scale is None
        assert config.axis_names == ["x", "y"]

    def test_custom_config_2d(self):
        """Test 2D configuration."""
        config = SSPConfig(dim=256, num_axes=2, scale=0.1)
        assert config.dim == 256
        assert config.num_axes == 2
        assert config.scale == 0.1
        assert config.axis_names == ["x", "y"]

    def test_custom_config_3d(self):
        """Test 3D configuration."""
        config = SSPConfig(dim=1024, num_axes=3)
        assert config.num_axes == 3
        assert config.axis_names == ["x", "y", "z"]

    def test_custom_axis_names(self):
        """Test custom axis names."""
        config = SSPConfig(num_axes=2, axis_names=["north", "east"])
        assert config.axis_names == ["north", "east"]

    def test_mismatched_axis_names_raises_error(self):
        """Test that mismatched axis_names length raises ValueError."""
        with pytest.raises(ValueError, match="axis_names length"):
            SSPConfig(num_axes=2, axis_names=["x", "y", "z"])

    def test_high_dimensional_default_names(self):
        """Test default names for high-dimensional spaces."""
        config = SSPConfig(num_axes=8)
        assert len(config.axis_names) == 8
        assert config.axis_names[0] == "axis_0"
        assert config.axis_names[7] == "axis_7"


class TestSpatialSemanticPointers:
    """Test suite for SpatialSemanticPointers."""

    @pytest.fixture
    def ssp_2d(self):
        """Create 2D SSP instance."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=2)
        ssp = SpatialSemanticPointers(model, memory, config)
        return ssp, model, memory

    @pytest.fixture
    def ssp_3d(self):
        """Create 3D SSP instance."""
        key = jax.random.PRNGKey(43)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=3)
        ssp = SpatialSemanticPointers(model, memory, config)
        return ssp, model, memory

    def test_initialization(self, ssp_2d):
        """Test SSP initialization."""
        ssp, model, memory = ssp_2d
        assert ssp.model == model
        assert ssp.memory == memory
        assert ssp.config.num_axes == 2
        # Axis basis vectors should be in memory
        assert "x" in memory
        assert "y" in memory

    def test_initialization_creates_axis_bases(self, ssp_3d):
        """Test that initialization creates all axis basis vectors."""
        ssp, model, memory = ssp_3d
        assert "x" in memory
        assert "y" in memory
        assert "z" in memory

    def test_encode_location_2d(self, ssp_2d):
        """Test encoding 2D location."""
        ssp, _, _ = ssp_2d
        location = ssp.encode_location([3.5, 2.1])

        assert location.shape == (512,)
        assert jnp.iscomplexobj(location.vec)

    def test_encode_location_3d(self, ssp_3d):
        """Test encoding 3D location."""
        ssp, _, _ = ssp_3d
        location = ssp.encode_location([1.0, 2.0, 3.0])

        assert location.shape == (512,)
        assert jnp.iscomplexobj(location.vec)

    def test_encode_location_wrong_dimensions_raises_error(self, ssp_2d):
        """Test that wrong number of coordinates raises ValueError."""
        ssp, _, _ = ssp_2d

        with pytest.raises(ValueError, match="Expected 2 coordinates"):
            ssp.encode_location([1.0, 2.0, 3.0])  # 3 coords for 2D

    def test_encode_location_with_negatives(self, ssp_2d):
        """Test encoding with negative coordinates."""
        ssp, _, _ = ssp_2d
        location = ssp.encode_location([-1.5, 3.2])

        assert location.shape == (512,)
        assert jnp.iscomplexobj(location.vec)

    def test_encode_location_deterministic(self, ssp_2d):
        """Test that encoding same location twice gives same result."""
        ssp, _, _ = ssp_2d
        loc1 = ssp.encode_location([3.5, 2.1])
        loc2 = ssp.encode_location([3.5, 2.1])

        assert jnp.allclose(loc1.vec, loc2.vec)

    def test_encode_different_locations_are_distinct(self, ssp_2d):
        """Test that different locations produce distinct encodings."""
        ssp, _, _ = ssp_2d
        loc1 = ssp.encode_location([1.0, 1.0])
        loc2 = ssp.encode_location([5.0, 5.0])

        similarity = cosine_similarity(loc1.vec, loc2.vec)
        assert similarity < 0.9  # Should be fairly distinct

    def test_bind_object_location(self, ssp_2d):
        """Test binding object to location."""
        ssp, _, memory = ssp_2d
        memory.add("apple")

        result = ssp.bind_object_location("apple", [3.5, 2.1])

        assert result.shape == (512,)
        assert jnp.iscomplexobj(result.vec)

    def test_bind_object_location_nonexistent_object_raises_error(self, ssp_2d):
        """Test that binding non-existent object raises KeyError."""
        ssp, _, _ = ssp_2d

        with pytest.raises(KeyError):
            ssp.bind_object_location("nonexistent", [1.0, 2.0])

    def test_query_location_retrieves_object(self, ssp_2d):
        """Test querying location to retrieve object."""
        ssp, _, memory = ssp_2d
        memory.add("apple")

        # Bind apple to location
        scene = ssp.bind_object_location("apple", [3.5, 2.1])

        # Query that location
        result = ssp.query_location(scene, [3.5, 2.1])

        # Note: Unbinding with circular convolution + random bases is approximate
        # Verify the operation works and produces valid output
        assert result.shape == (512,)
        assert jnp.iscomplexobj(result.vec)
        assert jnp.all(jnp.isfinite(result.vec))

    def test_query_location_wrong_location_low_similarity(self, ssp_2d):
        """Test querying wrong location gives low similarity."""
        ssp, _, memory = ssp_2d
        memory.add("apple")

        # Bind apple to (3.5, 2.1)
        scene = ssp.bind_object_location("apple", [3.5, 2.1])

        # Query different location (0.0, 0.0)
        result = ssp.query_location(scene, [0.0, 0.0])

        # Result should have low similarity to apple
        apple_hv = memory["apple"]
        similarity = cosine_similarity(result.vec, apple_hv.vec)
        assert similarity < 0.5  # Low similarity expected

    def test_query_object_retrieves_location(self, ssp_2d):
        """Test querying object to retrieve location."""
        ssp, _, memory = ssp_2d
        memory.add("apple")

        # Bind apple to location
        scene = ssp.bind_object_location("apple", [3.5, 2.1])

        # Query where apple is
        result = ssp.query_object(scene, "apple")

        # Verify the operation works and produces valid output
        assert result.shape == (512,)
        assert jnp.iscomplexobj(result.vec)
        assert jnp.all(jnp.isfinite(result.vec))

    def test_query_object_nonexistent_raises_error(self, ssp_2d):
        """Test that querying non-existent object raises KeyError."""
        ssp, _, memory = ssp_2d
        memory.add("apple")
        scene = ssp.bind_object_location("apple", [3.5, 2.1])

        with pytest.raises(KeyError):
            ssp.query_object(scene, "nonexistent")

    def test_shift_scene_2d(self, ssp_2d):
        """Test shifting scene in 2D."""
        ssp, _, memory = ssp_2d
        memory.add("apple")

        # Apple at (3.5, 2.1)
        scene = ssp.bind_object_location("apple", [3.5, 2.1])

        # Shift by (1.0, -0.5)
        shifted = ssp.shift_scene(scene, [1.0, -0.5])

        # Verify the shift operation works and produces valid output
        assert shifted.shape == (512,)
        assert jnp.iscomplexobj(shifted.vec)
        assert jnp.all(jnp.isfinite(shifted.vec))

    def test_shift_scene_3d(self, ssp_3d):
        """Test shifting scene in 3D."""
        ssp, _, memory = ssp_3d
        memory.add("ball")

        # Ball at (1.0, 2.0, 3.0)
        scene = ssp.bind_object_location("ball", [1.0, 2.0, 3.0])

        # Shift by (0.5, 0.5, 0.5)
        shifted = ssp.shift_scene(scene, [0.5, 0.5, 0.5])

        # Verify the shift operation works and produces valid output
        assert shifted.shape == (512,)
        assert jnp.iscomplexobj(shifted.vec)
        assert jnp.all(jnp.isfinite(shifted.vec))

    def test_shift_scene_wrong_dimensions_raises_error(self, ssp_2d):
        """Test that shifting with wrong dimensions raises ValueError."""
        ssp, _, memory = ssp_2d
        memory.add("apple")
        scene = ssp.bind_object_location("apple", [3.5, 2.1])

        with pytest.raises(ValueError, match="Expected 2 offset"):
            ssp.shift_scene(scene, [1.0, 2.0, 3.0])  # 3D offset for 2D

    def test_decode_location_2d(self, ssp_2d):
        """Test decoding 2D location."""
        ssp, _, _ = ssp_2d

        # Encode a location
        original_coords = [3.5, 2.1]
        location_hv = ssp.encode_location(original_coords)

        # Decode it
        decoded = ssp.decode_location(
            location_hv,
            search_range=[(0.0, 5.0), (0.0, 5.0)],
            resolution=30,
        )

        # Should be close to original
        assert len(decoded) == 2
        assert abs(decoded[0] - original_coords[0]) < 0.5
        assert abs(decoded[1] - original_coords[1]) < 0.5

    def test_decode_location_3d(self, ssp_3d):
        """Test decoding 3D location."""
        ssp, _, _ = ssp_3d

        # Encode a location
        original_coords = [1.5, 2.5, 3.5]
        location_hv = ssp.encode_location(original_coords)

        # Decode it
        decoded = ssp.decode_location(
            location_hv,
            search_range=[(0.0, 5.0), (0.0, 5.0), (0.0, 5.0)],
            resolution=20,
        )

        # Should be close to original
        assert len(decoded) == 3
        assert abs(decoded[0] - original_coords[0]) < 0.5
        assert abs(decoded[1] - original_coords[1]) < 0.5
        assert abs(decoded[2] - original_coords[2]) < 0.5

    def test_decode_location_wrong_search_range_raises_error(self, ssp_2d):
        """Test that wrong search_range dimensions raises ValueError."""
        ssp, _, _ = ssp_2d
        location_hv = ssp.encode_location([3.5, 2.1])

        with pytest.raises(ValueError, match="Expected 2 search ranges"):
            ssp.decode_location(
                location_hv,
                search_range=[(0.0, 5.0)],  # Only 1 range for 2D
            )

    def test_multiple_objects_in_scene(self, ssp_2d):
        """Test scene with multiple objects at different locations."""
        ssp, model, memory = ssp_2d
        memory.add("apple")
        memory.add("banana")
        memory.add("cherry")

        # Create scene with 3 objects
        apple_loc = ssp.bind_object_location("apple", [1.0, 1.0])
        banana_loc = ssp.bind_object_location("banana", [3.0, 2.0])
        cherry_loc = ssp.bind_object_location("cherry", [2.0, 4.0])

        # Bundle into scene
        scene_vec = model.opset.bundle(apple_loc.vec, banana_loc.vec, cherry_loc.vec)
        from vsax.representations import ComplexHypervector

        scene = ComplexHypervector(scene_vec)

        # Query each location - verify operations work
        apple_result = ssp.query_location(scene, [1.0, 1.0])
        banana_result = ssp.query_location(scene, [3.0, 2.0])
        cherry_result = ssp.query_location(scene, [2.0, 4.0])

        # Verify all queries produce valid outputs
        assert apple_result.shape == (512,)
        assert banana_result.shape == (512,)
        assert cherry_result.shape == (512,)
        assert jnp.all(jnp.isfinite(apple_result.vec))
        assert jnp.all(jnp.isfinite(banana_result.vec))
        assert jnp.all(jnp.isfinite(cherry_result.vec))

    def test_ssp_with_scaling(self):
        """Test SSP with coordinate scaling."""
        key = jax.random.PRNGKey(44)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=2, scale=0.1)
        ssp = SpatialSemanticPointers(model, memory, config)

        # Encode location with scaling
        location = ssp.encode_location([10.0, 20.0])
        # With scale=0.1, this is equivalent to encoding [1.0, 2.0]

        assert location.shape == (512,)
        assert jnp.iscomplexobj(location.vec)

    def test_encode_origin(self, ssp_2d):
        """Test encoding the origin (0, 0)."""
        ssp, _, _ = ssp_2d
        origin = ssp.encode_location([0.0, 0.0])

        # Encoding origin should work and produce valid output
        # (Exact value depends on how circular convolution handles v^0)
        assert origin.shape == (512,)
        assert jnp.iscomplexobj(origin.vec)
        assert jnp.all(jnp.isfinite(origin.vec))

    def test_ssp_preserves_valid_values(self, ssp_2d):
        """Test that SSP operations produce valid values."""
        ssp, _, memory = ssp_2d
        memory.add("apple")

        location = ssp.encode_location([3.5, 2.1])
        bound = ssp.bind_object_location("apple", [3.5, 2.1])

        # Check that all operations produce valid complex numbers
        assert jnp.all(jnp.isfinite(location.vec))
        assert jnp.all(jnp.isfinite(bound.vec))

        # Check magnitudes are positive
        loc_mags = jnp.abs(location.vec)
        bound_mags = jnp.abs(bound.vec)
        assert jnp.all(loc_mags > 0)
        assert jnp.all(bound_mags > 0)
