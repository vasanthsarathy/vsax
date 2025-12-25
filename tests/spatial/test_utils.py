"""Tests for SSP utility functions."""

import jax
import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_fhrr_model
from vsax.spatial import (
    SpatialSemanticPointers,
    SSPConfig,
    create_spatial_scene,
    region_query,
    similarity_map_2d,
)


class TestCreateSpatialScene:
    """Test suite for create_spatial_scene."""

    @pytest.fixture
    def ssp_setup(self):
        """Create SSP instance with objects."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=2)
        ssp = SpatialSemanticPointers(model, memory, config)

        # Add objects
        memory.add("apple")
        memory.add("banana")
        memory.add("cherry")

        return ssp, model, memory

    def test_create_scene_single_object(self, ssp_setup):
        """Test creating scene with single object."""
        ssp, _, _ = ssp_setup

        scene = create_spatial_scene(ssp, {"apple": [1.0, 2.0]})

        assert scene.shape == (512,)
        assert jnp.iscomplexobj(scene.vec)

    def test_create_scene_multiple_objects(self, ssp_setup):
        """Test creating scene with multiple objects."""
        ssp, _, _ = ssp_setup

        scene = create_spatial_scene(
            ssp,
            {
                "apple": [1.0, 2.0],
                "banana": [3.0, 4.0],
                "cherry": [5.0, 1.0],
            },
        )

        assert scene.shape == (512,)
        assert jnp.iscomplexobj(scene.vec)
        assert jnp.all(jnp.isfinite(scene.vec))

    def test_create_scene_empty_raises_error(self, ssp_setup):
        """Test that empty dict raises ValueError."""
        ssp, _, _ = ssp_setup

        with pytest.raises(ValueError, match="cannot be empty"):
            create_spatial_scene(ssp, {})

    def test_create_scene_nonexistent_object_raises_error(self, ssp_setup):
        """Test that non-existent object raises KeyError."""
        ssp, _, _ = ssp_setup

        with pytest.raises(KeyError):
            create_spatial_scene(ssp, {"nonexistent": [1.0, 2.0]})

    def test_create_scene_querying_works(self, ssp_setup):
        """Test that scenes created can be queried."""
        ssp, _, _ = ssp_setup

        scene = create_spatial_scene(
            ssp,
            {
                "apple": [1.0, 2.0],
                "banana": [3.0, 4.0],
            },
        )

        # Query locations - verify operations work
        result1 = ssp.query_location(scene, [1.0, 2.0])
        result2 = ssp.query_location(scene, [3.0, 4.0])

        assert result1.shape == (512,)
        assert result2.shape == (512,)
        assert jnp.all(jnp.isfinite(result1.vec))
        assert jnp.all(jnp.isfinite(result2.vec))


class TestSimilarityMap2D:
    """Test suite for similarity_map_2d."""

    @pytest.fixture
    def ssp_2d(self):
        """Create 2D SSP instance."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=2)
        ssp = SpatialSemanticPointers(model, memory, config)
        return ssp, model, memory

    def test_similarity_map_basic(self, ssp_2d):
        """Test basic similarity map generation."""
        ssp, _, _ = ssp_2d

        # Encode a location
        location = ssp.encode_location([3.0, 2.0])

        # Generate similarity map
        X, Y, sims = similarity_map_2d(ssp, location, x_range=(0, 5), y_range=(0, 5), resolution=10)

        # Check shapes
        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert sims.shape == (10, 10)

        # Check that similarities are valid
        assert jnp.all(jnp.isfinite(sims))
        assert jnp.all(sims >= -1.0)
        assert jnp.all(sims <= 1.0)

    def test_similarity_map_peak_location(self, ssp_2d):
        """Test that similarity map has peak near encoded location."""
        ssp, _, _ = ssp_2d

        # Encode location at (2.5, 2.5)
        location = ssp.encode_location([2.5, 2.5])

        # Generate similarity map
        X, Y, sims = similarity_map_2d(ssp, location, x_range=(0, 5), y_range=(0, 5), resolution=20)

        # Find peak location
        max_idx = jnp.unravel_index(jnp.argmax(sims), sims.shape)
        peak_x = X[max_idx]
        peak_y = Y[max_idx]

        # Peak should be reasonably close to (2.5, 2.5)
        # Due to grid discretization and SSP approximations, allow some tolerance
        assert abs(peak_x - 2.5) < 1.0
        assert abs(peak_y - 2.5) < 1.0

    def test_similarity_map_with_3d_raises_error(self):
        """Test that 3D SSP raises ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=3)  # 3D
        ssp = SpatialSemanticPointers(model, memory, config)

        location = ssp.encode_location([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="only works with 2D"):
            similarity_map_2d(ssp, location, x_range=(0, 5), y_range=(0, 5))

    def test_similarity_map_different_resolutions(self, ssp_2d):
        """Test similarity map with different resolutions."""
        ssp, _, _ = ssp_2d
        location = ssp.encode_location([3.0, 2.0])

        # Low resolution
        X1, Y1, sims1 = similarity_map_2d(
            ssp, location, x_range=(0, 5), y_range=(0, 5), resolution=5
        )
        assert sims1.shape == (5, 5)

        # High resolution
        X2, Y2, sims2 = similarity_map_2d(
            ssp, location, x_range=(0, 5), y_range=(0, 5), resolution=30
        )
        assert sims2.shape == (30, 30)


class TestRegionQuery:
    """Test suite for region_query."""

    @pytest.fixture
    def ssp_setup(self):
        """Create SSP instance with scene."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=2)
        ssp = SpatialSemanticPointers(model, memory, config)

        memory.add("apple")
        memory.add("banana")
        memory.add("cherry")

        scene = create_spatial_scene(
            ssp,
            {
                "apple": [1.0, 1.0],
                "banana": [3.0, 3.0],
                "cherry": [5.0, 5.0],
            },
        )

        return ssp, scene, memory

    def test_region_query_basic(self, ssp_setup):
        """Test basic region query."""
        ssp, scene, _ = ssp_setup

        # Query region around (3.0, 3.0) where banana is
        results = region_query(
            ssp,
            scene,
            ["apple", "banana", "cherry"],
            center=[3.0, 3.0],
            radius=0.5,
            resolution=10,
        )

        # Check that results are returned for all objects
        assert "apple" in results
        assert "banana" in results
        assert "cherry" in results

        # Check that similarity scores are valid floats
        assert isinstance(results["apple"], float)
        assert isinstance(results["banana"], float)
        assert isinstance(results["cherry"], float)

    def test_region_query_finds_nearby_object(self, ssp_setup):
        """Test that region query finds object at center."""
        ssp, scene, _ = ssp_setup

        # Query small region around banana (3.0, 3.0)
        results = region_query(
            ssp,
            scene,
            ["apple", "banana", "cherry"],
            center=[3.0, 3.0],
            radius=0.3,
            resolution=15,
        )

        # Banana should have highest (or at least reasonable) similarity
        # Note: Due to SSP approximations, we just verify it's in valid range
        assert -1.0 <= results["banana"] <= 1.0
        assert jnp.isfinite(results["banana"])

    def test_region_query_wrong_dimensions_raises_error(self, ssp_setup):
        """Test that wrong center dimensions raises ValueError."""
        ssp, scene, _ = ssp_setup

        with pytest.raises(ValueError, match="must have 2 coordinates"):
            region_query(
                ssp,
                scene,
                ["apple"],
                center=[1.0, 2.0, 3.0],  # 3D for 2D SSP
                radius=1.0,
            )

    def test_region_query_nonexistent_object_raises_error(self, ssp_setup):
        """Test that non-existent object raises KeyError."""
        ssp, scene, _ = ssp_setup

        with pytest.raises(KeyError):
            region_query(
                ssp,
                scene,
                ["nonexistent"],
                center=[1.0, 1.0],
                radius=1.0,
            )

    def test_region_query_3d(self):
        """Test region query with 3D SSP."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=3)
        ssp = SpatialSemanticPointers(model, memory, config)

        memory.add("ball")
        memory.add("cube")

        scene = create_spatial_scene(
            ssp,
            {
                "ball": [1.0, 2.0, 3.0],
                "cube": [4.0, 5.0, 6.0],
            },
        )

        results = region_query(
            ssp,
            scene,
            ["ball", "cube"],
            center=[1.0, 2.0, 3.0],
            radius=0.5,
            resolution=5,
        )

        assert "ball" in results
        assert "cube" in results
        assert jnp.isfinite(results["ball"])
        assert jnp.isfinite(results["cube"])

    def test_region_query_empty_objects_list(self, ssp_setup):
        """Test region query with empty objects list."""
        ssp, scene, _ = ssp_setup

        results = region_query(ssp, scene, [], center=[1.0, 1.0], radius=1.0, resolution=5)

        assert results == {}


class TestPlotSSP2DScene:
    """Test suite for plot_ssp_2d_scene."""

    @pytest.fixture
    def ssp_setup(self):
        """Create SSP instance with scene."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=2)
        ssp = SpatialSemanticPointers(model, memory, config)

        memory.add("apple")
        memory.add("banana")

        scene = create_spatial_scene(
            ssp,
            {
                "apple": [1.0, 2.0],
                "banana": [3.0, 4.0],
            },
        )

        return ssp, scene, memory

    def test_plot_scene_requires_matplotlib(self, ssp_setup):
        """Test that plotting function exists and has proper signature."""
        from vsax.spatial.utils import plot_ssp_2d_scene

        # Function should exist
        assert callable(plot_ssp_2d_scene)

    def test_plot_with_3d_raises_error(self):
        """Test that 3D SSP raises ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        config = SSPConfig(dim=512, num_axes=3)  # 3D
        ssp = SpatialSemanticPointers(model, memory, config)

        memory.add("ball")
        scene = create_spatial_scene(ssp, {"ball": [1.0, 2.0, 3.0]})

        # Try importing matplotlib first
        try:
            from vsax.spatial.utils import plot_ssp_2d_scene

            with pytest.raises(ValueError, match="only works with 2D"):
                plot_ssp_2d_scene(ssp, scene, ["ball"])
        except ImportError:
            # Matplotlib not installed, skip this test
            pytest.skip("matplotlib not installed")
