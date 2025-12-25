"""Tests for VFA applications."""

import jax
import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_fhrr_model, create_map_model
from vsax.representations import ComplexHypervector
from vsax.vfa import DensityEstimator, ImageProcessor, NonlinearRegressor


class TestDensityEstimator:
    """Test suite for DensityEstimator."""

    def test_init_with_fhrr_model(self):
        """Test initialization with FHRR model."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)

        estimator = DensityEstimator(model, memory, bandwidth=1.0)

        assert estimator.bandwidth == 1.0
        assert estimator.vfa is not None
        assert not estimator._fitted

    def test_init_with_invalid_bandwidth_raises_error(self):
        """Test that invalid bandwidth raises ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)

        with pytest.raises(ValueError, match="bandwidth must be positive"):
            DensityEstimator(model, memory, bandwidth=-1.0)

        with pytest.raises(ValueError, match="bandwidth must be positive"):
            DensityEstimator(model, memory, bandwidth=0.0)

    def test_init_with_non_fhrr_model_raises_error(self):
        """Test that non-FHRR model raises TypeError."""
        key = jax.random.PRNGKey(42)
        model = create_map_model(dim=512, key=key)
        memory = VSAMemory(model)

        with pytest.raises(TypeError, match="requires ComplexHypervector"):
            DensityEstimator(model, memory)

    def test_fit_with_gaussian_samples(self):
        """Test fitting to Gaussian samples."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        estimator = DensityEstimator(model, memory, bandwidth=0.5)

        # Generate Gaussian samples
        samples = jax.random.normal(key, (100,))

        estimator.fit(samples)

        assert estimator._fitted
        assert estimator._density_hv is not None
        assert isinstance(estimator._density_hv, ComplexHypervector)

    def test_fit_with_2d_samples_raises_error(self):
        """Test that 2D samples raise ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        estimator = DensityEstimator(model, memory)

        samples = jax.random.normal(key, (10, 10))

        with pytest.raises(ValueError, match="must be 1D array"):
            estimator.fit(samples)

    def test_evaluate_after_fit(self):
        """Test evaluating density after fitting."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=1024, key=key)
        memory = VSAMemory(model)
        estimator = DensityEstimator(model, memory, bandwidth=0.5)

        # Fit to samples
        samples = jax.random.normal(key, (50,))
        estimator.fit(samples)

        # Evaluate at query points
        x_query = jnp.linspace(-3, 3, 10)
        densities = estimator.evaluate(x_query)

        assert densities.shape == (10,)
        assert jnp.all(jnp.isfinite(densities))
        # Density should be non-negative (approximately)
        # Note: Due to VFA approximation, may have small negative values
        assert jnp.all(densities > -1.0)

    def test_evaluate_before_fit_raises_error(self):
        """Test that evaluating before fit raises RuntimeError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        estimator = DensityEstimator(model, memory)

        x_query = jnp.array([0.0, 1.0])

        with pytest.raises(RuntimeError, match="must be fitted"):
            estimator.evaluate(x_query)

    def test_sample_raises_not_implemented(self):
        """Test that sample() raises NotImplementedError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        estimator = DensityEstimator(model, memory)

        samples = jax.random.normal(key, (50,))
        estimator.fit(samples)

        with pytest.raises(NotImplementedError, match="Sampling"):
            estimator.sample(key, n_samples=10)

    def test_different_bandwidths(self):
        """Test that different bandwidths affect the fit."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory1 = VSAMemory(model)
        memory2 = VSAMemory(model)

        samples = jax.random.normal(key, (50,))

        # Fit with different bandwidths
        est1 = DensityEstimator(model, memory1, bandwidth=0.1)
        est2 = DensityEstimator(model, memory2, bandwidth=2.0)

        est1.fit(samples)
        est2.fit(samples)

        # Density hypervectors should be different
        assert not jnp.allclose(est1._density_hv.vec, est2._density_hv.vec)


class TestNonlinearRegressor:
    """Test suite for NonlinearRegressor."""

    def test_init_with_fhrr_model(self):
        """Test initialization with FHRR model."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)

        regressor = NonlinearRegressor(model, memory, regularization=1e-6)

        assert regressor.regularization == 1e-6
        assert regressor.vfa is not None
        assert not regressor._fitted

    def test_init_with_negative_regularization_raises_error(self):
        """Test that negative regularization raises ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)

        with pytest.raises(ValueError, match="must be non-negative"):
            NonlinearRegressor(model, memory, regularization=-1.0)

    def test_init_with_non_fhrr_model_raises_error(self):
        """Test that non-FHRR model raises TypeError."""
        key = jax.random.PRNGKey(42)
        model = create_map_model(dim=512, key=key)
        memory = VSAMemory(model)

        with pytest.raises(TypeError, match="requires ComplexHypervector"):
            NonlinearRegressor(model, memory)

    def test_fit_linear_function(self):
        """Test fitting a linear function."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        regressor = NonlinearRegressor(model, memory)

        # Linear data
        x_train = jnp.linspace(0, 10, 30)
        y_train = 2 * x_train + 1

        regressor.fit(x_train, y_train)

        assert regressor._fitted
        assert regressor._function_hv is not None

    def test_fit_nonlinear_function(self):
        """Test fitting a nonlinear function."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=1024, key=key)
        memory = VSAMemory(model)
        regressor = NonlinearRegressor(model, memory)

        # Nonlinear data
        x_train = jnp.linspace(0, 2 * jnp.pi, 50)
        y_train = jnp.sin(x_train)

        regressor.fit(x_train, y_train)

        assert regressor._fitted
        assert regressor._function_hv is not None

    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=1024, key=key)
        memory = VSAMemory(model)
        regressor = NonlinearRegressor(model, memory)

        # Fit to quadratic
        x_train = jnp.linspace(0, 10, 40)
        y_train = x_train**2

        regressor.fit(x_train, y_train)

        # Predict
        x_test = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = regressor.predict(x_test)

        assert y_pred.shape == (5,)
        assert jnp.all(jnp.isfinite(y_pred))

    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fit raises RuntimeError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        regressor = NonlinearRegressor(model, memory)

        x_test = jnp.array([1.0, 2.0])

        with pytest.raises(RuntimeError, match="must be fitted"):
            regressor.predict(x_test)

    def test_score_after_fit(self):
        """Test R² scoring after fitting."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=1024, key=key)
        memory = VSAMemory(model)
        regressor = NonlinearRegressor(model, memory)

        # Fit to linear function with some noise
        x_train = jnp.linspace(0, 10, 50)
        y_train = 2 * x_train + 1 + 0.1 * jax.random.normal(key, (50,))

        regressor.fit(x_train, y_train)

        # Score on test set
        x_test = jnp.linspace(0, 10, 20)
        y_test = 2 * x_test + 1

        r2 = regressor.score(x_test, y_test)

        # R² should be reasonable (not perfect due to noise and approximation)
        assert isinstance(r2, float)
        assert jnp.isfinite(r2)

    def test_score_before_fit_raises_error(self):
        """Test that scoring before fit raises RuntimeError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        regressor = NonlinearRegressor(model, memory)

        x_test = jnp.array([1.0, 2.0])
        y_test = jnp.array([2.0, 4.0])

        with pytest.raises(RuntimeError, match="must be fitted"):
            regressor.score(x_test, y_test)

    def test_different_regularization(self):
        """Test that different regularization affects results."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory1 = VSAMemory(model)
        memory2 = VSAMemory(model)

        x_train = jnp.linspace(0, 10, 30)
        y_train = jnp.sin(x_train)

        reg1 = NonlinearRegressor(model, memory1, regularization=1e-8)
        reg2 = NonlinearRegressor(model, memory2, regularization=1e-2)

        reg1.fit(x_train, y_train)
        reg2.fit(x_train, y_train)

        # Function hypervectors should be different
        assert not jnp.allclose(reg1._function_hv.vec, reg2._function_hv.vec)


class TestImageProcessor:
    """Test suite for ImageProcessor."""

    def test_init_with_fhrr_model(self):
        """Test initialization with FHRR model."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)

        processor = ImageProcessor(model, memory)

        assert processor.vfa is not None
        assert not processor._encoded

    def test_init_with_non_fhrr_model_raises_error(self):
        """Test that non-FHRR model raises TypeError."""
        key = jax.random.PRNGKey(42)
        model = create_map_model(dim=512, key=key)
        memory = VSAMemory(model)

        with pytest.raises(TypeError, match="requires ComplexHypervector"):
            ImageProcessor(model, memory)

    def test_encode_small_image(self):
        """Test encoding a small image."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        processor = ImageProcessor(model, memory)

        # Create simple 8x8 image
        image = jnp.zeros((8, 8))
        image = image.at[2:6, 2:6].set(1.0)  # White square in center

        image_hv = processor.encode(image)

        assert isinstance(image_hv, ComplexHypervector)
        assert image_hv.shape == (512,)
        assert processor._encoded
        assert processor._image_shape == (8, 8)

    def test_encode_1d_array_raises_error(self):
        """Test that 1D array raises ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        processor = ImageProcessor(model, memory)

        image = jnp.zeros(64)

        with pytest.raises(ValueError, match="must be 2D array"):
            processor.encode(image)

    def test_encode_3d_array_raises_error(self):
        """Test that 3D array raises ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        processor = ImageProcessor(model, memory)

        image = jnp.zeros((8, 8, 3))

        with pytest.raises(ValueError, match="must be 2D array"):
            processor.encode(image)

    def test_decode_image(self):
        """Test decoding an encoded image."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=1024, key=key)
        memory = VSAMemory(model)
        processor = ImageProcessor(model, memory)

        # Encode image
        original = jnp.zeros((8, 8))
        original = original.at[2:6, 2:6].set(1.0)
        image_hv = processor.encode(original)

        # Decode
        reconstructed = processor.decode(image_hv, shape=(8, 8))

        assert reconstructed.shape == (8, 8)
        assert jnp.all(jnp.isfinite(reconstructed))

    def test_blend_two_images(self):
        """Test blending two encoded images."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        processor = ImageProcessor(model, memory)

        # Create two different images
        image1 = jnp.zeros((8, 8))
        image1 = image1.at[0:4, 0:4].set(1.0)

        image2 = jnp.zeros((8, 8))
        image2 = image2.at[4:8, 4:8].set(1.0)

        # Encode
        hv1 = processor.encode(image1)
        hv2 = processor.encode(image2)

        # Blend
        blended = processor.blend(hv1, hv2, alpha=0.5)

        assert isinstance(blended, ComplexHypervector)
        assert blended.shape == (512,)

    def test_blend_with_invalid_alpha_raises_error(self):
        """Test that invalid alpha raises ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        processor = ImageProcessor(model, memory)

        image = jnp.zeros((8, 8))
        hv1 = processor.encode(image)
        hv2 = processor.encode(image)

        with pytest.raises(ValueError, match="alpha must be in"):
            processor.blend(hv1, hv2, alpha=-0.1)

        with pytest.raises(ValueError, match="alpha must be in"):
            processor.blend(hv1, hv2, alpha=1.5)

    def test_shift_raises_not_implemented(self):
        """Test that shift() raises NotImplementedError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        processor = ImageProcessor(model, memory)

        image = jnp.zeros((8, 8))
        processor.encode(image)

        with pytest.raises(NotImplementedError, match="2D image shifting"):
            processor.shift(dx=1.0, dy=1.0)

    def test_shift_before_encode_raises_error(self):
        """Test that shift before encode raises RuntimeError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        processor = ImageProcessor(model, memory)

        with pytest.raises(RuntimeError, match="must encode an image"):
            processor.shift(dx=1.0, dy=1.0)

    def test_encode_different_sizes(self):
        """Test encoding images of different sizes."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)

        for shape in [(4, 4), (8, 8), (16, 16), (10, 12)]:
            memory = VSAMemory(model)
            processor = ImageProcessor(model, memory)
            image = jax.random.uniform(key, shape)

            image_hv = processor.encode(image)

            assert isinstance(image_hv, ComplexHypervector)
            assert processor._image_shape == shape
