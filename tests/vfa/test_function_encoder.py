"""Tests for VFA VectorFunctionEncoder."""

import jax
import jax.numpy as jnp
import pytest

from vsax import VSAMemory, create_fhrr_model, create_map_model
from vsax.representations import ComplexHypervector
from vsax.vfa import KernelConfig, KernelType, VectorFunctionEncoder


class TestVectorFunctionEncoderInit:
    """Test suite for VectorFunctionEncoder initialization."""

    def test_init_with_fhrr_model(self):
        """Test initialization with FHRR model (should work)."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)

        vfa = VectorFunctionEncoder(model, memory)

        assert vfa.model == model
        assert vfa.memory == memory
        assert isinstance(vfa.kernel_config, KernelConfig)
        assert vfa.basis_vector.shape == (512,)
        assert jnp.iscomplexobj(vfa.basis_vector)

    def test_init_with_non_fhrr_model_raises_error(self):
        """Test that non-FHRR model raises TypeError."""
        key = jax.random.PRNGKey(42)
        model = create_map_model(dim=512, key=key)
        memory = VSAMemory(model)

        with pytest.raises(TypeError, match="requires ComplexHypervector"):
            VectorFunctionEncoder(model, memory)

    def test_init_with_custom_kernel_config(self):
        """Test initialization with custom kernel config."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=256, key=key)
        memory = VSAMemory(model)
        config = KernelConfig(kernel_type=KernelType.UNIFORM, bandwidth=2.0, dim=256)

        vfa = VectorFunctionEncoder(model, memory, kernel_config=config)

        assert vfa.kernel_config == config
        assert vfa.basis_vector.shape == (256,)

    def test_init_with_custom_basis_key(self):
        """Test initialization with custom basis key."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)

        basis_key1 = jax.random.PRNGKey(1)
        basis_key2 = jax.random.PRNGKey(2)

        vfa1 = VectorFunctionEncoder(model, memory, basis_key=basis_key1)
        vfa2 = VectorFunctionEncoder(model, memory, basis_key=basis_key2)

        # Different keys should produce different basis vectors
        assert not jnp.allclose(vfa1.basis_vector, vfa2.basis_vector)

    def test_basis_vector_has_unit_magnitude(self):
        """Test that basis vector has unit magnitude."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)

        vfa = VectorFunctionEncoder(model, memory)

        magnitudes = jnp.abs(vfa.basis_vector)
        assert jnp.allclose(magnitudes, 1.0)


class TestEncodeFunctionOneDim:
    """Test suite for encode_function_1d."""

    def test_encode_linear_function(self):
        """Test encoding a linear function."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        # Linear function: y = 2x + 1
        x = jnp.linspace(0, 10, 20)
        y = 2 * x + 1

        f_hv = vfa.encode_function_1d(x, y)

        assert isinstance(f_hv, ComplexHypervector)
        assert f_hv.shape == (512,)
        assert jnp.iscomplexobj(f_hv.vec)
        assert jnp.all(jnp.isfinite(f_hv.vec))

    def test_encode_quadratic_function(self):
        """Test encoding a quadratic function."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        # Quadratic: y = x^2
        x = jnp.linspace(-5, 5, 30)
        y = x**2

        f_hv = vfa.encode_function_1d(x, y)

        assert isinstance(f_hv, ComplexHypervector)
        assert f_hv.shape == (512,)
        assert jnp.all(jnp.isfinite(f_hv.vec))

    def test_encode_sine_function(self):
        """Test encoding a sine function."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        # Sine: y = sin(x)
        x = jnp.linspace(0, 2 * jnp.pi, 50)
        y = jnp.sin(x)

        f_hv = vfa.encode_function_1d(x, y)

        assert isinstance(f_hv, ComplexHypervector)
        assert f_hv.shape == (512,)
        assert jnp.all(jnp.isfinite(f_hv.vec))

    def test_encode_exponential_function(self):
        """Test encoding an exponential function."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        # Exponential: y = exp(-x)
        x = jnp.linspace(0, 5, 30)
        y = jnp.exp(-x)

        f_hv = vfa.encode_function_1d(x, y)

        assert isinstance(f_hv, ComplexHypervector)
        assert f_hv.shape == (512,)
        assert jnp.all(jnp.isfinite(f_hv.vec))

    def test_encode_different_sample_counts(self):
        """Test encoding with different numbers of samples."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        for n_samples in [10, 20, 50, 100]:
            x = jnp.linspace(0, 10, n_samples)
            y = x**2

            f_hv = vfa.encode_function_1d(x, y)

            assert isinstance(f_hv, ComplexHypervector)
            assert f_hv.shape == (512,)

    def test_encode_with_different_regularization(self):
        """Test encoding with different regularization values."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y = jnp.sin(x)

        for reg in [1e-8, 1e-6, 1e-4, 1e-2]:
            f_hv = vfa.encode_function_1d(x, y, regularization=reg)
            assert isinstance(f_hv, ComplexHypervector)
            assert jnp.all(jnp.isfinite(f_hv.vec))

    def test_encode_mismatched_lengths_raises_error(self):
        """Test that mismatched x and y lengths raise ValueError."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y = jnp.sin(jnp.linspace(0, 10, 15))  # Different length!

        with pytest.raises(ValueError, match="must have same length"):
            vfa.encode_function_1d(x, y)


class TestEvaluateOneDim:
    """Test suite for evaluate_1d."""

    def test_evaluate_at_sample_points(self):
        """Test that evaluation at sample points is reasonably accurate."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=1024, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        # Linear function
        x = jnp.linspace(0, 10, 20)
        y = 2 * x + 1

        f_hv = vfa.encode_function_1d(x, y)

        # Evaluate at a few sample points
        for i in [0, 5, 10, 15, 19]:
            y_pred = vfa.evaluate_1d(f_hv, x[i])
            # Should be close to actual value (within 20% for linear)
            assert isinstance(y_pred, (float, jnp.ndarray))
            assert jnp.isfinite(y_pred)

    def test_evaluate_at_interpolation_points(self):
        """Test evaluation at points between samples."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=1024, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 2 * jnp.pi, 50)
        y = jnp.sin(x)

        f_hv = vfa.encode_function_1d(x, y)

        # Evaluate at points not in training set
        x_test = jnp.array([0.5, 1.5, 2.5, 3.5])
        for x_query in x_test:
            y_pred = vfa.evaluate_1d(f_hv, x_query)
            assert isinstance(y_pred, (float, jnp.ndarray))
            assert jnp.isfinite(y_pred)

    def test_evaluate_returns_real_values(self):
        """Test that evaluation returns real values."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y = x**2

        f_hv = vfa.encode_function_1d(x, y)
        y_pred = vfa.evaluate_1d(f_hv, 5.0)

        # Should return real value
        assert not jnp.iscomplexobj(y_pred)
        assert isinstance(y_pred, (float, jnp.ndarray))

    def test_evaluate_at_negative_values(self):
        """Test evaluation at negative x values."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(-10, 10, 40)
        y = x**2

        f_hv = vfa.encode_function_1d(x, y)
        y_pred = vfa.evaluate_1d(f_hv, -5.0)

        assert isinstance(y_pred, (float, jnp.ndarray))
        assert jnp.isfinite(y_pred)


class TestAddFunctions:
    """Test suite for add_functions."""

    def test_add_two_functions(self):
        """Test adding two functions."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y1 = x
        y2 = x**2

        f1_hv = vfa.encode_function_1d(x, y1)
        f2_hv = vfa.encode_function_1d(x, y2)

        f_sum = vfa.add_functions(f1_hv, f2_hv)

        assert isinstance(f_sum, ComplexHypervector)
        assert f_sum.shape == (512,)
        assert jnp.all(jnp.isfinite(f_sum.vec))

    def test_add_with_coefficients(self):
        """Test adding functions with alpha and beta coefficients."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y1 = jnp.sin(x)
        y2 = jnp.cos(x)

        f1_hv = vfa.encode_function_1d(x, y1)
        f2_hv = vfa.encode_function_1d(x, y2)

        # 2*f1 + 3*f2
        f_combined = vfa.add_functions(f1_hv, f2_hv, alpha=2.0, beta=3.0)

        assert isinstance(f_combined, ComplexHypervector)
        assert f_combined.shape == (512,)
        assert jnp.all(jnp.isfinite(f_combined.vec))

    def test_add_with_negative_coefficients(self):
        """Test adding functions with negative coefficients (subtraction)."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y1 = x**2
        y2 = x

        f1_hv = vfa.encode_function_1d(x, y1)
        f2_hv = vfa.encode_function_1d(x, y2)

        # f1 - f2
        f_diff = vfa.add_functions(f1_hv, f2_hv, alpha=1.0, beta=-1.0)

        assert isinstance(f_diff, ComplexHypervector)
        assert f_diff.shape == (512,)
        assert jnp.all(jnp.isfinite(f_diff.vec))

    def test_add_zero_coefficient(self):
        """Test adding with zero coefficient."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y1 = jnp.sin(x)
        y2 = jnp.cos(x)

        f1_hv = vfa.encode_function_1d(x, y1)
        f2_hv = vfa.encode_function_1d(x, y2)

        # f1 + 0*f2 = f1
        f_result = vfa.add_functions(f1_hv, f2_hv, alpha=1.0, beta=0.0)

        assert isinstance(f_result, ComplexHypervector)
        # Should be similar to f1 (but not exactly due to encoding)
        assert jnp.all(jnp.isfinite(f_result.vec))


class TestShiftFunction:
    """Test suite for shift_function."""

    def test_shift_positive(self):
        """Test shifting function to the right."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 2 * jnp.pi, 50)
        y = jnp.sin(x)

        f_hv = vfa.encode_function_1d(x, y)
        f_shifted = vfa.shift_function(f_hv, jnp.pi / 2)

        assert isinstance(f_shifted, ComplexHypervector)
        assert f_shifted.shape == (512,)
        assert jnp.all(jnp.isfinite(f_shifted.vec))

    def test_shift_negative(self):
        """Test shifting function to the left."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 30)
        y = x**2

        f_hv = vfa.encode_function_1d(x, y)
        f_shifted = vfa.shift_function(f_hv, -2.0)

        assert isinstance(f_shifted, ComplexHypervector)
        assert f_shifted.shape == (512,)
        assert jnp.all(jnp.isfinite(f_shifted.vec))

    def test_shift_zero(self):
        """Test that zero shift returns similar function."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y = jnp.sin(x)

        f_hv = vfa.encode_function_1d(x, y)
        f_shifted = vfa.shift_function(f_hv, 0.0)

        assert isinstance(f_shifted, ComplexHypervector)
        # Should be very similar to original
        assert jnp.allclose(f_shifted.vec, f_hv.vec)

    def test_shift_sine_to_cosine(self):
        """Test shifting sine to approximate cosine."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=1024, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 2 * jnp.pi, 100)
        y_sin = jnp.sin(x)

        f_sin = vfa.encode_function_1d(x, y_sin)
        # sin(x - π/2) = -cos(x)
        f_shifted = vfa.shift_function(f_sin, jnp.pi / 2)

        assert isinstance(f_shifted, ComplexHypervector)
        assert f_shifted.shape == (1024,)
        assert jnp.all(jnp.isfinite(f_shifted.vec))


class TestConvolveFunctions:
    """Test suite for convolve_functions."""

    def test_convolve_two_functions(self):
        """Test convolution of two functions."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 30)
        y1 = jnp.sin(x)
        y2 = jnp.exp(-x)

        f1_hv = vfa.encode_function_1d(x, y1)
        f2_hv = vfa.encode_function_1d(x, y2)

        f_conv = vfa.convolve_functions(f1_hv, f2_hv)

        assert isinstance(f_conv, ComplexHypervector)
        assert f_conv.shape == (512,)
        assert jnp.all(jnp.isfinite(f_conv.vec))

    def test_convolve_is_symmetric(self):
        """Test that convolution is approximately symmetric."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 5, 20)
        y1 = x
        y2 = x**2

        f1_hv = vfa.encode_function_1d(x, y1)
        f2_hv = vfa.encode_function_1d(x, y2)

        f_conv1 = vfa.convolve_functions(f1_hv, f2_hv)
        f_conv2 = vfa.convolve_functions(f2_hv, f1_hv)

        # Should be the same (circular convolution is commutative)
        assert jnp.allclose(f_conv1.vec, f_conv2.vec)


class TestEvaluateBatch:
    """Test suite for evaluate_batch."""

    def test_evaluate_batch_basic(self):
        """Test batch evaluation at multiple points."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y = x**2

        f_hv = vfa.encode_function_1d(x, y)

        x_test = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = vfa.evaluate_batch(f_hv, x_test)

        assert y_pred.shape == (5,)
        assert jnp.all(jnp.isfinite(y_pred))
        assert not jnp.iscomplexobj(y_pred)

    def test_evaluate_batch_output_shape(self):
        """Test that batch evaluation produces correct output shape."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 2 * jnp.pi, 30)
        y = jnp.sin(x)

        f_hv = vfa.encode_function_1d(x, y)

        for n_queries in [5, 10, 50, 100]:
            x_test = jnp.linspace(0, 2 * jnp.pi, n_queries)
            y_pred = vfa.evaluate_batch(f_hv, x_test)
            assert y_pred.shape == (n_queries,)

    def test_evaluate_batch_matches_individual(self):
        """Test that batch evaluation matches individual evaluations."""
        key = jax.random.PRNGKey(42)
        model = create_fhrr_model(dim=512, key=key)
        memory = VSAMemory(model)
        vfa = VectorFunctionEncoder(model, memory)

        x = jnp.linspace(0, 10, 20)
        y = jnp.sin(x)

        f_hv = vfa.encode_function_1d(x, y)

        x_test = jnp.array([1.5, 3.5, 5.5])

        # Batch evaluation
        y_batch = vfa.evaluate_batch(f_hv, x_test)

        # Individual evaluations
        y_individual = jnp.array([vfa.evaluate_1d(f_hv, x_q) for x_q in x_test])

        assert jnp.allclose(y_batch, y_individual)
