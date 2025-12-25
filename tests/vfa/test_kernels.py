"""Tests for VFA kernel functions."""

import jax
import jax.numpy as jnp
import pytest

from vsax.vfa import (
    KernelConfig,
    KernelType,
    get_kernel_name,
    sample_kernel_basis,
    sample_kernel_basis_batch,
)


class TestKernelType:
    """Test suite for KernelType enum."""

    def test_kernel_types_exist(self):
        """Test that all kernel types are defined."""
        assert KernelType.UNIFORM
        assert KernelType.GAUSSIAN
        assert KernelType.LAPLACE
        assert KernelType.CUSTOM

    def test_kernel_type_values(self):
        """Test kernel type string values."""
        assert KernelType.UNIFORM.value == "uniform"
        assert KernelType.GAUSSIAN.value == "gaussian"
        assert KernelType.LAPLACE.value == "laplace"
        assert KernelType.CUSTOM.value == "custom"


class TestKernelConfig:
    """Test suite for KernelConfig."""

    def test_default_config(self):
        """Test KernelConfig with default values."""
        config = KernelConfig()
        assert config.kernel_type == KernelType.UNIFORM
        assert config.bandwidth == 1.0
        assert config.dim == 512
        assert config.custom_sampler is None

    def test_custom_config(self):
        """Test KernelConfig with custom values."""
        config = KernelConfig(
            kernel_type=KernelType.GAUSSIAN,
            bandwidth=2.0,
            dim=1024,
        )
        assert config.kernel_type == KernelType.GAUSSIAN
        assert config.bandwidth == 2.0
        assert config.dim == 1024

    def test_invalid_bandwidth_raises_error(self):
        """Test that negative bandwidth raises ValueError."""
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            KernelConfig(bandwidth=-1.0)

        with pytest.raises(ValueError, match="bandwidth must be positive"):
            KernelConfig(bandwidth=0.0)

    def test_invalid_dim_raises_error(self):
        """Test that non-positive dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            KernelConfig(dim=0)

        with pytest.raises(ValueError, match="dim must be positive"):
            KernelConfig(dim=-512)

    def test_custom_kernel_without_sampler_raises_error(self):
        """Test that CUSTOM kernel without sampler raises ValueError."""
        with pytest.raises(ValueError, match="custom_sampler must be provided"):
            KernelConfig(kernel_type=KernelType.CUSTOM)

    def test_custom_kernel_with_sampler(self):
        """Test CUSTOM kernel with valid sampler."""

        def custom_sampler(key, dim):
            phases = jax.random.uniform(key, (dim,), minval=0, maxval=2 * jnp.pi)
            return jnp.exp(1j * phases)

        config = KernelConfig(kernel_type=KernelType.CUSTOM, custom_sampler=custom_sampler)
        assert config.custom_sampler is not None


class TestSampleKernelBasis:
    """Test suite for sample_kernel_basis."""

    def test_sample_uniform_kernel(self):
        """Test sampling with UNIFORM kernel."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(kernel_type=KernelType.UNIFORM, dim=512)

        basis = sample_kernel_basis(config, key)

        assert basis.shape == (512,)
        assert jnp.iscomplexobj(basis)

    def test_sample_preserves_unit_magnitude(self):
        """Test that sampled basis has unit magnitude."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(kernel_type=KernelType.UNIFORM, dim=256)

        basis = sample_kernel_basis(config, key)

        magnitudes = jnp.abs(basis)
        assert jnp.allclose(magnitudes, 1.0)

    def test_sample_different_keys_produce_different_vectors(self):
        """Test that different keys produce different basis vectors."""
        config = KernelConfig(dim=512)

        basis1 = sample_kernel_basis(config, jax.random.PRNGKey(1))
        basis2 = sample_kernel_basis(config, jax.random.PRNGKey(2))

        # Should be different (very unlikely to be identical)
        assert not jnp.allclose(basis1, basis2)

    def test_sample_same_key_produces_same_vector(self):
        """Test that same key produces same basis vector."""
        config = KernelConfig(dim=512)
        key = jax.random.PRNGKey(42)

        basis1 = sample_kernel_basis(config, key)
        basis2 = sample_kernel_basis(config, key)

        assert jnp.allclose(basis1, basis2)

    def test_sample_gaussian_kernel(self):
        """Test sampling with GAUSSIAN kernel."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(kernel_type=KernelType.GAUSSIAN, bandwidth=2.0, dim=512)

        basis = sample_kernel_basis(config, key)

        assert basis.shape == (512,)
        assert jnp.iscomplexobj(basis)
        assert jnp.allclose(jnp.abs(basis), 1.0)

    def test_sample_laplace_kernel(self):
        """Test sampling with LAPLACE kernel."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(kernel_type=KernelType.LAPLACE, bandwidth=1.5, dim=512)

        basis = sample_kernel_basis(config, key)

        assert basis.shape == (512,)
        assert jnp.iscomplexobj(basis)
        assert jnp.allclose(jnp.abs(basis), 1.0)

    def test_sample_custom_kernel(self):
        """Test sampling with CUSTOM kernel."""

        def custom_sampler(key, dim):
            # Simple custom sampler: uniform phases
            phases = jax.random.uniform(key, (dim,), minval=0, maxval=jnp.pi)
            return jnp.exp(1j * phases)

        key = jax.random.PRNGKey(42)
        config = KernelConfig(kernel_type=KernelType.CUSTOM, custom_sampler=custom_sampler, dim=128)

        basis = sample_kernel_basis(config, key)

        assert basis.shape == (128,)
        assert jnp.iscomplexobj(basis)
        assert jnp.allclose(jnp.abs(basis), 1.0)

    def test_sample_different_dimensions(self):
        """Test sampling with different dimensions."""
        key = jax.random.PRNGKey(42)

        for dim in [128, 256, 512, 1024]:
            config = KernelConfig(dim=dim)
            basis = sample_kernel_basis(config, key)
            assert basis.shape == (dim,)


class TestSampleKernelBasisBatch:
    """Test suite for sample_kernel_basis_batch."""

    def test_batch_sampling(self):
        """Test basic batch sampling."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(dim=512)

        bases = sample_kernel_basis_batch(config, key, n_vectors=10)

        assert bases.shape == (10, 512)
        assert jnp.iscomplexobj(bases)

    def test_batch_sampling_all_unit_magnitude(self):
        """Test that all sampled vectors have unit magnitude."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(dim=256)

        bases = sample_kernel_basis_batch(config, key, n_vectors=20)

        magnitudes = jnp.abs(bases)
        assert jnp.allclose(magnitudes, 1.0)

    def test_batch_sampling_vectors_are_different(self):
        """Test that batch-sampled vectors are different from each other."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(dim=512)

        bases = sample_kernel_basis_batch(config, key, n_vectors=5)

        # Check that vectors are not identical
        for i in range(5):
            for j in range(i + 1, 5):
                assert not jnp.allclose(bases[i], bases[j])

    def test_batch_sampling_single_vector(self):
        """Test batch sampling with n_vectors=1."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(dim=512)

        bases = sample_kernel_basis_batch(config, key, n_vectors=1)

        assert bases.shape == (1, 512)

    def test_batch_sampling_large_batch(self):
        """Test batch sampling with large number of vectors."""
        key = jax.random.PRNGKey(42)
        config = KernelConfig(dim=128)

        bases = sample_kernel_basis_batch(config, key, n_vectors=100)

        assert bases.shape == (100, 128)
        assert jnp.all(jnp.isfinite(bases))


class TestGetKernelName:
    """Test suite for get_kernel_name."""

    def test_get_uniform_name(self):
        """Test getting name for UNIFORM kernel."""
        name = get_kernel_name(KernelType.UNIFORM)
        assert name == "Uniform"

    def test_get_gaussian_name(self):
        """Test getting name for GAUSSIAN kernel."""
        name = get_kernel_name(KernelType.GAUSSIAN)
        assert name == "Gaussian"

    def test_get_laplace_name(self):
        """Test getting name for LAPLACE kernel."""
        name = get_kernel_name(KernelType.LAPLACE)
        assert name == "Laplace"

    def test_get_custom_name(self):
        """Test getting name for CUSTOM kernel."""
        name = get_kernel_name(KernelType.CUSTOM)
        assert name == "Custom"

    def test_all_kernel_types_have_names(self):
        """Test that all kernel types have human-readable names."""
        for kernel_type in KernelType:
            name = get_kernel_name(kernel_type)
            assert isinstance(name, str)
            assert len(name) > 0
