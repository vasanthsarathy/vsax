"""Tests for device management utilities."""

import jax.numpy as jnp
import pytest

from vsax.utils.device import (
    benchmark_operation,
    compare_devices,
    ensure_gpu,
    get_array_device,
    get_device_info,
    print_benchmark_results,
    print_device_info,
)


def test_get_device_info() -> None:
    """Test getting device information."""
    info = get_device_info()

    assert "devices" in info
    assert "default_backend" in info
    assert "device_count" in info
    assert "gpu_available" in info

    assert isinstance(info["devices"], list)
    assert len(info["devices"]) > 0
    assert isinstance(info["default_backend"], str)
    assert isinstance(info["device_count"], int)
    assert isinstance(info["gpu_available"], bool)


def test_print_device_info() -> None:
    """Test printing device information."""
    # Should not raise any errors
    print_device_info()


def test_ensure_gpu() -> None:
    """Test GPU availability check."""
    result = ensure_gpu()
    assert isinstance(result, bool)


def test_get_array_device() -> None:
    """Test getting array device."""
    arr = jnp.array([1.0, 2.0, 3.0])
    device = get_array_device(arr)

    assert isinstance(device, str)
    # Should contain either 'cpu' or 'cuda' or 'gpu'
    assert any(x in device.lower() for x in ["cpu", "cuda", "gpu", "metal"])


def test_benchmark_operation() -> None:
    """Test benchmarking an operation."""

    def simple_operation() -> jnp.ndarray:
        """Simple matrix multiplication."""
        x = jnp.ones((100, 100))
        return jnp.dot(x, x)

    # Benchmark on default device
    results = benchmark_operation(simple_operation, n_iterations=5, warmup=2)

    assert "mean_time" in results
    assert "std_time" in results
    assert "device" in results
    assert "throughput" in results

    assert results["mean_time"] > 0
    assert results["std_time"] >= 0
    assert results["throughput"] > 0


def test_benchmark_operation_cpu() -> None:
    """Test benchmarking on CPU explicitly."""

    def simple_operation() -> jnp.ndarray:
        return jnp.ones((50, 50))

    results = benchmark_operation(simple_operation, n_iterations=3, device="cpu")

    assert "cpu" in results["device"].lower()
    assert results["mean_time"] > 0


def test_benchmark_operation_invalid_device() -> None:
    """Test benchmarking with invalid device."""

    def simple_operation() -> jnp.ndarray:
        return jnp.ones((10, 10))

    with pytest.raises(ValueError, match="Unknown device"):
        benchmark_operation(simple_operation, device="invalid")


def test_compare_devices() -> None:
    """Test comparing devices."""

    def simple_operation() -> jnp.ndarray:
        x = jnp.ones((100, 100))
        return jnp.dot(x, x)

    results = compare_devices(simple_operation, n_iterations=3)

    assert "cpu" in results
    assert results["cpu"]["mean_time"] > 0

    # If GPU available, should have GPU results and speedup
    info = get_device_info()
    if info["gpu_available"]:
        assert "gpu" in results
        assert "speedup" in results
        assert results["speedup"] > 0


def test_print_benchmark_results() -> None:
    """Test printing benchmark results."""
    results = {
        "cpu": {
            "device": "cpu:0",
            "mean_time": 0.001,
            "std_time": 0.0001,
            "throughput": 1000.0,
        },
    }

    # Should not raise any errors
    print_benchmark_results(results)


def test_print_benchmark_results_with_speedup() -> None:
    """Test printing benchmark results with GPU speedup."""
    results = {
        "cpu": {
            "device": "cpu:0",
            "mean_time": 0.01,
            "std_time": 0.001,
            "throughput": 100.0,
        },
        "gpu": {
            "device": "cuda:0",
            "mean_time": 0.001,
            "std_time": 0.0001,
            "throughput": 1000.0,
        },
        "speedup": 10.0,
    }

    # Should not raise any errors
    print_benchmark_results(results)
