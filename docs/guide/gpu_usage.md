# GPU Usage Guide

VSAX is built on JAX, which provides automatic GPU acceleration for all operations. This guide shows you how to leverage GPUs for maximum performance.

## Quick Start

### Check GPU Availability

```python
from vsax.utils import print_device_info, ensure_gpu

# Print detailed device information
print_device_info()

# Check if GPU is available (with warning if not)
ensure_gpu()
```

**Output example:**
```
============================================================
JAX Device Information
============================================================
Default backend: gpu
Device count: 1
GPU available: True

Available devices:
  [0] cuda:0
============================================================
```

## GPU Installation

### Installing JAX with GPU Support

VSAX requires JAX with CUDA support for GPU acceleration:

**CUDA 12:**
```bash
uv add jax[cuda12]
```

**CUDA 11:**
```bash
uv add jax[cuda11]
```

**Verify installation:**
```python
import jax
print(jax.devices())  # Should show: [cuda(id=0)]
```

## Controlling Device Placement

### Automatic (Recommended)

JAX automatically uses GPU if available:

```python
from vsax import create_fhrr_model, VSAMemory

# Automatically uses GPU if available
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)
memory.add("test")

# Check where vectors are stored
from vsax.utils import get_array_device
print(get_array_device(memory["test"].vec))  # cuda:0
```

### Environment Variables

Control device selection before running:

```bash
# Force CPU only
JAX_PLATFORMS=cpu python script.py

# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python script.py

# Use multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python script.py
```

### Programmatic Control

Force specific device in code:

```python
import jax

# Force CPU
with jax.default_device(jax.devices('cpu')[0]):
    model = create_fhrr_model(dim=1024)
    # All operations run on CPU

# Force specific GPU
with jax.default_device(jax.devices('gpu')[0]):
    model = create_fhrr_model(dim=1024)
    # All operations run on GPU 0
```

## Benchmarking Performance

### Single Operation Benchmark

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.utils import benchmark_operation
import jax.numpy as jnp

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)
memory.add_many(["a", "b", "c"])

# Define operation to benchmark
def bind_operation():
    return model.opset.bind(memory["a"].vec, memory["b"].vec)

# Benchmark on GPU
results = benchmark_operation(bind_operation, n_iterations=100)
print(f"Mean time: {results['mean_time']*1000:.2f} ms")
print(f"Throughput: {results['throughput']:.0f} ops/sec")
```

### CPU vs GPU Comparison

```python
from vsax.utils import compare_devices, print_benchmark_results

# Compare devices
results = compare_devices(bind_operation, n_iterations=50)

# Print formatted results
print_benchmark_results(results)
```

**Output example:**
```
============================================================
Benchmark Results
============================================================

CPU:
  Device: cpu:0
  Mean time: 2.45 ms
  Std time: 0.12 ms
  Throughput: 408.16 ops/sec

GPU:
  Device: cuda:0
  Mean time: 0.23 ms
  Std time: 0.01 ms
  Throughput: 4347.83 ops/sec

Speedup: 10.65x (GPU vs CPU)
============================================================
```

## GPU-Optimized Operations

All VSAX operations are GPU-accelerated through JAX:

### FFT Operations (FHRR)

```python
from vsax import create_fhrr_model

model = create_fhrr_model(dim=2048)
# Uses cuFFT on GPU for circular convolution
# 10-100x faster than CPU for large dimensions
```

### Matrix Operations

```python
from vsax.similarity import cosine_similarity
from vsax.utils import vmap_similarity

# Single similarity (uses cuBLAS on GPU)
sim = cosine_similarity(vec1, vec2)

# Batch similarity (parallel on GPU)
similarities = vmap_similarity(query_vec, candidate_vecs)
# GPU processes all candidates in parallel
```

### Batch Processing

```python
from vsax.utils import vmap_bind, vmap_bundle
import jax.numpy as jnp

# Stack vectors for batch processing
vectors_a = jnp.stack([memory[f"a{i}"].vec for i in range(100)])
vectors_b = jnp.stack([memory[f"b{i}"].vec for i in range(100)])

# GPU-accelerated batch binding
bound_vectors = vmap_bind(model.opset, vectors_a, vectors_b)
# All 100 bindings computed in parallel on GPU
```

## Performance Tips

### 1. Use Larger Dimensions

GPUs excel with larger vector dimensions:

```python
# CPU-friendly
small_model = create_fhrr_model(dim=512)   # ~5x speedup

# GPU-friendly
large_model = create_fhrr_model(dim=4096)  # ~20x speedup
```

### 2. Batch Operations

Always prefer batch operations over loops:

**❌ Slow (sequential):**
```python
results = []
for vec in vectors:
    result = model.opset.bind(query, vec)
    results.append(result)
```

**✅ Fast (parallel on GPU):**
```python
results = vmap_bind(model.opset, jnp.broadcast_to(query, (len(vectors), query.shape[0])), vectors)
```

### 3. JIT Compilation

JAX automatically JIT-compiles operations. For custom functions:

```python
import jax

@jax.jit
def custom_operation(a, b, c):
    """Custom VSA operation."""
    bound = model.opset.bind(a, b)
    return model.opset.bundle(bound, c)

# First call compiles (slow)
result1 = custom_operation(vec_a, vec_b, vec_c)

# Subsequent calls use compiled version (fast)
result2 = custom_operation(vec_d, vec_e, vec_f)
```

### 4. Warmup Iterations

First GPU operation includes initialization overhead:

```python
# Warmup
_ = model.opset.bind(memory["a"].vec, memory["b"].vec)

# Now benchmark
results = benchmark_operation(bind_operation)
```

## Monitoring GPU Usage

### In Python

```python
from vsax.utils import get_device_info

info = get_device_info()
if info['gpu_available']:
    print(f"Using GPU: {info['devices'][0]}")
else:
    print("Using CPU")
```

### External Monitoring

Monitor GPU utilization in real-time:

```bash
# NVIDIA GPUs
watch -n 1 nvidia-smi

# Or continuously
nvidia-smi -l 1
```

## Troubleshooting

### GPU Not Detected

**Problem:** `gpu_available: False`

**Solutions:**
1. Install JAX with GPU support: `uv add jax[cuda12]`
2. Check CUDA installation: `nvidia-smi`
3. Verify CUDA version matches JAX version
4. Check `LD_LIBRARY_PATH` includes CUDA libraries

### Out of Memory Errors

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce dimension: `dim=1024` instead of `dim=8192`
2. Reduce batch size
3. Clear JAX cache: `jax.clear_caches()`
4. Use CPU for prototyping: `JAX_PLATFORMS=cpu`

### Slow First Iteration

**Problem:** First operation is very slow

**Explanation:** JAX compiles operations on first use (XLA compilation)

**Solution:** Add warmup iterations:
```python
# Warmup
for _ in range(3):
    _ = operation()

# Now measure
results = benchmark_operation(operation)
```

## Performance Comparison

Typical speedups for common operations (GPU vs CPU):

| Operation | Dimension | CPU Time | GPU Time | Speedup |
|-----------|-----------|----------|----------|---------|
| FHRR Bind | 512 | 0.8 ms | 0.15 ms | 5.3x |
| FHRR Bind | 2048 | 3.2 ms | 0.25 ms | 12.8x |
| FHRR Bind | 8192 | 15.1 ms | 0.45 ms | 33.6x |
| Batch Bind (100) | 1024 | 82 ms | 3.2 ms | 25.6x |
| Similarity (1000) | 1024 | 45 ms | 1.8 ms | 25.0x |

*Benchmarked on: Intel i7-10700K (CPU) vs NVIDIA RTX 3080 (GPU)*

## See Also

- [JAX GPU Installation Guide](https://jax.readthedocs.io/en/latest/installation.html)
- [Batch Operations Guide](batch_operations.md)
- [MNIST Tutorial](../tutorials/01_mnist_classification.md) - Includes GPU benchmarking
- [API Reference: Device Utilities](../api/utils/index.md#device-management)
