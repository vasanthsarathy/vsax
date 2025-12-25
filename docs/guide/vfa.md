# Vector Function Architecture

**NEW in v1.2.0** - Function approximation in Reproducing Kernel Hilbert Space (RKHS).

## Overview

Vector Function Architecture (VFA) enables **encoding and manipulating functions** using hypervectors. Based on Frady et al. (2021), VFA represents functions `f(x)` in a Reproducing Kernel Hilbert Space, allowing function evaluation, arithmetic, shifting, and convolution using vector symbolic operations.

**Key insight:** Functions become first-class citizens in VSA - encode, query, combine, and transform them just like symbolic concepts.

| Capability | Example |
|------------|---------|
| **Encode function** | Fit sin(x) from samples |
| **Evaluate** | f(x_query) for any x |
| **Add functions** | h = f + g |
| **Shift** | f(x - shift) |
| **Convolve** | f * g |

## Why Vector Function Architecture?

### Problem: Functions Are Not Symbolic

**Without VFA**, functions are opaque:

```python
# Traditional function representation
def my_function(x):
    return jnp.sin(x) + 0.5 * jnp.cos(2*x)

# Cannot:
# - Bind function to a symbol
# - Compare function similarity
# - Compose functions symbolically
# - Store function in VSA memory
```

### Solution: VFA Vectorizes Functions

**With VFA**, functions become hypervectors:

```python
from vsax.vfa import VectorFunctionEncoder
import jax.numpy as jnp

vfa = VectorFunctionEncoder(model, memory)

# Sample function
x = jnp.linspace(0, 2*jnp.pi, 50)
y = jnp.sin(x)

# Encode as hypervector
f_hv = vfa.encode_function_1d(x, y)

# Evaluate at any point
y_pred = vfa.evaluate_1d(f_hv, 1.5)

# Combine functions
g_hv = vfa.encode_function_1d(x, jnp.cos(x))
h_hv = vfa.add_functions(f_hv, g_hv)  # h = f + g
```

### Advantages

✅ **Symbolic** - Functions are hypervectors (bind, bundle, compare)
✅ **Approximate** - Learn from samples, generalize to unseen points
✅ **Composable** - Add, shift, convolve functions algebraically
✅ **Memory-efficient** - Fixed-size representation regardless of complexity
✅ **GPU-accelerated** - JAX-native for fast evaluation

## Mathematical Foundation

### RKHS Representation

VFA represents functions in a **Reproducing Kernel Hilbert Space**:

```
f(x) ≈ Σ α_i * K(z, x)
```

For VSAX/VFA:
```
f(x) ≈ <α, z^x>
```

Where:
- `α` is the coefficient hypervector (learned from samples)
- `z` is a basis hypervector (randomly sampled)
- `z^x` means "raise z to power x" (Fractional Power Encoding)
- `<·,·>` is inner product

**Key properties:**
- **Kernel:** `K(z, x) = z^x` (fractional power kernel)
- **Linear in α:** Function space is a vector space
- **Continuous:** Small Δx → smooth change in z^x

### Learning Coefficients

Given samples `(x_1, y_1), ..., (x_n, y_n)`, solve for `α`:

```
Z * α = y
```

Where `Z[i, j] = (z_j)^(x_i)` is the design matrix.

Uses **regularized least squares**:
```
α = (Z^H Z + λI)^(-1) Z^H y
```

### Evaluation

To evaluate `f(x_query)`:

```
f(x_query) = <α, z^x_query> = Σ α_i * (z_i)^x_query
```

### Why FHRR Only?

VFA **requires ComplexHypervector (FHRR)** because:
1. Fractional powers (`z^x`) only work with complex phase representation
2. Inner product must be well-defined over complex numbers
3. Kernel smoothness requires continuous phase encoding

## Basic Usage

### Creating VFA Encoder

```python
import jax
import jax.numpy as jnp
from vsax import create_fhrr_model, VSAMemory
from vsax.vfa import VectorFunctionEncoder

# Create FHRR model
model = create_fhrr_model(dim=512, key=jax.random.PRNGKey(0))
memory = VSAMemory(model)

# Create VFA encoder
vfa = VectorFunctionEncoder(model, memory)
```

### Encoding 1D Functions

```python
# Sample a function
x_train = jnp.linspace(0, 2*jnp.pi, 30)
y_train = jnp.sin(x_train)

# Encode function
f_hv = vfa.encode_function_1d(x_train, y_train)
print(type(f_hv))  # ComplexHypervector

# Evaluate at test points
x_test = jnp.linspace(0, 2*jnp.pi, 100)
y_pred = vfa.evaluate_batch(f_hv, x_test)

# Compare with true function
y_true = jnp.sin(x_test)
error = jnp.mean((y_pred - y_true)**2)
print(f"MSE: {error:.6f}")
```

### Function Arithmetic

```python
# Encode two functions
x = jnp.linspace(0, 2*jnp.pi, 50)

f_hv = vfa.encode_function_1d(x, jnp.sin(x))
g_hv = vfa.encode_function_1d(x, jnp.cos(x))

# Add functions: h = f + g
h_hv = vfa.add_functions(f_hv, g_hv)

# Evaluate h
y_h = vfa.evaluate_1d(h_hv, 1.0)
y_expected = jnp.sin(1.0) + jnp.cos(1.0)
print(f"h(1.0) = {y_h:.3f}, expected = {y_expected:.3f}")

# Linear combination: h = 2*f - 0.5*g
h_hv = vfa.add_functions(f_hv, g_hv, alpha=2.0, beta=-0.5)
```

### Function Shifting

```python
# Encode sin(x)
x = jnp.linspace(0, 2*jnp.pi, 50)
f_hv = vfa.encode_function_1d(x, jnp.sin(x))

# Shift by π/2: sin(x - π/2) = -cos(x)
shift = jnp.pi / 2
f_shifted = vfa.shift_function(f_hv, shift)

# Evaluate
y_shifted = vfa.evaluate_1d(f_shifted, 1.0)
y_expected = jnp.sin(1.0 - jnp.pi/2)
print(f"Shifted: {y_shifted:.3f}, expected: {y_expected:.3f}")
```

### Function Convolution

```python
# Convolve two functions
f_hv = vfa.encode_function_1d(x, jnp.sin(x))
g_hv = vfa.encode_function_1d(x, jnp.exp(-x))

# Approximate convolution
conv_hv = vfa.convolve_functions(f_hv, g_hv)

# Note: This is an approximation using FHRR binding
```

## Common Use Cases

### 1. Function Approximation

Learn arbitrary functions from data:

```python
# Arbitrary nonlinear function
def target_func(x):
    return jnp.sin(x) + 0.3 * jnp.cos(3*x) - 0.1 * x**2

# Sample training data
x_train = jnp.linspace(-5, 5, 40)
y_train = target_func(x_train)

# Encode function
f_hv = vfa.encode_function_1d(x_train, y_train)

# Test generalization
x_test = jnp.linspace(-5, 5, 200)
y_pred = vfa.evaluate_batch(f_hv, x_test)
y_true = target_func(x_test)

# Measure error
mse = jnp.mean((y_pred - y_true)**2)
print(f"MSE: {mse:.6f}")
```

### 2. Density Estimation

Estimate probability density from samples:

```python
from vsax.vfa.applications import DensityEstimator

# Create estimator
density_est = DensityEstimator(model, memory)

# Sample from distribution (e.g., mixture of Gaussians)
samples = jnp.concatenate([
    jax.random.normal(key1, (100,)) * 0.5 + 2.0,
    jax.random.normal(key2, (100,)) * 0.5 - 1.0
])

# Fit density
density_est.fit(samples, bandwidth=0.5)

# Evaluate density at query points
x_query = jnp.linspace(-5, 5, 100)
density = density_est.evaluate_batch(x_query)

# Plot
import matplotlib.pyplot as plt
plt.plot(x_query, density)
plt.hist(samples, bins=30, density=True, alpha=0.5)
plt.show()
```

### 3. Nonlinear Regression

Fit regression models:

```python
from vsax.vfa.applications import NonlinearRegressor

# Create regressor
regressor = NonlinearRegressor(model, memory)

# Generate noisy data
x_train = jnp.linspace(0, 10, 50)
y_train = jnp.sin(x_train) + 0.1 * jax.random.normal(key, (50,))

# Fit model
regressor.fit(x_train, y_train, regularization=1e-3)

# Predict
x_test = jnp.linspace(0, 10, 200)
y_pred = regressor.predict_batch(x_test)

# Visualize
plt.scatter(x_train, y_train, label="Training data", alpha=0.5)
plt.plot(x_test, y_pred, label="VFA fit", color="red")
plt.legend()
plt.show()
```

### 4. Image Processing

Encode and manipulate images as 2D functions:

```python
from vsax.vfa.applications import ImageProcessor

# Create processor
img_proc = ImageProcessor(model, memory)

# Load grayscale image (e.g., 28x28 MNIST digit)
import numpy as np
image = np.random.rand(28, 28)  # Placeholder

# Encode image
img_hv = img_proc.encode(image)

# Decode image
reconstructed = img_proc.decode(img_hv, shape=(28, 28))

# Compute reconstruction error
error = np.mean((image - reconstructed)**2)
print(f"Reconstruction MSE: {error:.6f}")
```

## Advanced Features

### Kernel Configuration

Control the RKHS kernel:

```python
from vsax.vfa import VectorFunctionEncoder, KernelConfig, KernelType

# Uniform kernel (standard FHRR)
kernel_config = KernelConfig(
    dim=512,
    kernel_type=KernelType.UNIFORM,
    bandwidth=1.0
)

vfa = VectorFunctionEncoder(
    model, memory,
    kernel_config=kernel_config,
    basis_key=jax.random.PRNGKey(42)
)

# Future: Gaussian, Laplace kernels
# kernel_config = KernelConfig(kernel_type=KernelType.GAUSSIAN)
```

### Regularization

Control overfitting with regularization parameter:

```python
# Light regularization (fit training data closely)
f_hv = vfa.encode_function_1d(x, y, regularization=1e-8)

# Strong regularization (smoother fit, less overfitting)
f_hv = vfa.encode_function_1d(x, y, regularization=1e-3)
```

### Batch Evaluation

Efficiently evaluate at multiple points:

```python
# Single evaluation (slower)
y_vals = [vfa.evaluate_1d(f_hv, x) for x in x_test]

# Batch evaluation (faster)
y_vals = vfa.evaluate_batch(f_hv, x_test)
```

### Reproducibility

Use explicit basis key for reproducible encoding:

```python
# Same basis = same encoding
key = jax.random.PRNGKey(42)
vfa1 = VectorFunctionEncoder(model, memory, basis_key=key)
vfa2 = VectorFunctionEncoder(model, memory, basis_key=key)

# Encodings will be identical
f1 = vfa1.encode_function_1d(x, y)
f2 = vfa2.encode_function_1d(x, y)

similarity = cosine_similarity(f1.vec, f2.vec)
# similarity ≈ 1.0
```

## VFA vs Traditional Methods

| Aspect | VFA | Neural Networks | Gaussian Processes |
|--------|-----|-----------------|-------------------|
| **Representation** | Fixed-size hypervector | Variable weights | Kernel matrix |
| **Training** | Closed-form (least squares) | Iterative (gradient descent) | Closed-form |
| **Inference** | O(dim) | O(width × depth) | O(n²) or O(n³) |
| **Memory** | O(dim) | O(params) | O(n²) |
| **Symbolic ops** | ✅ (add, shift, bind) | ❌ | Limited |
| **Interpretability** | Medium | Low | High |

**When to use VFA:**
- Function comparison and similarity
- Symbolic function manipulation
- Memory-constrained applications
- Rapid prototyping and exploration

## Configuration Options

### VectorFunctionEncoder

```python
VectorFunctionEncoder(
    model,               # VSAModel (must be FHRR)
    memory,             # VSAMemory
    kernel_config=None, # KernelConfig (optional)
    basis_key=None      # JAX PRNGKey for reproducibility
)
```

### KernelConfig

```python
from vsax.vfa import KernelConfig, KernelType

KernelConfig(
    dim=512,                        # Hypervector dimension
    kernel_type=KernelType.UNIFORM, # Kernel type
    bandwidth=1.0                   # Kernel bandwidth (future)
)
```

**Available kernel types:**
- `KernelType.UNIFORM` - Standard FHRR (current)
- `KernelType.GAUSSIAN` - Concentrated frequencies (future)
- `KernelType.LAPLACE` - Exponential decay (future)

## Performance Considerations

### Encoding Cost

**1D encoding:** O(n × dim) where n = number of samples

Dominated by solving linear system `(Z^H Z + λI) α = Z^H y`:
- Dense solve: O(dim³)
- Can be optimized with iterative solvers for large dim

### Evaluation Cost

**Single point:** O(dim) - inner product
**Batch (m points):** O(m × dim)

All operations are GPU-accelerated via JAX.

### Approximation Quality

| Samples | Dimensionality | Typical Error |
|---------|----------------|---------------|
| 20 | 512 | ±0.1 |
| 50 | 512 | ±0.05 |
| 100 | 512 | ±0.02 |
| 50 | 1024 | ±0.01 |

More samples and higher dimensionality → better approximation.

### Memory Usage

**Per function:** O(dim) complex floats
- 512-dim: ~4 KB per function
- 1024-dim: ~8 KB per function

Can store thousands of functions in memory.

## Design Principles

### 1. FPE Foundation

VFA builds on FractionalPowerEncoder:

```python
# VFA uses fractional powers for kernel evaluation
query_vec = jnp.power(self.basis_vector, x_query)
```

### 2. RKHS Theory

VFA follows Reproducing Kernel Hilbert Space theory:
- Functions are linear combinations of kernel evaluations
- Inner product defines function evaluation
- Kernel is `K(z, x) = z^x`

### 3. FHRR-Only

VFA requires ComplexHypervector for:
- Fractional power operations
- Complex inner products
- Smooth kernel representation

```python
# Type checking enforced
if model.rep_cls != ComplexHypervector:
    raise TypeError("VFA requires ComplexHypervector (FHRR) model")
```

### 4. Immutable

All operations return new hypervectors; originals unchanged.

## Best Practices

### When to Use VFA

✅ **Use VFA for:**
- Function approximation from samples
- Symbolic function manipulation
- Density estimation
- Nonlinear regression
- Treating functions as first-class symbolic objects

❌ **Use other approaches for:**
- High-precision requirements → Neural networks or GPs
- Discrete functions → Standard VSA encoding
- Tabular data → DictEncoder, SetEncoder

### Choosing Regularization

Balance fitting vs smoothness:

```python
# Noisy data: use stronger regularization
vfa.encode_function_1d(x_noisy, y_noisy, regularization=1e-2)

# Clean data: use light regularization
vfa.encode_function_1d(x_clean, y_clean, regularization=1e-6)
```

### Sample Density

More samples in regions of high variation:

```python
# Adaptive sampling for sin(x)
x_dense = jnp.concatenate([
    jnp.linspace(0, jnp.pi/2, 20),  # Rising edge
    jnp.linspace(jnp.pi/2, 3*jnp.pi/2, 10),  # Flat regions
    jnp.linspace(3*jnp.pi/2, 2*jnp.pi, 20)  # Falling edge
])
y = jnp.sin(x_dense)
f_hv = vfa.encode_function_1d(x_dense, y)
```

### Dimensionality

Higher dimensions → better approximation but slower:

- **512**: Good for simple smooth functions
- **1024**: Better for complex or noisy functions
- **2048**: High-precision requirements

## Limitations

### Current Limitations

1. **FHRR-only** - Cannot use Binary or Real hypervectors
2. **1D only** - Multi-dimensional function encoding not yet implemented
3. **Closed-form learning** - No iterative refinement or online learning
4. **Uniform kernel only** - Gaussian/Laplace kernels planned for future

### Future Extensions

Planned features:
- Multi-dimensional functions: `f(x, y, z)`
- Learned kernels (adaptive bandwidth)
- Online/incremental learning
- Symbolic derivatives
- Function composition beyond linear combinations

## Related Topics

- **Tutorial 11:** [Analogical Reasoning with Conceptual Spaces](../tutorials/11_analogical_reasoning.md)
- **Guide:** [Fractional Power Encoding](fpe.md)
- **Guide:** [Spatial Semantic Pointers](spatial.md)
- **API Reference:** [VFA API](../api/vfa/index.md)
- **Examples:** [Density Estimation](../../examples/vfa/density_estimation.py), [Regression](../../examples/vfa/nonlinear_regression.py), [Image Processing](../../examples/vfa/image_processing.py)

## References

**Theoretical Foundation:**
- Frady et al. (2021) - "Computing on Functions Using Randomized Vector Representations"
- Plate (1995) - "Holographic Reduced Representations"
- Kanerva (2009) - "Hyperdimensional Computing"

**RKHS Theory:**
- Aronszajn (1950) - "Theory of Reproducing Kernels"
- Schölkopf & Smola (2002) - "Learning with Kernels"

**Applications:**
- Density estimation (Frady 2021 §7.2.1)
- Nonlinear regression (Frady 2021 §7.2.2)
- Image processing (Frady 2021 §7.1)
