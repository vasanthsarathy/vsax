# Hypervector Representations

VSAX provides three hypervector representations, each designed for a specific VSA algebra. All representations inherit from `AbstractHypervector` and provide a consistent interface.

## Overview

| Representation | Values | Use Case | Operations |
|---|---|---|---|
| `ComplexHypervector` | Complex unit-magnitude | FHRR (Fourier) | Circular convolution |
| `RealHypervector` | Real continuous | MAP | Element-wise multiply |
| `BinaryHypervector` | Bipolar {-1,+1} or Binary {0,1} | Binary VSA | XOR, majority vote |

## ComplexHypervector

Phase-based representation using complex numbers for FHRR (Fourier Holographic Reduced Representation).

### Features

- **Unit magnitude**: All elements have magnitude 1.0
- **Phase encoding**: Information stored in phase (angle)
- **Exact unbinding**: Circular convolution is invertible via conjugate
- **GPU-friendly**: Leverages JAX's complex number support

### Example

```python
import jax
import jax.numpy as jnp
from vsax import ComplexHypervector, sample_complex_random

# Sample a complex vector
key = jax.random.PRNGKey(42)
vec = sample_complex_random(dim=512, n=1, key=key)[0]

# Create hypervector
hv = ComplexHypervector(vec)

# Normalize to unit magnitude (phase-only)
normalized = hv.normalize()

# Access properties
print(f"Phase: {hv.phase}")           # Angles in [-π, π]
print(f"Magnitude: {hv.magnitude}")    # All should be ~1.0
print(f"Shape: {hv.shape}")            # (512,)
```

### Properties

- `phase`: Extract phase component (angles)
- `magnitude`: Extract magnitude component
- `vec`: Underlying JAX array
- `shape`: Vector shape
- `dtype`: Data type (complex64 or complex128)

### Methods

- `normalize()`: Normalize to unit magnitude (phase-only representation)
- `to_numpy()`: Convert to NumPy array

## RealHypervector

Continuous real-valued representation for MAP (Multiply-Add-Permute) operations.

### Features

- **L2 normalization**: Vectors normalized to unit length
- **Continuous values**: Real-valued elements
- **Approximate unbinding**: MAP unbinding is approximate, not exact
- **Simple operations**: Element-wise multiplication and mean

### Example

```python
from vsax import RealHypervector, sample_random

# Sample a real vector
key = jax.random.PRNGKey(42)
vec = sample_random(dim=512, n=1, key=key)[0]

# Create hypervector
hv = RealHypervector(vec)

# L2 normalize
normalized = hv.normalize()

# Properties
print(f"L2 norm: {jnp.linalg.norm(normalized.vec)}")  # Should be 1.0
print(f"Is complex: {jnp.iscomplexobj(hv.vec)}")      # False
```

### Methods

- `normalize()`: L2 normalization to unit length
- `to_numpy()`: Convert to NumPy array

## BinaryHypervector

Discrete binary representation for Binary VSA with XOR binding.

### Features

- **Exact unbinding**: XOR is self-inverse
- **Two modes**: Bipolar {-1, +1} or Binary {0, 1}
- **Hardware-friendly**: Efficient for digital hardware
- **Majority voting**: Robust bundling via majority vote

### Example

```python
from vsax import BinaryHypervector, sample_binary_random

# Sample bipolar vectors
key = jax.random.PRNGKey(42)
vec = sample_binary_random(dim=512, n=1, key=key, bipolar=True)[0]

# Create bipolar hypervector
hv = BinaryHypervector(vec, bipolar=True)

# Check mode
print(f"Is bipolar: {hv.bipolar}")  # True

# Convert between representations
binary_hv = hv.to_binary()      # Convert to {0, 1}
bipolar_hv = binary_hv.to_bipolar()  # Convert back to {-1, +1}

# Verify values
print(f"Values: {jnp.unique(hv.vec)}")  # Array([-1, 1])
```

### Conversion

```python
# Bipolar {-1, +1} to Binary {0, 1}
# Formula: (x + 1) / 2
# Example: -1 → 0, +1 → 1

# Binary {0, 1} to Bipolar {-1, +1}
# Formula: 2*x - 1
# Example: 0 → -1, 1 → +1
```

### Properties

- `bipolar`: Check if using bipolar encoding
- `vec`: Underlying JAX array
- `shape`: Vector shape
- `dtype`: Data type (typically int32)

### Methods

- `normalize()`: No-op for binary (already normalized)
- `to_bipolar()`: Convert to {-1, +1} representation
- `to_binary()`: Convert to {0, 1} representation
- `to_numpy()`: Convert to NumPy array

## Common Interface

All representations share a common interface via `AbstractHypervector`:

```python
class AbstractHypervector:
    @property
    def vec(self) -> jnp.ndarray:
        """Access underlying JAX array"""

    @property
    def shape(self) -> tuple[int, ...]:
        """Vector shape"""

    @property
    def dtype(self):
        """Data type"""

    def normalize(self) -> "AbstractHypervector":
        """Normalize the hypervector"""

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array"""
```

## Choosing a Representation

**Use ComplexHypervector when:**
- You need exact unbinding
- Working with sequences or structured data
- GPU acceleration is available
- Circular convolution is suitable for your task

**Use RealHypervector when:**
- You have continuous-valued data
- Approximate unbinding is acceptable
- Simple operations are preferred
- Working with embeddings or features

**Use BinaryHypervector when:**
- Deploying to hardware (FPGA, ASIC)
- Memory constraints are tight
- You need exact unbinding
- Working with symbolic/discrete data

## Performance Considerations

| Representation | Memory | Computation | Unbinding |
|---|---|---|---|
| Complex | 2x (real+imag) | FFT overhead | Exact |
| Real | 1x | Fast multiply/add | Approximate |
| Binary | 1/32x (int vs float) | Fastest | Exact |

## Next Steps

- Learn about [Operations](operations.md) for each representation
- See [Examples](../examples/fhrr.md) for complete workflows
- Check [API Reference](../api/representations/complex.md) for detailed docs
