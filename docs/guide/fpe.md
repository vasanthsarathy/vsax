# Fractional Power Encoding

**NEW in v1.2.0** - Continuous value encoding using fractional exponentiation.

## Overview

Fractional Power Encoding (FPE) enables encoding continuous real-valued data into hypervectors using **fractional exponentiation**: `v^r` where `v` is a basis hypervector and `r` is a real number.

**Key insight:** FPE transforms discrete symbolic operations into continuous representations, enabling smooth interpolation and spatial reasoning.

| Encoding Type | Example | Use Case |
|--------------|---------|----------|
| **Discrete** | "dog", "cat", "red" | Symbolic AI, discrete concepts |
| **Continuous** | 3.14, [1.5, 2.7, 3.2] | Spatial coordinates, real values, analog signals |

## Why Fractional Power Encoding?

### Problem: Traditional VSA is Discrete

**Without FPE**, encoding continuous values requires discretization:

```python
# Encode temperature 23.7°C
# Must discretize to nearest bin: 20-25°C
temp_bin = memory["temp_20_25"]  # Loses precision!
```

### Solution: FPE Encodes Continuously

**With FPE**, continuous values are encoded smoothly:

```python
from vsax.encoders import FractionalPowerEncoder

encoder = FractionalPowerEncoder(model, memory)

# Encode exact temperature: basis^23.7
temp_237 = encoder.encode("temperature", 23.7)
temp_238 = encoder.encode("temperature", 23.8)

# Small difference in value → small difference in representation
similarity = cosine_similarity(temp_237.vec, temp_238.vec)
# similarity ≈ 0.99 (very high!)
```

### Advantages

✅ **Continuous** - No discretization loss
✅ **Smooth** - Small changes in value → small changes in representation
✅ **Compositional** - `(v^r1)^r2 = v^(r1*r2)`
✅ **Invertible** - `v^r ⊗ v^(-r) = identity`
✅ **Multi-dimensional** - Encode spatial coordinates, color spaces, etc.

## Mathematical Foundation

### How It Works

FPE leverages the **complex exponential representation** of FHRR hypervectors:

```
v = exp(i*θ)  (unit complex numbers)
v^r = exp(i*r*θ)  (phase rotation by r)
```

**Properties:**
- Norm-preserving: `|v^r| = 1` for all r
- Continuous: small Δr → smooth change in output
- Compositional: `(v^r1) ⊗ (v^r2) = v^(r1+r2)` (for circular convolution)

### Why FHRR Only?

FPE **requires ComplexHypervector** because:

1. **Binary hypervectors** - Cannot raise {-1, +1} to fractional powers meaningfully
2. **Real hypervectors** - Fractional powers can produce negative/zero values, breaking norm
3. **Complex hypervectors** - Phase representation allows smooth rotation via `exp(i*r*θ)`

```python
# FPE only works with FHRR models
model = create_fhrr_model(dim=512)  # ✅ ComplexHypervector

# Will raise TypeError with other models:
# model = create_map_model(dim=512)    # ❌ RealHypervector
# model = create_binary_model(dim=512) # ❌ BinaryHypervector
```

## Basic Usage

### Creating an FPE Encoder

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import FractionalPowerEncoder
import jax

model = create_fhrr_model(dim=512, key=jax.random.PRNGKey(0))
memory = VSAMemory(model)

# Create FPE encoder
encoder = FractionalPowerEncoder(model, memory)
```

### Single-Dimension Encoding

```python
# Add basis vector
memory.add("temperature")

# Encode temperature = 23.7
temp_hv = encoder.encode("temperature", 23.7)
print(type(temp_hv))  # ComplexHypervector

# Encode different value
temp_hv2 = encoder.encode("temperature", 25.0)

# Check similarity (closer values → higher similarity)
from vsax.similarity import cosine_similarity
sim = cosine_similarity(temp_hv.vec, temp_hv2.vec)
print(f"Similarity: {sim:.3f}")  # ≈ 0.95
```

### Multi-Dimensional Encoding

Encode multi-dimensional coordinates:

```python
# Add basis vectors for each dimension
memory.add_many(["x", "y", "z"])

# Encode 3D point (1.5, 2.7, 3.2)
point_hv = encoder.encode_multi(
    symbol_names=["x", "y", "z"],
    values=[1.5, 2.7, 3.2]
)

# This computes: X^1.5 ⊗ Y^2.7 ⊗ Z^3.2
```

### Scaling Values

Use the `scale` parameter to normalize inputs:

```python
# Encode values in range [0, 100]
encoder = FractionalPowerEncoder(model, memory, scale=0.1)

# Now large values like 50 become 50*0.1 = 5.0
hv = encoder.encode("value", 50.0)  # Actually encodes basis^5.0
```

## Common Use Cases

### 1. Spatial Coordinates

Encode 2D/3D positions:

```python
# 2D positions
memory.add_many(["x", "y"])

pos1 = encoder.encode_multi(["x", "y"], [3.5, 2.1])
pos2 = encoder.encode_multi(["x", "y"], [3.6, 2.0])

# Nearby positions have high similarity
sim = cosine_similarity(pos1.vec, pos2.vec)
print(f"Spatial similarity: {sim:.3f}")  # High!
```

### 2. Color Representation

Encode colors in HSB space:

```python
# Hue-Saturation-Brightness
memory.add_many(["hue", "sat", "bright"])

# Purple: (hue=6.2, sat=-6.2, bright=5.3)
purple = encoder.encode_multi(
    ["hue", "sat", "bright"],
    [6.2, -6.2, 5.3]
)

# Blue: (hue=4.8, sat=-2.1, bright=4.5)
blue = encoder.encode_multi(
    ["hue", "sat", "bright"],
    [4.8, -2.1, 4.5]
)

# Similar colors → high similarity
sim = cosine_similarity(purple.vec, blue.vec)
```

### 3. Time Series

Encode temporal data:

```python
memory.add("time")

# Encode time points
t1 = encoder.encode("time", 1.5)
t2 = encoder.encode("time", 2.0)
t3 = encoder.encode("time", 2.5)

# Temporal ordering preserved via similarity decay
```

### 4. Sensor Readings

Encode continuous sensor data:

```python
memory.add_many(["sensor1", "sensor2", "sensor3"])

# Multi-sensor reading
reading = encoder.encode_multi(
    ["sensor1", "sensor2", "sensor3"],
    [0.75, 1.23, 0.89]
)
```

## Advanced Features

### Unbinding and Cleanup

Query which value was encoded:

```python
# Encode known value
x_hv = encoder.encode("x", 5.0)

# Later: recover x coordinate by unbinding
# If scene = X^5.0 ⊗ Y^3.0
# Then scene ⊗ Y^(-3.0) ≈ X^5.0

# Use grid search to find best match
from vsax.similarity import cosine_similarity
import jax.numpy as jnp

candidates = jnp.linspace(0, 10, 100)
similarities = []

for val in candidates:
    candidate_hv = encoder.encode("x", val)
    sim = cosine_similarity(x_hv.vec, candidate_hv.vec)
    similarities.append(sim)

# Peak similarity reveals encoded value
best_idx = jnp.argmax(jnp.array(similarities))
recovered = candidates[best_idx]
print(f"Encoded: 5.0, Recovered: {recovered:.2f}")
```

### Negative Values

FPE supports negative exponents:

```python
# Negative values work naturally
neg_hv = encoder.encode("x", -3.5)

# v^(-3.5) = (v^3.5)^(-1) = inverse of v^3.5
pos_hv = encoder.encode("x", 3.5)
inv_hv = model.opset.inverse(pos_hv.vec)

# These should be similar
sim = cosine_similarity(neg_hv.vec, inv_hv)
```

### Fractional Composition

Combine fractional powers algebraically:

```python
# (v^r1) ⊗ (v^r2) = v^(r1+r2) for circular convolution
hv1 = encoder.encode("x", 2.0)
hv2 = encoder.encode("x", 3.0)

# Bind them
combined = model.opset.bind(hv1.vec, hv2.vec)

# Should equal v^5.0
hv5 = encoder.encode("x", 5.0)
sim = cosine_similarity(combined, hv5.vec)
print(f"Composition: {sim:.3f}")  # Very high!
```

## Relationship to ScalarEncoder

FPE is a **specialized version** of ScalarEncoder:

| Feature | ScalarEncoder | FractionalPowerEncoder |
|---------|---------------|------------------------|
| **Single values** | ✅ `basis^value` | ✅ `basis^value` |
| **Multi-dimensional** | ❌ (manual binding) | ✅ `encode_multi()` |
| **Spatial focus** | ❌ | ✅ (designed for SSP, VFA) |
| **Scaling** | ❌ | ✅ Built-in `scale` param |

**When to use:**
- **ScalarEncoder** - Simple continuous encoding, one dimension
- **FractionalPowerEncoder** - Spatial reasoning, multi-dimensional data, SSP/VFA

## Integration with SSP and VFA

FPE is the **foundation** for advanced modules:

### Spatial Semantic Pointers (SSP)

SSP uses FPE for continuous spatial representation:

```python
from vsax.spatial import SpatialSemanticPointers, SSPConfig

# SSP uses FPE internally
config = SSPConfig(dim=512, num_axes=2)  # 2D space
ssp = SpatialSemanticPointers(model, memory, config)

# Encodes locations as: X^x ⊗ Y^y
location = ssp.encode_location([3.5, 2.1])
```

### Vector Function Architecture (VFA)

VFA uses FPE for function encoding:

```python
from vsax.vfa import VectorFunctionEncoder

vfa = VectorFunctionEncoder(model, memory)

# Represents functions as: f(x) = Σ α_i * z^x
# FPE enables the z^x operation
```

## Design Principles

### 1. FHRR-Only by Design

FPE is **intentionally FHRR-only** because phase representation is essential for smooth continuous encoding.

```python
# Type checking enforced
model = create_map_model(512)
encoder = FractionalPowerEncoder(model, memory)
# Raises: TypeError("FractionalPowerEncoder requires ComplexHypervector")
```

### 2. Immutable and Functional

Encoders are immutable; encoding creates new hypervectors:

```python
# Each encode() returns a new ComplexHypervector
hv1 = encoder.encode("x", 1.0)
hv2 = encoder.encode("x", 2.0)
# hv1 unchanged
```

### 3. JAX-Native

All operations are JAX-compatible for GPU acceleration:

```python
import jax

# Can JIT-compile encoding
@jax.jit
def encode_batch(values):
    return jax.vmap(lambda v: encoder.encode("x", v))(values)

values = jnp.array([1.0, 2.0, 3.0])
batch = encode_batch(values)
```

## Performance Considerations

### Computational Cost

**Single encoding:** O(dim) - element-wise exponentiation
**Multi-dimensional:** O(num_axes × dim) - sequential binding

All operations are GPU-accelerated via JAX.

### Precision vs Dimensionality

Higher dimensionality → better precision in recovered values:

| Dimensionality | Typical Recovery Error |
|----------------|----------------------|
| 128 | ±0.2 |
| 512 | ±0.05 |
| 1024 | ±0.02 |
| 2048 | ±0.01 |

## Best Practices

### When to Use FPE

✅ **Use FPE for:**
- Encoding continuous spatial coordinates
- Multi-dimensional real-valued data
- Smooth interpolation between values
- Building Spatial Semantic Pointers
- Vector Function Architecture applications

❌ **Use standard encoders for:**
- Discrete symbolic data (use VSAMemory)
- Binary features (use SetEncoder)
- Categorical data (use DictEncoder)
- Sequences (use SequenceEncoder)

### Choosing Scaling

Scale values to reasonable range (typically -10 to +10):

```python
# Bad: large exponents can cause numerical issues
encoder = FractionalPowerEncoder(model, memory)  # No scaling
hv = encoder.encode("x", 1000.0)  # basis^1000 - unstable!

# Good: scale to [-10, 10]
encoder = FractionalPowerEncoder(model, memory, scale=0.01)
hv = encoder.encode("x", 1000.0)  # basis^10.0 - stable!
```

### Multi-Dimensional Ordering

Order doesn't matter (binding is commutative):

```python
# These are equivalent
hv1 = encoder.encode_multi(["x", "y"], [1.0, 2.0])
hv2 = encoder.encode_multi(["y", "x"], [2.0, 1.0])

sim = cosine_similarity(hv1.vec, hv2.vec)
# sim ≈ 1.0
```

## Limitations

### Current Limitations

1. **FHRR-only** - Cannot use with Binary or Real hypervectors
2. **Numerical stability** - Very large exponents (>100) can cause issues
3. **Approximate decoding** - Grid search required to recover exact values

### Workarounds

**For large values:**
```python
# Use scaling
encoder = FractionalPowerEncoder(model, memory, scale=0.001)
```

**For precise decoding:**
```python
# Use finer grid resolution
candidates = jnp.linspace(min_val, max_val, 1000)  # More points
```

## Related Topics

- **Tutorial 11:** [Analogical Reasoning with Conceptual Spaces](../tutorials/11_analogical_reasoning.md)
- **Guide:** [Spatial Semantic Pointers](spatial.md)
- **Guide:** [Vector Function Architecture](vfa.md)
- **API Reference:** [FractionalPowerEncoder API](../api/encoders/fpe.md)

## References

**Theoretical Foundation:**
- Plate (1995) - "Holographic Reduced Representations"
- Komer et al. (2019) - "A neural representation of continuous space using fractional binding"
- Frady et al. (2021) - "Computing on Functions Using Randomized Vector Representations"

**Applications:**
- Spatial reasoning with SSP
- Function approximation with VFA
- Conceptual spaces (Gärdenfors 2004)
