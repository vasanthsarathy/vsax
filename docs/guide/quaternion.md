# Quaternion Hypervectors

Quaternion Hypervectors (QHV) use quaternion algebra for vector symbolic architectures, providing **non-commutative binding** via the Hamilton product. This makes them ideal for order-sensitive role/filler bindings where the order of arguments matters.

## Overview

| Property | Value |
|---|---|
| **Representation** | Unit quaternions (S³ manifold) |
| **Binding** | Hamilton product (non-commutative) |
| **Storage** | `(D, 4)` where D is dimensionality |
| **Unbinding** | Exact (left and right unbind) |
| **Use Case** | Order-sensitive bindings |

## Key Features

### Non-Commutative Binding

Unlike FHRR and MAP where `bind(x, y) = bind(y, x)`, quaternion binding is **order-sensitive**:

```python
from vsax import create_quaternion_model, VSAMemory, quaternion_similarity
import jax

# Create model and memory
model = create_quaternion_model(dim=512)
memory = VSAMemory(model, key=jax.random.PRNGKey(42))
memory.add_many(["role", "filler"])

role = memory["role"].vec
filler = memory["filler"].vec

# Non-commutative binding
role_filler = model.opset.bind(role, filler)  # role * filler
filler_role = model.opset.bind(filler, role)  # filler * role

# These are DIFFERENT!
similarity = quaternion_similarity(role_filler, filler_role)
print(f"Similarity: {similarity:.3f}")  # Low similarity (~0.0)
```

### Left and Right Unbinding

Because binding is non-commutative, you need two types of unbinding:

**Right-unbind**: Recover `x` from `z = bind(x, y)` given `y`

```python
# z = x * y, recover x: z * y⁻¹ = x
recovered_x = model.opset.unbind(z, y)
```

**Left-unbind**: Recover `y` from `z = bind(x, y)` given `x`

```python
# z = x * y, recover y: x⁻¹ * z = y
recovered_y = model.opset.unbind_left(x, z)
```

### Complete Example

```python
from vsax import create_quaternion_model, VSAMemory, quaternion_similarity
import jax

# Create quaternion model
model = create_quaternion_model(dim=512)
memory = VSAMemory(model, key=jax.random.PRNGKey(42))

# Add symbols
memory.add_many(["subject", "verb", "object", "john", "eats", "apple"])

# Get vectors
subject = memory["subject"].vec
verb = memory["verb"].vec
obj = memory["object"].vec
john = memory["john"].vec
eats = memory["eats"].vec
apple = memory["apple"].vec

# Encode "John eats apple" with role-filler bindings
# Order matters: role * filler
subj_bind = model.opset.bind(subject, john)   # subject → john
verb_bind = model.opset.bind(verb, eats)      # verb → eats
obj_bind = model.opset.bind(obj, apple)       # object → apple

# Bundle into single representation
sentence = model.opset.bundle(subj_bind, verb_bind, obj_bind)

# Query: "Who is the subject?"
# Use left-unbind with subject role to get filler
query = model.opset.unbind_left(subject, sentence)

# Find best match
print(f"john similarity: {quaternion_similarity(query, john):.3f}")
print(f"eats similarity: {quaternion_similarity(query, eats):.3f}")
print(f"apple similarity: {quaternion_similarity(query, apple):.3f}")
# john should have highest similarity
```

## Quaternion Representation

### Storage Format

Quaternion hypervectors are stored as shape `(D, 4)` arrays where:
- `D` is the number of quaternion coordinates (VSA dimensionality)
- `4` is the quaternion components `(a, b, c, d)` for `q = a + bi + cj + dk`

```python
from vsax.representations import QuaternionHypervector

# 512-dimensional quaternion hypervector
# Actual storage: 512 × 4 = 2048 floats
hv = memory["symbol"]
print(f"Shape: {hv.shape}")  # (512, 4)
print(f"Dimensionality: {hv.dim}")  # 512
```

### Unit Quaternions

All quaternions are normalized to unit length, living on the 3-sphere S³:

```python
import jax.numpy as jnp

# Check unit length
norms = jnp.linalg.norm(hv.vec, axis=-1)
print(f"All unit: {jnp.allclose(norms, 1.0)}")  # True
```

### QuaternionHypervector Properties

```python
hv = memory["symbol"]

# Quaternion components
scalar = hv.scalar_part     # Shape (D,) - the 'a' component
vector = hv.vector_part     # Shape (D, 3) - the 'b, c, d' components

# Check if normalized
print(f"Is unit: {hv.is_unit()}")

# Normalize if needed
normalized = hv.normalize()
```

## Operations

### Binding (Hamilton Product)

The Hamilton product is the core binding operation:

```python
ops = model.opset

# Bind two vectors
z = ops.bind(x, y)  # z = x * y

# Properties:
# - NON-COMMUTATIVE: bind(x, y) ≠ bind(y, x)
# - Associative: bind(bind(x, y), z) = bind(x, bind(y, z))
# - Preserves unit length
```

### Unbinding

```python
# Right-unbind: z * y⁻¹
recovered_x = ops.unbind(z, y)

# Left-unbind: x⁻¹ * z
recovered_y = ops.unbind_left(x, z)

# Both achieve >99% similarity recovery
```

### Bundling

Bundling creates a superposition of vectors:

```python
bundled = ops.bundle(x, y, z)

# Properties:
# - Similar to all constituents
# - Normalized to unit quaternions
# - Similarity decreases with more items
```

### Inverse

The quaternion inverse is used for unbinding:

```python
x_inv = ops.inverse(x)

# Property: bind(x, inverse(x)) = identity
identity = ops.bind(x, x_inv)  # (1, 0, 0, 0) for all coordinates
```

## Similarity

Use `quaternion_similarity` for comparing quaternion hypervectors:

```python
from vsax import quaternion_similarity

sim = quaternion_similarity(x, y)
# Returns average dot product across quaternion coordinates
# Range: [-1, 1]
# 1.0 = identical
# 0.0 = orthogonal
# -1.0 = opposite
```

## Sampling

Generate random quaternion hypervectors:

```python
from vsax import sample_quaternion_random
import jax

key = jax.random.PRNGKey(42)
vectors = sample_quaternion_random(dim=512, n=10, key=key)
# Shape: (10, 512, 4)
# All quaternions are unit length (S³ manifold)
```

## When to Use Quaternion VSA

**Use Quaternion when:**
- Order matters in bindings (role/filler, subject/object)
- You need asymmetric relationships
- Exact unbinding from both directions is required
- Working with structured knowledge where position matters

**Consider alternatives when:**
- Order doesn't matter (use FHRR or MAP)
- Memory is constrained (quaternions use 4× more storage)
- You don't need left/right unbind distinction

## Comparison with Other Representations

| Feature | FHRR | MAP | Binary | Quaternion |
|---|---|---|---|---|
| Binding | Commutative | Commutative | Commutative | **Non-commutative** |
| Unbinding | Exact | Approximate | Exact | **Exact (left/right)** |
| Storage | 2×D | 1×D | 1×D | **4×D** |
| Use Case | General | Embeddings | Hardware | **Order-sensitive** |

## API Reference

### Factory Function

```python
from vsax import create_quaternion_model

model = create_quaternion_model(dim=512)
```

### Classes

- `QuaternionHypervector`: Representation class
- `QuaternionOperations`: Operation set with `bind`, `unbind`, `unbind_left`, `bundle`, `inverse`

### Functions

- `sample_quaternion_random(dim, n, key)`: Sample random quaternion hypervectors
- `quaternion_similarity(a, b)`: Compute similarity between quaternion vectors
