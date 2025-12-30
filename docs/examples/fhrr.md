# FHRR Model Example

Complete example using the FHRR (Fourier Holographic Reduced Representation) model with complex-valued hypervectors.

## Setup

```python
import jax
import jax.numpy as jnp
from vsax import VSAModel, ComplexHypervector, FHRROperations, sample_fhrr_random

# Create FHRR model
model = VSAModel(
    dim=512,
    rep_cls=ComplexHypervector,
    opset=FHRROperations(),
    sampler=sample_fhrr_random  # Recommended: ensures >99% unbinding accuracy
)

# Alternative: Use sample_complex_random for general complex vectors (~70% unbinding)
```

## Basic Operations

### Sampling and Normalization

```python
# Sample basis vectors
key = jax.random.PRNGKey(42)
vectors = model.sampler(dim=model.dim, n=3, key=key)

# Create and normalize hypervectors
a = model.rep_cls(vectors[0]).normalize()
b = model.rep_cls(vectors[1]).normalize()
c = model.rep_cls(vectors[2]).normalize()

# Verify unit magnitude
print(f"Magnitude of a: {jnp.allclose(jnp.abs(a.vec), 1.0)}")  # True
```

### Binding (Circular Convolution)

```python
# Bind two vectors
bound = model.opset.bind(a.vec, b.vec)
bound_hv = model.rep_cls(bound).normalize()

print(f"Bound vector shape: {bound_hv.shape}")
print(f"Is complex: {jnp.iscomplexobj(bound_hv.vec)}")
```

### Unbinding (Exact Recovery)

```python
# NEW: Explicit unbind method (recommended)
recovered = model.opset.unbind(bound_hv.vec, b.vec)
recovered_hv = model.rep_cls(recovered).normalize()

# Check similarity (should be very high with FHRR sampling)
similarity = jnp.abs(jnp.vdot(a.vec, recovered_hv.vec)) / model.dim
print(f"Recovery similarity: {similarity:.4f}")  # >0.99 with sample_fhrr_random!

# Alternative: Using inverse (equivalent but less clear)
# inv_b = model.opset.inverse(b.vec)
# recovered = model.opset.bind(bound_hv.vec, inv_b)
```

### Bundling (Superposition)

```python
# Bundle multiple vectors
bundled = model.opset.bundle(a.vec, b.vec, c.vec)
bundled_hv = model.rep_cls(bundled)

# Result has unit magnitude
print(f"Bundled magnitude: {jnp.allclose(jnp.abs(bundled_hv.vec), 1.0)}")  # True
```

## Role-Filler Binding

Encode structured data using role-filler binding.

```python
# Define roles and fillers
key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 6)

# Roles
subject_role = model.sampler(dim=model.dim, n=1, key=keys[0])[0]
verb_role = model.sampler(dim=model.dim, n=1, key=keys[1])[0]
object_role = model.sampler(dim=model.dim, n=1, key=keys[2])[0]

# Fillers (concepts)
dog = model.sampler(dim=model.dim, n=1, key=keys[3])[0]
chase = model.sampler(dim=model.dim, n=1, key=keys[4])[0]
cat = model.sampler(dim=model.dim, n=1, key=keys[5])[0]

# Encode sentence: "The dog chased the cat"
sentence = model.opset.bundle(
    model.opset.bind(subject_role, dog),
    model.opset.bind(verb_role, chase),
    model.opset.bind(object_role, cat)
)

# Query: What is the subject? (NEW: using unbind)
query = model.opset.unbind(sentence, subject_role)
query_hv = model.rep_cls(query).normalize()
dog_hv = model.rep_cls(dog).normalize()

# Similarity to "dog" should be very high
similarity = jnp.abs(jnp.vdot(query_hv.vec, dog_hv.vec)) / model.dim
print(f"Subject query similarity to 'dog': {similarity:.4f}")  # >0.99 with FHRR!
```

## Sequence Encoding

Use permutation for positional information.

```python
# Encode sequence: [A, B, C]
sequence_keys = jax.random.split(key, 3)
A = model.sampler(dim=model.dim, n=1, key=sequence_keys[0])[0]
B = model.sampler(dim=model.dim, n=1, key=sequence_keys[1])[0]
C = model.sampler(dim=model.dim, n=1, key=sequence_keys[2])[0]

# Encode with positional information
sequence = model.opset.bundle(
    A,                              # Position 0
    model.opset.permute(B, 1),      # Position 1
    model.opset.permute(C, 2)       # Position 2
)

# Decode position 1
pos1_query = model.opset.permute(sequence, -1)
# High similarity to B
```

## Next Steps

- See [MAP Example](map.md) for real-valued operations
- See [Binary Example](binary.md) for discrete operations
- Check [API Reference](../api/ops/fhrr.md) for detailed documentation
