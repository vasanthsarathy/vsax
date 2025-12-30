# Binary Model Example

Complete example using Binary VSA with bipolar {-1, +1} hypervectors.

## Setup

```python
import jax
import jax.numpy as jnp
from vsax import VSAModel, BinaryHypervector, BinaryOperations, sample_binary_random

# Create Binary model
model = VSAModel(
    dim=512,
    rep_cls=BinaryHypervector,
    opset=BinaryOperations(),
    sampler=sample_binary_random
)
```

## Basic Operations

```python
# Sample bipolar vectors
key = jax.random.PRNGKey(42)
vectors = model.sampler(dim=model.dim, n=2, key=key, bipolar=True)

a = model.rep_cls(vectors[0], bipolar=True)
b = model.rep_cls(vectors[1], bipolar=True)

# Verify bipolar values
print(f"Unique values: {jnp.unique(a.vec)}")  # Array([-1, 1])

# Bind (XOR)
bound = model.opset.bind(a.vec, b.vec)

# NEW: Explicit unbind method (exact recovery with Binary VSA!)
recovered = model.opset.unbind(bound, b.vec)
print(f"Exact recovery: {jnp.array_equal(recovered, a.vec)}")  # True!

# Note: For Binary VSA, unbind(a,b) = bind(a,b) due to XOR self-inverse property
# Both work, but unbind() is clearer in intent

# Bundle (majority vote)
bundled = model.opset.bundle(a.vec, b.vec)
```

## Symbolic Reasoning Example

Encode logical facts using binary vectors.

```python
# Define symbols
keys = jax.random.split(key, 4)
alice = model.sampler(dim=model.dim, n=1, key=keys[0], bipolar=True)[0]
bob = model.sampler(dim=model.dim, n=1, key=keys[1], bipolar=True)[0]
likes = model.sampler(dim=model.dim, n=1, key=keys[2], bipolar=True)[0]
charlie = model.sampler(dim=model.dim, n=1, key=keys[3], bipolar=True)[0]

# Encode: "Alice likes Bob"
fact = model.opset.bind(model.opset.bind(alice, likes), bob)

# Query: Who does Alice like? (NEW: using unbind)
query = model.opset.unbind(fact, model.opset.bind(alice, likes))
# Exact match to bob (Binary VSA provides perfect unbinding!)
```

**Advantage:** Binary VSA provides exact unbinding and is hardware-friendly!
