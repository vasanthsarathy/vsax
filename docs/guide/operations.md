# VSA Operations

VSA operations define how hypervectors are combined and manipulated. VSAX provides three operation sets, each corresponding to a representation type.

## Overview

All operation sets implement the `AbstractOpSet` interface with four core operations:

| Operation | Purpose | Example |
|---|---|---|
| `bind(a, b)` | Combine/associate two vectors | Role-filler binding |
| `bundle(*vecs)` | Superposition of multiple vectors | Create composite representations |
| `inverse(a)` | Compute inverse for unbinding | Retrieve bound information |
| `permute(a, shift)` | Circular shift/rotation | Sequential encoding |

## FHRROperations

Operations for complex-valued hypervectors using FFT-based circular convolution.

### Binding (Circular Convolution)

Binds two complex vectors using circular convolution implemented via FFT.

```python
from vsax import FHRROperations
import jax.numpy as jnp

ops = FHRROperations()

# Create unit-magnitude complex vectors
a = jnp.exp(1j * jnp.array([0.1, 0.5, 1.0, 1.5]))
b = jnp.exp(1j * jnp.array([0.2, 0.6, 1.1, 1.6]))

# Bind via circular convolution
bound = ops.bind(a, b)

# Result is also complex
assert jnp.iscomplexobj(bound)
```

**Properties:**
- Commutative: `bind(a, b) = bind(b, a)`
- Associative: `bind(a, bind(b, c)) = bind(bind(a, b), c)`
- Invertible: Can recover `a` from `bind(a, b)` using `inverse(b)`

### Bundling (Sum and Normalize)

Bundles multiple vectors by summing and normalizing to unit magnitude.

```python
# Bundle three vectors
bundled = ops.bundle(a, b, c)

# All elements have unit magnitude
assert jnp.allclose(jnp.abs(bundled), 1.0)
```

**Properties:**
- Similarity preserving: Bundled vector is similar to constituents
- Approximate: Some information loss occurs
- Commutative: Order doesn't matter

### Inverse (Frequency-Domain Conjugate)

For FHRR circular convolution, the inverse uses **frequency-domain conjugate**:
`inverse(b) = ifft(conj(fft(b)))`

This ensures high-accuracy unbinding (>99% with proper FHRR vectors).

```python
# Method 1: Explicit unbind (RECOMMENDED)
recovered = ops.unbind(bound, b)

# Method 2: Using inverse (equivalent)
inv_b = ops.inverse(b)
recovered = ops.bind(bound, inv_b)

# With proper FHRR vectors (sample_fhrr_random):
# recovered ≈ a with >99% similarity!
```

**Note:** For optimal unbinding accuracy, use `sample_fhrr_random()` which generates
vectors with conjugate symmetry. General complex phasors achieve ~70% accuracy,
while proper FHRR vectors achieve >99% accuracy.

### Example: Role-Filler Binding

```python
# Represent "The dog chased the cat"
subject = jnp.exp(1j * jax.random.uniform(key1, (512,)))
verb = jnp.exp(1j * jax.random.uniform(key2, (512,)))
object_ = jnp.exp(1j * jax.random.uniform(key3, (512,)))

dog = jnp.exp(1j * jax.random.uniform(key4, (512,)))
chase = jnp.exp(1j * jax.random.uniform(key5, (512,)))
cat = jnp.exp(1j * jax.random.uniform(key6, (512,)))

# Create sentence representation
sentence = ops.bundle(
    ops.bind(subject, dog),
    ops.bind(verb, chase),
    ops.bind(object_, cat)
)

# Query: What was the subject?
query = ops.bind(sentence, ops.inverse(subject))
# query ≈ dog (high similarity)
```

## MAPOperations

Operations for real-valued hypervectors using element-wise operations.

### Binding (Element-wise Multiplication)

Simplest binding operation - just multiply element-wise.

```python
from vsax import MAPOperations

ops = MAPOperations()

# Real vectors
a = jax.random.normal(key1, (512,))
b = jax.random.normal(key2, (512,))

# Bind via multiplication
bound = ops.bind(a, b)
assert bound.shape == a.shape
assert jnp.array_equal(bound, a * b)
```

**Properties:**
- Commutative: `bind(a, b) = bind(b, a)`
- Associative: `bind(a, bind(b, c)) = bind(bind(a, b), c)`
- Approximate unbinding: Cannot perfectly recover original

### Bundling (Element-wise Mean)

Average of all input vectors.

```python
# Bundle three vectors
bundled = ops.bundle(a, b, c)

# Result is the mean
assert jnp.allclose(bundled, (a + b + c) / 3)
```

**Properties:**
- Order-independent
- Lossy: Individual vectors cannot be perfectly recovered
- Preserves similarity: Bundled vector similar to constituents

### Inverse (Approximate)

MAP uses an approximate inverse based on normalization.

```python
# Method 1: Explicit unbind (RECOMMENDED)
recovered = ops.unbind(bound, b)

# Method 2: Using inverse (equivalent)
inv_b = ops.inverse(b)
recovered = ops.bind(bound, inv_b)

# recovered ≈ a (but not exact, typically ~30% similarity)
```

**Note:** MAP unbinding is approximate - use for applications where exact recovery
isn't critical. Typical unbinding similarity is ~30-40%, sufficient for many
pattern matching and classification tasks.

### Example: Feature Binding

```python
# Represent a data point: {age: 25, income: 50000, city: "SF"}
age_role = jax.random.normal(key1, (512,))
income_role = jax.random.normal(key2, (512,))
city_role = jax.random.normal(key3, (512,))

age_25 = jax.random.normal(key4, (512,))
income_50k = jax.random.normal(key5, (512,))
sf = jax.random.normal(key6, (512,))

# Create record
record = ops.bundle(
    ops.bind(age_role, age_25),
    ops.bind(income_role, income_50k),
    ops.bind(city_role, sf)
)
```

## BinaryOperations

Operations for binary hypervectors using XOR and majority voting.

### Binding (XOR)

In bipolar {-1, +1} representation, XOR is implemented as multiplication.

```python
from vsax import BinaryOperations

ops = BinaryOperations()

# Bipolar vectors
a = jnp.array([1, -1, 1, -1, 1, 1, -1, -1])
b = jnp.array([1, 1, -1, -1, 1, -1, 1, -1])

# Bind via XOR (multiplication in bipolar)
bound = ops.bind(a, b)

# Result: element-wise multiplication
# Same values → +1, different values → -1
```

**Properties:**
- Commutative: `bind(a, b) = bind(b, a)`
- Associative: `bind(a, bind(b, c)) = bind(bind(a, b), c)`
- Self-inverse: `bind(bind(a, b), b) = a` (exact unbinding!)

### Bundling (Majority Vote)

Each position in the bundled vector is determined by majority vote.

```python
a = jnp.array([1, -1, 1, -1])
b = jnp.array([1, 1, -1, -1])
c = jnp.array([1, 1, 1, 1])

bundled = ops.bundle(a, b, c)

# Position 0: [1, 1, 1] → majority 1
# Position 1: [-1, 1, 1] → majority 1
# Position 2: [1, -1, 1] → majority 1
# Position 3: [-1, -1, 1] → majority -1
# Result: [1, 1, 1, -1]
```

**Tie Breaking:** For even numbers of vectors, ties default to +1.

### Inverse (Self-Inverse Property)

XOR is its own inverse, so `inverse(a) = a`. This means unbinding is
identical to binding for Binary VSA!

```python
# Method 1: Explicit unbind (RECOMMENDED - clearer intent)
recovered = ops.unbind(bound, b)
assert jnp.array_equal(recovered, a)  # Exact recovery!

# Method 2: Using inverse (equivalent due to self-inverse)
inv_b = ops.inverse(b)  # inv_b == b
recovered = ops.bind(bound, inv_b)
assert jnp.array_equal(recovered, a)  # Same result!

# Method 3: Direct binding (works because XOR is self-inverse)
recovered = ops.bind(bound, b)
assert jnp.array_equal(recovered, a)  # Also works!
```

**Key insight:** For Binary VSA, `unbind(a, b) = bind(a, b) = a XOR b`
due to XOR's self-inverse property.

### Example: Symbolic Reasoning

```python
# Encode facts: "Alice likes Bob", "Bob likes Charlie"
alice = jax.random.choice(key1, jnp.array([-1, 1]), (512,))
bob = jax.random.choice(key2, jnp.array([-1, 1]), (512,))
charlie = jax.random.choice(key3, jnp.array([-1, 1]), (512,))
likes = jax.random.choice(key4, jnp.array([-1, 1]), (512,))

# Create knowledge base
fact1 = ops.bind(ops.bind(alice, likes), bob)
fact2 = ops.bind(ops.bind(bob, likes), charlie)
kb = ops.bundle(fact1, fact2)

# Query: Who does Alice like?
query = ops.bind(ops.bind(kb, alice), likes)
# High similarity to bob
```

## Permutation

All operation sets support circular permutation (rotation).

```python
vec = jnp.array([1, 2, 3, 4, 5])

# Rotate right by 2
shifted = ops.permute(vec, 2)
# Result: [4, 5, 1, 2, 3]

# Rotate left by 2
shifted = ops.permute(vec, -2)
# Result: [3, 4, 5, 1, 2]
```

**Use cases:**
- Sequence encoding
- Temporal ordering
- Positional information

## Comparison

| Feature | FHRR | MAP | Binary |
|---|---|---|---|
| Binding | FFT convolution | Element-wise × | XOR |
| Unbinding | Exact (conjugate) | Approximate | Exact (self-inverse) |
| Bundling | Sum + normalize | Mean | Majority vote |
| Complexity | O(n log n) | O(n) | O(n) |
| Memory | 2x (complex) | 1x | 1/32x |

## Best Practices

1. **Normalize inputs**: Ensure vectors are properly normalized before operations
2. **Consistent types**: Don't mix operation sets with wrong representations
3. **Batch operations**: Use JAX's `vmap` for processing multiple vectors
4. **Numerical stability**: Be aware of numerical precision, especially with FHRR

## Next Steps

- Learn about [Sampling](sampling.md) to create basis vectors
- See [Models](models.md) to combine representations and operations
- Check [Examples](../examples/fhrr.md) for complete workflows
