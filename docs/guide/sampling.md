# Sampling Hypervectors

VSAX provides sampling functions to generate random basis hypervectors for each representation type.

## Overview

| Function | Output | Distribution | Use With |
|---|---|---|---|
| `sample_random` | Real vectors | Normal N(0,1) | RealHypervector, MAP |
| `sample_complex_random` | Complex vectors | Uniform phase | ComplexHypervector, FHRR |
| `sample_binary_random` | Binary vectors | Uniform {-1,+1} or {0,1} | BinaryHypervector, Binary |

## sample_random

Samples real-valued vectors from standard normal distribution.

```python
from vsax.sampling import sample_random
import jax

key = jax.random.PRNGKey(42)
vectors = sample_random(dim=512, n=10, key=key)

# Shape: (10, 512)
# Elements: drawn from N(0, 1)
```

**Parameters:**
- `dim`: Vector dimensionality
- `n`: Number of vectors to sample
- `key`: JAX random key (optional, defaults to PRNGKey(0))

## sample_complex_random

Samples unit-magnitude complex vectors with uniformly random phases.

```python
from vsax.sampling import sample_complex_random

key = jax.random.PRNGKey(42)
vectors = sample_complex_random(dim=512, n=10, key=key)

# All magnitudes are 1.0
assert jnp.allclose(jnp.abs(vectors), 1.0)

# Phases uniformly distributed in [0, 2π)
phases = jnp.angle(vectors)
```

**Properties:**
- All elements have magnitude 1.0
- Phases uniformly distributed in [0, 2π)
- Suitable for FHRR operations

## sample_binary_random

Samples binary vectors with values from {-1, +1} (bipolar) or {0, 1} (binary).

```python
from vsax.sampling import sample_binary_random

key = jax.random.PRNGKey(42)

# Bipolar sampling (default)
bipolar_vecs = sample_binary_random(dim=512, n=10, key=key, bipolar=True)
assert jnp.all(jnp.isin(bipolar_vecs, jnp.array([-1, 1])))

# Binary sampling
binary_vecs = sample_binary_random(dim=512, n=10, key=key, bipolar=False)
assert jnp.all(jnp.isin(binary_vecs, jnp.array([0, 1])))
```

**Parameters:**
- `dim`: Vector dimensionality
- `n`: Number of vectors to sample
- `key`: JAX random key (optional)
- `bipolar`: If True, sample from {-1, +1}; if False, sample from {0, 1}

## Reproducibility

Use JAX's PRNG system for reproducible sampling:

```python
# Same key = same samples
key = jax.random.PRNGKey(42)
samples1 = sample_random(dim=100, n=5, key=key)
samples2 = sample_random(dim=100, n=5, key=key)
assert jnp.array_equal(samples1, samples2)

# Different keys = different samples
key2 = jax.random.PRNGKey(43)
samples3 = sample_random(dim=100, n=5, key=key2)
assert not jnp.array_equal(samples1, samples3)
```

## Complete Example

```python
import jax
from vsax import VSAModel, ComplexHypervector, FHRROperations, sample_complex_random

# Create model with sampler
model = VSAModel(
    dim=512,
    rep_cls=ComplexHypervector,
    opset=FHRROperations(),
    sampler=sample_complex_random
)

# Use model's sampler
key = jax.random.PRNGKey(42)
basis_vectors = model.sampler(dim=model.dim, n=100, key=key)

# Create hypervectors
hvs = [model.rep_cls(vec).normalize() for vec in basis_vectors]
```

## Next Steps

- Learn about [Models](models.md) to combine samplers with representations
- See [Examples](../examples/fhrr.md) for complete workflows
