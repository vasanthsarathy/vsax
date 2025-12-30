# MAP Model Example

Complete example using the MAP (Multiply-Add-Permute) model with real-valued hypervectors.

## Setup

```python
import jax
import jax.numpy as jnp
from vsax import VSAModel, RealHypervector, MAPOperations, sample_random

# Create MAP model
model = VSAModel(
    dim=512,
    rep_cls=RealHypervector,
    opset=MAPOperations(),
    sampler=sample_random
)
```

## Basic Operations

```python
# Sample and normalize
key = jax.random.PRNGKey(42)
vectors = model.sampler(dim=model.dim, n=2, key=key)

a = model.rep_cls(vectors[0]).normalize()
b = model.rep_cls(vectors[1]).normalize()

# Verify L2 normalization
print(f"L2 norm of a: {jnp.linalg.norm(a.vec):.4f}")  # 1.0

# Bind (element-wise multiplication)
bound = model.opset.bind(a.vec, b.vec)

# Bundle (element-wise mean)
bundled = model.opset.bundle(a.vec, b.vec)

# Unbind (approximate recovery with MAP)
recovered = model.opset.unbind(bound, b.vec)
recovered_hv = model.rep_cls(recovered).normalize()

# Check similarity - MAP unbinding is approximate
similarity = jnp.dot(a.vec, recovered_hv.vec)
print(f"Recovery similarity: {similarity:.4f}")  # ~0.30-0.40 (approximate!)
```

## Feature Binding Example

Encode structured records with real-valued features.

```python
# Define feature roles
age_role = model.sampler(dim=model.dim, n=1, key=jax.random.PRNGKey(1))[0]
income_role = model.sampler(dim=model.dim, n=1, key=jax.random.PRNGKey(2))[0]

# Encode feature values (simplified - normally you'd use encoders)
age_25 = model.sampler(dim=model.dim, n=1, key=jax.random.PRNGKey(3))[0]
income_50k = model.sampler(dim=model.dim, n=1, key=jax.random.PRNGKey(4))[0]

# Create record
record = model.opset.bundle(
    model.opset.bind(age_role, age_25),
    model.opset.bind(income_role, income_50k)
)
```

**Note:** MAP unbinding is approximate - use for similarity-based retrieval rather than exact recovery.
