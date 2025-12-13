# VSA Models

The `VSAModel` is an immutable container that defines a complete VSA algebra by combining a representation type, operation set, and sampler.

## VSAModel Structure

```python
@dataclass(frozen=True)
class VSAModel:
    dim: int                          # Dimensionality
    rep_cls: type[AbstractHypervector]  # Representation class
    opset: AbstractOpSet              # Operation set
    sampler: Callable                 # Sampling function
```

## Creating Models

### FHRR Model

```python
from vsax import VSAModel, ComplexHypervector, FHRROperations, sample_complex_random

fhrr_model = VSAModel(
    dim=512,
    rep_cls=ComplexHypervector,
    opset=FHRROperations(),
    sampler=sample_complex_random
)
```

### MAP Model

```python
from vsax import RealHypervector, MAPOperations, sample_random

map_model = VSAModel(
    dim=512,
    rep_cls=RealHypervector,
    opset=MAPOperations(),
    sampler=sample_random
)
```

### Binary Model

```python
from vsax import BinaryHypervector, BinaryOperations, sample_binary_random

binary_model = VSAModel(
    dim=512,
    rep_cls=BinaryHypervector,
    opset=BinaryOperations(),
    sampler=sample_binary_random
)
```

## Using Models

```python
import jax

# Sample basis vectors
key = jax.random.PRNGKey(42)
vectors = fhrr_model.sampler(dim=fhrr_model.dim, n=2, key=key)

# Create hypervectors using model's representation
a = fhrr_model.rep_cls(vectors[0]).normalize()
b = fhrr_model.rep_cls(vectors[1]).normalize()

# Perform operations using model's opset
bound = fhrr_model.opset.bind(a.vec, b.vec)
bundled = fhrr_model.opset.bundle(a.vec, b.vec)
```

## Model Properties

**Immutability:** Models are frozen dataclasses - cannot be modified after creation.

```python
model = VSAModel(dim=512, ...)

# This will raise an error
model.dim = 1024  # FrozenInstanceError!
```

**Type Safety:** The model ensures all components work together correctly.

## Next: VSAMemory (Coming in Iteration 3)

In the next iteration, you'll be able to use `VSAMemory` to manage named basis vectors:

```python
# Coming soon!
memory = VSAMemory(model)
memory.add("dog")
memory.add("cat")

dog = memory["dog"]  # Access by name
```

## Next Steps

- See [Examples](../examples/fhrr.md) for complete model usage
- Check [API Reference](../api/core/model.md) for detailed documentation
