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
from vsax import VSAModel, ComplexHypervector, FHRROperations, sample_fhrr_random

fhrr_model = VSAModel(
    dim=512,
    rep_cls=ComplexHypervector,
    opset=FHRROperations(),
    sampler=sample_fhrr_random  # Recommended: ensures >99% unbinding accuracy
)

# Alternative: Use sample_complex_random for general complex vectors (~70% unbinding)
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

# Unbind to recover original (NEW: explicit unbind method)
recovered = fhrr_model.opset.unbind(bound, b.vec)
# With FHRR + sample_fhrr_random: >99% similarity to a.vec!
```

## Model Properties

**Immutability:** Models are frozen dataclasses - cannot be modified after creation.

```python
model = VSAModel(dim=512, ...)

# This will raise an error
model.dim = 1024  # FrozenInstanceError!
```

**Type Safety:** The model ensures all components work together correctly.

## Next: VSAMemory

Use `VSAMemory` to manage named basis vectors:

```python
from vsax import VSAMemory

memory = VSAMemory(model)
memory.add("dog")
memory.add("cat")

dog = memory["dog"]  # Access by name
```

See the [VSAMemory guide](memory.md) for more details.

## Next Steps

- See [Examples](../examples/fhrr.md) for complete model usage
- Check [API Reference](../api/core/model.md) for detailed documentation
