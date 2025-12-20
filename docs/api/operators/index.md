# Operators API Reference

**NEW in v1.1.0** - Exact, compositional, invertible transformations for reasoning.

## Overview

The operators module provides Clifford-inspired operators for exact transformations in VSA reasoning.

**Key classes:**
- `CliffordOperator` - Phase-based operator for FHRR hypervectors
- `AbstractOperator` - Base interface for all operators
- `OperatorKind` - Enum for semantic operator types
- `OperatorMetadata` - Metadata dataclass for operators

## Module Structure

```
vsax.operators/
├── CliffordOperator      # Core operator implementation
├── AbstractOperator      # Abstract base class
├── OperatorKind          # Semantic type enum
└── OperatorMetadata      # Metadata dataclass
```

## Quick Reference

### Creating Operators

```python
from vsax.operators import CliffordOperator, OperatorKind
import jax

# Create random operator
op = CliffordOperator.random(
    dim=512,
    kind=OperatorKind.SPATIAL,
    name="LEFT_OF",
    key=jax.random.PRNGKey(42)
)
```

### Applying Transformations

```python
from vsax import create_fhrr_model, VSAMemory

model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
memory.add("concept")

# Apply operator
transformed = op.apply(memory["concept"])

# Exact inversion
recovered = op.inverse().apply(transformed)
```

### Composing Operators

```python
# Create two operators
op1 = CliffordOperator.random(512, key=jax.random.PRNGKey(1))
op2 = CliffordOperator.random(512, key=jax.random.PRNGKey(2))

# Compose
composed = op1.compose(op2)
```

## Detailed API

### CliffordOperator

::: vsax.operators.clifford.CliffordOperator
    options:
      show_source: true
      members:
        - __init__
        - apply
        - inverse
        - compose
        - random
        - dim

### AbstractOperator

::: vsax.operators.base.AbstractOperator
    options:
      show_source: true
      members:
        - apply
        - inverse
        - compose
        - dim

### OperatorKind

::: vsax.operators.kinds.OperatorKind
    options:
      show_source: true

### OperatorMetadata

::: vsax.operators.kinds.OperatorMetadata
    options:
      show_source: true
      members:
        - kind
        - name
        - description
        - invertible
        - commutative

## Usage Examples

### Spatial Reasoning

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.operators import CliffordOperator, OperatorKind
from vsax.similarity import cosine_similarity
import jax

# Setup
model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
memory.add_many(["cup", "plate", "table"])

# Create spatial operators
LEFT_OF = CliffordOperator.random(
    512, kind=OperatorKind.SPATIAL, name="LEFT_OF", key=jax.random.PRNGKey(100)
)
RIGHT_OF = LEFT_OF.inverse()

# Encode: "cup LEFT_OF plate"
scene = model.opset.bundle(
    memory["cup"].vec,
    LEFT_OF.apply(memory["plate"]).vec
)

# Query: What's LEFT_OF plate?
answer = RIGHT_OF.apply(model.rep_cls(scene))
similarity = cosine_similarity(answer.vec, memory["cup"].vec)
print(f"Similarity to 'cup': {similarity:.3f}")  # High similarity
```

### Semantic Roles

```python
# Create semantic operators
AGENT = CliffordOperator.random(
    512, kind=OperatorKind.SEMANTIC, name="AGENT", key=jax.random.PRNGKey(200)
)
PATIENT = CliffordOperator.random(
    512, kind=OperatorKind.SEMANTIC, name="PATIENT", key=jax.random.PRNGKey(201)
)

memory.add_many(["dog", "cat", "chase"])

# Encode: "dog chases cat"
sentence = model.opset.bundle(
    AGENT.apply(memory["dog"]).vec,
    memory["chase"].vec,
    PATIENT.apply(memory["cat"]).vec
)

# Query: Who is the AGENT?
who = AGENT.inverse().apply(model.rep_cls(sentence))
similarity = cosine_similarity(who.vec, memory["dog"].vec)
print(f"AGENT is 'dog': {similarity:.3f}")  # High similarity
```

### Operator Composition

```python
# Compose spatial relations
left_and_up = LEFT_OF.compose(ABOVE)

# Apply composed transformation
transformed = left_and_up.apply(memory["origin"])

# Exact inverse
recovered = left_and_up.inverse().apply(transformed)
similarity = cosine_similarity(recovered.vec, memory["origin"].vec)
print(f"Recovery: {similarity:.6f}")  # > 0.999
```

## Properties

### Exact Inversion

CliffordOperator provides exact inversion with similarity > 0.999:

```python
op = CliffordOperator.random(512, key=jax.random.PRNGKey(0))
hv = memory["test"]

transformed = op.apply(hv)
recovered = op.inverse().apply(transformed)

similarity = cosine_similarity(recovered.vec, hv.vec)
assert similarity > 0.999  # Exact recovery
```

### Associativity

Composition is associative:

```python
op1 = CliffordOperator.random(512, key=jax.random.PRNGKey(1))
op2 = CliffordOperator.random(512, key=jax.random.PRNGKey(2))
op3 = CliffordOperator.random(512, key=jax.random.PRNGKey(3))

hv = memory["test"]

# (op1 ∘ op2) ∘ op3
left = op1.compose(op2).compose(op3).apply(hv)

# op1 ∘ (op2 ∘ op3)
right = op1.compose(op2.compose(op3)).apply(hv)

similarity = cosine_similarity(left.vec, right.vec)
assert similarity > 0.999  # Associative
```

### Commutativity

For phase-based operators, composition is commutative:

```python
# op1 ∘ op2
comp_12 = op1.compose(op2).apply(hv)

# op2 ∘ op1
comp_21 = op2.compose(op1).apply(hv)

similarity = cosine_similarity(comp_12.vec, comp_21.vec)
assert similarity > 0.999  # Commutative
```

### Norm Preservation

Operators preserve the unit magnitude of FHRR vectors:

```python
hv = memory["test"]
transformed = op.apply(hv)

# Both have unit magnitude
assert jnp.allclose(jnp.abs(hv.vec), 1.0, atol=1e-5)
assert jnp.allclose(jnp.abs(transformed.vec), 1.0, atol=1e-5)
```

## Type Safety

CliffordOperator only works with ComplexHypervector (FHRR):

```python
from vsax.representations import ComplexHypervector, RealHypervector

# Works with ComplexHypervector ✅
complex_hv = memory["test"]  # ComplexHypervector from FHRR model
result = op.apply(complex_hv)

# Error with other types ❌
real_hv = RealHypervector(jnp.ones(512))
result = op.apply(real_hv)  # TypeError with helpful message
```

## Performance

### Computational Complexity

| Operation | Complexity | JAX-native | GPU-accelerated |
|-----------|-----------|------------|-----------------|
| `apply()` | O(dim) | ✅ | ✅ |
| `inverse()` | O(dim) | ✅ | ✅ |
| `compose()` | O(dim) | ✅ | ✅ |

### Test Coverage

- **96% coverage** on CliffordOperator module
- **23 comprehensive tests** covering all properties
- Properties tested: inversion, composition, associativity, commutativity, type safety

## Design Principles

### JAX-Native

All operations use JAX for GPU acceleration:

```python
import jax

# JIT-compile operator application
@jax.jit
def transform(op, hv):
    return op.apply(hv)

# GPU-accelerated
result = transform(op, memory["test"])
```

### Immutable

Operators are immutable (frozen dataclasses):

```python
op = CliffordOperator.random(512, key=jax.random.PRNGKey(0))

# Cannot modify ❌
# op.params = new_params  # Raises FrozenInstanceError

# Create new operator instead ✅
new_op = CliffordOperator(params=new_params)
```

### FHRR-Compatible

Operators compile to FHRR's phase algebra:

```python
# Phase-based implementation
# apply(v) = v * exp(i * params)

# Compatible with FHRR circular convolution
bound = model.opset.bind(
    op.apply(hv1).vec,
    hv2.vec
)
```

## Comparison with Other Approaches

### VSA Bundling

| Aspect | Bundling | CliffordOperator |
|--------|----------|-----------------|
| **Inversion** | ~0.6-0.7 similarity | >0.999 similarity |
| **Directionality** | Lost | Preserved |
| **Use case** | Symmetric relations | Asymmetric transformations |
| **Example** | "dog AND cat" | "dog CHASES cat" |

### Other VSA Libraries

VSAX is the **first VSA library** to provide:
- Clifford-inspired operators with exact inversion
- Compositional algebra for transformations
- Semantic typing with OperatorKind enum
- Full integration with VSA operations

## Future Extensions

Planned for future releases:

- Pre-defined spatial operators (LEFT_OF, ABOVE, NEAR, etc.)
- Pre-defined semantic operators (AGENT, PATIENT, THEME, etc.)
- Operator learning from data
- Batch operator application with vmap
- Non-commutative operators for sequences
- Visualization tools

## Related Documentation

- **User Guide:** [Operators Guide](../../guide/operators.md)
- **Tutorial:** [Tutorial 10: Clifford Operators](../../tutorials/10_clifford_operators.md)
- **Design Spec:** [VSAX Design Specification](../../design-spec.md)

## References

**Clifford Algebra:**
- Hestenes & Sobczyk (1984) - "Clifford Algebra to Geometric Calculus"
- Dorst et al. (2007) - "Geometric Algebra for Computer Science"

**VSA Theory:**
- Kanerva (2009) - "Hyperdimensional Computing"
- Plate (1995) - "Holographic Reduced Representations"
- Gayler (2004) - "Vector Symbolic Architectures answer Jackendoff's challenges"
