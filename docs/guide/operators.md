# Operators

**NEW in v1.1.0** - Exact, compositional, invertible transformations for reasoning.

## Overview

Operators extend VSA with **exact transformations** that represent "what happens" (actions, relations, transformations), complementing hypervectors which represent "what exists" (concepts, objects, symbols).

**Key insight:** Hypervectors + Operators = Complete symbolic reasoning system

| Component | Represents | Example |
|-----------|-----------|---------|
| **Hypervectors** | Concepts, objects, symbols | "cup", "dog", "3" |
| **Operators** | Transformations, relations, actions | LEFT_OF, AGENT, ROTATE |

## Why Operators?

### Problem: Bundling Loses Directionality

**Without operators**, encoding asymmetric relations loses directionality:

```python
# Encode "cup LEFT_OF plate" using only bundling
scene = bundle(cup, left_role, plate)

# Query: what's on the left?
# Problem: Cannot distinguish "cup left of plate" from "plate left of cup"
# Both cup and plate have similar similarity!
```

### Solution: Operators Preserve Direction

**With operators**, we get exact, directional transformations:

```python
# Encode "cup LEFT_OF plate" using operators
scene = bundle(cup, LEFT_OF.apply(plate))

# Query: what's LEFT_OF plate?
answer = LEFT_OF.inverse().apply(scene)  # → cup (similarity > 0.7)

# Query: what's RIGHT_OF cup?
answer = RIGHT_OF.apply(scene)  # → plate (similarity > 0.7)
```

### Advantages

✅ **Exact inversion** - Similarity > 0.999 (vs ~0.6-0.7 with bundling)
✅ **Directional** - Preserves asymmetric relationships
✅ **Compositional** - Combine transformations algebraically
✅ **Typed** - Semantic metadata (SPATIAL, SEMANTIC, TEMPORAL)
✅ **JAX-native** - GPU-accelerated, JIT-compatible

## CliffordOperator

The `CliffordOperator` is VSAX's implementation of Clifford-inspired operators.

### Mathematical Foundation

**Phase-based transformation:**
```
apply(v) = v * exp(i * params)
```

Where:
- `v` is a ComplexHypervector (FHRR representation)
- `params` is a vector of phase rotations (shape: dim)
- `exp(i * params)` applies element-wise phase shift

**Properties:**
- Norm-preserving: |result| = |v|
- Exact inverse: `exp(-i * params)`
- Compositional: `exp(i * (params1 + params2))`

### Clifford Algebra Inspiration

**Not full geometric algebra**, but inspired by key concepts:

| Clifford Concept | VSAX Implementation |
|------------------|---------------------|
| **Bivectors** | Elementary operators (phase generators) |
| **Rotors** | Composed operators (sum of generators) |
| **Geometric product** | Operator composition |
| **Reverse/inverse** | Phase negation |

**Explicitly out of scope:**
- Full multivectors
- Blade arithmetic
- 2^n basis expansion

## Basic Usage

### Creating Operators

```python
from vsax.operators import CliffordOperator, OperatorKind
import jax

# Create random operator
op = CliffordOperator.random(
    dim=512,
    kind=OperatorKind.GENERAL,
    name="MY_OP",
    key=jax.random.PRNGKey(42)
)

print(op)  # MY_OP(dim=512, kind=general)
```

### Applying Transformations

```python
from vsax import create_fhrr_model, VSAMemory

model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
memory.add("concept")

# Apply operator
transformed = op.apply(memory["concept"])
print(type(transformed))  # ComplexHypervector
```

### Exact Inversion

```python
from vsax.similarity import cosine_similarity

# Apply and invert
original = memory["concept"]
transformed = op.apply(original)
recovered = op.inverse().apply(transformed)

# Check recovery
similarity = cosine_similarity(recovered.vec, original.vec)
print(f"Recovery: {similarity:.6f}")  # > 0.999
```

### Composition

```python
# Create two operators
op1 = CliffordOperator.random(512, name="OP1", key=jax.random.PRNGKey(1))
op2 = CliffordOperator.random(512, name="OP2", key=jax.random.PRNGKey(2))

# Compose them
composed = op1.compose(op2)

# Apply composed = apply sequentially
hv = memory["concept"]
result1 = composed.apply(hv)
result2 = op2.apply(op1.apply(hv))

similarity = cosine_similarity(result1.vec, result2.vec)
print(f"Composition: {similarity:.6f}")  # 1.0
```

## Operator Types

The `OperatorKind` enum provides semantic typing:

```python
from vsax.operators import OperatorKind

# Available kinds:
OperatorKind.RELATION    # Relational operators (e.g., PART_OF)
OperatorKind.TRANSFORM   # Transformations (e.g., ROTATE)
OperatorKind.LOGICAL     # Logical operators (e.g., NOT, AND)
OperatorKind.SPATIAL     # Spatial relations (e.g., LEFT_OF, ABOVE)
OperatorKind.TEMPORAL    # Temporal relations (e.g., BEFORE, AFTER)
OperatorKind.SEMANTIC    # Semantic roles (e.g., AGENT, PATIENT)
OperatorKind.GENERAL     # General-purpose operators
```

### Creating Typed Operators

```python
# Spatial operator
LEFT_OF = CliffordOperator.random(
    dim=512,
    kind=OperatorKind.SPATIAL,
    name="LEFT_OF",
    key=jax.random.PRNGKey(100)
)

# Semantic operator
AGENT = CliffordOperator.random(
    dim=512,
    kind=OperatorKind.SEMANTIC,
    name="AGENT",
    key=jax.random.PRNGKey(200)
)

# Check metadata
print(LEFT_OF.metadata.kind)  # OperatorKind.SPATIAL
print(AGENT.metadata.kind)    # OperatorKind.SEMANTIC
```

## Common Use Cases

### 1. Spatial Reasoning

Encode spatial layouts and query relations:

```python
# Create spatial operators
LEFT_OF = CliffordOperator.random(512, kind=OperatorKind.SPATIAL,
                                  name="LEFT_OF", key=jax.random.PRNGKey(100))
ABOVE = CliffordOperator.random(512, kind=OperatorKind.SPATIAL,
                                name="ABOVE", key=jax.random.PRNGKey(101))

# Inverses
RIGHT_OF = LEFT_OF.inverse()
BELOW = ABOVE.inverse()

# Encode: "cup LEFT_OF plate, plate ABOVE table"
scene = model.opset.bundle(
    memory["cup"].vec,
    LEFT_OF.apply(memory["plate"]).vec,
    ABOVE.apply(memory["table"]).vec
)

# Query: What's LEFT_OF plate?
answer = RIGHT_OF.apply(model.rep_cls(scene))
# Similarity to "cup" will be high
```

### 2. Semantic Role Labeling

Encode who-did-what-to-whom:

```python
# Create semantic operators
AGENT = CliffordOperator.random(512, kind=OperatorKind.SEMANTIC,
                                name="AGENT", key=jax.random.PRNGKey(200))
PATIENT = CliffordOperator.random(512, kind=OperatorKind.SEMANTIC,
                                  name="PATIENT", key=jax.random.PRNGKey(201))
ACTION = CliffordOperator.random(512, kind=OperatorKind.SEMANTIC,
                                 name="ACTION", key=jax.random.PRNGKey(202))

# Encode: "dog chases cat"
sentence = model.opset.bundle(
    AGENT.apply(memory["dog"]).vec,
    ACTION.apply(memory["chase"]).vec,
    PATIENT.apply(memory["cat"]).vec
)

# Query: Who is the AGENT?
who = AGENT.inverse().apply(model.rep_cls(sentence))
# Similarity to "dog" will be high
```

### 3. Graph Reasoning with Typed Edges

Encode knowledge graphs with typed relations:

```python
# Create relation operators
IS_A = CliffordOperator.random(512, kind=OperatorKind.RELATION,
                               name="IS_A", key=jax.random.PRNGKey(300))
HAS_PART = CliffordOperator.random(512, kind=OperatorKind.RELATION,
                                   name="HAS_PART", key=jax.random.PRNGKey(301))

# Encode facts
memory.add_many(["dog", "animal", "mammal", "tail", "legs"])

fact1 = model.opset.bundle(
    memory["dog"].vec,
    IS_A.apply(memory["mammal"]).vec
)

fact2 = model.opset.bundle(
    memory["dog"].vec,
    HAS_PART.apply(memory["tail"]).vec
)

# Knowledge base
kb = model.opset.bundle(fact1, fact2)

# Query: What IS_A mammal?
answer = IS_A.inverse().apply(model.rep_cls(kb))
# Returns "dog"

# Query: What parts does dog have?
parts = HAS_PART.apply(model.rep_cls(kb))
# Returns "tail"
```

### 4. Compositional Transformations

Chain multiple transformations:

```python
# Move left then up
left_and_up = LEFT_OF.compose(ABOVE)

# Apply to a point in space
transformed = left_and_up.apply(memory["origin"])

# Inverse to recover
recovered = left_and_up.inverse().apply(transformed)
```

## Advanced Features

### Operator Metadata

```python
from vsax.operators import OperatorMetadata

# Create operator with full metadata
metadata = OperatorMetadata(
    kind=OperatorKind.SPATIAL,
    name="NORTH_OF",
    description="Represents northward spatial relation",
    invertible=True,
    commutative=False
)

# Use in operator creation (manual)
params = jax.random.uniform(jax.random.PRNGKey(0), (512,), minval=0, maxval=2*jnp.pi)
op = CliffordOperator(params=params, metadata=metadata)
```

### Reproducibility

Use consistent random keys for reproducible operators:

```python
# Same key = same operator
key = jax.random.PRNGKey(42)
op1 = CliffordOperator.random(512, key=key)
op2 = CliffordOperator.random(512, key=key)

# Operators are identical
assert jnp.allclose(op1.params, op2.params)
```

### Batch Operations (Future)

While not yet implemented, operators are designed for batch operations:

```python
# Future: Apply operator to batch of hypervectors
# batch = jnp.stack([hv1.vec, hv2.vec, hv3.vec])
# results = jax.vmap(op.apply)(batch)
```

## Design Principles

### 1. JAX-Native

All operations use JAX for GPU acceleration and JIT compilation:

```python
# JIT-compile operator application
@jax.jit
def transform_many(op, hvs):
    return jax.vmap(op.apply)(hvs)
```

### 2. Immutable

Operators are immutable (frozen dataclasses):

```python
# Cannot modify operators
op = CliffordOperator.random(512, key=jax.random.PRNGKey(0))
# op.params = new_params  # ❌ Raises error

# Create new operator instead
new_op = CliffordOperator(params=new_params)
```

### 3. Type-Safe

Operators only work with ComplexHypervector (FHRR):

```python
from vsax.representations import RealHypervector

# Works with ComplexHypervector
complex_hv = memory["concept"]  # ComplexHypervector
result = op.apply(complex_hv)  # ✅ Works

# Error with other types
real_hv = RealHypervector(jnp.ones(512))
result = op.apply(real_hv)  # ❌ TypeError
```

### 4. FHRR-Compatible

Operators compile to FHRR's phase algebra:

```python
# Operator applies phase rotation
# This is compatible with FHRR's circular convolution
bound = model.opset.bind(op.apply(hv1).vec, hv2.vec)
```

## Performance

### Test Coverage

- **96% coverage** on CliffordOperator module
- **23 comprehensive tests** covering all properties
- **410 total tests** in VSAX (including operators)

### Computational Cost

**Operator application:** O(dim) - element-wise phase rotation
**Composition:** O(dim) - element-wise phase addition
**Inversion:** O(dim) - phase negation

All operations are JAX-native and GPU-accelerated.

## Comparison with Bundling

| Aspect | Bundling | Operators |
|--------|----------|-----------|
| **Use case** | Symmetric relations, prototypes | Asymmetric relations, transformations |
| **Inversion** | Approximate (~0.6-0.7) | Exact (>0.999) |
| **Directionality** | Lost | Preserved |
| **Composition** | Limited | Full algebraic composition |
| **Example** | "dog AND cat" | "dog CHASES cat" |

## Limitations

### Current Limitations

1. **FHRR-only** - Operators require ComplexHypervector (phase representation)
2. **No pre-defined operators yet** - Must create operators manually with `random()`
3. **No learned operators** - Parameters are randomly sampled, not learned from data
4. **Single operator per application** - Cannot apply multiple operators simultaneously

### Future Extensions

Planned for future releases:

- Pre-defined spatial operators (LEFT_OF, ABOVE, NEAR, etc.)
- Pre-defined semantic operators (AGENT, PATIENT, THEME, etc.)
- Operator learning from data
- Batch operator application with vmap
- Non-commutative operators for sequence modeling
- Operator visualization tools

## Best Practices

### When to Use Operators

✅ **Use operators for:**
- Encoding directional relations (LEFT_OF, ABOVE, BEFORE)
- Semantic role labeling (AGENT, PATIENT)
- Transformations requiring exact inversion
- Compositional reasoning tasks
- Typed graph edges

❌ **Use bundling/binding for:**
- Symmetric relations ("similar to", "same color as")
- Building prototypes from examples
- Encoding unordered collections
- Feature combination

### Operator Naming

Use descriptive names with semantic meaning:

```python
# Good ✅
LEFT_OF = CliffordOperator.random(512, name="LEFT_OF", ...)
AGENT = CliffordOperator.random(512, name="AGENT", ...)

# Less clear ❌
op1 = CliffordOperator.random(512, name="op1", ...)
x = CliffordOperator.random(512, name="x", ...)
```

### Reproducibility

Always use explicit random keys for reproducibility:

```python
# Good ✅
LEFT_OF = CliffordOperator.random(512, key=jax.random.PRNGKey(100))

# Non-reproducible ❌
LEFT_OF = CliffordOperator.random(512)  # Uses key(0) by default
```

### Inverse Pairs

Create inverse pairs for bidirectional relations:

```python
# Create forward and inverse together
LEFT_OF = CliffordOperator.random(512, key=jax.random.PRNGKey(100))
RIGHT_OF = LEFT_OF.inverse()

ABOVE = CliffordOperator.random(512, key=jax.random.PRNGKey(101))
BELOW = ABOVE.inverse()
```

## Related Topics

- **Tutorial 10:** [Clifford Operators Tutorial](../tutorials/10_clifford_operators.md)
- **API Reference:** [Operators API](../api/operators/index.md)
- **Design Spec:** [VSAX Design Specification](../design-spec.md)

## Further Reading

**Clifford Algebra:**
- Hestenes & Sobczyk (1984) - "Clifford Algebra to Geometric Calculus"
- Dorst et al. (2007) - "Geometric Algebra for Computer Science"

**VSA Theory:**
- Kanerva (2009) - "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors"
- Plate (1995) - "Holographic Reduced Representations"

**Operator-Based Reasoning:**
- Gayler (2004) - "Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience"
- Kleyko et al. (2021) - "A Survey on Hyperdimensional Computing"
