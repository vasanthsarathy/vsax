# Lesson 4.1: Clifford Operators

**Duration:** ~50 minutes (25 min theory + 25 min tutorial)

**Learning Objectives:**

- Understand operators vs hypervectors (transformations vs concepts)
- Learn when to use operators vs binding
- Master exact, invertible transformations
- Apply operators for spatial relations and semantic roles
- Complete the Clifford operators tutorial
- Build directional reasoning systems

---

## Introduction

So far, you've used **hypervectors** to represent concepts ("dog", "cat", "red") and **binding** to create associations. But what about **transformations** and **directed relationships**?

**Operators** represent "what happens" rather than "what exists":
- **Hypervectors:** Concepts, objects, symbols ("cup", "dog", "3")
- **Operators:** Transformations, relations, actions (LEFT_OF, AGENT, ROTATE)

**Key insight:** Hypervectors + Operators = Complete symbolic reasoning system

---

## The Problem: Binding Loses Direction

### Asymmetric Relations

Consider encoding the spatial relation "cup is left of plate":

```python
from vsax import create_fhrr_model, VSAMemory

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)
memory.add_many(["cup", "plate", "left_of", "right_of"])

# Attempt 1: Bind cup with left_of and plate
scene = model.opset.bind(
    model.opset.bind(memory["cup"].vec, memory["left_of"].vec),
    memory["plate"].vec
)
```

**Problem:** How do we distinguish:
- "cup left_of plate" from
- "plate left_of cup"?

Both cup and plate are bound together with left_of, losing **directionality**!

---

### Query Ambiguity

If we try to query "what is left of the plate?":

```python
# Unbind plate and left_of to find what's on the left
plate_inv = model.opset.inverse(memory["plate"].vec)
left_inv = model.opset.inverse(memory["left_of"].vec)

result = model.opset.bind(scene, plate_inv)
result = model.opset.bind(result, left_inv)

# Check similarity to cup
from vsax.similarity import cosine_similarity
sim_cup = cosine_similarity(result, memory["cup"].vec)
sim_plate = cosine_similarity(result, memory["plate"].vec)

print(f"Cup:   {sim_cup:.4f}")   # ~0.35 (low!)
print(f"Plate: {sim_plate:.4f}") # ~0.30 (also low!)
```

**Issue:** Regular binding produces **approximate, symmetric** results. We can't reliably recover directional information.

---

## The Solution: Clifford Operators

**Clifford operators** provide **exact, invertible, directional** transformations.

### Operators as Transformations

Think of operators as **functions** that transform hypervectors:

$$\text{Operator}(v) = v'$$

**Properties:**
- ✅ **Exact inversion:** $O^{-1}(O(v)) = v$ with similarity > 0.999
- ✅ **Directional:** $O_{\text{LEFT}}(plate) \neq O_{\text{RIGHT}}(plate)$
- ✅ **Compositional:** $O_1 \circ O_2 = O_3$
- ✅ **Norm-preserving:** $|O(v)| = |v|$

---

### Encoding with Operators

Using operators, we encode "cup left_of plate" as:

```python
from vsax.operators import CliffordOperator, OperatorKind
import jax

# Create spatial operator for LEFT_OF
LEFT_OF = CliffordOperator.random(
    dim=2048,
    kind=OperatorKind.SPATIAL,
    name="LEFT_OF",
    key=jax.random.PRNGKey(42)
)

# Encode: cup and (LEFT_OF applied to plate)
cup = memory["cup"]
plate = memory["plate"]

# LEFT_OF.apply(plate) means "what's at this spatial relation to plate"
scene = model.opset.bundle(
    cup.vec,
    LEFT_OF.apply(plate).vec
)
```

**Interpretation:** The scene contains "cup" and "the left-of-plate position" (which should match cup).

---

### Querying with Operators

Now we can query directionally:

```python
# Query: What is left of the plate?
# Apply LEFT_OF to plate, then check similarity to scene
left_of_plate = LEFT_OF.apply(plate)

# Find what matches this position
candidates = ["cup", "plate", "spoon", "fork"]
for item in candidates:
    sim = cosine_similarity(left_of_plate.vec, memory[item].vec)
    print(f"{item}: {sim:.4f}")
```

**Expected:** Cup has high similarity (~0.7-0.9), others have low similarity (~0.0).

---

### Inverse Operators

Operators have **exact inverses**:

```python
# Create operator
AGENT = CliffordOperator.random(
    dim=2048,
    kind=OperatorKind.SEMANTIC,
    name="AGENT",
    key=jax.random.PRNGKey(1)
)

# Apply and invert
original = memory["dog"]
transformed = AGENT.apply(original)
recovered = AGENT.inverse().apply(transformed)

# Check recovery
sim = cosine_similarity(recovered.vec, original.vec)
print(f"Recovery similarity: {sim:.6f}")  # > 0.999 (exact!)
```

**Compare to regular binding:** Binding gives ~0.7-0.8 similarity after unbinding. Operators give >0.999!

---

## When to Use Operators vs Binding

| Use Case | Use Binding | Use Operators |
|----------|-------------|---------------|
| **Symmetric relations** | ✅ (color ⊗ red) | ❌ |
| **Asymmetric relations** | ❌ | ✅ (LEFT_OF(plate)) |
| **Set membership** | ✅ (tag ⊗ value) | ❌ |
| **Spatial relations** | ❌ | ✅ (ABOVE, BELOW) |
| **Semantic roles** | ❌ | ✅ (AGENT, PATIENT) |
| **Transformations** | ❌ | ✅ (ROTATE, NEGATE) |
| **Simple composition** | ✅ | ❌ |
| **Exact recovery needed** | ❌ | ✅ |

**Rule of thumb:**
- **Binding:** Symmetric associations, flexible composition
- **Operators:** Directed relations, exact transformations

---

## Mathematical Foundation

### Phase-Based Transformations

Clifford operators work via **element-wise phase rotation** on complex hypervectors:

$$O(v) = v \odot e^{i\theta}$$

Where:
- $v$ is a ComplexHypervector (unit complex numbers)
- $\theta$ is a vector of phase shifts (shape: `dim`)
- $\odot$ is element-wise multiplication

**Example:**
```python
import jax.numpy as jnp

# Single complex element
v_elem = jnp.exp(1j * 2.5)  # e^(i*2.5)

# Operator phase shift
theta = 1.0

# Apply operator
result = v_elem * jnp.exp(1j * theta)  # e^(i*3.5)

# Inverse
inverse_result = result * jnp.exp(-1j * theta)  # e^(i*2.5) = v_elem
```

**Why it works:** Phase rotation is **reversible** (just rotate back) and **norm-preserving** ($|e^{i\theta}| = 1$).

---

### Composition

Operators compose algebraically:

$$O_1 \circ O_2 = O_3$$

where $\theta_3 = \theta_1 + \theta_2$

```python
# Create two operators
SPATIAL = CliffordOperator.random(2048, name="SPATIAL", key=jax.random.PRNGKey(1))
TEMPORAL = CliffordOperator.random(2048, name="TEMPORAL", key=jax.random.PRNGKey(2))

# Compose
SPATIOTEMPORAL = SPATIAL.compose(TEMPORAL)

# Apply composed = apply sequentially
v = memory["event"]
result1 = SPATIAL.apply(TEMPORAL.apply(v))  # Sequential
result2 = SPATIOTEMPORAL.apply(v)            # Composed

# These are equivalent
sim = cosine_similarity(result1.vec, result2.vec)
print(f"Composition equivalence: {sim:.6f}")  # ~1.0
```

---

## Operator Types (OperatorKind)

VSAX provides semantic typing for operators:

```python
from vsax.operators import OperatorKind

# Spatial relations
LEFT_OF = CliffordOperator.random(2048, kind=OperatorKind.SPATIAL, name="LEFT_OF")
ABOVE = CliffordOperator.random(2048, kind=OperatorKind.SPATIAL, name="ABOVE")

# Semantic roles (NLP)
AGENT = CliffordOperator.random(2048, kind=OperatorKind.SEMANTIC, name="AGENT")
PATIENT = CliffordOperator.random(2048, kind=OperatorKind.SEMANTIC, name="PATIENT")

# Temporal relations
BEFORE = CliffordOperator.random(2048, kind=OperatorKind.TEMPORAL, name="BEFORE")
AFTER = CliffordOperator.random(2048, kind=OperatorKind.TEMPORAL, name="AFTER")

# General purpose
TRANSFORM = CliffordOperator.random(2048, kind=OperatorKind.GENERAL, name="TRANSFORM")
```

**Note:** `kind` is metadata for documentation—doesn't affect operator behavior.

---

## Use Cases for Operators

### 1. Spatial Scene Encoding

```python
# Scene: cup is left of plate, spoon is above plate
scene = model.opset.bundle(
    memory["cup"].vec,
    LEFT_OF.apply(memory["plate"]).vec,
    memory["spoon"].vec,
    ABOVE.apply(memory["plate"]).vec
)

# Query: What is left of the plate?
left_of_plate_pos = LEFT_OF.apply(memory["plate"])
# Cup should have high similarity to this position
```

---

### 2. Semantic Role Labeling (NLP)

```python
# Sentence: "Alice gave Bob a book"
# AGENT(Alice), ACTION(gave), RECIPIENT(Bob), THEME(book)

AGENT = CliffordOperator.random(2048, kind=OperatorKind.SEMANTIC, name="AGENT")
RECIPIENT = CliffordOperator.random(2048, kind=OperatorKind.SEMANTIC, name="RECIPIENT")
THEME = CliffordOperator.random(2048, kind=OperatorKind.SEMANTIC, name="THEME")

sentence = model.opset.bundle(
    AGENT.apply(memory["Alice"]).vec,
    memory["gave"].vec,
    RECIPIENT.apply(memory["Bob"]).vec,
    THEME.apply(memory["book"]).vec
)

# Query: Who is the AGENT?
agent_role = AGENT.inverse().apply(...)
# Alice should be recovered
```

---

### 3. Temporal Ordering

```python
# Events: breakfast BEFORE lunch BEFORE dinner
BEFORE = CliffordOperator.random(2048, kind=OperatorKind.TEMPORAL, name="BEFORE")

timeline = model.opset.bundle(
    memory["breakfast"].vec,
    BEFORE.apply(memory["lunch"]).vec,
    BEFORE.compose(BEFORE).apply(memory["dinner"]).vec  # 2x BEFORE
)

# Query: What comes BEFORE dinner?
# Apply BEFORE^(-1) to dinner to find lunch
```

---

## Hands-On: Complete Operators Tutorial

Now dive deep into Clifford operators!

**📓 [Tutorial 10: Clifford Operators](../../tutorials/10_clifford_operators.md)**

**What you'll learn:**
- Creating and applying Clifford operators
- Exact inversion (similarity > 0.999)
- Operator composition
- Spatial scene understanding
- Semantic role encoding
- Comparing operators vs binding
- Building directional reasoning systems

**Time estimate:** 25-30 minutes

**Prerequisites:**
- Understanding of binding and bundling (Module 1)
- FHRR operations (Module 2)

---

**Additional Reference:**

**📖 [Operators Guide](../../guide/operators.md)**

Complete technical documentation on:
- Mathematical foundations
- Clifford algebra inspiration
- Advanced composition
- Performance optimization
- Design patterns

---

## Key Concepts from the Tutorial

### 1. Exact Inversion Property

```python
# Create operator
op = CliffordOperator.random(2048, name="OP", key=jax.random.PRNGKey(42))

# Apply and invert
original = memory["concept"]
transformed = op.apply(original)
recovered = op.inverse().apply(transformed)

# Measure recovery
sim = cosine_similarity(recovered.vec, original.vec)
print(f"Recovery: {sim:.6f}")  # > 0.999 (exact!)
```

---

### 2. Directional Encoding

```python
# Asymmetric relation: A relates-to B
scene = model.opset.bundle(
    memory["A"].vec,
    RELATION.apply(memory["B"]).vec
)

# Query: What does RELATION map B to?
# Answer: Should retrieve A
```

---

### 3. Operator Library

Build a reusable library of operators:

```python
class SpatialOperators:
    """Collection of spatial relation operators."""

    def __init__(self, dim, seed=0):
        import jax
        key = jax.random.PRNGKey(seed)

        # Generate operators
        keys = jax.random.split(key, 6)
        self.LEFT_OF = CliffordOperator.random(dim, kind=OperatorKind.SPATIAL, name="LEFT_OF", key=keys[0])
        self.RIGHT_OF = CliffordOperator.random(dim, kind=OperatorKind.SPATIAL, name="RIGHT_OF", key=keys[1])
        self.ABOVE = CliffordOperator.random(dim, kind=OperatorKind.SPATIAL, name="ABOVE", key=keys[2])
        self.BELOW = CliffordOperator.random(dim, kind=OperatorKind.SPATIAL, name="BELOW", key=keys[3])
        self.IN_FRONT = CliffordOperator.random(dim, kind=OperatorKind.SPATIAL, name="IN_FRONT", key=keys[4])
        self.BEHIND = CliffordOperator.random(dim, kind=OperatorKind.SPATIAL, name="BEHIND", key=keys[5])

# Usage
spatial = SpatialOperators(dim=2048)
scene = spatial.LEFT_OF.apply(memory["plate"])
```

---

## Comparison: Operators vs Binding

| Feature | Binding (⊗) | Operators (O) |
|---------|-------------|---------------|
| **Inversion accuracy** | ~0.7-0.8 similarity | >0.999 similarity |
| **Directionality** | Symmetric | Asymmetric |
| **Use case** | Associations, composition | Relations, transformations |
| **Composition** | Approximate | Exact (algebraic) |
| **Requirement** | Works with all models | FHRR only (complex vectors) |
| **Speed** | Fast (FFT) | Fast (element-wise) |
| **Memory** | Same as vectors | Small (just phase params) |

**When operators excel:**
- Spatial reasoning (LEFT_OF, ABOVE)
- Semantic roles (AGENT, PATIENT, THEME)
- Exact recovery requirements
- Directional relationships
- Transformation chains

**When binding excels:**
- Symmetric associations
- Quick prototyping
- Non-FHRR models (MAP, Binary)
- Flexible composition
- Similarity-based matching

---

## Advanced: Operator Composition Patterns

### Pattern 1: Chaining Relations

```python
# A is-left-of B, B is-left-of C
# Therefore, A is (2× left-of) C

DOUBLE_LEFT = LEFT_OF.compose(LEFT_OF)

# Encode
spatial_chain = model.opset.bundle(
    memory["A"].vec,
    LEFT_OF.apply(memory["B"]).vec,
    DOUBLE_LEFT.apply(memory["C"]).vec
)
```

---

### Pattern 2: Inverse Relations

```python
# LEFT_OF and RIGHT_OF are inverses
# LEFT_OF^(-1) should approximate RIGHT_OF

# Check
sample = memory["plate"]
left_then_inverse = LEFT_OF.inverse().apply(LEFT_OF.apply(sample))
right_direct = RIGHT_OF.apply(sample)

# These should be similar (depending on operator design)
```

---

### Pattern 3: Hierarchical Roles

```python
# AGENT of ACTION1, ACTION1 is-part-of EVENT
AGENT_OF_EVENT = AGENT.compose(PART_OF)

# Apply hierarchically
event_agent = AGENT_OF_EVENT.apply(memory["complex_event"])
```

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Explain the difference between hypervectors and operators
- [ ] Identify when to use operators vs binding
- [ ] Create and apply Clifford operators
- [ ] Use operator inversion for exact recovery
- [ ] Compose operators for complex relations
- [ ] Encode spatial scenes with directional relations
- [ ] Complete the Clifford operators tutorial
- [ ] Build operator libraries for specific domains

---

## Quick Quiz

**Q1:** What is the main advantage of operators over regular binding?

a) Operators are faster
b) Operators provide exact inversion (>0.999 similarity)
c) Operators use less memory
d) Operators work with all VSA models

<details>
<summary>Answer</summary>
**b) Operators provide exact inversion** - Clifford operators can recover the original hypervector with >0.999 similarity after applying and inverting, compared to ~0.7-0.8 with binding. This enables exact directional reasoning.
</details>

**Q2:** Why are operators necessary for encoding "cup is left of plate"?

a) Binding is too slow
b) We need exact numerical precision
c) Binding loses directionality (can't distinguish "cup left of plate" from "plate left of cup")
d) Binding doesn't work with spatial data

<details>
<summary>Answer</summary>
**c) Binding loses directionality** - Regular binding creates symmetric associations. Using operators like LEFT_OF.apply(plate) preserves the directional relationship, allowing us to query "what is left of plate?" and correctly retrieve "cup".
</details>

**Q3:** How do Clifford operators work mathematically?

a) Matrix multiplication
b) Element-wise phase rotation on complex vectors
c) Gradient descent optimization
d) Discrete Fourier transforms

<details>
<summary>Answer</summary>
**b) Element-wise phase rotation** - Clifford operators apply element-wise phase shifts to complex hypervectors: O(v) = v * exp(i*θ). This is norm-preserving and exactly invertible (rotate back by -θ).
</details>

**Q4:** Can operators be used with Binary or MAP models?

a) Yes, operators work with all models
b) No, operators require complex vectors (FHRR only)
c) Yes, but only Binary
d) Yes, but only MAP

<details>
<summary>Answer</summary>
**b) No, operators require FHRR** - Clifford operators use phase rotation on complex numbers, which requires ComplexHypervector (FHRR). Binary and MAP use real-valued or binary vectors that don't support phase arithmetic.
</details>

---

## Key Takeaways

1. **Operators complement hypervectors** - Hypervectors = concepts, Operators = transformations
2. **Exact inversion** - Similarity >0.999 after apply + inverse (vs ~0.7 for binding)
3. **Directional relations** - Essential for asymmetric spatial/semantic roles
4. **Phase-based** - Work via element-wise phase rotation on complex vectors
5. **Compositional** - Operators compose algebraically (O₁ ∘ O₂)
6. **FHRR-only** - Require ComplexHypervector representation
7. **When to use** - Spatial relations, semantic roles, exact recovery, transformations

---

**Next:** [Lesson 4.2: Spatial Semantic Pointers](02_ssp.md)

Learn how to encode continuous spatial coordinates for navigation, robotics, and spatial reasoning.

**Previous:** [Module 3: Encoders & Applications](../03_encoders/05_analogies.md)
