# VSAX Clifford Operator Integration Specification

## 1. Purpose and Scope

This document specifies a **Clifford-inspired operator layer** for the VSAX library. The goal is to introduce **typed, compositional, invertible operators**—inspired by Clifford algebra—that act on VSAX hypervectors (especially FHRR) **without replacing or re-implementing full Clifford algebra**.

The design preserves VSAX’s strengths (scalability, robustness, GPU-friendliness) while adding:
- exact operator inverses
- explicit operator composition
- interpretable, typed transformations
- clean reasoning over relations and actions

This is **not** a full geometric algebra implementation.

---

## 2. Design Principles

1. **VSA-first**: Hypervectors remain the core representational substrate.
2. **Operator algebra, not memory algebra**: Clifford ideas apply to operators, not stored symbols.
3. **Restricted Clifford subset**: Focus on bivectors and rotors; avoid full multivectors.
4. **Compilation to FHRR**: All operators compile to phase-based actions compatible with existing VSAX ops.
5. **Typed semantics**: Operators carry semantic types (relation, transform, logical, etc.).

---

## 3. Conceptual Mapping

| Clifford Concept | VSAX Implementation |
|-----------------|---------------------|
| Vector          | Hypervector (existing) |
| Bivector        | Elementary Operator (phase generator) |
| Rotor           | Composed Operator (sum of generators) |
| Geometric product | Operator composition |
| Reverse / inverse | Operator inverse |
| Grade           | OperatorType metadata |

---

## 4. Core Abstractions

### 4.1 Operator Interface

```python
class Operator:
    def apply(self, v: HyperVector) -> HyperVector:
        """Apply operator action to a hypervector."""

    def inverse(self) -> "Operator":
        """Return the exact inverse operator."""

    def compose(self, other: "Operator") -> "Operator":
        """Return the composed operator (self ∘ other)."""
```

---

### 4.2 CliffordOperator

Represents a **restricted Clifford element** compiled to an FHRR-compatible phase action.

```python
class CliffordOperator(Operator):
    kind: OperatorKind
    params: np.ndarray  # shape (D,), phase angles
```

- `params[d]` is the phase rotation applied to dimension `d`.
- Dimensionality `D` must match the VSAX hypervector dimension.

---

### 4.3 OperatorKind (Typed Semantics)

```python
from enum import Enum

class OperatorKind(Enum):
    RELATION = "relation"
    TRANSFORM = "transform"
    LOGICAL = "logical"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
```

OperatorKind is inspired by **Clifford grade**, but implemented as explicit metadata.

---

## 5. Operator Semantics

### 5.1 Application (Action on Hypervectors)

For FHRR hypervectors:

```python
def apply(self, v):
    return v * np.exp(1j * self.params)
```

- Elementwise complex multiplication
- Norm-preserving
- Exact inverse

---

### 5.2 Inverse

```python
def inverse(self):
    return CliffordOperator(
        kind=self.kind,
        params=-self.params
    )
```

This corresponds to Clifford **reverse / inverse**.

---

### 5.3 Composition

```python
def compose(self, other):
    assert self.params.shape == other.params.shape
    return CliffordOperator(
        kind=OperatorKind.TRANSFORM,
        params=self.params + other.params
    )
```

- Phase addition corresponds to bivector composition
- Non-commutativity may be modeled by metadata constraints

---

## 6. Construction Patterns

### 6.1 Elementary Operators (Bivector-like)

```python
@staticmethod
def random_relation(name, D):
    params = np.random.uniform(0, 2*np.pi, size=D)
    return CliffordOperator(
        kind=OperatorKind.RELATION,
        params=params
    )
```

Used for:
- semantic roles (AGENT, PATIENT)
- spatial relations (LEFT_OF, ABOVE)
- graph edges

---

### 6.2 Rotors (Composed Operators)

Rotors are created via operator composition:

```python
ROT = ROT_X.compose(ROT_Y)
```

Equivalent to Clifford rotor multiplication.

---

## 7. Example Usage

### 7.1 Scene Encoding

```python
cup = vsax.random()
plate = vsax.random()

LEFT_OF = CliffordOperator.random_relation("left_of", D)

scene = LEFT_OF.apply(plate) + cup
```

---

### 7.2 Viewpoint Change

```python
ROTATE_180 = CliffordOperator.random_transform("rotate_180", D)
scene2 = ROTATE_180.apply(scene)
```

---

### 7.3 Inversion and Reasoning

```python
RIGHT_OF = LEFT_OF.inverse()
assert RIGHT_OF.apply(LEFT_OF.apply(v)) ≈ v
```

---

## 8. Constraints and Validation

- Operator dimension must match hypervector dimension
- Invalid compositions (e.g., RELATION ∘ LOGICAL) may be restricted
- Optional commutation rules may be enforced via metadata

---

## 9. What Is Explicitly Out of Scope

- Full Clifford multivectors
- Blade arithmetic
- 2^n basis expansion
- Symbolic geometric algebra

These are intentionally excluded to preserve scalability.

---

## 10. Integration into VSAX

Suggested module structure:

```
vsax/
 ├── operators/
 │    ├── base.py
 │    ├── clifford.py
 │    ├── spatial.py
 │    ├── semantic.py
```

Existing VSAX hypervector APIs remain unchanged.

---

## 11. Summary

This specification introduces **Clifford-inspired operators** as a lightweight, exact, and interpretable action algebra layered on top of VSAX’s VSA substrate. The result is a system that supports:

- scalable symbolic memory (VSA)
- exact, compositional transformations (Clifford)
- interpretable reasoning over relations

without sacrificing performance or robustness.

---

## 12. Guiding Principle

> *Use VSA to represent what exists; use Clifford-inspired operators to represent what happens.*

