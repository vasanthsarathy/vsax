

# Resonator Networks

Resonator networks solve the **factorization problem** in Vector Symbolic Architectures: given a composite vector formed by binding multiple factors, recover the original factors.

This implementation is based on:
> Frady, E. P., Kleyko, D., & Sommer, F. T. (2020). A Theory of Sequence Indexing and Working Memory in Recurrent Neural Networks. Neural Computation.

## The Factorization Problem

### Problem Statement

Given a composite vector:
```
s = a ⊙ b ⊙ c
```

Where `a`, `b`, `c` are vectors from known codebooks `A`, `B`, `C`, find the specific vectors that were bound together.

### Why It's Hard

- **Superposition**: After binding, the composite is a new vector that doesn't obviously contain the factors
- **Search space**: With codebooks of size N, there are N³ possible combinations for 3 factors
- **Noise**: Binding isn't always perfectly reversible (especially for MAP model)

### The Solution: Resonator Networks

Resonator networks use an iterative algorithm that alternates between:
1. **Unbinding**: Remove current estimates of other factors
2. **Cleanup**: Project result onto codebook (find nearest clean vector)

The algorithm converges to the correct factors through **resonance** - mutual reinforcement of consistent estimates.

## Quick Start

### Basic Two-Factor Factorization

```python
from vsax import create_binary_model, VSAMemory
from vsax.resonator import CleanupMemory, Resonator

# Create model and memory
model = create_binary_model(dim=10000, bipolar=True)
memory = VSAMemory(model)
memory.add_many(["red", "blue", "circle", "square"])

# Create composite: red ⊙ circle
composite = model.opset.bind(
    memory["red"].vec,
    memory["circle"].vec
)

# Create codebooks for each factor position
colors = CleanupMemory(["red", "blue"], memory)
shapes = CleanupMemory(["circle", "square"], memory)

# Factorize!
resonator = Resonator([colors, shapes], model.opset)
factors = resonator.factorize(composite)

print(factors)  # ["red", "circle"]
```

### Three-Factor Factorization

```python
# Add size attribute
memory.add_many(["large", "small"])

# Create composite: red ⊙ circle ⊙ large
composite = model.opset.bind(
    model.opset.bind(memory["red"].vec, memory["circle"].vec),
    memory["large"].vec
)

# Create codebooks
colors = CleanupMemory(["red", "blue"], memory)
shapes = CleanupMemory(["circle", "square"], memory)
sizes = CleanupMemory(["large", "small"], memory)

# Factorize with three factors
resonator = Resonator([colors, shapes, sizes], model.opset)
factors = resonator.factorize(composite)

print(factors)  # ["red", "circle", "large"]
```

## Core Components

### CleanupMemory

`CleanupMemory` implements codebook projection - finding the nearest vector from a set of known vectors.

#### Creating a Cleanup Memory

```python
from vsax.resonator import CleanupMemory

# Define codebook symbols
colors = CleanupMemory(["red", "blue", "green"], memory)

# With similarity threshold
colors = CleanupMemory(
    ["red", "blue", "green"],
    memory,
    threshold=0.5  # Return None if similarity < 0.5
)
```

#### Querying

```python
# Simple query
result = colors.query(noisy_vector)
print(result)  # "red"

# With similarity score
result, similarity = colors.query(noisy_vector, return_similarity=True)
print(f"{result}: {similarity:.3f}")

# Top-k matches
top_3 = colors.query_top_k(noisy_vector, k=3)
for symbol, sim in top_3:
    print(f"{symbol}: {sim:.3f}")
```

#### How It Works

For binary/bipolar vectors, cleanup uses dot product similarity:
```python
similarities = codebook_matrix @ query_vector
best_idx = argmax(similarities)
```

This is equivalent to the projection operation from the paper: `g(XX^T v)`

### Resonator

`Resonator` implements the iterative factorization algorithm.

#### Creating a Resonator

```python
from vsax.resonator import Resonator

resonator = Resonator(
    codebooks=[colors, shapes, sizes],  # One per factor
    opset=model.opset,                  # Defines bind/unbind
    max_iterations=100,                 # Stop after N iterations
    convergence_threshold=3             # Stop if stable for N iterations
)
```

#### Factorization

```python
# Basic factorization
factors = resonator.factorize(composite)

# With initial estimates (optional)
factors = resonator.factorize(
    composite,
    initial_estimates=["red", "circle", "large"]
)

# Get convergence history
factors, history = resonator.factorize(composite, return_history=True)
print(f"Converged in {len(history)} iterations")
for i, step in enumerate(history):
    print(f"  Iteration {i}: {step}")

# Batch factorization
import jax.numpy as jnp
composites = jnp.stack([comp1, comp2, comp3])
all_factors = resonator.factorize_batch(composites)
```

## The Algorithm

### Resonance Equations

For a 3-factor composite `s = a ⊙ b ⊙ c`, the resonator updates are:

```
â(t+1) = cleanup_A(s ⊙ inv(b̂(t)) ⊙ inv(ĉ(t)))
b̂(t+1) = cleanup_B(s ⊙ inv(â(t)) ⊙ inv(ĉ(t)))
ĉ(t+1) = cleanup_C(s ⊙ inv(â(t)) ⊙ inv(b̂(t)))
```

Where:
- `â, b̂, ĉ` are current estimates
- `cleanup_X` projects onto codebook X
- `inv(·)` is the unbinding operation

### Initialization

On the first iteration (when estimates are None), the algorithm uses **superposition initialization**:
```python
superposition = sum(all_vectors_in_codebook)
```

This gives the algorithm information about all possible factors simultaneously.

### Convergence

The algorithm stops when:
1. Estimates don't change for `convergence_threshold` iterations (default: 3), OR
2. `max_iterations` is reached (default: 100)

For binary VSA with exact unbinding, convergence is typically very fast (< 10 iterations).

## Best Practices

### Model Selection

**Binary VSA (Recommended for Resonator)**
- ✅ Exact unbinding (self-inverse property)
- ✅ Fast convergence
- ✅ High accuracy
- ⚠️ Requires high dimensionality (≥10,000)

```python
model = create_binary_model(dim=10000, bipolar=True)
```

**FHRR (Complex)**
- ✅ Exact unbinding (complex conjugate)
- ✅ Lower dimensionality needed (≥512)
- ⚠️ More complex operations

```python
model = create_fhrr_model(dim=512)
```

**MAP (Real)**
- ⚠️ Approximate unbinding
- ⚠️ May require more iterations
- ✅ Simple operations

```python
model = create_map_model(dim=512)
```

### Codebook Design

**Separate Semantic Spaces**
```python
# Good: Codebooks represent different semantic categories
colors = CleanupMemory(["red", "blue", "green"], memory)
shapes = CleanupMemory(["circle", "square"], memory)
sizes = CleanupMemory(["large", "small"], memory)
```

**Avoid Overlap**
```python
# Bad: Same symbols in multiple codebooks creates ambiguity
codebook1 = CleanupMemory(["red", "blue"], memory)
codebook2 = CleanupMemory(["red", "green"], memory)  # "red" appears twice!
```

**Balanced Sizes**
```python
# Codebooks don't need to be the same size
colors = CleanupMemory(["red", "blue", "green", "yellow"], memory)  # 4 items
shapes = CleanupMemory(["circle", "square"], memory)                # 2 items
```

### Performance Tips

**Use Binary Model for Best Performance**
```python
# Binary VSA is fastest and most accurate for resonator
model = create_binary_model(dim=10000, bipolar=True)
```

**Batch Processing**
```python
# Process multiple composites efficiently
composites = jnp.stack([c1, c2, c3, c4])
results = resonator.factorize_batch(composites)
```

**Monitor Convergence**
```python
# Check if convergence is too slow
factors, history = resonator.factorize(composite, return_history=True)
if len(history) > 50:
    print("Warning: Slow convergence, may need higher dimensionality")
```

## Common Use Cases

### Structured Data Decoding

Decode attribute-value structures:
```python
# Encode: object ⊙ color ⊙ shape ⊙ size
objects = ["obj1", "obj2", "obj3"]
colors = ["red", "blue", "green"]
shapes = ["circle", "square", "triangle"]
sizes = ["large", "small"]

# ... factorize to recover attributes
```

### Sequence Indexing

Recover elements from indexed sequences:
```python
# Encode: item ⊙ position
# Example: "apple" ⊙ position_1 ⊙ "banana" ⊙ position_2
```

### Tree Decoding

Decode hierarchical tree structures:
```python
# Encode: parent ⊙ left_child ⊙ right_child
# See examples/resonator_tree_search.py for details
```

### Graph Structure Recovery

Decode graph edges:
```python
# Encode: edge ⊙ source_node ⊙ target_node
```

## Advanced Topics

### Custom Convergence Criteria

```python
class CustomResonator(Resonator):
    def factorize(self, composite, **kwargs):
        # Custom convergence logic
        # Check similarity scores, add early stopping, etc.
        ...
```

### Hierarchical Factorization

For nested structures, factorize recursively:
```python
# First level: Get high-level factors
factors_L1 = resonator_L1.factorize(composite)

# Second level: Factorize one of the factors
factors_L2 = resonator_L2.factorize(factors_L1[0])
```

### Error Analysis

```python
# Check which factors are uncertain
factors, history = resonator.factorize(composite, return_history=True)

# Look for oscillation (indicates ambiguity)
for i in range(len(history) - 5):
    if history[i] == history[i + 4]:
        print(f"Factor {i} may be ambiguous")
```

## Examples

See `examples/resonator_tree_search.py` for complete working examples:

1. **Simple tree decoding** - Basic two-factor case
2. **Multiple trees** - Decoding different structures
3. **Convergence history** - Monitoring the iterative process
4. **Nested trees** - Hierarchical structures
5. **Batch processing** - Multiple composites at once
6. **Error correction** - Robustness to noise

## References

- Frady, E. P., Kleyko, D., & Sommer, F. T. (2020). A Theory of Sequence Indexing and Working Memory in Recurrent Neural Networks. *Neural Computation*.
- Plate, T. A. (1995). Holographic reduced representations. *IEEE Transactions on Neural Networks*.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*.
