# Resonator Networks API

## Overview

The resonator module implements resonator networks for VSA factorization.

Given a composite vector `s = a ⊙ b ⊙ c`, resonator networks iteratively recover the factors `a`, `b`, `c` from known codebooks.

## CleanupMemory

::: vsax.resonator.CleanupMemory
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

### Example

```python
from vsax import create_binary_model, VSAMemory
from vsax.resonator import CleanupMemory

model = create_binary_model(dim=10000, bipolar=True)
memory = VSAMemory(model)
memory.add_many(["red", "blue", "green"])

# Create cleanup memory
cleanup = CleanupMemory(["red", "blue", "green"], memory)

# Query with noisy vector
result = cleanup.query(noisy_vector)
print(result)  # "red"

# Get top-k matches
top_3 = cleanup.query_top_k(noisy_vector, k=3)
for symbol, similarity in top_3:
    print(f"{symbol}: {similarity:.3f}")
```

## Resonator

::: vsax.resonator.Resonator
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

### Example

```python
from vsax import create_binary_model, VSAMemory
from vsax.resonator import CleanupMemory, Resonator

# Setup
model = create_binary_model(dim=10000, bipolar=True)
memory = VSAMemory(model)
memory.add_many(["red", "blue", "circle", "square"])

# Create composite
composite = model.opset.bind(
    memory["red"].vec,
    memory["circle"].vec
)

# Create codebooks
colors = CleanupMemory(["red", "blue"], memory)
shapes = CleanupMemory(["circle", "square"], memory)

# Factorize
resonator = Resonator([colors, shapes], model.opset)
factors = resonator.factorize(composite)

print(factors)  # ["red", "circle"]
```

## Algorithm Details

### Resonance Equations

For a composite `s = a ⊙ b ⊙ c`, the update equations are:

```
x̂(t+1) = g(XX^T(s ⊙ ŷ(t) ⊙ ẑ(t)))
ŷ(t+1) = g(YY^T(s ⊙ x̂(t) ⊙ ẑ(t)))
ẑ(t+1) = g(ZZ^T(s ⊙ x̂(t) ⊙ ŷ(t)))
```

Where:
- `x̂, ŷ, ẑ` are factor estimates
- `X, Y, Z` are codebook matrices
- `g(XX^T·)` is the cleanup operation (codebook projection)
- `⊙` is the binding operation

### Cleanup Operation

The cleanup operation `g(XX^T v)` projects vector `v` onto the nearest vector in codebook `X`.

For binary/bipolar vectors:
```python
similarities = codebook_matrix @ v
best_idx = argmax(similarities)
result = codebook[best_idx]
```

For complex/real vectors:
```python
similarities = [cosine_similarity(v, c) for c in codebook]
best_idx = argmax(similarities)
result = codebook[best_idx]
```

### Initialization

On the first iteration (no prior estimates), the algorithm uses **superposition initialization**:

```python
initial_estimate = sum(all_vectors_in_codebook)
```

This provides information about all possible factors simultaneously.

### Convergence

The algorithm stops when:

1. **Stable convergence**: Estimates unchanged for `convergence_threshold` iterations (default: 3)
2. **Max iterations**: Reached `max_iterations` (default: 100)

Binary VSA typically converges in < 10 iterations due to exact unbinding.

## Performance Characteristics

### Time Complexity

Per iteration for N factors with codebook size M and dimension D:
- **Unbinding**: O(N × D) - binding operations
- **Cleanup**: O(M × D) - dot products with codebook
- **Total**: O(N × M × D) per iteration

Typical iterations to convergence: 5-20

### Space Complexity

- **Codebooks**: O(M × D) per codebook
- **Estimates**: O(N × D)
- **Total**: O((N + M) × D)

### Recommendations

**Dimensionality**:
- Binary VSA: ≥10,000 dimensions
- FHRR: ≥512 dimensions
- MAP: ≥512 dimensions

**Codebook Size**:
- Works well with codebooks of 2-100 items
- Larger codebooks may require more iterations

**Number of Factors**:
- Tested with 2-3 factors
- Can handle more but convergence may slow

## Common Patterns

### Two-Factor Factorization

```python
# Encode
composite = bind(a, b)

# Setup codebooks
codebook_a = CleanupMemory(["a1", "a2", "a3"], memory)
codebook_b = CleanupMemory(["b1", "b2", "b3"], memory)

# Factorize
resonator = Resonator([codebook_a, codebook_b], opset)
factors = resonator.factorize(composite)
```

### Three-Factor Factorization

```python
# Encode
composite = bind(bind(a, b), c)

# Setup
codebook_a = CleanupMemory([...], memory)
codebook_b = CleanupMemory([...], memory)
codebook_c = CleanupMemory([...], memory)

# Factorize
resonator = Resonator([codebook_a, codebook_b, codebook_c], opset)
factors = resonator.factorize(composite)
```

### Batch Processing

```python
import jax.numpy as jnp

# Create multiple composites
composites = jnp.stack([comp1, comp2, comp3, comp4])

# Batch factorize
results = resonator.factorize_batch(composites)
# results[i] contains factors for composites[i]
```

### Monitoring Convergence

```python
factors, history = resonator.factorize(
    composite,
    return_history=True
)

print(f"Converged in {len(history)} iterations")
for i, step in enumerate(history):
    print(f"Iteration {i}: {step}")
```

## Error Handling

### Invalid Codebook

```python
# Raises ValueError if symbol not in memory
cleanup = CleanupMemory(["missing_symbol"], memory)
# ValueError: Symbol 'missing_symbol' not found in memory
```

### Wrong Number of Estimates

```python
# Raises ValueError if initial estimates don't match codebook count
resonator = Resonator([codebook1, codebook2], opset)
factors = resonator.factorize(composite, initial_estimates=["a"])
# ValueError: Expected 2 initial estimates, got 1
```

### Invalid Initial Estimate

```python
# Raises ValueError if estimate not in corresponding codebook
factors = resonator.factorize(
    composite,
    initial_estimates=["valid", "not_in_codebook"]
)
# ValueError: Initial estimate 'not_in_codebook' not in codebook 1
```

## See Also

- [User Guide: Resonator Networks](../../guide/resonator.md)
- [Example: Tree Search](https://github.com/vasanthsarathy/vsax/blob/main/examples/resonator_tree_search.py)
- [Paper: Frady et al. (2020)](https://direct.mit.edu/neco/article/32/9/1651/95535/A-Theory-of-Sequence-Indexing-and-Working-Memory)
