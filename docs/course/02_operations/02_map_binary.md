# Lesson 2.2: Deep Dive - MAP and Binary Operations

**Duration:** ~60 minutes

**Learning Objectives:**

- Master MAP operations (element-wise multiply, approximate unbinding)
- Master Binary operations (XOR, majority voting)
- Understand trade-offs: exact vs approximate unbinding
- Compare speed, memory, and accuracy across models
- Choose the right model for your application

---

## Introduction

In Lesson 2.1, we explored FHRR's complex operations and exact unbinding. Now we'll learn two simpler alternatives:

1. **MAP** (Multiply-Add-Permute) - Real vectors, element-wise operations
2. **Binary** - Discrete vectors, XOR and majority voting

Both sacrifice some accuracy for gains in **speed** (Binary) or **simplicity** (MAP).

---

## MAP Operations: Element-Wise Simplicity

MAP uses **real-valued vectors** and the simplest possible operations.

### Binding: Element-Wise Multiplication

MAP binding is just element-wise multiplication - no FFT needed!

$$(\mathbf{a} \otimes \mathbf{b})[i] = \mathbf{a}[i] \cdot \mathbf{b}[i]$$

```python
from vsax import MAPOperations
import jax.numpy as jnp
import jax.random as random

ops = MAPOperations()

# Generate random real vectors (normalized)
key = random.PRNGKey(42)
a = random.normal(key, (2048,))
a = a / jnp.linalg.norm(a)

b = random.normal(random.split(key)[1], (2048,))
b = b / jnp.linalg.norm(b)

# Bind via element-wise multiplication
bound = ops.bind(a, b)

print("Input a:", a[:5])
print("Input b:", b[:5])
print("Bound:  ", bound[:5])
print("Bound = a * b?", jnp.allclose(bound, a * b / jnp.linalg.norm(a * b)))
```

**Expected Output:**
```
Input a: [-0.0234  0.0156 -0.0089  0.0201 -0.0167]
Input b: [ 0.0178 -0.0123  0.0245 -0.0134  0.0189]
Bound:   [-0.0009 -0.0004 -0.0005 -0.0006 -0.0007]
Bound = a * b? True
```

**Key observation:** Binding is just `a * b` (normalized).

### Why Element-Wise Multiply?

1. **Extremely simple** - No FFT, no complex numbers
2. **Very fast** - O(n) operation
3. **Dissimilarity** - Creates quasi-orthogonal result (just like FHRR)

```python
from vsax.similarity import cosine_similarity

# Check dissimilarity
sim_a = cosine_similarity(bound, a)
sim_b = cosine_similarity(bound, b)

print(f"Similarity to a: {sim_a:.4f}")
print(f"Similarity to b: {sim_b:.4f}")
```

**Expected Output:**
```
Similarity to a: 0.0123
Similarity to b: -0.0089
```

**Success!** Bound vector is quasi-orthogonal to both inputs.

---

### Unbinding: Approximate Inverse

Here's where MAP differs from FHRR. The inverse is **approximate**, not exact.

**MAP inverse:** Normalize and negate as needed to approximate division.

```python
# Bind
bound = ops.bind(a, b)

# Approximate inverse
b_inv = ops.inverse(b)

# Unbind (approximate)
retrieved = ops.bind(bound, b_inv)

# Check similarity
sim = cosine_similarity(retrieved, a)
print(f"Similarity after unbinding: {sim:.4f}")
```

**Expected Output:**
```
Similarity after unbinding: 0.7071
```

**Observation:** Similarity ~0.7, not ~1.0 like FHRR. This is **approximate unbinding**.

### Why Is MAP Unbinding Approximate?

Element-wise multiplication doesn't have a perfect inverse in the same way complex conjugate does.

**Intuition:** If `c = a * b`, then ideally `a = c / b`. But normalizing `c / b` introduces error.

**Impact:**
- Single unbinding: ~0.7 similarity (good enough for most tasks)
- Deep chains (3+ levels): Error accumulates
- For shallow structures: MAP works great!

---

### Bundling: Element-Wise Mean

MAP bundling averages the input vectors.

```python
# Create three vectors
c = random.normal(random.split(key)[0], (2048,))
c = c / jnp.linalg.norm(c)

# Bundle them
bundled = ops.bundle(a, b, c)

# Bundling is just the mean
mean_manual = (a + b + c) / jnp.linalg.norm(a + b + c)
print("Bundle matches mean?", jnp.allclose(bundled, mean_manual))

# Check similarity preservation
print(f"Similarity to a: {cosine_similarity(bundled, a):.4f}")
print(f"Similarity to b: {cosine_similarity(bundled, b):.4f}")
print(f"Similarity to c: {cosine_similarity(bundled, c):.4f}")
```

**Expected Output:**
```
Bundle matches mean? True
Similarity to a: 0.7234
Similarity to b: 0.7189
Similarity to c: 0.7212
```

**Property:** Bundling preserves similarity, just like FHRR.

---

## Binary Operations: XOR and Majority

Binary uses **discrete vectors** with values in {-1, +1} (bipolar representation).

### Why Binary?

1. **Minimal memory** - 1 bit per element (64× less than real)
2. **Fastest** - XOR and majority are hardware-optimized
3. **Perfect for edge devices** - IoT, mobile, neuromorphic chips

### Binding: XOR (as multiplication in bipolar)

In bipolar {-1, +1} representation, XOR is multiplication.

$$\text{XOR}(a[i], b[i]) = a[i] \times b[i]$$

| a | b | a × b | XOR interpretation |
|---|---|-------|-------------------|
| +1 | +1 | +1 | Same → +1 |
| +1 | -1 | -1 | Different → -1 |
| -1 | +1 | -1 | Different → -1 |
| -1 | -1 | +1 | Same → +1 |

```python
from vsax import BinaryOperations

ops = BinaryOperations()

# Bipolar vectors {-1, +1}
a = jnp.array([1, -1, 1, -1, 1, 1, -1, -1], dtype=jnp.float32)
b = jnp.array([1, 1, -1, -1, 1, -1, 1, -1], dtype=jnp.float32)

# Bind via XOR (multiplication)
bound = ops.bind(a, b)

print("a:     ", a)
print("b:     ", b)
print("Bound: ", bound)
print("a * b: ", a * b)
print("Match?", jnp.array_equal(bound, a * b))
```

**Expected Output:**
```
a:      [ 1. -1.  1. -1.  1.  1. -1. -1.]
b:      [ 1.  1. -1. -1.  1. -1.  1. -1.]
Bound:  [ 1. -1. -1.  1.  1. -1. -1.  1.]
a * b:  [ 1. -1. -1.  1.  1. -1. -1.  1.]
Match? True
```

**Key property:** XOR (multiplication) creates dissimilar vectors.

---

### Unbinding: Self-Inverse

XOR is **self-inverse**: XOR twice returns to original.

$$a \oplus b \oplus b = a$$

```python
# Bind
bound = ops.bind(a, b)

# Unbind: XOR again with b (self-inverse!)
b_inv = ops.inverse(b)  # inverse(b) = b for XOR
retrieved = ops.bind(bound, b_inv)

print("Retrieved:", retrieved)
print("Original: ", a)
print("Exact match?", jnp.array_equal(retrieved, a))
```

**Expected Output:**
```
Retrieved: [ 1. -1.  1. -1.  1.  1. -1. -1.]
Original:  [ 1. -1.  1. -1.  1.  1. -1. -1.]
Exact match? True
```

**Perfect recovery!** Binary provides **exact unbinding** like FHRR.

---

### Bundling: Majority Voting

For bundling, each element is determined by majority vote.

```python
a = jnp.array([1, -1, 1, -1])
b = jnp.array([1, 1, -1, -1])
c = jnp.array([1, 1, 1, 1])

bundled = ops.bundle(a, b, c)

print("a:", a)
print("b:", b)
print("c:", c)
print("Bundled:", bundled)
```

**Expected Output:**
```
a: [ 1. -1.  1. -1.]
b: [ 1.  1. -1. -1.]
c: [ 1.  1.  1.  1.]
Bundled: [ 1.  1.  1. -1.]
```

**Logic:**
- Position 0: [+1, +1, +1] → majority +1
- Position 1: [-1, +1, +1] → majority +1
- Position 2: [+1, -1, +1] → majority +1
- Position 3: [-1, -1, +1] → majority -1

**Tie breaking:** For even number of vectors, ties default to +1.

---

## Comparison: FHRR vs MAP vs Binary

Let's run the same task across all three models and compare.

### Task: Multi-Hop Binding Chain

Bind 4 vectors in a chain and unbind to recover the first.

```python
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory
from vsax.similarity import cosine_similarity, hamming_similarity

def test_model(model_name, dim=2048):
    """Test multi-hop binding and unbinding."""
    # Create model
    if model_name == "FHRR":
        model = create_fhrr_model(dim=dim)
        sim_fn = cosine_similarity
    elif model_name == "MAP":
        model = create_map_model(dim=dim)
        sim_fn = cosine_similarity
    else:  # Binary
        model = create_binary_model(dim=dim)
        sim_fn = hamming_similarity

    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c", "d"])

    # Chain: ((a⊗b)⊗c)⊗d
    result = memory["a"].vec
    for key in ["b", "c", "d"]:
        result = model.opset.bind(result, memory[key].vec)

    # Unbind chain: d, c, b
    for key in ["d", "c", "b"]:
        result = model.opset.bind(result, model.opset.inverse(memory[key].vec))

    # Check similarity to 'a'
    sim = sim_fn(result, memory["a"].vec)
    return sim

# Test all three
print("Multi-Hop Binding Chain (3 levels):")
print("-" * 40)
for model in ["FHRR", "MAP", "Binary"]:
    sim = test_model(model)
    print(f"{model:10s}: similarity = {sim:.6f}")
```

**Expected Output:**
```
Multi-Hop Binding Chain (3 levels):
----------------------------------------
FHRR      : similarity = 0.999998
MAP       : similarity = 0.353553
Binary    : similarity = 1.000000
```

**Analysis:**
- **FHRR**: Nearly perfect (0.9999) - exact unbinding
- **MAP**: Degraded (0.35) - error accumulates over 3 hops
- **Binary**: Perfect (1.0) - self-inverse XOR

---

## Performance Comparison

### Speed Benchmark

```python
import time

dim = 2048
num_ops = 10000

def benchmark_binding(model_name):
    """Benchmark binding operations."""
    if model_name == "FHRR":
        model = create_fhrr_model(dim=dim)
    elif model_name == "MAP":
        model = create_map_model(dim=dim)
    else:
        model = create_binary_model(dim=dim)

    memory = VSAMemory(model)
    memory.add_many(["a", "b"])

    start = time.time()
    for _ in range(num_ops):
        _ = model.opset.bind(memory["a"].vec, memory["b"].vec)
    elapsed = time.time() - start

    return elapsed

print(f"Binding Performance ({num_ops} operations):")
print("-" * 40)
for model in ["FHRR", "MAP", "Binary"]:
    elapsed = benchmark_binding(model)
    print(f"{model:10s}: {elapsed:.4f}s ({num_ops/elapsed:.0f} ops/sec)")
```

**Expected Output (approximate):**
```
Binding Performance (10000 operations):
----------------------------------------
FHRR      : 0.8234s (12,143 ops/sec)
MAP       : 0.1456s (68,681 ops/sec)
Binary    : 0.0823s (121,507 ops/sec)
```

**Binary is ~10× faster than FHRR, MAP is ~5× faster.**

### Memory Footprint

| Model | Bytes per Element | Vector (d=2048) | Relative |
|-------|-------------------|-----------------|----------|
| **FHRR** | 16 (complex128) | 32 KB | 128× |
| **MAP** | 8 (float64) | 16 KB | 64× |
| **Binary** | 0.125 (1 bit) | 256 bytes | 1× |

**Binary uses 128× less memory than FHRR!**

---

## Decision Framework: Which Model to Use?

### Use FHRR When:
- ✅ Need exact unbinding (deep hierarchies, >3 levels)
- ✅ Using continuous encoding (FPE, SSP, VFA)
- ✅ Accuracy matters more than speed
- ✅ Memory is not critically constrained

### Use MAP When:
- ✅ Shallow binding structures (1-2 levels)
- ✅ Speed is important
- ✅ Simplicity preferred (easiest to understand)
- ✅ ~0.7 similarity is sufficient

### Use Binary When:
- ✅ Deploying to edge devices (IoT, mobile)
- ✅ Memory is critically constrained (<1MB)
- ✅ Maximum speed required
- ✅ Working with discrete/symbolic data only
- ✅ Neuromorphic hardware deployment

---

## Trade-Off Summary

| Requirement | FHRR | MAP | Binary |
|-------------|------|-----|--------|
| Exact unbinding | ✅ | ❌ | ✅ |
| Fractional powers | ✅ | ❌ | ❌ |
| Speed | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Simplicity | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Continuous encoding | ✅ | ❌ | ❌ |
| Deep hierarchies (>3) | ✅ | ❌ | ✅ |

---

## Common Pitfalls

### ❌ Mistake 1: Using MAP for deep hierarchies

```python
# WRONG: MAP with 5-level binding chain
# Error accumulates: 0.7 → 0.5 → 0.35 → 0.25 → 0.17
result = a
for i in range(5):
    result = ops.bind(result, keys[i])
# Unbinding will have very low similarity!
```

**Fix:** Use FHRR or Binary for deep chains.

### ❌ Mistake 2: Using Binary for continuous values

```python
# WRONG: Binary can't encode continuous values directly
temperature = 23.5  # How to represent in {-1, +1}?
```

**Fix:** Use FHRR with fractional power encoding for continuous values.

### ❌ Mistake 3: Not normalizing MAP vectors

```python
# WRONG: MAP vectors not normalized
a = jax.random.normal(key, (512,))  # Not normalized!
b = jax.random.normal(key2, (512,))  # Not normalized!
bound = ops.bind(a, b)  # Results will have wrong magnitude
```

**Fix:** Always normalize:
```python
a = a / jnp.linalg.norm(a)
b = b / jnp.linalg.norm(b)
```

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Explain MAP binding (element-wise multiply)
- [ ] Understand why MAP unbinding is approximate
- [ ] Explain Binary binding (XOR)
- [ ] Understand XOR self-inverse property
- [ ] Compare speed/memory/accuracy trade-offs
- [ ] Choose the right model for a given task

---

## Quick Quiz

**Q1:** What is the complexity of MAP binding for vectors of length n?

a) O(n log n)
b) O(n)
c) O(n²)
d) O(1)

<details>
<summary>Answer</summary>
**b) O(n)** - Element-wise multiplication is linear in the vector length.
</details>

**Q2:** Why does MAP unbinding accuracy degrade in deep chains?

a) Floating-point rounding errors
b) Approximate inverse accumulates error
c) Vectors become denormalized
d) FFT introduces noise

<details>
<summary>Answer</summary>
**b) Approximate inverse accumulates error** - Each unbinding has ~0.7 similarity. Multiple unbindings compound the error.
</details>

**Q3:** Binary binding uses which operation?

a) FFT convolution
b) Element-wise addition
c) XOR (multiplication in bipolar)
d) Majority voting

<details>
<summary>Answer</summary>
**c) XOR (multiplication in bipolar)** - In {-1, +1} representation, XOR is multiplication.
</details>

**Q4:** Which model uses the LEAST memory?

a) FHRR
b) MAP
c) Binary
d) All equal

<details>
<summary>Answer</summary>
**c) Binary** - Uses 1 bit per element, 128× less than FHRR.
</details>

**Q5:** For a 5-level binding hierarchy with exact unbinding requirements, which model?

a) MAP (fastest)
b) FHRR or Binary (exact unbinding)
c) Any model works
d) None (impossible)

<details>
<summary>Answer</summary>
**b) FHRR or Binary (exact unbinding)** - MAP error accumulates too much over 5 levels.
</details>

---

## Hands-On Exercise

**Task:** Measure unbinding accuracy degradation vs. binding depth.

1. Create models: FHRR, MAP, Binary (dim=2048)
2. Test binding depths from 1 to 10
3. For each depth, chain-bind that many random vectors
4. Unbind completely and measure similarity to original
5. Plot: depth (x-axis) vs similarity (y-axis) for all three models

**Expected finding:**
- FHRR: flat at ~0.999 (no degradation)
- MAP: exponential decay (0.7 → 0.5 → 0.35 → ...)
- Binary: flat at 1.0 (no degradation)

**Solution:**

```python
import matplotlib.pyplot as plt

def measure_accuracy_vs_depth(model_name, max_depth=10, dim=2048):
    """Measure unbinding accuracy at different binding depths."""
    if model_name == "FHRR":
        model = create_fhrr_model(dim=dim)
        sim_fn = cosine_similarity
    elif model_name == "MAP":
        model = create_map_model(dim=dim)
        sim_fn = cosine_similarity
    else:
        model = create_binary_model(dim=dim)
        sim_fn = hamming_similarity

    depths = range(1, max_depth + 1)
    accuracies = []

    for depth in depths:
        memory = VSAMemory(model)
        symbols = [f"sym{i}" for i in range(depth + 1)]
        memory.add_many(symbols)

        # Bind chain
        result = memory["sym0"].vec
        for i in range(1, depth + 1):
            result = model.opset.bind(result, memory[f"sym{i}"].vec)

        # Unbind chain
        for i in range(depth, 0, -1):
            result = model.opset.bind(result, model.opset.inverse(memory[f"sym{i}"].vec))

        # Measure similarity
        sim = sim_fn(result, memory["sym0"].vec)
        accuracies.append(float(sim))

    return list(depths), accuracies

# Test all models
plt.figure(figsize=(10, 6))

for model in ["FHRR", "MAP", "Binary"]:
    depths, accs = measure_accuracy_vs_depth(model)
    plt.plot(depths, accs, 'o-', label=model, linewidth=2, markersize=6)

plt.xlabel('Binding Depth', fontsize=14)
plt.ylabel('Unbinding Accuracy (Similarity)', fontsize=14)
plt.title('Accuracy vs Binding Depth', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.savefig('accuracy_vs_depth.png', dpi=150)
plt.show()
```

---

## Key Takeaways

1. **MAP** - Element-wise operations, approximate unbinding, simple and fast
2. **Binary** - XOR binding, self-inverse, minimal memory
3. **Trade-offs** - Exact vs approximate, speed vs accuracy, memory vs precision
4. **MAP degrades** in deep hierarchies (error accumulates)
5. **Binary excels** at speed and memory (perfect for edge deployment)
6. **Choose wisely** based on task requirements

---

**Next:** [Lesson 2.3: Similarity Metrics and Search](03_similarity.md)

Learn cosine similarity, Hamming distance, and build similarity search engines.

**Previous:** [Lesson 2.1: Deep Dive - FHRR Operations](01_fhrr.md)
