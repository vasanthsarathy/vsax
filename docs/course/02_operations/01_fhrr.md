# Lesson 2.1: Deep Dive - FHRR Operations

**Duration:** ~60 minutes

**Learning Objectives:**

- Understand circular convolution and why it works for binding
- Master FFT-based implementation in JAX
- Grasp phase representation of complex vectors
- Implement exact unbinding using complex conjugate
- Appreciate permutation for sequence encoding

---

## Introduction

FHRR (Fourier Holographic Reduced Representations) uses **complex-valued vectors** and **circular convolution** for binding. This might seem complicated, but it provides two critical advantages:

1. **Exact unbinding** - Complex conjugate is a perfect inverse
2. **Fractional powers** - Enables continuous encoding (spatial/function encoding)

In this lesson, we'll understand the mathematics and implementation of FHRR operations.

---

## Complex Vectors: A Quick Review

FHRR vectors are complex numbers: each element has a **real** and **imaginary** part.

### Unit-Magnitude Complex Vectors

FHRR vectors are normalized to unit magnitude. In polar form:

$$z = e^{i\theta} = \cos(\theta) + i\sin(\theta)$$

where $\theta$ is the **phase angle**.

**Key property:** $|e^{i\theta}| = 1$ for all $\theta$.

```python
import jax.numpy as jnp

# Create a unit-magnitude complex vector
phases = jnp.array([0.0, jnp.pi/4, jnp.pi/2, jnp.pi])
vector = jnp.exp(1j * phases)

print("Vector:", vector)
print("Magnitudes:", jnp.abs(vector))  # All 1.0
```

**Expected Output:**
```
Vector: [1.+0.j  0.707+0.707j  0.+1.j  -1.+0.j]
Magnitudes: [1. 1. 1. 1.]
```

**Intuition:** Complex vectors encode information in **phase angles** rather than magnitudes.

---

## Binding: Circular Convolution via FFT

FHRR binding is **circular convolution**, efficiently computed using the FFT.

### What is Circular Convolution?

For two vectors $\mathbf{a}$ and $\mathbf{b}$ of length $n$, circular convolution is:

$$(a \circledast b)[k] = \sum_{j=0}^{n-1} a[j] \cdot b[(k-j) \mod n]$$

**Intuition:** Each output element is a weighted sum of rotated versions of the inputs.

### Why Use Convolution for Binding?

**The Convolution Theorem** states:

$$\mathcal{F}(a \circledast b) = \mathcal{F}(a) \odot \mathcal{F}(b)$$

where:
- $\mathcal{F}$ is the Fourier transform (FFT)
- $\odot$ is element-wise multiplication
- $\circledast$ is circular convolution

**Key insight:** Convolution in the spatial domain = multiplication in the frequency domain!

This means we can compute convolution efficiently:
1. FFT both vectors: $O(n \log n)$
2. Multiply element-wise: $O(n)$
3. Inverse FFT: $O(n \log n)$

Total: $O(n \log n)$ instead of $O(n^2)$ for direct convolution.

---

## Implementing FHRR Binding

Let's implement binding step-by-step.

### Step 1: Naive Circular Convolution (Don't Use This!)

```python
def circular_convolution_naive(a, b):
    """
    Naive O(n^2) circular convolution - for demonstration only!
    """
    n = len(a)
    result = jnp.zeros(n, dtype=complex)

    for k in range(n):
        for j in range(n):
            result = result.at[k].add(a[j] * b[(k - j) % n])

    return result

# Test
a = jnp.exp(1j * jnp.array([0.1, 0.5, 1.0, 1.5]))
b = jnp.exp(1j * jnp.array([0.2, 0.6, 1.1, 1.6]))

result_naive = circular_convolution_naive(a, b)
print("Naive result:", result_naive)
```

**Problem:** $O(n^2)$ complexity - too slow for d=2048!

### Step 2: FFT-Based Convolution (Fast!)

```python
def circular_convolution_fft(a, b):
    """
    Fast O(n log n) circular convolution using FFT.
    """
    # Transform to frequency domain
    a_fft = jnp.fft.fft(a)
    b_fft = jnp.fft.fft(b)

    # Multiply in frequency domain (binding!)
    result_fft = a_fft * b_fft

    # Transform back to spatial domain
    result = jnp.fft.ifft(result_fft)

    # Normalize to unit magnitude per element
    result = result / jnp.abs(result)

    return result

# Test - should match naive
result_fft = circular_convolution_fft(a, b)
print("FFT result:", result_fft)
print("Match naive?", jnp.allclose(result_naive / jnp.abs(result_naive), result_fft))
```

**Expected Output:**
```
FFT result: [...]
Match naive? True
```

**Performance:** For d=2048, FFT is ~100x faster!

### Step 3: VSAX's Implementation

VSAX's `FHRROperations.bind()` uses the FFT approach with additional optimizations:

```python
from vsax import FHRROperations

ops = FHRROperations()

# Create random complex vectors
import jax.random as random
key = random.PRNGKey(42)

a = jnp.exp(1j * random.uniform(key, (2048,), minval=0, maxval=2*jnp.pi))
b = jnp.exp(1j * random.uniform(random.split(key)[1], (2048,), minval=0, maxval=2*jnp.pi))

# VSAX binding
bound = ops.bind(a, b)

print("Bound shape:", bound.shape)
print("Bound dtype:", bound.dtype)
print("All unit magnitude?", jnp.allclose(jnp.abs(bound), 1.0))
```

**Expected Output:**
```
Bound shape: (2048,)
Bound dtype: complex64
All unit magnitude? True
```

---

## Phase Interpretation

The magic of FHRR is that binding operates on **phase angles**.

### Visualizing Phase Arithmetic

For unit-magnitude complex numbers:

$$e^{i\theta_1} \cdot e^{i\theta_2} = e^{i(\theta_1 + \theta_2)}$$

**Binding adds phases!**

```python
# Example: Simple 2D case
theta_a = jnp.array([0.5, 1.0])
theta_b = jnp.array([0.3, 0.7])

a = jnp.exp(1j * theta_a)
b = jnp.exp(1j * theta_b)

# Element-wise multiply in frequency domain (after FFT, simplified for demo)
result_phases = theta_a + theta_b

print("Input phases A:", theta_a)
print("Input phases B:", theta_b)
print("Result phases:", result_phases)
print("Result phases (mod 2π):", result_phases % (2 * jnp.pi))
```

**Observation:** Binding creates new phase angles that encode the relationship between inputs.

---

## Unbinding: Complex Conjugate

The **complex conjugate** flips the sign of the imaginary part:

$$\overline{e^{i\theta}} = e^{-i\theta}$$

This gives us **exact inversion**:

$$e^{i\theta_1} \cdot e^{i\theta_2} \cdot e^{-i\theta_2} = e^{i\theta_1}$$

### Implementation

```python
from vsax.similarity import cosine_similarity

# Bind two vectors
bound = ops.bind(a, b)

# Create inverse of b (complex conjugate)
b_inv = ops.inverse(b)

# Unbind to recover a
recovered = ops.bind(bound, b_inv)

# Check similarity
sim = cosine_similarity(recovered, a)
print(f"Similarity after unbinding: {sim:.8f}")
```

**Expected Output:**
```
Similarity after unbinding: 0.99999994
```

**Nearly perfect recovery!** This is why FHRR is powerful for deep hierarchies.

---

## Bundling: Sum and Normalize

FHRR bundling is simpler: element-wise sum, then normalize to unit magnitude.

```python
# Create three vectors
c = jnp.exp(1j * random.uniform(random.split(key)[0], (2048,), minval=0, maxval=2*jnp.pi))

# Bundle them
bundled = ops.bundle(a, b, c)

# Check properties
print("Bundled shape:", bundled.shape)
print("Unit magnitude?", jnp.allclose(jnp.abs(bundled), 1.0))

# Check similarity to inputs
sim_a = cosine_similarity(bundled, a)
sim_b = cosine_similarity(bundled, b)
sim_c = cosine_similarity(bundled, c)

print(f"Similarity to a: {sim_a:.4f}")
print(f"Similarity to b: {sim_b:.4f}")
print(f"Similarity to c: {sim_c:.4f}")
```

**Expected Output:**
```
Bundled shape: (2048,)
Unit magnitude? True
Similarity to a: 0.7234
Similarity to b: 0.7189
Similarity to c: 0.7212
```

**Observation:** Bundled vector is similar (~0.7) to all inputs.

---

## Permutation: Circular Shift

Permutation rotates vector elements - useful for sequence encoding.

```python
vec = jnp.array([1+0j, 2+0j, 3+0j, 4+0j, 5+0j])

# Rotate right by 2
shifted_right = ops.permute(vec, 2)
print("Original:      ", vec)
print("Shifted right 2:", shifted_right)

# Rotate left by 2
shifted_left = ops.permute(vec, -2)
print("Shifted left 2: ", shifted_left)
```

**Expected Output:**
```
Original:       [1.+0.j 2.+0.j 3.+0.j 4.+0.j 5.+0.j]
Shifted right 2: [4.+0.j 5.+0.j 1.+0.j 2.+0.j 3.+0.j]
Shifted left 2:  [3.+0.j 4.+0.j 5.+0.j 1.+0.j 2.+0.j]
```

### Use Case: Sequence Encoding

```python
# Encode the sequence "cat sat mat"
from vsax import create_fhrr_model, VSAMemory

model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
memory.add_many(["cat", "sat", "mat"])

# Bind each word with its position (using permutation)
pos0 = memory["cat"].vec
pos1 = model.opset.permute(memory["sat"].vec, 1)
pos2 = model.opset.permute(memory["mat"].vec, 2)

# Bundle to create sequence
sequence = model.opset.bundle(pos0, pos1, pos2)

# Query: What's at position 1?
query = model.opset.permute(sequence, -1)  # Undo the shift
sim_sat = cosine_similarity(query, memory["sat"].vec)
print(f"Similarity to 'sat': {sim_sat:.4f}")
```

**Expected:** High similarity to "sat"!

---

## Complete Example: Encoding and Querying

Let's put it all together with the "cat sat on mat" example from Lesson 1.4, but now understanding the FFT magic.

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

# Create model and memory
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)
memory.add_many(["cat", "mat", "sat", "subject", "object", "verb"])

# Bind concepts with roles using FFT-based circular convolution
cat_subj = model.opset.bind(memory["cat"].vec, memory["subject"].vec)
mat_obj = model.opset.bind(memory["mat"].vec, memory["object"].vec)
sat_verb = model.opset.bind(memory["sat"].vec, memory["verb"].vec)

# Bundle to create sentence (element-wise sum + normalize)
sentence = model.opset.bundle(cat_subj, mat_obj, sat_verb)

# Query using complex conjugate inverse
subject_inv = model.opset.inverse(memory["subject"].vec)  # Complex conjugate
retrieved = model.opset.bind(sentence, subject_inv)

# Measure similarity
sim = cosine_similarity(retrieved, memory["cat"].vec)
print(f"Retrieved 'cat' with similarity: {sim:.4f}")
```

**Expected Output:**
```
Retrieved 'cat' with similarity: 0.9567
```

**Why so accurate?** Complex conjugate provides exact inverse!

---

## Mathematical Properties Summary

| Property | FHRR Binding | FHRR Bundling |
|----------|--------------|---------------|
| **Operation** | Circular convolution (FFT) | Element-wise sum + normalize |
| **Commutative** | ✅ Yes: a⊗b = b⊗a | ✅ Yes: a⊕b = b⊕a |
| **Associative** | ✅ Yes: (a⊗b)⊗c = a⊗(b⊗c) | ✅ Yes: (a⊕b)⊕c = a⊕(b⊕c) |
| **Inverse** | ✅ Exact: conjugate | ❌ Approximate |
| **Similarity** | ~0.0 (dissimilar) | ~0.7 (similar) |
| **Complexity** | O(n log n) | O(n) |

---

## Common Pitfalls

### ❌ Mistake 1: Mixing real and complex

```python
# WRONG: Using real vectors with FHRR
a_real = jax.random.normal(key, (512,))  # Real!
b_complex = jnp.exp(1j * jax.random.uniform(key, (512,)))  # Complex!
bound = ops.bind(a_real, b_complex)  # Type error!
```

**Fix:** Always use complex vectors with FHRR.

### ❌ Mistake 2: Not normalizing

```python
# WRONG: Complex vectors not normalized to unit magnitude
a = jax.random.normal(key, (512,)) + 1j * jax.random.normal(key2, (512,))
# Magnitudes are not 1.0!
```

**Fix:** Use polar form with unit magnitude:
```python
phases = jax.random.uniform(key, (512,), minval=0, maxval=2*jnp.pi)
a = jnp.exp(1j * phases)  # All unit magnitude
```

### ❌ Mistake 3: Forgetting IFFT

```python
# WRONG: Staying in frequency domain
a_fft = jnp.fft.fft(a)
b_fft = jnp.fft.fft(b)
bound = a_fft * b_fft  # Still in frequency domain!
```

**Fix:** Apply inverse FFT to return to spatial domain.

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Explain circular convolution and why FFT makes it fast
- [ ] Understand phase arithmetic in complex vectors
- [ ] Implement FHRR binding using JAX FFT
- [ ] Use complex conjugate for exact unbinding
- [ ] Apply permutation for sequence encoding
- [ ] Appreciate why FHRR provides exact unbinding

---

## Quick Quiz

**Q1:** What is the complexity of FFT-based circular convolution for vectors of length n?

a) O(n)
b) O(n log n)
c) O(n²)
d) O(2^n)

<details>
<summary>Answer</summary>
**b) O(n log n)** - FFT has O(n log n) complexity, much faster than O(n²) naive convolution.
</details>

**Q2:** For unit-magnitude complex vectors, binding (multiplication) operates on:

a) Real parts only
b) Imaginary parts only
c) Phase angles
d) Magnitudes

<details>
<summary>Answer</summary>
**c) Phase angles** - Multiplying $e^{i\theta_1} \cdot e^{i\theta_2} = e^{i(\theta_1 + \theta_2)}$ adds phases.
</details>

**Q3:** The inverse operation for FHRR is:

a) Element-wise division
b) Inverse FFT
c) Complex conjugate
d) Negation

<details>
<summary>Answer</summary>
**c) Complex conjugate** - Conjugate flips the phase sign, providing exact inverse.
</details>

**Q4:** Why is FHRR unbinding exact while MAP unbinding is approximate?

a) FFT is more precise than multiplication
b) Complex conjugate is a perfect mathematical inverse
c) FHRR uses higher dimensions
d) MAP uses lossy compression

<details>
<summary>Answer</summary>
**b) Complex conjugate is a perfect mathematical inverse** - For $e^{i\theta}$, conjugate $e^{-i\theta}$ gives exact cancellation: $e^{i\theta} \cdot e^{-i\theta} = e^{i0} = 1$.
</details>

---

## Hands-On Exercise

**Task:** Implement a multi-level binding chain and test unbinding accuracy.

1. Create 4 random FHRR vectors: a, b, c, d
2. Bind them in a chain: result = ((a⊗b)⊗c)⊗d
3. Unbind step-by-step to recover a
4. Measure similarity at each unbinding step
5. Compare to MAP (expect FHRR to be more accurate)

**Solution:**

```python
from vsax import create_fhrr_model, create_map_model, VSAMemory
from vsax.similarity import cosine_similarity
import jax.random as random

def test_unbinding_chain(model_name, dim=2048):
    """Test multi-level binding and unbinding."""
    if model_name == "FHRR":
        from vsax import create_fhrr_model
        model = create_fhrr_model(dim=dim)
    else:
        from vsax import create_map_model
        model = create_map_model(dim=dim)

    memory = VSAMemory(model)
    memory.add_many(["a", "b", "c", "d"])

    # Chain binding: ((a⊗b)⊗c)⊗d
    step1 = model.opset.bind(memory["a"].vec, memory["b"].vec)
    step2 = model.opset.bind(step1, memory["c"].vec)
    step3 = model.opset.bind(step2, memory["d"].vec)

    # Unbind step-by-step (NEW: using unbind method)
    unbind1 = model.opset.unbind(step3, memory["d"].vec)
    unbind2 = model.opset.unbind(unbind1, memory["c"].vec)
    unbind3 = model.opset.unbind(unbind2, memory["b"].vec)

    # Measure similarities
    sim1 = cosine_similarity(unbind1, step2)
    sim2 = cosine_similarity(unbind2, step1)
    sim3 = cosine_similarity(unbind3, memory["a"].vec)

    print(f"\n{model_name} Unbinding Chain:")
    print(f"  After unbind d: similarity = {sim1:.6f}")
    print(f"  After unbind c: similarity = {sim2:.6f}")
    print(f"  After unbind b: similarity = {sim3:.6f} (recover 'a')")

    return sim3

# Test both models
fhrr_sim = test_unbinding_chain("FHRR")
map_sim = test_unbinding_chain("MAP")

print(f"\nComparison:")
print(f"  FHRR final similarity: {fhrr_sim:.6f}")
print(f"  MAP final similarity:  {map_sim:.6f}")
print(f"  FHRR advantage: {(fhrr_sim - map_sim):.6f}")
```

**Expected:** FHRR maintains >0.99 similarity, MAP degrades to ~0.5-0.7.

---

## Key Takeaways

1. **FHRR uses FFT** for fast O(n log n) circular convolution
2. **Phase arithmetic** - binding adds phase angles
3. **Complex conjugate** provides exact inverse (unbinding)
4. **Bundling** is simple: sum and normalize
5. **Permutation** enables sequence/temporal encoding
6. **Exact unbinding** makes FHRR ideal for deep hierarchies

---

**Next:** [Lesson 2.2: Deep Dive - MAP and Binary Operations](02_map_binary.md)

Learn element-wise operations (MAP) and XOR-based binding (Binary).

**Previous:** [Module 2 Overview](index.md)
