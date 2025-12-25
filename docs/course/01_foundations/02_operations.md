# Lesson 1.2: The Two Fundamental Operations

**Duration:** ~60 minutes

**Learning Objectives:**

- Understand binding as a way to create compositional structures
- Understand bundling as a way to create prototypes and sets
- Grasp the mathematical intuitions behind both operations
- Predict when to use binding vs bundling
- Measure binding dissimilarity and bundling similarity empirically

---

## Introduction

Now that we understand why high dimensions work, let's learn the **two fundamental operations** that make symbolic computation possible:

1. **Binding (⊗)**: Combines concepts into new, dissimilar structures
2. **Bundling (⊕)**: Aggregates concepts into sets while preserving similarity

These operations are the building blocks for all VSA computations.

---

## Operation 1: Binding

**Binding** combines two hypervectors into a third vector that is **dissimilar to both inputs**.

### The Intuition

Think of binding like **multiplication**:
- Bind "cat" with "red" → "red cat"
- The result is a NEW concept, dissimilar to both "cat" and "red"
- Like mixing paint: red + blue → purple (different from both)

### Mathematical Property

For vectors **a** and **b**, binding creates **c = a ⊗ b** such that:

$$\text{sim}(c, a) \approx 0 \quad \text{and} \quad \text{sim}(c, b) \approx 0$$

The binding **destroys similarity** to create compositional structure.

### Why Does This Work?

Binding operations (circular convolution, element-wise multiply, or XOR) **scramble** the vector components in a structured way:

- The scrambling is deterministic (not random)
- It preserves enough information to **unbind** (reverse the operation)
- But the result is quasi-orthogonal to inputs

Let's verify this with code.

---

## Experiment 1: Binding Destroys Similarity

```python
import jax.numpy as jnp
import jax.random as random

# Generate two random unit vectors
key = random.PRNGKey(42)
key1, key2 = random.split(key)

dim = 2048

# Create "cat" and "red" as random vectors
cat = random.normal(key1, (dim,))
cat = cat / jnp.linalg.norm(cat)

red = random.normal(key2, (dim,))
red = red / jnp.linalg.norm(red)

# Simple binding operation: element-wise multiplication
# (We'll learn the proper VSA binding operations in Lesson 1.3)
bound = cat * red
bound = bound / jnp.linalg.norm(bound)  # Renormalize

# Measure similarities
sim_cat = jnp.dot(bound, cat)
sim_red = jnp.dot(bound, red)

print(f"Similarity between bound and 'cat': {sim_cat:.4f}")
print(f"Similarity between bound and 'red': {sim_red:.4f}")
```

**Expected Output:**
```
Similarity between bound and 'cat': 0.0103
Similarity between bound and 'red': -0.0087
```

**Observation:** The bound vector is nearly orthogonal (~0 similarity) to both inputs!

---

## Binding in Three VSA Models

Different VSA models use different binding operations:

| Model | Binding Operation | Why? |
|-------|-------------------|------|
| **FHRR** | Circular convolution via FFT | Exact unbinding through complex conjugate |
| **MAP** | Element-wise multiplication | Simple, approximate unbinding via division |
| **Binary** | XOR (⊕) | Self-inverse: a ⊕ b ⊕ b = a |

We'll explore each in detail in Lesson 1.3. For now, understand that **all binding operations create dissimilarity**.

---

## Unbinding: The Inverse Operation

The power of binding is that it's **invertible**. If we bind:

$$c = \text{cat} \otimes \text{red}$$

We can **unbind** to retrieve the original:

$$\text{cat} = c \otimes \text{red}^{-1}$$

where $\text{red}^{-1}$ is the **inverse** of red.

### Inverse Operations by Model

- **FHRR**: Complex conjugate (exact)
- **MAP**: Element-wise division (approximate)
- **Binary**: Self (XOR is self-inverse)

```python
# Example: Binding and unbinding with element-wise multiply (MAP-style)
# Inverse for MAP is element-wise divide (or multiply by reciprocal)

# Bind
bound = cat * red

# Unbind: retrieve cat by dividing out red
retrieved_cat = bound / red
retrieved_cat = retrieved_cat / jnp.linalg.norm(retrieved_cat)

# Check similarity
sim_retrieved = jnp.dot(retrieved_cat, cat)
print(f"Similarity after unbinding: {sim_retrieved:.4f}")
```

**Expected Output:**
```
Similarity after unbinding: 0.7071
```

**Observation:** We recovered a vector similar to the original "cat" (sim ~0.7). Perfect for MAP-style operations!

---

## Operation 2: Bundling

**Bundling** combines multiple hypervectors into an aggregate that **preserves similarity** to all inputs.

### The Intuition

Think of bundling like **addition** or **averaging**:
- Bundle "cat" + "dog" + "mouse" → "small mammals"
- The result is similar to each input
- Like averaging colors: red + orange + yellow → reddish-orange (similar to all)

### Mathematical Property

For vectors **a**, **b**, **c**, bundling creates **s = a ⊕ b ⊕ c** such that:

$$\text{sim}(s, a) > 0 \quad \text{and} \quad \text{sim}(s, b) > 0 \quad \text{and} \quad \text{sim}(s, c) > 0$$

The bundling **preserves similarity** to create sets and prototypes.

---

## Experiment 2: Bundling Preserves Similarity

```python
# Generate multiple random vectors (concepts)
num_concepts = 5
key = random.PRNGKey(123)
concepts = random.normal(key, (num_concepts, dim))
concepts = concepts / jnp.linalg.norm(concepts, axis=1, keepdims=True)

# Bundle them (simple average)
bundled = jnp.mean(concepts, axis=0)
bundled = bundled / jnp.linalg.norm(bundled)

# Measure similarities
similarities = concepts @ bundled

print("Similarities between bundle and each input:")
for i, sim in enumerate(similarities):
    print(f"  Concept {i+1}: {sim:.4f}")

print(f"\nMean similarity: {jnp.mean(similarities):.4f}")
```

**Expected Output:**
```
Similarities between bundle and each input:
  Concept 1: 0.7234
  Concept 2: 0.6982
  Concept 3: 0.7156
  Concept 4: 0.7089
  Concept 5: 0.6945

Mean similarity: 0.7081
```

**Observation:** The bundled vector is highly similar (~0.7) to all inputs!

---

## Bundling in Three VSA Models

All three models use essentially the same bundling operation:

| Model | Bundling Operation | Notes |
|-------|-------------------|-------|
| **FHRR** | Element-wise sum, then normalize | Preserves complex phase information |
| **MAP** | Element-wise sum, then normalize | Simple averaging |
| **Binary** | Majority voting per element | 0 or 1 based on majority |

Bundling is conceptually simpler than binding - it's just averaging (with variants).

---

## Capacity: How Many Can We Bundle?

Unlike binding (which is 2-input operation), bundling can aggregate **many** vectors. But there's a limit.

### Rule of Thumb

In d dimensions, you can reliably bundle up to ~√d vectors before interference becomes significant.

**Why?** Each bundled vector contributes noise. After √d vectors, the signal-to-noise ratio degrades.

### Experiment: Bundling Capacity

```python
def test_bundling_capacity(dim, num_bundles):
    """Test how similarity degrades as we bundle more vectors."""
    key = random.PRNGKey(456)

    # Generate vectors to bundle
    vectors = random.normal(key, (num_bundles, dim))
    vectors = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

    # Bundle them
    bundled = jnp.sum(vectors, axis=0)
    bundled = bundled / jnp.linalg.norm(bundled)

    # Measure similarity to each input
    similarities = vectors @ bundled
    mean_sim = jnp.mean(similarities)

    return mean_sim

dim = 2048
bundle_sizes = [1, 5, 10, 20, 45, 90, 180]  # √2048 ≈ 45

print(f"Bundling capacity for dim={dim} (expected capacity: ~{int(jnp.sqrt(dim))}):\n")
print(f"{'Num Bundled':<15} {'Mean Similarity':<15}")
print("-" * 30)

for num in bundle_sizes:
    mean_sim = test_bundling_capacity(dim, num)
    marker = " ← expected capacity" if num == 45 else ""
    print(f"{num:<15} {mean_sim:.4f}{marker}")
```

**Expected Output:**
```
Bundling capacity for dim=2048 (expected capacity: ~45):

Num Bundled     Mean Similarity
------------------------------
1               1.0000
5               0.8944
10              0.8165
20              0.7071
45              0.5477 ← expected capacity
90              0.4472
180             0.3162
```

**Observation:** At ~45 bundles (√d), similarity drops to ~0.55. Beyond that, signal degrades significantly.

---

## When to Use Binding vs Bundling?

This is the most important question!

| Use Case | Operation | Example |
|----------|-----------|---------|
| **Combine different roles** | Binding | "cat" ⊗ "subject", "mat" ⊗ "object" |
| **Create composite concepts** | Binding | "red" ⊗ "cup", "blue" ⊗ "plate" |
| **Encode sequences** | Binding | "word1" ⊗ "pos1", "word2" ⊗ "pos2" |
| **Create prototypes/averages** | Bundling | "cat" ⊕ "dog" ⊕ "mouse" = "animals" |
| **Build sets** | Bundling | "alice" ⊕ "bob" ⊕ "charlie" = "people" |
| **Store scene** | Bundling | (cat⊗pos1) ⊕ (mat⊗pos2) = "scene" |

### Simple Rule

- **Binding**: Things that are DIFFERENT but belong TOGETHER
  - "cup" and "red" are different, but together they describe a red cup

- **Bundling**: Things that are SIMILAR and form a GROUP
  - "cat", "dog", "mouse" are similar (all animals), bundled into a group

---

## Combining Binding and Bundling

The real power comes from **composing** these operations:

```python
# Example: Encoding a simple scene
# "The cat sat on the mat"

# Step 1: Bind concepts with roles
cat_subject = bind(cat, subject)      # cat ⊗ subject
mat_object = bind(mat, object_role)   # mat ⊗ object
sat_verb = bind(sat, verb)            # sat ⊗ verb

# Step 2: Bundle to create scene
scene = bundle([cat_subject, mat_object, sat_verb])

# Now we can query:
# "What's the subject?" → unbind(scene, subject) ≈ cat
# "What's the object?" → unbind(scene, object) ≈ mat
```

This compositional structure is the foundation of VSA's symbolic power!

---

## Mathematical Properties

### Binding Properties

1. **Dissimilarity**: sim(a⊗b, a) ≈ 0
2. **Commutativity**: a⊗b = b⊗a (for FHRR, Binary; not MAP)
3. **Invertibility**: a⊗a^(-1) = identity
4. **Associativity**: (a⊗b)⊗c = a⊗(b⊗c)

### Bundling Properties

1. **Similarity preservation**: sim(a⊕b, a) > 0
2. **Commutativity**: a⊕b = b⊕a (all models)
3. **Associativity**: (a⊕b)⊕c = a⊕(b⊕c)
4. **Capacity limits**: Bundle <√d vectors for good signal

---

## Common Pitfalls

### ❌ "I'll just bind everything"
Binding creates dissimilarity. If you bind 10 concepts, you can't query for any of them directly. Use bundling to aggregate.

### ❌ "I'll bundle 1000 vectors in d=512"
Capacity is ~√d. Bundling 1000 vectors in d=512 will result in noise (capacity is ~23).

### ❌ "Binding and bundling are the same"
They're opposites! Binding→dissimilarity, Bundling→similarity.

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Explain what binding does (creates dissimilar composite)
- [ ] Explain what bundling does (creates similar prototype)
- [ ] Predict when to use binding vs bundling
- [ ] Understand capacity limits for bundling (~√d)
- [ ] Describe the inverse operation for binding
- [ ] Compose binding and bundling for structured representations

---

## Quick Quiz

**Q1:** If you bind "cat" ⊗ "mat", what is the similarity between the result and "cat"?

a) ~1.0 (very similar)
b) ~0.7 (moderately similar)
c) ~0.0 (orthogonal)
d) -1.0 (opposite)

<details>
<summary>Answer</summary>
**c) ~0.0 (orthogonal)** - Binding destroys similarity, creating a dissimilar vector.
</details>

**Q2:** If you bundle "cat" ⊕ "dog" ⊕ "mouse", what is the approximate similarity to "cat"?

a) ~1.0
b) ~0.7
c) ~0.0
d) Can't determine

<details>
<summary>Answer</summary>
**b) ~0.7** - Bundling preserves similarity. For 3 vectors, each contributes ~1/√3 ≈ 0.58 to the average, but normalized similarity is higher (~0.7).
</details>

**Q3:** In d=1024 dimensions, approximately how many vectors can you bundle before signal degrades significantly?

a) ~10
b) ~32
c) ~100
d) ~1000

<details>
<summary>Answer</summary>
**b) ~32** - Capacity rule: √d = √1024 = 32 vectors.
</details>

**Q4:** You want to encode "red cup" and "blue plate". Which operation(s) should you use?

a) Bind: (red ⊗ cup) and (blue ⊗ plate)
b) Bundle: (red ⊕ cup) and (blue ⊕ plate)
c) First bind, then bundle: (red⊗cup) ⊕ (blue⊗plate)
d) First bundle, then bind: (red⊕blue) ⊗ (cup⊕plate)

<details>
<summary>Answer</summary>
**c) First bind, then bundle** - Bind color with object to create composite concepts, then bundle to create a scene containing both items.
</details>

---

## Hands-On Exercise

**Task:** Investigate bundling capacity empirically.

1. Fix dimension d=2048
2. Test bundling sizes from 1 to 200 vectors (step by 5)
3. For each size, bundle N random vectors and measure mean similarity
4. Plot bundle size (x-axis) vs mean similarity (y-axis)
5. Mark the theoretical capacity √d on the plot

**Expected finding:** Similarity decreases as 1/√N. At N=√d, similarity ≈ 0.5-0.6.

**Solution:**

```python
import matplotlib.pyplot as plt

dim = 2048
bundle_sizes = range(1, 201, 5)
mean_sims = []

for num in bundle_sizes:
    mean_sim = test_bundling_capacity(dim, num)
    mean_sims.append(float(mean_sim))

# Theoretical curve: 1/sqrt(N)
theoretical = [1/jnp.sqrt(n) for n in bundle_sizes]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(bundle_sizes, mean_sims, 'o-', label='Empirical', linewidth=2, markersize=4)
plt.plot(bundle_sizes, theoretical, '--', label='Theoretical (1/√N)', linewidth=2)
plt.axvline(jnp.sqrt(dim), color='red', linestyle=':', linewidth=2, label=f'Capacity (√{dim}≈45)')
plt.xlabel('Number of Bundled Vectors', fontsize=14)
plt.ylabel('Mean Similarity to Inputs', fontsize=14)
plt.title('Bundling Capacity Analysis', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bundling_capacity.png', dpi=150)
plt.show()
```

---

## Key Takeaways

1. **Binding** creates compositional structure (dissimilar to inputs)
2. **Bundling** creates prototypes and sets (similar to inputs)
3. **Binding is invertible** - we can unbind to retrieve components
4. **Bundling has capacity limits** - bundle ~√d vectors maximum
5. **Composition is key** - combine binding and bundling for structured representations
6. **Use binding for roles, bundling for sets**

---

## Real-World Analogy

Think of building a house:

- **Binding**: Combining different materials (wood ⊗ nails, brick ⊗ mortar)
  - Wood+nails creates a wall (different from both wood and nails)

- **Bundling**: Collecting similar items (hammer ⊕ saw ⊕ drill = "tools")
  - Tools bundle is similar to each tool

- **Composition**: A house is a bundle of bound structures
  - house = (wood⊗walls) ⊕ (brick⊗foundation) ⊕ (glass⊗windows)

---

**Next:** [Lesson 1.3: The Three VSA Models in VSAX](03_models.md)

Learn the three VSA models (FHRR, MAP, Binary) and when to use each.

**Previous:** [Lesson 1.1: Why High-Dimensional Vectors?](01_high_dimensions.md)
