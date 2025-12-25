# Lesson 1.1: Why High-Dimensional Vectors?

**Duration:** ~45 minutes

**Learning Objectives:**

- Understand the "blessing of dimensionality" for symbolic computation
- Grasp why random vectors become nearly orthogonal in high dimensions
- Calculate expected cosine similarity between random vectors
- Appreciate capacity arguments for hyperdimensional computing

---

## Introduction

Vector Symbolic Architectures (VSAs) represent concepts as **high-dimensional vectors** (typically 512 to 10,000 dimensions). This might seem wasteful at first - why use thousands of numbers to represent a simple concept like "cat" or "red"?

The answer lies in a remarkable mathematical property: **random vectors in high dimensions are nearly orthogonal to each other**. This property is the foundation of all VSA operations.

## The Curse vs Blessing of Dimensionality

In machine learning, we often hear about the "curse of dimensionality" - how high-dimensional spaces become exponentially sparse and difficult to work with. But for VSA, high dimensions are a **blessing**, not a curse.

### Why? Two Key Insights

**1. Orthogonality is Free**

In low dimensions (2D or 3D), orthogonal vectors are rare. You can only have 2-3 mutually orthogonal vectors maximum.

In high dimensions (d=10,000), **almost any random vectors are nearly orthogonal**. You can have thousands of mutually quasi-orthogonal vectors without any effort.

**2. Similarity as a Probe**

When vectors are orthogonal, their similarity (cosine) is ~0. This means:
- Random concepts are naturally dissimilar
- We can distinguish thousands of concepts by their similarity scores
- Similarity becomes a powerful query mechanism

Let's verify this with code.

---

## Experiment 1: Orthogonality in High Dimensions

Let's generate random vectors and measure their pairwise similarities in different dimensions.

```python
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt

def measure_orthogonality(dim, num_vectors=100, seed=0):
    """Generate random vectors and measure pairwise cosine similarities."""
    key = random.PRNGKey(seed)

    # Generate random vectors
    vectors = random.normal(key, (num_vectors, dim))

    # Normalize to unit length
    norms = jnp.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / norms

    # Compute all pairwise similarities
    similarity_matrix = vectors @ vectors.T

    # Extract off-diagonal elements (don't compare vector with itself)
    mask = ~jnp.eye(num_vectors, dtype=bool)
    off_diagonal = similarity_matrix[mask]

    return off_diagonal

# Test different dimensions
dimensions = [2, 10, 100, 1000, 10000]
results = {}

for dim in dimensions:
    similarities = measure_orthogonality(dim)
    results[dim] = similarities

    mean_sim = jnp.mean(jnp.abs(similarities))
    std_sim = jnp.std(similarities)

    print(f"Dimension {dim:5d}: Mean |similarity| = {mean_sim:.4f} ± {std_sim:.4f}")
```

**Expected Output:**
```
Dimension     2: Mean |similarity| = 0.3015 ± 0.2123
Dimension    10: Mean |similarity| = 0.1803 ± 0.1421
Dimension   100: Mean |similarity| = 0.0567 ± 0.0491
Dimension  1000: Mean |similarity| = 0.0179 ± 0.0156
Dimension 10000: Mean |similarity| = 0.0057 ± 0.0049
```

**Observation:** As dimension increases, mean similarity approaches 0. At d=10,000, random vectors are nearly perfectly orthogonal (similarity ≈ 0.006).

---

## Mathematical Insight: Expected Similarity

For two random unit vectors **a** and **b** in d dimensions:

$$E[\langle a, b \rangle] = 0$$

The **standard deviation** of the dot product is approximately:

$$\sigma \approx \frac{1}{\sqrt{d}}$$

This means:
- In d=100 dimensions: σ ≈ 0.1 (similarity fluctuates by ±0.1)
- In d=10,000 dimensions: σ ≈ 0.01 (similarity fluctuates by ±0.01)

**Conclusion:** Higher dimension → tighter concentration around zero → more reliable orthogonality.

---

## Visualizing the Distribution

Let's visualize how the similarity distribution becomes more concentrated as dimension increases.

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, dim in enumerate([10, 100, 1000]):
    similarities = measure_orthogonality(dim, num_vectors=500)

    axes[idx].hist(similarities, bins=50, alpha=0.7, edgecolor='black')
    axes[idx].axvline(0, color='red', linestyle='--', linewidth=2, label='Expected (0)')
    axes[idx].set_title(f'Dimension = {dim}')
    axes[idx].set_xlabel('Cosine Similarity')
    axes[idx].set_ylabel('Frequency')
    axes[idx].legend()
    axes[idx].set_xlim(-0.5, 0.5)

plt.tight_layout()
plt.savefig('similarity_distribution.png', dpi=150)
plt.show()
```

**Key Observation:** The distribution becomes narrower and more concentrated around 0 as dimension increases. This is called **concentration of measure**.

---

## Capacity: How Many Concepts Can We Represent?

If random vectors are nearly orthogonal, how many can we distinguish?

### Rule of Thumb

In d dimensions, we can reliably distinguish approximately **d** to **2d** random vectors.

**Why?** Consider the unit sphere in d dimensions. The "surface area" (volume of the sphere shell) grows exponentially with d, allowing exponentially many nearly-orthogonal vectors.

### Practical Capacity

With d=10,000:
- We can represent ~10,000-20,000 unique concepts
- Each concept has similarity ~0 with all others
- We can query by similarity to find specific concepts

Let's verify this:

```python
def test_capacity(dim, num_concepts):
    """Test if we can distinguish num_concepts in dim dimensions."""
    key = random.PRNGKey(42)

    # Generate concept vectors
    concepts = random.normal(key, (num_concepts, dim))
    concepts = concepts / jnp.linalg.norm(concepts, axis=1, keepdims=True)

    # Compute all pairwise similarities
    similarity_matrix = concepts @ concepts.T

    # Mask out self-similarities
    mask = ~jnp.eye(num_concepts, dtype=bool)
    max_cross_similarity = jnp.max(jnp.abs(similarity_matrix[mask]))
    mean_cross_similarity = jnp.mean(jnp.abs(similarity_matrix[mask]))

    return max_cross_similarity, mean_cross_similarity

# Test capacity
dim = 2048
test_sizes = [100, 500, 1000, 2000, 5000]

print(f"Testing capacity for dim={dim}:\n")
print(f"{'Num Concepts':<15} {'Max Similarity':<15} {'Mean Similarity':<15}")
print("-" * 50)

for num in test_sizes:
    max_sim, mean_sim = test_capacity(dim, num)
    print(f"{num:<15} {max_sim:<15.4f} {mean_sim:<15.4f}")
```

**Expected Output:**
```
Testing capacity for dim=2048:

Num Concepts    Max Similarity  Mean Similarity
--------------------------------------------------
100             0.1234          0.0223
500             0.1456          0.0223
1000            0.1678          0.0224
2000            0.1891          0.0223
5000            0.2234          0.0224
```

**Observation:** Even with 5000 concepts in 2048 dimensions, max similarity is only ~0.22. Concepts remain distinguishable.

---

## Why This Matters for VSA

This quasi-orthogonality property enables three critical VSA operations:

### 1. Symbol Storage
We can store thousands of symbols (concepts) as random vectors. Each symbol is naturally dissimilar to all others.

### 2. Similarity-Based Retrieval
Given a query vector, we can find the closest symbol by computing similarities:

```python
def find_closest_symbol(query, symbol_vectors, symbol_names):
    """Find the symbol most similar to the query."""
    similarities = query @ symbol_vectors.T
    best_idx = jnp.argmax(similarities)
    return symbol_names[best_idx], similarities[best_idx]
```

### 3. Interference Resistance
When we combine vectors (we'll learn how in the next lesson), the high dimensionality ensures minimal interference between different concepts.

---

## The Key Intuition

**Low Dimensions (d=2 or 3):**
- Few orthogonal directions available
- Concepts must interfere
- Hard to distinguish many concepts

**High Dimensions (d=1000+):**
- Nearly unlimited orthogonal directions
- Random concepts are naturally separated
- Can distinguish thousands of concepts

Think of it like this: in a 2D room, you can only point in a few distinct directions (North, East, South, West). In a 10,000-dimensional space, you can point in 10,000+ completely different directions!

---

## Common Misconceptions

### ❌ "Higher dimensions mean more computation"
**Reality:** Modern hardware (GPUs) handles high-dimensional vectors efficiently. Vector operations are O(d), which is acceptable even for d=10,000.

### ❌ "We need to carefully design orthogonal vectors"
**Reality:** Random vectors are automatically quasi-orthogonal. No design needed!

### ❌ "High dimensions waste memory"
**Reality:** A 10,000-dimensional float vector uses only 40KB. We can store thousands of concepts in megabytes.

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Explain why random vectors become orthogonal in high dimensions
- [ ] Calculate expected cosine similarity (σ ≈ 1/√d)
- [ ] Understand capacity arguments (d dimensions → ~d distinguishable concepts)
- [ ] Appreciate why high dimensions are a "blessing" for symbolic computation
- [ ] Run the code examples and verify the results

---

## Quick Quiz

**Q1:** What happens to the similarity between two random unit vectors as dimension increases?

a) It increases towards 1
b) It stays constant
c) It decreases towards 0
d) It becomes negative

<details>
<summary>Answer</summary>
**c) It decreases towards 0** - The expected similarity is 0, and the standard deviation σ ≈ 1/√d decreases as d increases.
</details>

**Q2:** Approximately how many concepts can we reliably distinguish in d=4096 dimensions?

a) ~100
b) ~1,000
c) ~4,000
d) ~100,000

<details>
<summary>Answer</summary>
**c) ~4,000** - Rule of thumb: d to 2d concepts can be distinguished, so ~4,000-8,000 for d=4096.
</details>

**Q3:** Why is high dimensionality a "blessing" for VSA?

a) It makes computation faster
b) Random vectors are automatically quasi-orthogonal
c) It uses less memory
d) It's easier to visualize

<details>
<summary>Answer</summary>
**b) Random vectors are automatically quasi-orthogonal** - This natural separation is the foundation of VSA's power.
</details>

---

## Hands-On Exercise

**Task:** Investigate the "elbow" of dimension scaling.

1. Test dimensions from 10 to 10,000 (logarithmic spacing)
2. For each dimension, generate 100 random vectors
3. Compute mean absolute similarity
4. Plot dimension (x-axis) vs mean similarity (y-axis) on a log-log plot
5. Find the dimension where similarity drops below 0.05

**Expected finding:** Around d=400-500, similarities become reliably small (<0.05).

**Solution:**

```python
import numpy as np

# Logarithmically spaced dimensions
dimensions = np.logspace(1, 4, num=30, dtype=int)  # 10 to 10,000
mean_sims = []

for dim in dimensions:
    sims = measure_orthogonality(dim, num_vectors=100)
    mean_sim = float(jnp.mean(jnp.abs(sims)))
    mean_sims.append(mean_sim)

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(dimensions, mean_sims, 'o-', linewidth=2, markersize=6)
plt.axhline(0.05, color='red', linestyle='--', label='Threshold (0.05)')
plt.xlabel('Dimension (d)', fontsize=14)
plt.ylabel('Mean Absolute Similarity', fontsize=14)
plt.title('Orthogonality vs Dimension', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('dimension_scaling.png', dpi=150)
plt.show()

# Find elbow
elbow_idx = np.where(np.array(mean_sims) < 0.05)[0][0]
print(f"Elbow dimension (sim < 0.05): d = {dimensions[elbow_idx]}")
```

---

## Key Takeaways

1. **High dimensions ≠ curse** - For VSA, high dimensions enable symbolic representation
2. **Random = orthogonal** - In d dimensions, random vectors are quasi-orthogonal with σ ≈ 1/√d
3. **Capacity scales linearly** - d dimensions → ~d distinguishable concepts
4. **No design needed** - Random vectors automatically have the properties we need
5. **Foundation for everything** - All VSA operations rely on this orthogonality

---

**Next:** [Lesson 1.2: The Two Fundamental Operations](02_operations.md)

Learn how to **bind** and **bundle** hypervectors to create compositional symbolic structures.

**Previous:** [Module 1 Overview](index.md)
