# Lesson 2.3: Similarity Metrics and Search

**Duration:** ~45 minutes

**Learning Objectives:**

- Master cosine similarity, dot product, and Hamming distance
- Understand when to use each metric
- Build efficient similarity search engines
- Debug common similarity issues
- Use batch operations with vmap for performance

---

## Introduction

Similarity metrics are how we **query** VSA systems. After encoding information with binding and bundling, we use similarity to:

- Find the most related concept
- Retrieve information from composite structures
- Build search and recommendation systems
- Debug VSA operations

Let's master the three similarity metrics in VSAX.

---

## Metric 1: Cosine Similarity

**Cosine similarity** measures the angle between two vectors, normalized to [-1, 1].

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{||\mathbf{a}|| \cdot ||\mathbf{b}||}$$

**Interpretation:**
- **1.0**: Identical direction (perfect match)
- **0.7-0.9**: Strong similarity
- **0.5**: Moderate similarity
- **0.0**: Orthogonal (unrelated)
- **-1.0**: Opposite direction

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)
memory.add_many(["dog", "cat", "car"])

# Compare similarities
sim_dog_cat = cosine_similarity(memory["dog"].vec, memory["cat"].vec)
sim_dog_car = cosine_similarity(memory["dog"].vec, memory["car"].vec)

print(f"Dog-Cat similarity: {sim_dog_cat:.4f}")
print(f"Dog-Car similarity: {sim_dog_car:.4f}")
```

**Expected Output:**
```
Dog-Cat similarity: 0.0123
Dog-Car similarity: -0.0089
```

**Observation:** Random vectors have similarity ~0 (orthogonal).

### When to Use Cosine Similarity

✅ **Use for:**
- FHRR and MAP models (real/complex vectors)
- General-purpose similarity comparisons
- When you need normalized scores [-1, 1]
- Comparing vectors of different magnitudes

❌ **Don't use for:**
- Binary vectors (use Hamming instead)

---

## Metric 2: Dot Product Similarity

**Dot product** is unnormalized similarity - just the raw inner product.

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b} = \sum_{i} a_i \cdot b_i$$

```python
from vsax.similarity import dot_similarity

# Works with all hypervector types
dot_sim = dot_similarity(memory["dog"].vec, memory["cat"].vec)
print(f"Dot product: {dot_sim:.4f}")
```

**Key difference:** Dot product is **not normalized**, so it depends on vector magnitudes.

### When to Use Dot Product

✅ **Use for:**
- When vectors are already unit-normalized (most VSAX vectors are)
- Slightly faster than cosine (no division)
- When you need raw similarity scores

❌ **Don't use for:**
- Vectors with varying magnitudes (unnormalized)

**Note:** For VSAX, cosine and dot product give similar results since vectors are normalized!

---

## Metric 3: Hamming Similarity

**Hamming similarity** counts the fraction of matching elements (for binary vectors).

$$\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{1}{n} \sum_{i} \mathbb{1}[a_i = b_i]$$

```python
from vsax import create_binary_model
from vsax.similarity import hamming_similarity

model = create_binary_model(dim=2048)
memory = VSAMemory(model)
memory.add_many(["dog", "cat"])

# Hamming similarity for binary vectors
ham_sim = hamming_similarity(memory["dog"].vec, memory["cat"].vec)
print(f"Hamming similarity: {ham_sim:.4f}")
```

**Expected Output:**
```
Hamming similarity: 0.5012
```

**Observation:** Random binary vectors have ~50% matching bits.

### When to Use Hamming Similarity

✅ **Use for:**
- Binary vectors (required!)
- Hardware-optimized operations (bit counting)

❌ **Don't use for:**
- Real or complex vectors

---

## Comparing the Three Metrics

Let's compare all three on the same data:

```python
from vsax import create_fhrr_model, create_binary_model, VSAMemory
from vsax.similarity import cosine_similarity, dot_similarity, hamming_similarity

# Test with FHRR
fhrr_model = create_fhrr_model(dim=2048)
fhrr_memory = VSAMemory(fhrr_model)
fhrr_memory.add_many(["a", "b"])

print("FHRR model:")
print(f"  Cosine:      {cosine_similarity(fhrr_memory['a'].vec, fhrr_memory['b'].vec):.4f}")
print(f"  Dot product: {dot_similarity(fhrr_memory['a'].vec, fhrr_memory['b'].vec):.4f}")

# Test with Binary
binary_model = create_binary_model(dim=2048)
binary_memory = VSAMemory(binary_model)
binary_memory.add_many(["a", "b"])

print("\nBinary model:")
print(f"  Hamming:     {hamming_similarity(binary_memory['a'].vec, binary_memory['b'].vec):.4f}")
```

**Key takeaway:** Use the metric that matches your model!

---

## Building a Similarity Search Engine

Now let's build a practical search engine using similarity.

### Task: Find Most Similar Concept

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

# Create knowledge base
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

animals = ["dog", "cat", "wolf", "lion", "eagle", "snake", "fish"]
memory.add_many(animals)

def find_most_similar(query_name, candidates, memory, top_k=3):
    """
    Find the top-k most similar concepts to the query.

    Args:
        query_name: Name of the query concept
        candidates: List of candidate concept names
        memory: VSAMemory containing all concepts
        top_k: Number of top results to return

    Returns:
        List of (name, similarity) tuples
    """
    query_vec = memory[query_name].vec

    # Compute similarities to all candidates
    similarities = []
    for candidate in candidates:
        if candidate == query_name:
            continue  # Skip self
        sim = cosine_similarity(query_vec, memory[candidate].vec)
        similarities.append((candidate, float(sim)))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]

# Query: What's similar to "wolf"?
results = find_most_similar("wolf", animals, memory, top_k=3)

print("Most similar to 'wolf':")
for name, sim in results:
    print(f"  {name}: {sim:.4f}")
```

**Expected Output:**
```
Most similar to 'wolf':
  dog: 0.0234
  lion: 0.0189
  cat: 0.0156
```

**Note:** For random vectors, similarities are all ~0. In a real system with semantic embeddings, "dog" would have much higher similarity to "wolf"!

---

## Batch Similarity with vmap

For efficiency, compute many similarities at once using JAX's `vmap`.

```python
import jax.numpy as jnp
from vsax.utils import vmap_similarity

# Stack all candidate vectors into a matrix
candidate_names = [a for a in animals if a != "wolf"]
candidate_vecs = jnp.stack([memory[name].vec for name in candidate_names])

# Compute all similarities at once (batched!)
query_vec = memory["wolf"].vec
similarities = vmap_similarity(None, query_vec, candidate_vecs)

print("Batch similarities:")
for name, sim in zip(candidate_names, similarities):
    print(f"  {name}: {sim:.4f}")
```

**Performance:** Batch operations are **10-100× faster** than loops!

---

## Common Debugging Issues

### Problem 1: "All my similarities are ~0.5"

**Symptom:**
```python
sim = cosine_similarity(a, b)
print(sim)  # 0.5123
```

**Possible causes:**
1. Vectors aren't normalized
2. Using wrong similarity metric
3. Dimension is too low

**Debug:**
```python
# Check normalization
print(f"||a|| = {jnp.linalg.norm(a):.4f}")  # Should be ~1.0
print(f"||b|| = {jnp.linalg.norm(b):.4f}")  # Should be ~1.0

# Check vector type
print(f"a dtype: {a.dtype}")  # complex64, float32, or int?

# Try higher dimension
model = create_fhrr_model(dim=4096)  # Instead of 512
```

---

### Problem 2: "Unbinding gives low similarity"

**Symptom:**
```python
bound = model.opset.bind(a, b)
retrieved = model.opset.bind(bound, model.opset.inverse(b))
sim = cosine_similarity(retrieved, a)
print(sim)  # 0.35 (too low!)
```

**Possible causes:**
1. Using MAP with deep binding chain (error accumulates)
2. Not using inverse correctly
3. Mixing different operation sets

**Debug:**
```python
# Check model type
print(f"Model: {model.rep_cls.__name__}")  # FHRR, MAP, or Binary?

# For MAP, check binding depth
# If depth > 3, consider switching to FHRR

# Verify inverse is correct
if model.rep_cls.__name__ == "ComplexHypervector":
    # FHRR: inverse should be complex conjugate
    b_inv = model.opset.inverse(b)
    product = b * jnp.conj(b)
    print(f"b * conj(b) all ~1? {jnp.allclose(jnp.abs(product), 1.0)}")
```

---

### Problem 3: "Hamming similarity is always 0.5"

**Symptom:**
```python
sim = hamming_similarity(a, b)
print(sim)  # Always ~0.5
```

**Cause:** Random binary vectors have ~50% matching bits (expected!)

**Solution:** This is normal for unrelated concepts. Hamming ~0.5 means orthogonal.

---

## Building a Similarity Matrix

Visualize relationships between multiple concepts:

```python
def similarity_matrix(concepts, memory):
    """
    Compute pairwise similarity matrix for concepts.
    """
    n = len(concepts)
    matrix = jnp.zeros((n, n))

    for i, c1 in enumerate(concepts):
        for j, c2 in enumerate(concepts):
            if i == j:
                matrix = matrix.at[i, j].set(1.0)  # Self-similarity
            else:
                sim = cosine_similarity(memory[c1].vec, memory[c2].vec)
                matrix = matrix.at[i, j].set(sim)

    return matrix

# Test
concepts = ["dog", "cat", "wolf", "eagle"]
matrix = similarity_matrix(concepts, memory)

# Pretty print
print("\nSimilarity Matrix:")
print("       " + "".join(f"{c:>8s}" for c in concepts))
for i, concept in enumerate(concepts):
    print(f"{concept:>8s}", end="")
    for j in range(len(concepts)):
        print(f"{matrix[i, j]:8.3f}", end="")
    print()
```

**Expected Output:**
```
Similarity Matrix:
            dog     cat    wolf   eagle
     dog   1.000   0.012   0.023  -0.008
     cat   0.012   1.000  -0.015   0.019
    wolf   0.023  -0.015   1.000   0.011
   eagle  -0.008   0.019   0.011   1.000
```

---

## Performance Optimization

### Tip 1: Pre-stack Candidates

```python
# SLOW: Stacking inside the query loop
for query in queries:
    candidates = jnp.stack([memory[c].vec for c in candidate_names])
    sims = vmap_similarity(None, query, candidates)

# FAST: Stack once, reuse
candidates = jnp.stack([memory[c].vec for c in candidate_names])
for query in queries:
    sims = vmap_similarity(None, query, candidates)
```

### Tip 2: JIT Compilation

```python
import jax

@jax.jit
def fast_similarity_search(query, candidates):
    """JIT-compiled similarity search."""
    return vmap_similarity(None, query, candidates)

# First call compiles (slow)
sims = fast_similarity_search(query_vec, candidate_vecs)

# Subsequent calls are FAST
sims = fast_similarity_search(query_vec2, candidate_vecs)  # ~10× faster
```

### Tip 3: GPU Acceleration

VSAX automatically uses GPU through JAX when available:

```python
import jax
print(f"Devices: {jax.devices()}")  # Check for GPU

# If GPU available, vmap automatically parallelizes
# No code changes needed!
```

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Explain cosine similarity, dot product, and Hamming distance
- [ ] Choose the right metric for FHRR, MAP, and Binary
- [ ] Build a similarity search function
- [ ] Use vmap for batch similarity computation
- [ ] Debug low similarity issues
- [ ] Optimize similarity search performance

---

## Quick Quiz

**Q1:** What is the range of cosine similarity?

a) [0, 1]
b) [-1, 1]
c) [0, ∞)
d) (-∞, ∞)

<details>
<summary>Answer</summary>
**b) [-1, 1]** - Cosine similarity is the cosine of the angle, which ranges from -1 (opposite) to 1 (identical).
</details>

**Q2:** Which similarity metric should you use for Binary models?

a) Cosine similarity
b) Dot product
c) Hamming similarity
d) Euclidean distance

<details>
<summary>Answer</summary>
**c) Hamming similarity** - Binary vectors use Hamming distance (fraction of matching bits).
</details>

**Q3:** For random unit vectors in high dimensions, expected cosine similarity is:

a) ~1.0
b) ~0.7
c) ~0.0
d) ~-1.0

<details>
<summary>Answer</summary>
**c) ~0.0** - Random vectors are quasi-orthogonal in high dimensions.
</details>

**Q4:** What's the advantage of vmap_similarity over a loop?

a) More accurate
b) Much faster (10-100×)
c) Uses less memory
d) Works on GPU only

<details>
<summary>Answer</summary>
**b) Much faster (10-100×)** - Batch operations parallelize computation and avoid Python loops.
</details>

---

## Hands-On Exercise

**Task:** Build a k-Nearest Neighbors (k-NN) classifier using similarity.

1. Create a dataset with 3 classes (each with 10 examples)
2. Encode all examples as random hypervectors
3. Implement k-NN: find k most similar training examples
4. Classify test examples by majority vote of k neighbors
5. Measure accuracy

**Solution:**

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity
import jax.numpy as jnp

def knn_classifier(train_vecs, train_labels, test_vec, k=3):
    """
    Classify test_vec using k-nearest neighbors.

    Args:
        train_vecs: Training vectors (stacked)
        train_labels: Training labels
        test_vec: Test vector to classify
        k: Number of neighbors

    Returns:
        Predicted label
    """
    # Compute similarities to all training examples
    from vsax.utils import vmap_similarity
    similarities = vmap_similarity(None, test_vec, train_vecs)

    # Find k nearest neighbors
    top_k_indices = jnp.argsort(similarities)[-k:]

    # Get their labels
    neighbor_labels = [train_labels[int(idx)] for idx in top_k_indices]

    # Majority vote
    from collections import Counter
    prediction = Counter(neighbor_labels).most_common(1)[0][0]

    return prediction

# Create dataset
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# 3 classes, 10 examples each
train_vecs = []
train_labels = []

for class_id in range(3):
    for example_id in range(10):
        name = f"class{class_id}_ex{example_id}"
        memory.add(name)
        train_vecs.append(memory[name].vec)
        train_labels.append(class_id)

train_vecs = jnp.stack(train_vecs)

# Test
test_names = ["test1", "test2", "test3"]
memory.add_many(test_names)

print("k-NN Classification (k=3):")
for test_name in test_names:
    test_vec = memory[test_name].vec
    prediction = knn_classifier(train_vecs, train_labels, test_vec, k=3)
    print(f"  {test_name} → Class {prediction}")
```

**Expected:** Random assignment (no semantic structure in random vectors).

**Extension:** Try with real semantic embeddings (word vectors) for better results!

---

## Key Takeaways

1. **Three metrics** - Cosine (FHRR/MAP), Hamming (Binary), Dot product (all)
2. **Cosine is default** - Normalized to [-1, 1], works for most cases
3. **vmap for batches** - 10-100× faster than loops
4. **Similarity ~0 is normal** - Random vectors are orthogonal
5. **Debug systematically** - Check normalization, model type, dimension
6. **Optimize with JIT** - JAX compilation for repeated operations

---

**Next:** [Lesson 2.4: Model Selection Decision Framework](04_selection.md)

Learn systematic decision-making for choosing FHRR, MAP, or Binary.

**Previous:** [Lesson 2.2: Deep Dive - MAP and Binary Operations](02_map_binary.md)
