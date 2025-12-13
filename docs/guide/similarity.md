# Similarity Metrics

Similarity metrics allow you to compare hypervectors and find related concepts. VSAX provides three main similarity functions that work across all VSA models (FHRR, MAP, Binary).

## Available Metrics

### Cosine Similarity

Cosine similarity measures the cosine of the angle between two vectors, ranging from -1 (opposite) to 1 (identical direction).

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

model = create_fhrr_model(dim=512)
memory = VSAMemory(model)
memory.add_many(["dog", "cat", "wolf"])

# Compare similarity
sim_dog_cat = cosine_similarity(memory["dog"], memory["cat"])
sim_dog_wolf = cosine_similarity(memory["dog"], memory["wolf"])

print(f"Dog-Cat similarity: {sim_dog_cat:.3f}")
print(f"Dog-Wolf similarity: {sim_dog_wolf:.3f}")
```

**When to use:** Best for general-purpose similarity comparisons. Normalized to [-1, 1] range.

### Dot Product Similarity

Dot product provides an unnormalized similarity measure. Higher values indicate more similarity.

```python
from vsax.similarity import dot_similarity

# Works with all hypervector types
similarity = dot_similarity(memory["dog"], memory["cat"])
print(f"Dot product: {similarity:.3f}")
```

**When to use:** When you need raw similarity scores or when vectors are already normalized.

### Hamming Similarity

Hamming similarity measures the proportion of matching elements, ranging from 0 (completely different) to 1 (identical).

```python
from vsax import create_binary_model
from vsax.similarity import hamming_similarity

model = create_binary_model(dim=10000, bipolar=True)
memory = VSAMemory(model)
memory.add_many(["dog", "cat"])

similarity = hamming_similarity(memory["dog"], memory["cat"])
print(f"Hamming similarity: {similarity:.3f}")
```

**When to use:** Best for binary hypervectors. Counts matching bits.

## Batch Similarity Search

For efficient similarity search across multiple candidates, use `vmap_similarity`:

```python
import jax.numpy as jnp
from vsax.utils import vmap_similarity, format_similarity_results

# Create query and candidates
query = memory["dog"].vec
candidates = jnp.stack([
    memory["cat"].vec,
    memory["wolf"].vec,
    memory["lion"].vec,
])

# Compute all similarities at once
similarities = vmap_similarity(None, query, candidates)

# Find best match
best_match_idx = jnp.argmax(similarities)
print(f"Best match: {['cat', 'wolf', 'lion'][int(best_match_idx)]}")

# Format results nicely
results = format_similarity_results(
    "dog",
    ["cat", "wolf", "lion"],
    similarities,
    top_k=3
)
print(results)
```

## Use Cases

### 1. Finding Similar Concepts

```python
# Build knowledge base
animals = ["dog", "cat", "wolf", "lion", "eagle", "snake"]
memory.add_many(animals)

# Query for similar animals
query = memory["wolf"]
candidates = jnp.stack([memory[a].vec for a in animals if a != "wolf"])
similarities = vmap_similarity(None, query.vec, candidates)

# Top 3 most similar
top_indices = jnp.argsort(similarities)[-3:][::-1]
for idx in top_indices:
    animal = [a for a in animals if a != "wolf"][int(idx)]
    sim = float(similarities[int(idx)])
    print(f"  {animal}: {sim:.3f}")
```

### 2. Concept Retrieval

```python
# Encode structured data
from vsax.encoders import DictEncoder

encoder = DictEncoder(model, memory)

# Add concepts to memory
memory.add_many(["subject", "action", "object"])
memory.add_many(["dog", "cat", "runs", "sleeps", "bone", "mouse"])

# Encode facts
fact1 = encoder.encode({"subject": "dog", "action": "runs"})
fact2 = encoder.encode({"subject": "cat", "action": "sleeps"})
fact3 = encoder.encode({"subject": "dog", "object": "bone"})

# Query: What does the dog do?
query_concepts = model.opset.bind(memory["subject"].vec, memory["dog"].vec)

# Find most similar fact
facts = jnp.stack([fact1.vec, fact2.vec, fact3.vec])
similarities = vmap_similarity(None, query_concepts, facts)

best_fact = int(jnp.argmax(similarities))
print(f"Most similar fact: {['dog runs', 'cat sleeps', 'dog-bone'][best_fact]}")
```

### 3. Similarity Matrix

```python
# Compute all pairwise similarities
concepts = ["dog", "cat", "wolf", "eagle"]
n = len(concepts)

print("\nSimilarity Matrix:")
print("       " + "".join(f"{c:>8s}" for c in concepts))

for i, concept1 in enumerate(concepts):
    print(f"{concept1:>8s}", end="")
    for j, concept2 in enumerate(concepts):
        sim = cosine_similarity(memory[concept1], memory[concept2])
        print(f"{sim:8.3f}", end="")
    print()
```

## Comparison of Metrics

| Metric | Range | Best For | Complexity |
|--------|-------|----------|------------|
| Cosine | [-1, 1] | General similarity | O(n) |
| Dot Product | Unbounded | Normalized vectors | O(n) |
| Hamming | [0, 1] | Binary vectors | O(n) |

## Performance Tips

1. **Use vmap_similarity for batch queries:** Much faster than loop
ing with individual similarity calls

2. **Pre-stack candidates:** Stack candidate vectors once, reuse for multiple queries

3. **JIT compilation:** For repeated similarity computations, wrap in `jax.jit`

```python
import jax

@jax.jit
def fast_similarity_search(query, candidates):
    return vmap_similarity(None, query, candidates)

# First call compiles, subsequent calls are fast
similarities = fast_similarity_search(query_vec, candidate_vecs)
```

4. **GPU acceleration:** VSAX automatically uses GPU when available through JAX

## Complete Example

See [`examples/similarity_search.py`](../../examples/similarity_search.py) for a comprehensive demonstration of similarity search techniques.
