# Batch Operations

VSAX provides efficient batch operations using JAX's `vmap` for parallel processing on GPU/TPU. These operations allow you to process multiple hypervectors simultaneously.

## Overview

Batch operations are essential for:
- Processing large datasets efficiently
- Encoding multiple items at once
- Parallel similarity computations
- GPU/TPU acceleration

## Core Batch Functions

### vmap_bind

Vectorized binding of two batches of hypervectors.

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.utils import vmap_bind
import jax.numpy as jnp

model = create_fhrr_model(dim=512)
memory = VSAMemory(model)

# Create nouns and verbs
nouns = ["dog", "cat", "bird"]
verbs = ["runs", "jumps", "flies"]
memory.add_many(nouns + verbs)

# Batch bind nouns with verbs
noun_vecs = jnp.stack([memory[n].vec for n in nouns])
verb_vecs = jnp.stack([memory[v].vec for v in verbs])

actions = vmap_bind(model.opset, noun_vecs, verb_vecs)
print(f"Created {actions.shape[0]} action vectors")
# Output: Created 3 action vectors
```

### vmap_bundle

Vectorized bundling of multiple hypervectors into one.

```python
from vsax.utils import vmap_bundle

# Bundle related concepts
colors = ["red", "green", "blue"]
memory.add_many(colors)

color_vecs = jnp.stack([memory[c].vec for c in colors])
color_concept = vmap_bundle(model.opset, color_vecs)

print(f"Bundled concept shape: {color_concept.shape}")
# Output: Bundled concept shape: (512,)
```

### vmap_similarity

Vectorized similarity computation between a query and multiple candidates.

```python
from vsax.utils import vmap_similarity

# Find most similar color
query = memory["red"].vec
candidates = jnp.stack([memory[c].vec for c in ["green", "blue", "yellow"]])

similarities = vmap_similarity(None, query, candidates)
best_match = jnp.argmax(similarities)
print(f"Most similar: {['green', 'blue', 'yellow'][int(best_match)]}")
```

## Use Cases

### 1. Sequential Composition

Combine bind and bundle for structured encoding:

```python
# Encode multiple role-filler pairs
roles = ["subject", "verb", "object"]
fillers = ["Alice", "helps", "Bob"]
memory.add_many(roles + fillers)

# Bind roles with fillers
role_vecs = jnp.stack([memory[r].vec for r in roles])
filler_vecs = jnp.stack([memory[f].vec for f in fillers])
pairs = vmap_bind(model.opset, role_vecs, filler_vecs)

# Bundle into sentence
sentence = vmap_bundle(model.opset, pairs)
print(f"Sentence encoding: {sentence.shape}")
```

### 2. Batch Encoding

Encode multiple items efficiently:

```python
# Encode many facts
subjects = ["dog", "cat", "bird", "fish"]
actions = ["runs", "sleeps", "flies", "swims"]
memory.add_many(subjects + actions)

# Batch encode all subject-action pairs
subj_vecs = jnp.stack([memory[s].vec for s in subjects])
act_vecs = jnp.stack([memory[a].vec for a in actions])

facts = vmap_bind(model.opset, subj_vecs, act_vecs)
print(f"Encoded {facts.shape[0]} facts in parallel")
```

### 3. Hierarchical Structures

Build nested representations:

```python
# Create taxonomy
mammals = ["dog", "cat", "whale"]
birds = ["eagle", "sparrow", "penguin"]
reptiles = ["snake", "lizard"]

all_animals = mammals + birds + reptiles
memory.add_many(all_animals)

# Bundle each category
mammal_vecs = jnp.stack([memory[m].vec for m in mammals])
mammal_concept = vmap_bundle(model.opset, mammal_vecs)

bird_vecs = jnp.stack([memory[b].vec for b in birds])
bird_concept = vmap_bundle(model.opset, bird_vecs)

reptile_vecs = jnp.stack([memory[r].vec for r in reptiles])
reptile_concept = vmap_bundle(model.opset, reptile_vecs)

# Bundle categories into higher-level concept
categories = jnp.stack([mammal_concept, bird_concept, reptile_concept])
animal_concept = vmap_bundle(model.opset, categories)

print("Created hierarchical animal concept")
```

### 4. Knowledge Graphs

Encode graph structures with batch operations:

```python
# Knowledge graph: (subject, predicate, object) triples
subjects = ["Alice", "Bob", "Charlie"]
predicates = ["knows", "likes", "helps"]
objects = ["Bob", "Alice", "Alice"]

# Add to memory
all_concepts = list(set(subjects + predicates + objects))
memory.add_many(all_concepts)

# Batch encode triples
subj_vecs = jnp.stack([memory[s].vec for s in subjects])
pred_vecs = jnp.stack([memory[p].vec for p in predicates])
obj_vecs = jnp.stack([memory[o].vec for o in objects])

# Encode as: bind(subject, bind(predicate, object))
pred_obj = vmap_bind(model.opset, pred_vecs, obj_vecs)
triples = vmap_bind(model.opset, subj_vecs, pred_obj)

# Bundle all triples into knowledge graph
knowledge_graph = vmap_bundle(model.opset, triples)
print(f"Knowledge graph: {knowledge_graph.shape}")
```

## Performance Comparison

Batch operations provide significant speedups:

```python
import time

# Individual operations (slow)
start = time.time()
results = []
for i in range(100):
    result = model.opset.bind(memory["a"].vec, memory["b"].vec)
    results.append(result)
individual_time = time.time() - start

# Batch operation (fast)
X = jnp.stack([memory["a"].vec] * 100)
Y = jnp.stack([memory["b"].vec] * 100)

start = time.time()
batch_result = vmap_bind(model.opset, X, Y)
jax.block_until_ready(batch_result)
batch_time = time.time() - start

print(f"Individual: {individual_time:.4f}s")
print(f"Batch: {batch_time:.4f}s")
print(f"Speedup: {individual_time/batch_time:.1f}x")
```

## Best Practices

### 1. Pre-allocate Arrays

Stack vectors once, reuse for multiple operations:

```python
# Good: Pre-stack
candidates = jnp.stack([memory[name].vec for name in names])
for query in queries:
    similarities = vmap_similarity(None, query, candidates)

# Bad: Re-stack every time
for query in queries:
    candidates = jnp.stack([memory[name].vec for name in names])
    similarities = vmap_similarity(None, query, candidates)
```

### 2. Use JIT Compilation

For repeated batch operations, use `jax.jit`:

```python
@jax.jit
def batch_encode_facts(subjects, actions, opset):
    pairs = vmap(opset.bind, in_axes=(0, 0))(subjects, actions)
    return vmap_bundle(opset, pairs)

# First call compiles
result = batch_encode_facts(subj_vecs, act_vecs, model.opset)

# Subsequent calls are fast
result = batch_encode_facts(subj_vecs2, act_vecs2, model.opset)
```

### 3. Batch Size Considerations

For very large batches, consider chunking:

```python
def chunk_vmap_bind(opset, X, Y, chunk_size=1000):
    """Process large batches in chunks."""
    n = X.shape[0]
    results = []

    for i in range(0, n, chunk_size):
        chunk_X = X[i:i+chunk_size]
        chunk_Y = Y[i:i+chunk_size]
        chunk_result = vmap_bind(opset, chunk_X, chunk_Y)
        results.append(chunk_result)

    return jnp.concatenate(results, axis=0)

# Process 10000 bindings in chunks
large_X = jnp.stack([memory["a"].vec] * 10000)
large_Y = jnp.stack([memory["b"].vec] * 10000)
result = chunk_vmap_bind(model.opset, large_X, large_Y)
```

### 4. GPU Memory Management

Monitor GPU memory when processing large batches:

```python
import jax

# Check available devices
print(f"Devices: {jax.devices()}")

# Clear cached compilations if needed
jax.clear_caches()

# Force garbage collection
import gc
gc.collect()
```

## Working with All VSA Models

Batch operations work seamlessly across all VSA models:

```python
from vsax import create_map_model, create_binary_model

models = {
    "FHRR": create_fhrr_model(dim=512),
    "MAP": create_map_model(dim=512),
    "Binary": create_binary_model(dim=10000, bipolar=True),
}

for name, test_model in models.items():
    mem = VSAMemory(test_model)
    mem.add_many(["a", "b", "c", "x", "y", "z"])

    X = jnp.stack([mem["a"].vec, mem["b"].vec, mem["c"].vec])
    Y = jnp.stack([mem["x"].vec, mem["y"].vec, mem["z"].vec])

    result = vmap_bind(test_model.opset, X, Y)
    print(f"{name}: {result.shape}")
```

## Complete Example

See [`examples/batch_operations.py`](../../examples/batch_operations.py) for a comprehensive demonstration of batch processing techniques.
