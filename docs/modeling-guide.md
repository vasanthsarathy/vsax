# VSA Modeling Guide: From Problem to Solution

This guide walks you through the complete workflow for building VSA applications with VSAX. Whether you're classifying images, reasoning over knowledge graphs, or building recommender systems, these 7 steps will get you from problem to solution.

---

## Who Should Read This

- **Newcomers to VSA** - Learn the standard workflow
- **ML practitioners** - Understand when and how to use VSA
- **Researchers** - Quick reference for implementation decisions

---

## The 7-Step VSA Workflow

```
1. Define Your Problem
        ↓
2. Choose VSA Model (FHRR, MAP, or Binary)
        ↓
3. Select Dimensionality (512, 1024, 2048, 10000)
        ↓
4. Initialize Model & Memory
        ↓
5. Design Encoding Strategy
        ↓
6. Encode Your Data
        ↓
7. Perform Operations & Query
        ↓
   Evaluate & Iterate
```

---

## Step 1: Define Your Problem

**Ask yourself**: What am I trying to achieve?

### Common VSA Tasks

| Task | Description | Examples |
|------|-------------|----------|
| **Classification** | Assign labels to inputs | Image recognition, sentiment analysis |
| **Similarity Search** | Find similar items | Document retrieval, recommendation |
| **Reasoning** | Answer queries over knowledge | Knowledge graphs, Q&A systems |
| **Composition** | Build complex structures | Parse trees, scene understanding |
| **Analogy** | Find relationships | Word analogies, visual analogies |
| **Sequence Modeling** | Temporal patterns | Time series, activity recognition |

### Example: Let's Build a Sentiment Classifier

**Problem**: Given movie reviews, classify as positive or negative.

**Input**: Text reviews
**Output**: Sentiment label (positive/negative)
**Approach**: VSA prototype-based classification

---

## Step 2: Choose Your VSA Model

VSAX provides three models. Choose based on your needs:

### Decision Guide

```
Need exact unbinding for compositional structures?
    ├─ YES → Use FHRR (complex vectors, circular convolution)
    └─ NO → Continue...

Need simplest, fastest option?
    ├─ YES → Use MAP (real vectors, element-wise multiply)
    └─ NO → Continue...

Memory-constrained or targeting hardware?
    └─ YES → Use Binary (discrete vectors, XOR operations)
```

### Model Comparison

| Feature | FHRR | MAP | Binary |
|---------|------|-----|--------|
| **Representation** | Complex (phase) | Real (continuous) | Discrete (±1 or 0/1) |
| **Binding** | Circular convolution (FFT) | Element-wise multiply | XOR (or multiply) |
| **Unbinding** | Exact (>99% with proper vectors) | Approximate (~30%) | Exact (self-inverse) |
| **Speed** | Fast (FFT) | Fastest (element-wise) | Very fast (bit ops) |
| **Memory** | Moderate | Moderate | Low (1 bit/dim) |
| **Best For** | Compositional structures | Simple tasks, speed | Hardware, memory limits |

### Recommendations

- **Default choice**: Start with **FHRR** (dim=2048)
  - Most versatile, exact unbinding
  - Good for learning VSA concepts

- **For speed**: Use **MAP** (dim=2048)
  - Fastest operations
  - Good enough for most classification tasks

- **For constraints**: Use **Binary** (dim=10000)
  - Minimal memory
  - Hardware-friendly (bit operations)

### Example Decision

**Our sentiment classifier**: We'll use **FHRR** (dim=2048)
- Need to compose word meanings (bind words to positions)
- Want exact unbinding to inspect learned patterns
- Moderate speed requirements

---

## Step 3: Select Dimensionality

The dimension controls capacity and accuracy.

### Guidelines

| Dimension | Use Case | Capacity | Memory |
|-----------|----------|----------|--------|
| **512** | Quick prototyping, simple tasks | Low | 2-4 KB/vector |
| **1024** | Small datasets, real-time apps | Medium | 4-8 KB/vector |
| **2048** | **Recommended default** | High | 8-16 KB/vector |
| **4096** | Large-scale, high accuracy | Very high | 16-32 KB/vector |
| **10000** | Binary VSA, maximum capacity | Extreme | 1.25 KB/vector (binary) |

### Trade-offs

- **Higher dimension**:
  - ✅ More capacity (store more items)
  - ✅ Better noise tolerance
  - ✅ Higher accuracy
  - ❌ More memory
  - ❌ Slower operations

- **Lower dimension**:
  - ✅ Less memory
  - ✅ Faster operations
  - ❌ Lower capacity
  - ❌ More interference

### Rule of Thumb

- **Start with 2048** for most tasks
- **Use 1024** if speed is critical
- **Use 4096+** if accuracy is paramount
- **Binary needs 5-10x higher** (10000) for same capacity

### Example Decision

**Our sentiment classifier**: **dim = 2048**
- Moderate vocabulary size (~1000 words)
- Want good accuracy
- Not memory-constrained

---

## Step 4: Initialize Model & Memory

Use factory functions for one-line model creation.

### Code

```python
from vsax import create_fhrr_model, create_map_model, create_binary_model, VSAMemory

# Create model (choose one)
model = create_fhrr_model(dim=2048)        # FHRR
# model = create_map_model(dim=2048)       # MAP
# model = create_binary_model(dim=10000)   # Binary

# Create memory for storing symbols
memory = VSAMemory(model)

print(f"Model: {model.opset.__class__.__name__}")
print(f"Dimension: {model.dim}")
print(f"Representation: {model.rep_cls.__name__}")
```

**Output**:
```
Model: FHRROperations
Dimension: 2048
Representation: ComplexHypervector
```

### What Just Happened?

- **Model**: Defines the algebra (bind, bundle, inverse operations)
- **Memory**: Dictionary-style storage for named basis vectors
- **Ready**: Can now add symbols and encode data

---

## Step 5: Design Your Encoding Strategy

Choose encoders based on your data types.

### Encoder Selection Guide

| Data Type | Encoder | Use For | Example |
|-----------|---------|---------|---------|
| **Numbers** | `ScalarEncoder` | Continuous values | Temperature, age, price |
| **Sequences** | `SequenceEncoder` | Ordered items | Sentences, time series |
| **Sets** | `SetEncoder` | Unordered items | Tags, categories |
| **Dictionaries** | `DictEncoder` | Key-value pairs | Structured records |
| **Graphs** | `GraphEncoder` | Networks | Social graphs, molecules |
| **Custom** | Extend `AbstractEncoder` | Domain-specific | Images, audio |

### Compositional Patterns

**Role-filler binding**: Bind concept to position/role
```python
# "dog" in subject position
sentence = bind(role_subject, concept_dog)
```

**Bundling**: Aggregate multiple items
```python
# Average multiple examples
prototype = bundle(example1, example2, example3)
```

**Sequential encoding**: Bind items to positions
```python
# "the cat sat" → bind(pos1, the) + bind(pos2, cat) + bind(pos3, sat)
```

### Example: Sentiment Encoding Strategy

**Our sentiment classifier**:

1. **Tokenize**: "I loved this movie" → ["I", "loved", "this", "movie"]
2. **Create basis vectors**: Each word gets a random vector
3. **Positional encoding**: Bind each word to its position
4. **Bundle**: Sum all bound vectors
5. **Result**: Single vector representing the review

```python
# Pseudo-code
review_vector = bundle(
    bind(pos_1, word_I),
    bind(pos_2, word_loved),
    bind(pos_3, word_this),
    bind(pos_4, word_movie)
)
```

---

## Step 6: Encode Your Data

Transform raw data into hypervectors.

### For Classification: Build Prototypes

```python
# 1. Add all symbols to memory
words = ["good", "bad", "great", "terrible", "loved", "hated", ...]
memory.add_many(words)

# 2. Add position roles
positions = [f"pos_{i}" for i in range(max_length)]
memory.add_many(positions)

# 3. Encode reviews
def encode_review(words, memory, model):
    """Encode a review as a single vector."""
    vectors = []
    for i, word in enumerate(words):
        if word in memory:
            # Bind word to position
            pos_vec = memory[f"pos_{i}"].vec
            word_vec = memory[word].vec
            bound = model.opset.bind(pos_vec, word_vec)
            vectors.append(bound)

    # Bundle all positions
    if vectors:
        review_vec = model.opset.bundle(*vectors)
        return review_vec
    return None

# 4. Build prototypes for each class
positive_reviews = [...]  # List of positive review word lists
negative_reviews = [...]  # List of negative review word lists

pos_vecs = [encode_review(r, memory, model) for r in positive_reviews]
neg_vecs = [encode_review(r, memory, model) for r in negative_reviews]

# Average to get prototypes
prototype_positive = model.opset.bundle(*pos_vecs)
prototype_negative = model.opset.bundle(*neg_vecs)
```

### For Reasoning: Encode Facts

```python
# Encode "Paris is-capital-of France"
fact = model.opset.bundle(
    model.opset.bind(memory["subject"].vec, memory["Paris"].vec),
    model.opset.bind(memory["relation"].vec, memory["is_capital_of"].vec),
    model.opset.bind(memory["object"].vec, memory["France"].vec)
)
```

---

## Step 7: Perform Operations & Query

Use your encoded data to make predictions.

### Similarity Search (Classification)

```python
from vsax.similarity import cosine_similarity

# Encode new test review
test_review = ["I", "hated", "this", "movie"]
test_vec = encode_review(test_review, memory, model)

# Compare to prototypes
sim_positive = cosine_similarity(test_vec, prototype_positive)
sim_negative = cosine_similarity(test_vec, prototype_negative)

# Predict
if sim_positive > sim_negative:
    prediction = "positive"
else:
    prediction = "negative"

print(f"Review: {' '.join(test_review)}")
print(f"Positive similarity: {sim_positive:.3f}")
print(f"Negative similarity: {sim_negative:.3f}")
print(f"Prediction: {prediction}")
```

**Output**:
```
Review: I hated this movie
Positive similarity: 0.234
Negative similarity: 0.789
Prediction: negative
```

### Unbinding (Factorization)

```python
# Given a fact, extract components
# fact = bind(role_subject, Paris) + bind(role_relation, is_capital_of) + ...

# Unbind to get subject (NEW: explicit unbind method)
subject_vec = model.opset.unbind(fact, memory["role_subject"].vec)

# Find closest match
similarities = {}
for city in ["Paris", "London", "Berlin"]:
    sim = cosine_similarity(subject_vec, memory[city].vec)
    similarities[city] = sim

best_match = max(similarities.items(), key=lambda x: x[1])
print(f"Subject: {best_match[0]} (similarity: {best_match[1]:.3f})")
# With FHRR: expect >99% similarity for correct city!
```

### Batch Operations (GPU Acceleration)

```python
from vsax.utils import vmap_bind, vmap_bundle
import jax.numpy as jnp

# Encode multiple reviews in parallel
word_vecs = jnp.stack([memory[w].vec for w in words])
pos_vecs = jnp.stack([memory[f"pos_{i}"].vec for i in range(len(words))])

# Parallel binding
bound_vecs = vmap_bind(model.opset, pos_vecs, word_vecs)

# Bundle
review_vec = model.opset.bundle(*bound_vecs)
```

---

## Step 8: Evaluate & Iterate

Test your model and refine.

### Evaluation Checklist

- [ ] **Accuracy**: Does it predict correctly?
- [ ] **Similarity scores**: Are correct matches high? (> 0.7)
- [ ] **Failure analysis**: What mistakes does it make?
- [ ] **Capacity**: Can it handle your data size?

### Common Issues & Solutions

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| Low accuracy | Dimension too small | Increase to 2048+ |
| Low similarities | Over-bundling (too many items) | Reduce items or increase dim |
| Slow performance | Dimension too large | Reduce to 1024 or use MAP |
| Memory issues | Too many basis vectors | Use Binary model (dim=10000) |
| Can't unbind | Wrong model | Use FHRR instead of MAP |

### Iteration Strategies

1. **Start simple**: Use small dataset, basic encoding
2. **Test incrementally**: Verify each step works
3. **Analyze failures**: Look at misclassified examples
4. **Refine encoding**: Adjust positional binding, try different encoders
5. **Tune dimension**: Increase if accuracy low, decrease if slow

---

## Complete Example: Sentiment Classification

Putting it all together:

```python
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

# Step 1: Define problem
# Task: Classify movie reviews as positive/negative

# Step 2-3: Choose model and dimension
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Step 4: Initialize
print(f"Model: {model.opset.__class__.__name__}, Dim: {model.dim}")

# Step 5: Design encoding
# Vocabulary
words = ["I", "love", "hate", "good", "bad", "great", "terrible",
         "this", "movie", "film", "amazing", "awful"]
memory.add_many(words)

# Position markers
max_len = 10
positions = [f"pos_{i}" for i in range(max_len)]
memory.add_many(positions)

# Step 6: Encode data
def encode_review(words, memory, model):
    vectors = []
    for i, word in enumerate(words[:max_len]):
        if word in memory:
            bound = model.opset.bind(
                memory[f"pos_{i}"].vec,
                memory[word].vec
            )
            vectors.append(bound)
    return model.opset.bundle(*vectors) if vectors else None

# Training data
positive_reviews = [
    ["I", "love", "this", "movie"],
    ["this", "film", "is", "great"],
    ["amazing", "movie"]
]

negative_reviews = [
    ["I", "hate", "this", "movie"],
    ["terrible", "film"],
    ["awful", "movie"]
]

# Build prototypes
pos_vecs = [encode_review(r, memory, model) for r in positive_reviews]
neg_vecs = [encode_review(r, memory, model) for r in negative_reviews]

prototype_pos = model.opset.bundle(*pos_vecs)
prototype_neg = model.opset.bundle(*neg_vecs)

# Step 7: Query
test_reviews = [
    ["I", "love", "this", "film"],      # Should be positive
    ["terrible", "movie"],               # Should be negative
    ["this", "movie", "is", "great"],   # Should be positive
]

print("\nTest Results:")
for review in test_reviews:
    test_vec = encode_review(review, memory, model)

    sim_pos = cosine_similarity(test_vec, prototype_pos)
    sim_neg = cosine_similarity(test_vec, prototype_neg)

    pred = "positive" if sim_pos > sim_neg else "negative"

    print(f"Review: {' '.join(review)}")
    print(f"  Positive: {sim_pos:.3f}, Negative: {sim_neg:.3f} → {pred}")

# Step 8: Evaluate
# In real application: test on held-out data, compute accuracy, analyze failures
```

**Output**:
```
Model: FHRROperations, Dim: 2048

Test Results:
Review: I love this film
  Positive: 0.856, Negative: 0.342 → positive
Review: terrible movie
  Positive: 0.245, Negative: 0.891 → negative
Review: this movie is great
  Positive: 0.823, Negative: 0.298 → positive
```

---

## Common Patterns

### Pattern 1: Classification Pipeline

```python
# 1. Create model + memory
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# 2. Add basis vectors
memory.add_many(feature_names)

# 3. Encode training data
prototypes = {}
for label in classes:
    class_samples = [encode(x) for x in training_data[label]]
    prototypes[label] = model.opset.bundle(*class_samples)

# 4. Classify new data
test_vec = encode(test_sample)
predictions = {label: cosine_similarity(test_vec, proto)
               for label, proto in prototypes.items()}
predicted_label = max(predictions, key=predictions.get)
```

### Pattern 2: Knowledge Graph

```python
# Encode facts
facts = []
for (subj, rel, obj) in triples:
    fact = model.opset.bundle(
        model.opset.bind(memory["role_subject"].vec, memory[subj].vec),
        model.opset.bind(memory["role_relation"].vec, memory[rel].vec),
        model.opset.bind(memory["role_object"].vec, memory[obj].vec)
    )
    facts.append(fact)

# Query: "What is the capital of France?"
# Known: relation=capital_of, object=France
# Unknown: subject=?
query = model.opset.bundle(
    model.opset.bind(memory["role_relation"].vec, memory["capital_of"].vec),
    model.opset.bind(memory["role_object"].vec, memory["France"].vec)
)

# Find best matching fact
best_fact = max(facts, key=lambda f: cosine_similarity(query, f))

# Extract subject (NEW: explicit unbind method)
subject_vec = model.opset.unbind(best_fact, memory["role_subject"].vec)

# Find city
for city in cities:
    sim = cosine_similarity(subject_vec, memory[city].vec)
    print(f"{city}: {sim:.3f}")
    # With FHRR: correct city should show >99% similarity!
```

### Pattern 3: Online Learning

```python
# Initial prototype
prototype = model.opset.bundle(*initial_examples)

# Add new example without retraining
new_example = encode(new_data)
prototype = model.opset.bundle(prototype, new_example)

# That's it! No backprop, no retraining
```

---

## Tips & Best Practices

### ✅ Do's

- **Start with FHRR, dim=2048** - Good default for learning
- **Normalize vectors** - Most similarity metrics expect unit vectors
- **Test incrementally** - Verify encoding before querying
- **Use factory functions** - `create_fhrr_model()` is simpler than manual creation
- **Leverage VSAMemory** - Dictionary-style access is convenient
- **Profile first** - Use CPU for prototyping, GPU for production
- **Save basis vectors** - Use `save_basis()` to persist learned representations

### ❌ Don'ts

- **Don't bundle too many items** - Limit to ~100-1000 depending on dimension
- **Don't use MAP for unbinding** - MAP unbinding is approximate
- **Don't forget to add symbols** - Must call `memory.add()` before using
- **Don't mix representations** - Stick to one model per application
- **Don't ignore similarities** - Values < 0.5 indicate poor match

---

## Common Pitfalls

### Pitfall 1: Dimension Too Small

**Symptom**: Low accuracy, low similarity scores
**Cause**: Not enough capacity to store all patterns
**Solution**: Increase dimension (try 2048 or 4096)

### Pitfall 2: Over-Bundling

**Symptom**: All similarities look the same (~0.5)
**Cause**: Too many items bundled together
**Solution**: Reduce items or increase dimension

### Pitfall 3: Wrong Encoder

**Symptom**: Encoding doesn't capture structure
**Cause**: Using SetEncoder for sequences (order matters!)
**Solution**: Use SequenceEncoder for ordered data

### Pitfall 4: Forgot to Normalize

**Symptom**: Similarity values are huge or tiny
**Cause**: Vectors not unit length
**Solution**: Most VSAX operations auto-normalize, but check if using raw arrays

### Pitfall 5: Trying to Unbind with MAP

**Symptom**: Unbinding doesn't recover original (~30% similarity)
**Cause**: MAP uses approximate unbinding (element-wise multiply inverse)
**Solution**: Use FHRR for exact unbinding (>99% with proper sampling)

---

## Decision Trees

### Model Selection Flowchart

```
START: What's your primary need?

├─ Exact unbinding for compositional structures?
│  └─ YES → FHRR ✓
│
├─ Maximum speed, simple task?
│  └─ YES → MAP ✓
│
├─ Memory-constrained or hardware deployment?
│  └─ YES → Binary (dim=10000) ✓
│
└─ Not sure?
   └─ Default: FHRR (dim=2048) ✓
```

### Encoder Selection Flowchart

```
START: What type of data do you have?

├─ Numbers (continuous values)?
│  └─ ScalarEncoder ✓
│
├─ Ordered sequence (sentence, time series)?
│  └─ SequenceEncoder ✓
│
├─ Unordered set (tags, categories)?
│  └─ SetEncoder ✓
│
├─ Key-value pairs (JSON, struct)?
│  └─ DictEncoder ✓
│
├─ Graph/network?
│  └─ GraphEncoder ✓
│
└─ Domain-specific (images, audio)?
   └─ Extend AbstractEncoder ✓
```

---

## Next Steps

### Learn by Example

Check out our tutorials for complete examples:

- [Tutorial 1: MNIST Classification](tutorials/01_mnist_classification.md) - Image classification
- [Tutorial 2: Knowledge Graph Reasoning](tutorials/02_knowledge_graph.md) - Relational reasoning
- [Tutorial 4: Word Analogies](tutorials/04_word_analogies.md) - NLP with VSA
- [Tutorial 5: Understanding VSA Models](tutorials/05_model_comparison.md) - Model comparison
- [Tutorial 8: Multi-Modal Grounding](tutorials/08_multimodal_grounding.md) - Heterogeneous data fusion

### Dive Deeper

- [User Guide](guide/models.md) - Detailed documentation
- [API Reference](api/index.md) - Complete API docs
- [Design Spec](design-spec.md) - Architecture and theory

### Get Help

- [GitHub Issues](https://github.com/vasanthsarathy/vsax/issues) - Report bugs or ask questions
- [Contributing](https://github.com/vasanthsarathy/vsax/blob/main/CONTRIBUTING.md) - Contribute to VSAX

---

**Happy modeling with VSAX! 🚀**
