# Lesson 3.2: Structured Data - Dictionaries and Sets

**Duration:** ~45 minutes

**Learning Objectives:**

- Master DictEncoder for key-value pairs (role-filler binding)
- Use SetEncoder for unordered collections
- Understand GraphEncoder for relational data
- Build structured representations (records, frames, knowledge)
- Query structured data by unbinding
- Debug common structured encoding issues

---

## Introduction

Real-world data is rarely just numbers or sequences—it's **structured**:
- **Records:** `{"name": "Alice", "age": 30, "city": "NYC"}`
- **Sets:** `{"red", "round", "sweet"}` (order doesn't matter)
- **Graphs:** `[(Alice, knows, Bob), (Bob, likes, Coffee)]`

In this lesson, we'll learn how to encode structured data using VSA's core operations: **binding** (for associations) and **bundling** (for aggregation).

---

## Dictionary Encoding: Key-Value Pairs

### The Problem: Structured Records

How do we encode a person with multiple attributes?

```python
person = {
    "name": "Alice",
    "age": 30,
    "city": "NYC"
}
```

**Naive bundling (wrong!):**
```python
# BAD: Loses key-value associations
person_hv = model.opset.bundle(
    memory["Alice"].vec,
    memory["30"].vec,
    memory["NYC"].vec
)
# Can't tell which value belongs to which key!
```

**Correct approach (role-filler binding):**
```python
# GOOD: Bind each key (role) with its value (filler)
name_pair = model.opset.bind(memory["name"].vec, memory["Alice"].vec)
age_pair = model.opset.bind(memory["age"].vec, memory["30"].vec)
city_pair = model.opset.bind(memory["city"].vec, memory["NYC"].vec)

# Bundle all key-value pairs
person_hv = model.opset.bundle(name_pair, age_pair, city_pair)
# Now can query: "What is the name?" by unbinding
```

---

### DictEncoder: Basic Usage

**DictEncoder** automates role-filler binding for dictionaries:

```python
from vsax import create_fhrr_model, VSAMemory, DictEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add all symbols (keys and values)
memory.add_many(["name", "age", "city", "Alice", "30", "NYC"])

# Create dictionary encoder
encoder = DictEncoder(model, memory)

# Encode person record
person = encoder.encode({
    "name": "Alice",
    "age": "30",
    "city": "NYC"
})

print(type(person))  # ComplexHypervector
```

**What it does:**
$$\text{person} = (\text{name} \otimes \text{Alice}) \oplus (\text{age} \otimes \text{30}) \oplus (\text{city} \otimes \text{NYC})$$

---

### Querying Dictionaries by Unbinding

Retrieve values by unbinding keys:

```python
# Query: "What is the name?"
name_inv = model.opset.inverse(memory["name"].vec)
retrieved = model.opset.bind(person.vec, name_inv)

# Find most similar value
from vsax.similarity import cosine_similarity
values = ["Alice", "30", "NYC", "Bob", "London"]

similarities = {}
for value in values:
    sim = cosine_similarity(retrieved, memory[value].vec)
    similarities[value] = float(sim)

best_match = max(similarities, key=similarities.get)
print(f"Name: {best_match}")  # "Alice"
```

**Expected Output:**
```
Name: Alice
```

---

### DictEncoder Use Cases

**1. Entity Records (Databases)**

```python
memory.add_many([
    "product_id", "price", "category", "stock",
    "P001", "29.99", "electronics", "50"
])

encoder = DictEncoder(model, memory)

# Encode product
product = encoder.encode({
    "product_id": "P001",
    "price": "29.99",
    "category": "electronics",
    "stock": "50"
})

# Later: query by product_id to get category
prod_id_inv = model.opset.inverse(memory["product_id"].vec)
retrieved_prod = model.opset.bind(product.vec, prod_id_inv)

# Should retrieve "P001"
```

---

**2. Semantic Frames (NLP)**

```python
# Sentence: "Alice gave Bob a book"
memory.add_many([
    "agent", "action", "recipient", "theme",
    "Alice", "gave", "Bob", "book"
])

encoder = DictEncoder(model, memory)

# Encode semantic frame
frame = encoder.encode({
    "agent": "Alice",      # Who did it
    "action": "gave",      # What happened
    "recipient": "Bob",    # To whom
    "theme": "book"        # What was given
})

# Query: Who was the recipient?
recip_inv = model.opset.inverse(memory["recipient"].vec)
retrieved = model.opset.bind(frame.vec, recip_inv)

# Find best match
candidates = ["Alice", "Bob", "book"]
sims = {c: cosine_similarity(retrieved, memory[c].vec) for c in candidates}
print(f"Recipient: {max(sims, key=sims.get)}")  # "Bob"
```

---

**3. Configuration Objects**

```python
memory.add_many([
    "model_type", "learning_rate", "batch_size", "epochs",
    "FHRR", "0.001", "32", "100"
])

encoder = DictEncoder(model, memory)

# Encode ML configuration
config = encoder.encode({
    "model_type": "FHRR",
    "learning_rate": "0.001",
    "batch_size": "32",
    "epochs": "100"
})

# Store multiple configs
configs = {
    "config1": config,
    "config2": encoder.encode({"model_type": "MAP", "learning_rate": "0.01", ...}),
}
```

---

**4. JSON Object Encoding**

```python
import json

# Sample JSON
data = {
    "user": "alice123",
    "action": "login",
    "timestamp": "2024-01-15",
    "status": "success"
}

# Add all symbols
symbols = set()
for key, value in data.items():
    symbols.add(key)
    symbols.add(str(value))

memory.add_many(list(symbols))

# Encode JSON
encoder = DictEncoder(model, memory)
json_hv = encoder.encode(data)

# Now can query any field by unbinding
```

---

## Set Encoding: Unordered Collections

### The Problem: Order-Invariant Groups

How do we encode tags for a photo: `{"outdoor", "sunny", "beach"}`?

**Key property:** Order doesn't matter!
- `{"outdoor", "sunny", "beach"}`
- `{"beach", "outdoor", "sunny"}`
These should be **identical** representations.

**Solution:** Use **bundling only** (no positional binding):

```python
# Bundling is commutative: a ⊕ b ⊕ c = c ⊕ a ⊕ b
tags_hv = model.opset.bundle(
    memory["outdoor"].vec,
    memory["sunny"].vec,
    memory["beach"].vec
)
```

---

### SetEncoder: Basic Usage

**SetEncoder** encodes unordered collections:

```python
from vsax import create_fhrr_model, VSAMemory, SetEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add tags
memory.add_many(["outdoor", "sunny", "beach", "water", "sand"])

# Create set encoder
encoder = SetEncoder(model, memory)

# Encode sets (order doesn't matter)
set1 = encoder.encode({"outdoor", "sunny", "beach"})
set2 = encoder.encode({"beach", "outdoor", "sunny"})  # Same set, different order

# Should be identical
from vsax.similarity import cosine_similarity
sim = cosine_similarity(set1.vec, set2.vec)
print(f"Similarity: {sim:.6f}")  # ~1.0 (identical!)
```

**Expected Output:**
```
Similarity: 0.999998
```

---

### Querying Sets: Membership Testing

Check if an element is in the set:

```python
# Encode set
photo_tags = encoder.encode({"outdoor", "sunny", "beach"})

# Test membership: Is "sunny" in the set?
test_element = memory["sunny"].vec
similarity = cosine_similarity(photo_tags.vec, test_element)

print(f"'sunny' membership score: {similarity:.4f}")  # High!

# Test non-member
test_element2 = memory["water"].vec
similarity2 = cosine_similarity(photo_tags.vec, test_element2)

print(f"'water' membership score: {similarity2:.4f}")  # Low
```

**Rule:** Similarity > threshold → element is member

---

### SetEncoder Use Cases

**1. Document Tags**

```python
memory.add_many([
    "machine_learning", "neural_networks", "NLP", "computer_vision",
    "reinforcement_learning", "transformers"
])

encoder = SetEncoder(model, memory)

# Encode document tags
doc1 = encoder.encode({"machine_learning", "neural_networks", "computer_vision"})
doc2 = encoder.encode({"NLP", "transformers", "neural_networks"})

# Find similar documents
sim = cosine_similarity(doc1.vec, doc2.vec)
print(f"Document similarity: {sim:.4f}")  # Higher if more overlapping tags
```

---

**2. User Interests**

```python
memory.add_many(["music", "sports", "cooking", "travel", "reading", "gaming"])

encoder = SetEncoder(model, memory)

# User profiles
alice = encoder.encode({"music", "cooking", "travel"})
bob = encoder.encode({"sports", "gaming", "music"})
carol = encoder.encode({"cooking", "reading", "travel"})

# Find most similar users to Alice
users = {"Bob": bob, "Carol": carol}
for name, profile in users.items():
    sim = cosine_similarity(alice.vec, profile.vec)
    print(f"Alice-{name} similarity: {sim:.4f}")

# Carol should be more similar (2 common interests vs 1)
```

---

**3. Product Features**

```python
memory.add_many([
    "wireless", "bluetooth", "noise_cancelling", "over_ear",
    "portable", "waterproof", "long_battery"
])

encoder = SetEncoder(model, memory)

# Headphones with features
headphone1 = encoder.encode({"wireless", "bluetooth", "noise_cancelling", "over_ear"})
headphone2 = encoder.encode({"wireless", "portable", "waterproof"})

# Find products with similar features
sim = cosine_similarity(headphone1.vec, headphone2.vec)
print(f"Product similarity: {sim:.4f}")
```

---

**4. Chemical Properties**

```python
memory.add_many([
    "flammable", "toxic", "corrosive", "reactive",
    "explosive", "oxidizer"
])

encoder = SetEncoder(model, memory)

# Chemical hazard sets
chemical_a = encoder.encode({"flammable", "toxic"})
chemical_b = encoder.encode({"corrosive", "reactive", "toxic"})

# Shared hazard: toxic
sim = cosine_similarity(chemical_a.vec, chemical_b.vec)
```

---

## Graph Encoding: Relational Data

### The Problem: Encoding Relationships

How do we encode a knowledge graph?

```
(Alice, knows, Bob)
(Alice, likes, Coffee)
(Bob, lives_in, NYC)
```

**Solution:** Encode each **triple** (subject, predicate, object) as:
$$\text{triple} = \text{subject} \otimes \text{predicate} \otimes \text{object}$$

Then **bundle** all triples into a graph.

---

### GraphEncoder: Basic Usage

**GraphEncoder** encodes graphs as collections of triples:

```python
from vsax import create_fhrr_model, VSAMemory, GraphEncoder

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add all entities and relations
memory.add_many([
    "Alice", "Bob", "Coffee", "NYC",
    "knows", "likes", "lives_in"
])

# Create graph encoder
encoder = GraphEncoder(model, memory)

# Encode knowledge graph
knowledge_graph = encoder.encode([
    ("Alice", "knows", "Bob"),
    ("Alice", "likes", "Coffee"),
    ("Bob", "lives_in", "NYC")
])

print(type(knowledge_graph))  # ComplexHypervector
```

**What it does:**
$$\text{graph} = (\text{Alice} \otimes \text{knows} \otimes \text{Bob}) \oplus (\text{Alice} \otimes \text{likes} \otimes \text{Coffee}) \oplus (\text{Bob} \otimes \text{lives\_in} \otimes \text{NYC})$$

---

### Querying Graphs

**Query 1: Who does Alice know?**

```python
# Unbind Alice and knows to retrieve object
alice_knows = model.opset.bind(
    model.opset.bind(
        knowledge_graph.vec,
        model.opset.inverse(memory["Alice"].vec)
    ),
    model.opset.inverse(memory["knows"].vec)
)

# Find most similar entity
from vsax.similarity import cosine_similarity
entities = ["Alice", "Bob", "Coffee", "NYC"]

sims = {e: cosine_similarity(alice_knows, memory[e].vec) for e in entities}
answer = max(sims, key=sims.get)

print(f"Alice knows: {answer}")  # "Bob"
```

**Query 2: What does Alice like?**

```python
alice_likes = model.opset.bind(
    model.opset.bind(
        knowledge_graph.vec,
        model.opset.inverse(memory["Alice"].vec)
    ),
    model.opset.inverse(memory["likes"].vec)
)

sims = {e: cosine_similarity(alice_likes, memory[e].vec) for e in entities}
answer = max(sims, key=sims.get)

print(f"Alice likes: {answer}")  # "Coffee"
```

---

### GraphEncoder Use Cases

**1. Social Networks**

```python
memory.add_many([
    "Alice", "Bob", "Carol", "Dave",
    "follows", "likes", "shares", "comments"
])

encoder = GraphEncoder(model, memory)

# Social graph
social = encoder.encode([
    ("Alice", "follows", "Bob"),
    ("Alice", "likes", "Carol"),
    ("Bob", "follows", "Carol"),
    ("Carol", "shares", "Dave")
])

# Query: Who does Alice follow?
# Unbind (Alice, follows, ?)
```

---

**2. Knowledge Bases**

```python
memory.add_many([
    "Paris", "France", "London", "England",
    "capital_of", "part_of", "larger_than"
])

encoder = GraphEncoder(model, memory)

# Geographic knowledge
geo_kb = encoder.encode([
    ("Paris", "capital_of", "France"),
    ("London", "capital_of", "England"),
    ("France", "larger_than", "England")
])

# Multi-hop reasoning: What is the capital of France?
# (?, capital_of, France)
```

---

**3. Dependency Graphs (Software)**

```python
memory.add_many([
    "moduleA", "moduleB", "moduleC", "moduleD",
    "depends_on", "imports", "calls"
])

encoder = GraphEncoder(model, memory)

# Code dependencies
deps = encoder.encode([
    ("moduleA", "depends_on", "moduleB"),
    ("moduleA", "imports", "moduleC"),
    ("moduleB", "depends_on", "moduleD")
])

# Query: What does moduleA depend on?
```

---

**4. Biological Networks**

```python
memory.add_many([
    "geneA", "geneB", "proteinX", "proteinY",
    "codes_for", "interacts_with", "regulates"
])

encoder = GraphEncoder(model, memory)

# Gene regulatory network
bio_net = encoder.encode([
    ("geneA", "codes_for", "proteinX"),
    ("geneB", "codes_for", "proteinY"),
    ("proteinX", "interacts_with", "proteinY"),
    ("proteinX", "regulates", "geneB")
])
```

---

## Combining Dict, Set, and Graph Encoders

Real-world applications combine multiple encoder types:

### Example: Product Catalog with Reviews

```python
from vsax import create_fhrr_model, VSAMemory, DictEncoder, SetEncoder, GraphEncoder

model = create_fhrr_model(dim=4096)  # Higher dim for complex structure
memory = VSAMemory(model)

# Add all symbols
symbols = [
    # Products
    "product_id", "name", "price", "category",
    "P001", "headphones", "99.99", "electronics",
    # Features (for sets)
    "wireless", "bluetooth", "noise_cancelling",
    # Reviews (for graphs)
    "Alice", "Bob", "rated", "reviewed"
]
memory.add_many(symbols)

# 1. Encode product metadata with DictEncoder
dict_enc = DictEncoder(model, memory)
product_meta = dict_enc.encode({
    "product_id": "P001",
    "name": "headphones",
    "price": "99.99",
    "category": "electronics"
})

# 2. Encode product features with SetEncoder
set_enc = SetEncoder(model, memory)
product_features = set_enc.encode({
    "wireless", "bluetooth", "noise_cancelling"
})

# 3. Encode reviews with GraphEncoder
graph_enc = GraphEncoder(model, memory)
product_reviews = graph_enc.encode([
    ("Alice", "rated", "P001"),
    ("Bob", "reviewed", "P001")
])

# 4. Combine everything
product_complete = model.opset.bundle(
    product_meta.vec,
    product_features.vec,
    product_reviews.vec
)

print("Complete product representation created!")
```

**This creates a rich structured representation combining:**
- **Metadata** (key-value pairs via DictEncoder)
- **Features** (unordered set via SetEncoder)
- **Reviews** (relations via GraphEncoder)

---

## Common Structured Encoding Issues

### Issue 1: "Query returns wrong value from dictionary"

**Symptom:**
```python
person = encoder.encode({"name": "Alice", "age": "30"})

# Query name
name_inv = model.opset.inverse(memory["name"].vec)
retrieved = model.opset.bind(person.vec, name_inv)

# Best match is "30" instead of "Alice"!
```

**Causes:**
1. Symbols not added to memory
2. Dimension too low (similarity degrades)
3. Inverse operation incorrect

**Fixes:**
```python
# 1. Ensure all symbols are in memory
memory.add_many(["name", "age", "Alice", "30"])

# 2. Increase dimension
model = create_fhrr_model(dim=4096)  # Instead of 512

# 3. Use correct model (FHRR recommended for unbinding accuracy)
```

---

### Issue 2: "Set order affects similarity"

**Symptom:**
```python
set1 = encoder.encode({"a", "b", "c"})
set2 = encoder.encode({"c", "b", "a"})
sim = cosine_similarity(set1.vec, set2.vec)
# sim = 0.85 (should be ~1.0!)
```

**Cause:** Using SequenceEncoder instead of SetEncoder.

**Fix:**
```python
from vsax import SetEncoder  # NOT SequenceEncoder!
encoder = SetEncoder(model, memory)
```

---

### Issue 3: "Graph queries return low similarity"

**Symptom:**
```python
# Encoded (Alice, knows, Bob)
# Query: Who does Alice know?
# All similarities are ~0.3
```

**Causes:**
1. Triple binding depth too high (3 bindings: s ⊗ p ⊗ o)
2. Model doesn't support deep binding well (MAP)
3. Dimension too low

**Fixes:**
```python
# 1. Use FHRR for exact unbinding
model = create_fhrr_model(dim=4096)

# 2. Increase dimension for graph encoding
dim = 4096  # Or 8192 for complex graphs

# 3. Alternative: use 2-hop encoding (s ⊗ p) bundled with o
# Less compositional but more robust
```

---

## Performance Considerations

### Memory Footprint

**Complex structures** require higher dimensions:

| Structure Complexity | Recommended Dimension |
|---------------------|----------------------|
| Simple dict (3-5 keys) | 2048 |
| Medium dict (10-20 keys) | 4096 |
| Large dict (50+ keys) | 8192 |
| Small graphs (10-50 triples) | 4096 |
| Large graphs (100+ triples) | 8192-16384 |

### Encoding Speed

**Encoding time** scales with structure size:

```python
import time

# Simple dict: ~0.1 ms
small_dict = {"a": "1", "b": "2"}

# Large dict: ~1 ms
large_dict = {f"key{i}": f"val{i}" for i in range(100)}

# Measure
start = time.time()
encoder.encode(large_dict)
print(f"Encoding time: {(time.time() - start) * 1000:.2f} ms")
```

**Optimization:** Pre-encode common structures and reuse.

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Use DictEncoder to encode key-value pairs
- [ ] Query dictionaries by unbinding keys
- [ ] Use SetEncoder for order-invariant collections
- [ ] Test set membership using similarity
- [ ] Use GraphEncoder to encode relational triples
- [ ] Query graphs with multi-step unbinding
- [ ] Combine multiple encoder types for complex structures
- [ ] Debug common structured encoding issues

---

## Quick Quiz

**Q1:** What VSA operation does DictEncoder use for key-value association?

a) Bundling (⊕)
b) Binding (⊗)
c) Permutation
d) Inverse

<details>
<summary>Answer</summary>
**b) Binding (⊗)** - DictEncoder binds each key with its value (role-filler binding), then bundles all pairs: (k₁ ⊗ v₁) ⊕ (k₂ ⊗ v₂) ⊕ ...
</details>

**Q2:** Why does SetEncoder produce the same output regardless of element order?

a) Binding is commutative
b) Bundling is commutative
c) Sets are sorted before encoding
d) Special normalization step

<details>
<summary>Answer</summary>
**b) Bundling is commutative** - SetEncoder uses bundling only (no positional binding). Since a ⊕ b ⊕ c = c ⊕ a ⊕ b, order doesn't matter.
</details>

**Q3:** For encoding graph triple (Alice, knows, Bob), what operations are used?

a) Bundle all three
b) Bind all three: Alice ⊗ knows ⊗ Bob
c) Bind pairs: (Alice ⊗ knows) ⊕ (knows ⊗ Bob)
d) Sequence encoding with positions

<details>
<summary>Answer</summary>
**b) Bind all three** - Graph triples are encoded as subject ⊗ predicate ⊗ object, creating a compositional representation that can be queried by unbinding.
</details>

**Q4:** Which model is best for complex graph encoding with deep unbinding?

a) Binary (fastest operations)
b) MAP (real-valued vectors)
c) FHRR (exact unbinding)
d) All models work equally well

<details>
<summary>Answer</summary>
**c) FHRR (exact unbinding)** - Graph queries require unbinding 2-3 levels deep (s ⊗ p ⊗ o). FHRR maintains >0.99 similarity after deep unbinding, while MAP accumulates error.
</details>

---

## Hands-On Exercise: Build a Mini Knowledge Base

**Task:** Create a knowledge base about animals and query it.

```python
from vsax import create_fhrr_model, VSAMemory, DictEncoder, SetEncoder, GraphEncoder
from vsax.similarity import cosine_similarity

model = create_fhrr_model(dim=4096)
memory = VSAMemory(model)

# Knowledge base facts:
# 1. Animals with attributes (DictEncoder)
#    - Dog: {species: "canine", size: "medium", lifespan: "12"}
#    - Cat: {species: "feline", size: "small", lifespan: "15"}
#
# 2. Animals with features (SetEncoder)
#    - Dog: {furry, domesticated, loyal}
#    - Cat: {furry, domesticated, independent}
#
# 3. Relationships (GraphEncoder)
#    - (Dog, chases, Cat)
#    - (Cat, catches, Mouse)
#    - (Dog, larger_than, Cat)

# YOUR CODE HERE:
# 1. Add all necessary symbols to memory
# 2. Create encoders (DictEncoder, SetEncoder, GraphEncoder)
# 3. Encode animal attributes, features, and relationships
# 4. Query: What features do Dog and Cat share?
# 5. Query: What does Dog chase?
```

<details>
<summary>Solution</summary>

```python
# Step 1: Add all symbols
symbols = [
    # Animal names
    "Dog", "Cat", "Mouse",
    # Attribute keys
    "species", "size", "lifespan",
    # Attribute values
    "canine", "feline", "rodent", "medium", "small", "tiny", "12", "15", "2",
    # Features
    "furry", "domesticated", "loyal", "independent", "nocturnal",
    # Relations
    "chases", "catches", "larger_than"
]
memory.add_many(symbols)

# Step 2: Create encoders
dict_enc = DictEncoder(model, memory)
set_enc = SetEncoder(model, memory)
graph_enc = GraphEncoder(model, memory)

# Step 3: Encode animals
# Dog attributes
dog_attrs = dict_enc.encode({
    "species": "canine",
    "size": "medium",
    "lifespan": "12"
})

# Dog features
dog_features = set_enc.encode({"furry", "domesticated", "loyal"})

# Combine dog representation
dog = model.opset.bundle(dog_attrs.vec, dog_features.vec)

# Cat attributes
cat_attrs = dict_enc.encode({
    "species": "feline",
    "size": "small",
    "lifespan": "15"
})

# Cat features
cat_features = set_enc.encode({"furry", "domesticated", "independent"})

# Combine cat representation
cat = model.opset.bundle(cat_attrs.vec, cat_features.vec)

# Encode relationships
relationships = graph_enc.encode([
    ("Dog", "chases", "Cat"),
    ("Cat", "catches", "Mouse"),
    ("Dog", "larger_than", "Cat")
])

# Query 1: What features do Dog and Cat share?
shared_sim = cosine_similarity(dog_features.vec, cat_features.vec)
print(f"Dog-Cat feature similarity: {shared_sim:.4f}")
print("Shared features: furry, domesticated")

# Query 2: What does Dog chase?
dog_chases = model.opset.bind(
    model.opset.bind(
        relationships.vec,
        model.opset.inverse(memory["Dog"].vec)
    ),
    model.opset.inverse(memory["chases"].vec)
)

animals = ["Dog", "Cat", "Mouse"]
sims = {a: float(cosine_similarity(dog_chases, memory[a].vec)) for a in animals}

print(f"\nWhat does Dog chase?")
for animal, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True):
    print(f"  {animal}: {sim:.4f}")

answer = max(sims, key=sims.get)
print(f"\nAnswer: Dog chases {answer}")
```

**Expected Output:**
```
Dog-Cat feature similarity: 0.7071
Shared features: furry, domesticated

What does Dog chase?
  Cat: 0.8234
  Mouse: 0.0123
  Dog: 0.0089

Answer: Dog chases Cat
```
</details>

---

## Key Takeaways

1. **DictEncoder for key-value pairs** - Role-filler binding: (key ⊗ value) ⊕ ...
2. **SetEncoder for unordered collections** - Bundling only (order-invariant)
3. **GraphEncoder for relational data** - Triple binding: subject ⊗ predicate ⊗ object
4. **Querying by unbinding** - Retrieve values by unbinding keys/roles
5. **Combine encoders** - Real structures use multiple encoder types together
6. **Use FHRR for complex structures** - Deep unbinding requires exact operations
7. **Higher dimensions for complexity** - Complex graphs need d ≥ 4096

---

**Next:** [Lesson 3.3: Application - Image Classification](03_images.md)

Put encoders into action with a complete MNIST classification application.

**Previous:** [Lesson 3.1: Scalar and Sequence Encoding](01_scalar_sequence.md)
