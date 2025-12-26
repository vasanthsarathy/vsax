# Lesson 3.4: Application - Knowledge Graph Reasoning

**Duration:** ~60 minutes (30 min theory + 30 min tutorial)

**Learning Objectives:**

- Understand knowledge graphs as relational triples
- Encode facts using GraphEncoder (subject ⊗ relation ⊗ object)
- Query knowledge bases by unbinding
- Use resonator networks for factorization
- Perform multi-hop reasoning
- Complete a knowledge graph reasoning project

---

## Introduction

Knowledge graphs represent relationships between entities:
- **WordNet:** (dog, isA, mammal)
- **Wikidata:** (Paris, capitalOf, France)
- **Medical ontologies:** (aspirin, treats, headache)

In this lesson, we'll build and query knowledge graphs using VSA. This demonstrates how **binding** and **unbinding** enable symbolic reasoning over relational data.

**What makes VSA good for knowledge graphs?**
- ✅ **Compositional:** Facts compose via binding
- ✅ **Distributed:** Knowledge spread across high dimensions
- ✅ **Robust:** Handles noise and partial information
- ✅ **Efficient:** Constant-time operations (vs graph search)
- ✅ **Analogical:** Similar facts have similar representations

---

## Knowledge Graphs as Triples

### Structure: (Subject, Relation, Object)

Every fact is a triple:

```
(dog, isA, mammal)
(Paris, capitalOf, France)
(Alice, knows, Bob)
```

**Components:**
- **Subject:** Entity the fact is about
- **Relation:** Type of relationship
- **Object:** What the subject relates to

---

### Encoding Triples with VSA

Each triple is encoded using **3-way binding**:

$$\text{fact} = \text{subject} \otimes \text{relation} \otimes \text{object}$$

**Example:**
```python
from vsax import create_fhrr_model, VSAMemory

model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

# Add concepts
memory.add_many(["dog", "isA", "mammal"])

# Encode fact: (dog, isA, mammal)
dog = memory["dog"].vec
isA = memory["isA"].vec
mammal = memory["mammal"].vec

# Bind all three
fact = model.opset.bind(
    model.opset.bind(dog, isA),
    mammal
)

print(f"Fact encoded: dog ⊗ isA ⊗ mammal")
```

**Key insight:** The fact is now a **single hypervector** that can be bundled with other facts!

---

### Building a Knowledge Base

A knowledge base is a **bundle** of all facts:

```python
from vsax.encoders import GraphEncoder

# Create graph encoder
encoder = GraphEncoder(model, memory)

# Define knowledge base as list of triples
knowledge = encoder.encode([
    ("dog", "isA", "mammal"),
    ("cat", "isA", "mammal"),
    ("bird", "isA", "animal"),
    ("dog", "hasProperty", "fur"),
    ("bird", "hasProperty", "feathers"),
    ("dog", "can", "bark"),
    ("bird", "can", "fly")
])

print(f"Knowledge base: {len([...])} facts encoded")
```

**What it does:**
$$\text{KB} = (\text{dog} \otimes \text{isA} \otimes \text{mammal}) \oplus (\text{cat} \otimes \text{isA} \otimes \text{mammal}) \oplus \ldots$$

---

## Querying Knowledge Graphs

### Query Type 1: (Subject, Relation, ?)

**Question:** "What is dog?"

```python
# Query: (dog, isA, ?)
# Unbind dog and isA to retrieve object

dog_inv = model.opset.inverse(memory["dog"].vec)
isA_inv = model.opset.inverse(memory["isA"].vec)

# Unbind twice
retrieved = model.opset.bind(knowledge.vec, dog_inv)
retrieved = model.opset.bind(retrieved, isA_inv)

# Find most similar concept
from vsax.similarity import cosine_similarity
candidates = ["mammal", "animal", "fur", "bark", "cat"]

similarities = {c: cosine_similarity(retrieved, memory[c].vec)
                for c in candidates}

answer = max(similarities, key=similarities.get)
print(f"(dog, isA, ?) → {answer}")  # "mammal"
```

---

### Query Type 2: (Subject, ?, Object)

**Question:** "What relationship does dog have with mammal?"

```python
# Query: (dog, ?, mammal)
# Unbind dog and mammal to retrieve relation

dog_inv = model.opset.inverse(memory["dog"].vec)
mammal_inv = model.opset.inverse(memory["mammal"].vec)

retrieved = model.opset.bind(knowledge.vec, dog_inv)
retrieved = model.opset.bind(retrieved, mammal_inv)

# Find most similar relation
relations = ["isA", "hasProperty", "can"]
similarities = {r: cosine_similarity(retrieved, memory[r].vec)
                for r in relations}

answer = max(similarities, key=similarities.get)
print(f"(dog, ?, mammal) → {answer}")  # "isA"
```

---

### Query Type 3: (?, Relation, Object)

**Question:** "What has property fur?"

```python
# Query: (?, hasProperty, fur)
# Unbind hasProperty and fur to retrieve subject

hasProperty_inv = model.opset.inverse(memory["hasProperty"].vec)
fur_inv = model.opset.inverse(memory["fur"].vec)

retrieved = model.opset.bind(knowledge.vec, hasProperty_inv)
retrieved = model.opset.bind(retrieved, fur_inv)

# Find most similar subject
animals = ["dog", "cat", "bird", "fish"]
similarities = {a: cosine_similarity(retrieved, memory[a].vec)
                for a in animals}

answer = max(similarities, key=similarities.get)
print(f"(?, hasProperty, fur) → {answer}")  # "dog" or "cat"
```

---

## Multi-Hop Reasoning

**Challenge:** Answer questions requiring multiple inference steps.

**Question:** "What properties do mammals have?"

**Reasoning chain:**
1. Find what is a mammal: (?, isA, mammal) → dog, cat
2. Find properties: (dog, hasProperty, ?) → fur

### Approach 1: Sequential Queries

```python
# Step 1: Find mammals
isA_inv = model.opset.inverse(memory["isA"].vec)
mammal_inv = model.opset.inverse(memory["mammal"].vec)

mammals_vec = model.opset.bind(knowledge.vec, isA_inv)
mammals_vec = model.opset.bind(mammals_vec, mammal_inv)

# Step 2: For each mammal, find properties
# (This is simplified; use resonator for robust multi-hop)

hasProperty_inv = model.opset.inverse(memory["hasProperty"].vec)

properties_vec = model.opset.bind(knowledge.vec, mammals_vec)
properties_vec = model.opset.bind(properties_vec, hasProperty_inv)

# Find most similar properties
properties = ["fur", "feathers", "scales"]
similarities = {p: cosine_similarity(properties_vec, memory[p].vec)
                for p in properties}

answer = max(similarities, key=similarities.get)
print(f"Mammals have: {answer}")  # "fur"
```

---

### Approach 2: Resonator Networks

**Resonator networks** iteratively clean up noisy retrievals for **robust multi-hop reasoning**.

**How resonators work:**

1. Start with noisy query result
2. Find closest match in memory (cleanup)
3. Use cleaned result for next hop
4. Repeat until convergence

```python
from vsax.resonator import CleanupMemory, Resonator

# Create cleanup memory with all concepts
cleanup = CleanupMemory(model)
for concept in memory.symbols.keys():
    cleanup.add(concept, memory[concept].vec)

# Create resonator
resonator = Resonator(model, cleanup, max_iterations=10)

# Query with resonator for robust retrieval
query_vec = model.opset.bind(knowledge.vec, dog_inv)
query_vec = model.opset.bind(query_vec, isA_inv)

# Resonate to clean up
cleaned = resonator.resonate(query_vec)

# Find best match
best_match = cleanup.query(cleaned)
print(f"Resonator result: {best_match}")  # "mammal" (more robust!)
```

**Benefits:**
- More robust to noise
- Better multi-hop accuracy
- Handles deep reasoning chains

---

## Comparison: VSA vs Traditional Knowledge Graphs

| Feature | VSA Knowledge Graphs | Traditional (RDF, Neo4j) |
|---------|---------------------|--------------------------|
| **Storage** | Single hypervector (constant size) | Graph structure (scales with facts) |
| **Query time** | O(1) unbinding | O(E) graph traversal |
| **Multi-hop** | Compositional binding | Breadth-first search |
| **Noise tolerance** | Robust (distributed representation) | Brittle (exact match required) |
| **Analogical reasoning** | Natural (similar facts → similar vectors) | Requires explicit rules |
| **Scalability** | Constant memory & time | Scales with graph size |

**VSA advantages:**
- ✅ Constant-time queries (no search)
- ✅ Handles incomplete/noisy data
- ✅ Analogical reasoning built-in

**VSA challenges:**
- ❌ Approximate retrieval (not exact)
- ❌ Dimension selection matters
- ❌ Deep chains (>5 hops) degrade

---

## Hands-On: Complete Knowledge Graph Tutorial

Now build and query a complete knowledge base!

**📓 [Tutorial 2: Knowledge Graph Reasoning](../../tutorials/02_knowledge_graph.md)**

**What you'll do in the tutorial:**

1. **Build knowledge base:** Animal taxonomy with isA, hasProperty, can relations
2. **Encode facts:** Use GraphEncoder for triple encoding
3. **Simple queries:** Retrieve objects, relations, subjects
4. **Multi-hop reasoning:** Chain queries together
5. **Resonator networks:** Use cleanup for robust retrieval
6. **Model comparison:** Benchmark FHRR, MAP, Binary on reasoning tasks
7. **Visualization:** Inspect fact similarities and query results

**Time estimate:** 30-45 minutes

**Prerequisites:**
- Completed Lesson 3.2 (DictEncoder, GraphEncoder)
- Understanding of binding/unbinding operations

---

## Key Concepts from the Tutorial

### 1. Triple Encoding

Facts as compositional bindings:

```python
# Fact: (Paris, capitalOf, France)
fact = model.opset.bind(
    model.opset.bind(paris, capitalOf),
    france
)
```

### 2. Knowledge Base as Bundle

All facts combined:

```python
KB = fact1 ⊕ fact2 ⊕ fact3 ⊕ ... ⊕ factN
```

### 3. Query by Unbinding

Retrieve missing element:

```python
# Query: (Paris, capitalOf, ?)
result = KB ⊗ Paris^(-1) ⊗ capitalOf^(-1)
# Cleanup result to find: France
```

### 4. Resonator Cleanup

Iterative cleanup for noisy queries:

```python
resonator = Resonator(model, cleanup_memory, max_iterations=10)
cleaned_result = resonator.resonate(noisy_query_result)
```

### 5. Similarity-Based Inference

Analogical reasoning via hypervector similarity:

```python
# If (dog, isA, mammal) and (cat, isA, mammal),
# then dog ≈ cat (same category)
sim = cosine_similarity(dog_fact, cat_fact)  # High!
```

---

## Extensions and Experiments

After completing the tutorial, try these:

### 1. Larger Knowledge Bases

Scale to hundreds or thousands of facts:

```python
# Load from external ontology
import json

with open('animal_ontology.json') as f:
    triples = json.load(f)

# Encode all
kb = encoder.encode(triples)
```

**Challenge:** How does query accuracy scale with KB size?

---

### 2. Transitive Relations

Encode transitive properties (if A→B and B→C, then A→C):

```python
# Encode: (Paris, partOf, France) and (France, partOf, Europe)
# Query: (Paris, partOf?, Europe) via multi-hop
```

---

### 3. Negative Facts

Handle negation:

```python
# Positive: (bird, can, fly)
# Negative: (penguin, cannot, fly)

# Use separate "cannot" relation
memory.add("cannot")
negative_fact = encoder.encode([("penguin", "cannot", "fly")])
```

---

### 4. Temporal Knowledge

Add time dimension:

```python
# (Alice, married, Bob, 2010)
# 4-way binding: subject ⊗ relation ⊗ object ⊗ time
```

---

### 5. Confidence/Weights

Weight facts by confidence:

```python
# High confidence: (water, freezes_at, 0C)
# Low confidence: (Pluto, isA?, planet)

# Weight by multiplying fact vector by confidence
weighted_fact = 0.9 * high_confidence_fact + 0.3 * low_confidence_fact
```

---

## Real-World Applications

VSA knowledge graphs are used in:

**1. Question Answering**
- Encode facts from text
- Query with natural language questions
- Multi-hop reasoning for complex queries

**2. Drug Discovery**
- Encode biochemical interactions
- Query: "What proteins does drug X bind to?"
- Analogical reasoning: "Drugs similar to X?"

**3. Robotics**
- Spatial knowledge: (kitchen, nextTo, livingRoom)
- Object properties: (cup, contains, liquid)
- Action planning via multi-hop queries

**4. Recommendation Systems**
- User preferences: (Alice, likes, scienceFiction)
- Content features: (Interstellar, hasGenre, scienceFiction)
- Query: "What does Alice like?" → Interstellar

**5. Ontology Reasoning**
- Taxonomies: (Chihuahua, isA, Dog, isA, Mammal)
- Property inheritance: "What properties do Chihuahuas have?"
- Classify new entities via similarity

---

## Self-Assessment

Before moving to the next lesson, ensure you can:

- [ ] Encode knowledge as relational triples (s ⊗ r ⊗ o)
- [ ] Build knowledge bases using GraphEncoder
- [ ] Query by unbinding to retrieve missing elements
- [ ] Perform multi-hop reasoning chains
- [ ] Use resonator networks for robust cleanup
- [ ] Complete the knowledge graph tutorial
- [ ] Compare VSA vs traditional graph databases
- [ ] Extend knowledge bases with new relations

---

## Quick Quiz

**Q1:** How is a knowledge triple (Alice, knows, Bob) encoded in VSA?

a) Alice ⊕ knows ⊕ Bob (bundling)
b) Alice ⊗ knows ⊗ Bob (binding)
c) [Alice, knows, Bob] (sequence)
d) {Alice, knows, Bob} (set)

<details>
<summary>Answer</summary>
**b) Alice ⊗ knows ⊗ Bob** - Triples are encoded using 3-way binding to create a compositional representation that can be queried by unbinding any element.
</details>

**Q2:** To query "(dog, isA, ?)" from a knowledge base, what operations do you perform?

a) Bind KB with dog and isA
b) Unbind dog and isA from KB
c) Bundle KB with dog and isA
d) Permute KB by dog and isA

<details>
<summary>Answer</summary>
**b) Unbind dog and isA** - To retrieve the missing object, unbind (inverse bind) the known subject and relation from the knowledge base: KB ⊗ dog^(-1) ⊗ isA^(-1).
</details>

**Q3:** What is the advantage of resonator networks for knowledge graph queries?

a) Faster queries
b) Less memory usage
c) Iterative cleanup for robust multi-hop reasoning
d) Exact (non-approximate) results

<details>
<summary>Answer</summary>
**c) Iterative cleanup for robust multi-hop** - Resonators iteratively clean up noisy query results by finding the closest match in memory, enabling more accurate multi-hop reasoning chains.
</details>

**Q4:** Why is FHRR recommended over MAP for knowledge graph reasoning?

a) FHRR is faster
b) FHRR uses less memory
c) FHRR provides exact unbinding (critical for deep reasoning chains)
d) FHRR supports more relations

<details>
<summary>Answer</summary>
**c) FHRR provides exact unbinding** - Knowledge queries often require 2-3 levels of unbinding (s ⊗ r ⊗ o). FHRR maintains >0.99 similarity after deep unbinding, while MAP accumulates error with each unbind operation.
</details>

---

## Key Takeaways

1. **Knowledge as triples** - (subject, relation, object) encoded via binding
2. **KB is bundle of facts** - All facts bundled into single hypervector
3. **Query by unbinding** - Retrieve missing elements: KB ⊗ s^(-1) ⊗ r^(-1)
4. **Multi-hop reasoning** - Chain queries for complex inference
5. **Resonators for robustness** - Iterative cleanup improves accuracy
6. **Constant-time queries** - No graph search needed (vs traditional DBs)
7. **Analogical reasoning** - Similar facts have similar representations

---

**Next:** [Lesson 3.5: Application - Analogical Reasoning](05_analogies.md)

Learn how VSA naturally supports analogy solving and conceptual spaces.

**Previous:** [Lesson 3.3: Application - Image Classification](03_images.md)
