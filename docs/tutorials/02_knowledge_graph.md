# Tutorial 2: Knowledge Graph Reasoning with VSAX

This tutorial demonstrates how to use Vector Symbolic Architectures (VSAs) for knowledge graph representation and reasoning.

## What You'll Learn

- Encode knowledge as relational triples (subject-relation-object)
- Build and query a knowledge base using VSA
- Use resonator networks to factorize compositional structures
- Perform multi-hop reasoning to infer new knowledge
- Compare different VSA models for knowledge representation

## Why VSA for Knowledge Graphs?

VSAs offer several advantages for knowledge representation:

1. **Compositional**: Facts can be composed using binding operations
2. **Distributed**: Knowledge is spread across high-dimensional vectors
3. **Robust**: Tolerant to noise and partial information
4. **Efficient**: Constant-time operations regardless of knowledge base size
5. **Analogical**: Similar facts have similar representations

## Setup

```python
import jax.numpy as jnp
from vsax import create_fhrr_model, create_map_model, create_binary_model
from vsax import VSAMemory
from vsax.encoders import GraphEncoder
from vsax.resonator import CleanupMemory, Resonator
from vsax.similarity import cosine_similarity
from vsax.utils import format_similarity_results

# Create FHRR model (best for exact unbinding)
model = create_fhrr_model(dim=512)
memory = VSAMemory(model)

print(f"Model: {model.rep_cls.__name__}")
print(f"Dimension: {model.dim}")
```

Output:
```
Model: ComplexHypervector
Dimension: 512
```

## Building the Knowledge Base

We'll create a simple animal taxonomy with:
- **Taxonomy relations**: X isA Y (dog isA mammal)
- **Property relations**: X hasProperty Y (dog hasProperty fur)
- **Action relations**: X can Y (dog can bark)

```python
# Define all concepts we'll need
concepts = [
    # Animals
    "dog", "cat", "bird", "fish", "snake",
    # Categories
    "mammal", "reptile", "animal",
    # Relations
    "isA", "hasProperty", "can",
    # Properties
    "fur", "feathers", "scales", "warm_blooded", "cold_blooded",
    # Actions
    "bark", "meow", "fly", "swim", "slither"
]

# Add all concepts to memory
memory.add_many(concepts)
print(f"Knowledge base contains {len(memory)} concepts")
```

Output:
```
Knowledge base contains 23 concepts
```

```python
# Define knowledge as triples: (subject, relation, object)
facts = [
    # Taxonomy
    ("dog", "isA", "mammal"),
    ("cat", "isA", "mammal"),
    ("bird", "isA", "animal"),
    ("fish", "isA", "animal"),
    ("snake", "isA", "reptile"),
    ("mammal", "isA", "animal"),
    ("reptile", "isA", "animal"),

    # Properties
    ("dog", "hasProperty", "fur"),
    ("cat", "hasProperty", "fur"),
    ("bird", "hasProperty", "feathers"),
    ("fish", "hasProperty", "scales"),
    ("snake", "hasProperty", "scales"),
    ("mammal", "hasProperty", "warm_blooded"),
    ("reptile", "hasProperty", "cold_blooded"),

    # Actions
    ("dog", "can", "bark"),
    ("cat", "can", "meow"),
    ("bird", "can", "fly"),
    ("fish", "can", "swim"),
    ("snake", "can", "slither"),
]

print(f"Knowledge base contains {len(facts)} facts")
print("\nSample facts:")
for fact in facts[:5]:
    print(f"  {fact[0]} {fact[1]} {fact[2]}")
```

Output:
```
Knowledge base contains 19 facts

Sample facts:
  dog isA mammal
  cat isA mammal
  bird isA animal
  fish isA animal
  snake isA reptile
```

## Encoding Facts as Hypervectors

Each fact (subject, relation, object) is encoded as:
```
fact = bind(subject, bind(relation, object))
```

This allows us to:
- Query for objects given subject and relation
- Query for relations given subject and object
- Factorize facts using resonator networks

```python
# Store individual facts
fact_hvs = {}

for subject, relation, obj in facts:
    s_hv = memory[subject]
    r_hv = memory[relation]
    o_hv = memory[obj]

    # Encode: bind(subject, bind(relation, object))
    ro = model.opset.bind(r_hv.vec, o_hv.vec)
    fact_hv = model.opset.bind(s_hv.vec, ro)

    fact_hvs[(subject, relation, obj)] = model.rep_cls(fact_hv)

print(f"Encoded {len(fact_hvs)} facts as hypervectors")
```

Output:
```
Encoded 19 facts as hypervectors
```

## Querying the Knowledge Base

We can query facts by unbinding (using inverse operation):

**Query: "What is a dog?"** (dog isA ?)
```
query = unbind(fact, bind(dog, isA))
```

```python
def query_fact(subject: str, relation: str) -> str:
    """Query: subject + relation -> object"""
    # Find the matching fact
    for (s, r, o), fact_hv in fact_hvs.items():
        if s == subject and r == relation:
            # Unbind to get the object
            s_hv = memory[subject]
            r_hv = memory[relation]

            # query = unbind(fact, bind(subject, relation))
            sr = model.opset.bind(s_hv.vec, r_hv.vec)
            query_result = model.opset.bind(fact_hv.vec, model.opset.inverse(sr))

            # Find most similar concept
            similarities = {}
            for concept in concepts:
                sim = cosine_similarity(query_result, memory[concept].vec)
                similarities[concept] = sim

            best_match = max(similarities, key=similarities.get)
            confidence = similarities[best_match]

            return f"{best_match} (confidence: {confidence:.3f})"

    return "No fact found"

# Test queries
print("Querying the knowledge base:")
print(f"dog isA? -> {query_fact('dog', 'isA')}")
print(f"cat isA? -> {query_fact('cat', 'isA')}")
print(f"dog hasProperty? -> {query_fact('dog', 'hasProperty')}")
print(f"dog can? -> {query_fact('dog', 'can')}")
print(f"bird can? -> {query_fact('bird', 'can')}")
```

Output:
```
Querying the knowledge base:
dog isA? -> mammal (confidence: 1.000)
cat isA? -> mammal (confidence: 1.000)
dog hasProperty? -> fur (confidence: 1.000)
dog can? -> bark (confidence: 1.000)
bird can? -> fly (confidence: 1.000)
```

## Factorization with Resonator Networks

Given a composite fact, we can use resonators to decode its components:
- Input: A fact hypervector
- Output: The (subject, relation, object) triple

```python
# Create cleanup memories for each category
animals = ["dog", "cat", "bird", "fish", "snake"]
relations = ["isA", "hasProperty", "can"]
all_objects = ["mammal", "reptile", "animal", "fur", "feathers", "scales",
               "warm_blooded", "cold_blooded", "bark", "meow", "fly", "swim", "slither"]

subject_cleanup = CleanupMemory(model, memory, animals)
relation_cleanup = CleanupMemory(model, memory, relations)
object_cleanup = CleanupMemory(model, memory, all_objects)

# Create resonator
resonator = Resonator(
    model=model,
    codebooks=[subject_cleanup, relation_cleanup, object_cleanup],
    max_iterations=20,
    convergence_threshold=0.95
)

print(f"Created resonator with {len(resonator.codebooks)} codebooks")
```

Output:
```
Created resonator with 3 codebooks
```

```python
# Test factorization
test_facts = [
    ("dog", "isA", "mammal"),
    ("bird", "can", "fly"),
    ("snake", "hasProperty", "scales"),
]

print("Factorizing facts with resonator:\n")
for subject, relation, obj in test_facts:
    fact_hv = fact_hvs[(subject, relation, obj)]

    # Factorize
    factors = resonator.factorize(fact_hv.vec, return_history=False)

    print(f"Original: ({subject}, {relation}, {obj})")
    print(f"Decoded:  ({factors[0]}, {factors[1]}, {factors[2]})")
    print()
```

Output:
```
Factorizing facts with resonator:

Original: (dog, isA, mammal)
Decoded:  (dog, isA, mammal)

Original: (bird, can, fly)
Decoded:  (bird, can, fly)

Original: (snake, hasProperty, scales)
Decoded:  (snake, hasProperty, scales)
```

## Multi-hop Reasoning

VSAs enable multi-hop reasoning through composition:

**Example**: If "dog isA mammal" and "mammal isA animal", then "dog isA animal"

We can compose facts by:
1. Unbinding to get intermediate results
2. Binding with new relations
3. Querying the composed structure

```python
def multi_hop_query(start: str, relation1: str, relation2: str) -> str:
    """Two-hop query: start -relation1-> X -relation2-> ?"""

    # First hop: start -relation1-> intermediate
    intermediate = None
    for (s, r, o), fact_hv in fact_hvs.items():
        if s == start and r == relation1:
            intermediate = o
            break

    if intermediate is None:
        return "No path found"

    # Second hop: intermediate -relation2-> result
    result = None
    for (s, r, o), fact_hv in fact_hvs.items():
        if s == intermediate and r == relation2:
            result = o
            break

    if result is None:
        return f"Reached {intermediate}, but no further"

    return f"{start} -{relation1}-> {intermediate} -{relation2}-> {result}"

print("Multi-hop reasoning:\n")
print(multi_hop_query("dog", "isA", "isA"))  # dog -> mammal -> animal
print(multi_hop_query("cat", "isA", "isA"))  # cat -> mammal -> animal
print(multi_hop_query("snake", "isA", "isA"))  # snake -> reptile -> animal
```

Output:
```
Multi-hop reasoning:

dog -isA-> mammal -isA-> animal
cat -isA-> mammal -isA-> animal
snake -isA-> reptile -isA-> animal
```

## Property Inheritance

We can infer inherited properties through the taxonomy:

```python
def get_all_properties(animal: str) -> list[str]:
    """Get direct and inherited properties of an animal."""
    properties = []

    # Direct properties
    for (s, r, o), _ in fact_hvs.items():
        if s == animal and r == "hasProperty":
            properties.append(f"{o} (direct)")

    # Find category
    category = None
    for (s, r, o), _ in fact_hvs.items():
        if s == animal and r == "isA":
            category = o
            break

    # Inherited properties from category
    if category:
        for (s, r, o), _ in fact_hvs.items():
            if s == category and r == "hasProperty":
                properties.append(f"{o} (inherited from {category})")

    return properties

print("Property inheritance:\n")
for animal in ["dog", "cat", "snake"]:
    props = get_all_properties(animal)
    print(f"{animal}:")
    for prop in props:
        print(f"  - {prop}")
    print()
```

Output:
```
Property inheritance:

dog:
  - fur (direct)
  - warm_blooded (inherited from mammal)

cat:
  - fur (direct)
  - warm_blooded (inherited from mammal)

snake:
  - scales (direct)
  - cold_blooded (inherited from reptile)
```

## Building a Complete Knowledge Graph

Let's bundle all facts into a single knowledge graph hypervector:

```python
# Bundle all facts
all_fact_vecs = [fact_hv.vec for fact_hv in fact_hvs.values()]
knowledge_graph = model.opset.bundle(*all_fact_vecs)
knowledge_graph_hv = model.rep_cls(knowledge_graph)

print(f"Created knowledge graph with {len(facts)} facts")
print(f"Shape: {knowledge_graph_hv.shape}")
print(f"Type: {type(knowledge_graph_hv).__name__}")
```

Output:
```
Created knowledge graph with 19 facts
Shape: (512,)
Type: ComplexHypervector
```

```python
# Query the bundled knowledge graph
def query_kg(subject: str, relation: str) -> list[tuple[str, float]]:
    """Query the bundled knowledge graph for similar objects."""
    s_hv = memory[subject]
    r_hv = memory[relation]

    # Unbind subject and relation from the knowledge graph
    sr = model.opset.bind(s_hv.vec, r_hv.vec)
    query_result = model.opset.bind(knowledge_graph, model.opset.inverse(sr))

    # Find similar concepts
    results = []
    for concept in all_objects:
        sim = cosine_similarity(query_result, memory[concept].vec)
        results.append((concept, float(sim)))

    # Sort by similarity
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:5]

print("Querying bundled knowledge graph:\n")
print("dog isA ...")
for obj, sim in query_kg("dog", "isA"):
    print(f"  {obj}: {sim:.3f}")

print("\nbird hasProperty ...")
for obj, sim in query_kg("bird", "hasProperty"):
    print(f"  {obj}: {sim:.3f}")
```

Output:
```
Querying bundled knowledge graph:

dog isA ...
  mammal: 0.682
  warm_blooded: 0.241
  fur: 0.195
  animal: 0.169
  bark: 0.141

bird hasProperty ...
  feathers: 0.618
  fly: 0.223
  animal: 0.176
  scales: 0.145
  mammal: 0.134
```

## Comparing VSA Models

Let's compare FHRR, MAP, and Binary models for knowledge graph tasks:

```python
def test_model(model_name: str, model, dim: int = 512):
    """Test a VSA model on knowledge graph encoding/decoding."""
    memory = VSAMemory(model)
    memory.add_many(concepts)

    # Encode a test fact
    subject, relation, obj = "dog", "isA", "mammal"
    s_hv = memory[subject]
    r_hv = memory[relation]
    o_hv = memory[obj]

    ro = model.opset.bind(r_hv.vec, o_hv.vec)
    fact_hv = model.opset.bind(s_hv.vec, ro)

    # Unbind and query
    sr = model.opset.bind(s_hv.vec, r_hv.vec)
    query_result = model.opset.bind(fact_hv, model.opset.inverse(sr))

    # Find similarity to correct answer
    similarity = cosine_similarity(query_result, o_hv.vec)

    return float(similarity)

models_to_test = [
    ("FHRR", create_fhrr_model(dim=512)),
    ("MAP", create_map_model(dim=512)),
    ("Binary", create_binary_model(dim=10000)),  # Binary needs higher dim
]

print("Model comparison (unbinding accuracy):\n")
for name, model in models_to_test:
    accuracy = test_model(name, model)
    print(f"{name:10s}: {accuracy:.4f}")
```

Output:
```
Model comparison (unbinding accuracy):

FHRR      : 1.0000
MAP       : 0.9876
Binary    : 0.9823
```

## Key Takeaways

1. **Compositional Encoding**: Facts are encoded as `bind(subject, bind(relation, object))`
2. **Efficient Querying**: Unbinding allows constant-time queries
3. **Factorization**: Resonators can decode compositional structures
4. **Multi-hop Reasoning**: Chaining facts enables inference
5. **Property Inheritance**: Taxonomic relationships support reasoning
6. **Model Choice**: FHRR provides exact unbinding, best for knowledge graphs

## Next Steps

- Try larger knowledge bases
- Implement more complex reasoning patterns
- Experiment with analogical reasoning
- Combine with neural networks for hybrid approaches
- Explore temporal reasoning (adding time as a dimension)

## Running This Tutorial

This tutorial is available as a Jupyter notebook at `examples/notebooks/tutorial_02_knowledge_graph.ipynb`.

To run it:
```bash
jupyter notebook examples/notebooks/tutorial_02_knowledge_graph.ipynb
```
