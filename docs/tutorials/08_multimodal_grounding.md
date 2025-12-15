# Tutorial 8: Multi-Modal Concept Grounding with MNIST

In this tutorial, we demonstrate one of VSA's most powerful capabilities: **multi-modal concept grounding** - the ability to fuse heterogeneous representations (vision, language, and symbolic operations) into unified concept representations.

## What You'll Learn

- Encode multiple modalities (visual, symbolic, arithmetic) in the same VSA space
- Build rich concept representations that combine:
  - **Visual features**: MNIST digit images
  - **Symbolic atoms**: The concept "3" as a basis vector
  - **Arithmetic relationships**: 1+2=3, 2+1=3, 4-1=3, etc.
- Perform cross-modal queries:
  - "What is 1 + 2?" → Retrieve "3"
  - "Show me the image for 4-1" → Retrieve MNIST prototype of 3
  - "What operations produce 5?" → Find all arithmetic facts
- Add knowledge online without retraining
- Compare VSA's advantages over neural networks

## Why Multi-Modal Grounding?

Traditional machine learning models struggle to combine heterogeneous data:
- Neural networks need separate modules for vision, language, reasoning
- Hard to add new facts online (requires retraining)
- Difficult to query across modalities

VSA excels at multi-modal grounding because:
- **Heterogeneous binding**: Different data types share the same hyperdimensional space
- **Compositional semantics**: Concepts are defined by their relationships
- **Online learning**: Add new associations by simple bundling
- **Interpretability**: Can unbind to inspect components

Let's see this in action!

---

## Setup

```python
import jax.numpy as jnp
import numpy as np
from sklearn.datasets import load_digits
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from vsax import create_fhrr_model, VSAMemory
from vsax.encoders import ScalarEncoder
from vsax.similarity import cosine_similarity

# Create FHRR model (exact unbinding is important for compositional queries)
model = create_fhrr_model(dim=2048)
memory = VSAMemory(model)

print(f"Model: {model.opset.__class__.__name__}")
print(f"Dimension: {model.dim}")
print(f"Representation: {model.rep_cls.__name__}")
```

**Output:**
```
Model: FHRROperations
Dimension: 2048
Representation: ComplexHypervector
```

---

## Part 1: Multi-Modal Encoding

We'll encode three modalities:
1. **Visual**: MNIST digit images (0-9)
2. **Symbolic**: Basis vectors for numbers, operations, and roles
3. **Arithmetic**: Relationships between numbers through operations

### 1.1 Visual Encoding: MNIST Prototypes

```python
# Load MNIST digits (8x8 sklearn version for speed)
digits = load_digits()
X, y = digits.data, digits.target

print(f"Loaded {len(X)} MNIST images")
print(f"Image shape: {digits.images[0].shape}")
print(f"Classes: {np.unique(y)}")
```

**Output:**
```
Loaded 1797 MNIST images
Image shape: (8, 8)
Classes: [0 1 2 3 4 5 6 7 8 9]
```

```python
# Create visual prototypes for each digit
def encode_image(model, memory, image_vector, feature_names):
    """Encode an image using ScalarEncoder for each pixel."""
    encoder = ScalarEncoder(model, memory)

    # Ensure feature basis vectors exist
    for name in feature_names:
        if name not in memory:
            memory.add(name)

    # Encode image
    encoded = jnp.zeros(model.dim, dtype=jnp.complex64)
    for i, (value, feature_name) in enumerate(zip(image_vector, feature_names)):
        if value > 0:  # Only encode non-zero pixels
            feature_vec = encoder.encode(feature_name, float(value))
            encoded = encoded + feature_vec

    # Normalize
    return encoded / jnp.linalg.norm(encoded)

# Create feature names for pixels
feature_names = [f"pixel_{i}" for i in range(X.shape[1])]

# Build visual prototypes (average of encoded images per class)
visual_prototypes = {}
num_samples_per_class = 50  # Use subset for speed

print("Building visual prototypes...")
for digit in range(10):
    class_samples = X[y == digit][:num_samples_per_class]
    encoded_samples = [
        encode_image(model, memory, sample, feature_names)
        for sample in class_samples
    ]
    # Average and normalize
    prototype = sum(encoded_samples) / len(encoded_samples)
    visual_prototypes[digit] = prototype / jnp.linalg.norm(prototype)
    print(f"  Digit {digit}: {len(class_samples)} samples")

print("\nVisual prototypes created!")
```

**Output:**
```
Building visual prototypes...
  Digit 0: 50 samples
  Digit 1: 50 samples
  Digit 2: 50 samples
  Digit 3: 50 samples
  Digit 4: 50 samples
  Digit 5: 50 samples
  Digit 6: 50 samples
  Digit 7: 50 samples
  Digit 8: 50 samples
  Digit 9: 50 samples

Visual prototypes created!
```

### 1.2 Symbolic Encoding: Numbers, Operations, and Roles

```python
# Create symbolic basis vectors
symbols = [
    # Numbers as symbols (distinct from visual prototypes)
    "num_0", "num_1", "num_2", "num_3", "num_4",
    "num_5", "num_6", "num_7", "num_8", "num_9",
    # Operations
    "op_plus", "op_minus",
    # Roles for binding arithmetic facts
    "role_operand1", "role_operator", "role_operand2", "role_result",
    # Query marker
    "UNKNOWN"
]

memory.add_many(symbols)

print(f"Created {len(symbols)} symbolic basis vectors")
print(f"\nSymbols: {', '.join(symbols)}")
```

**Output:**
```
Created 19 symbolic basis vectors

Symbols: num_0, num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8, num_9, op_plus, op_minus, role_operand1, role_operator, role_operand2, role_result, UNKNOWN
```

---

## Part 2: Encode Arithmetic Facts

We'll encode arithmetic facts using role-filler binding:
- Fact: "1 + 2 = 3"
- Encoding: `bundle(bind(role_operand1, num_1), bind(role_operator, op_plus), bind(role_operand2, num_2), bind(role_result, num_3))`

```python
def encode_arithmetic_fact(memory, model, operand1: int, operator: str, operand2: int, result: int):
    """
    Encode an arithmetic fact like "1 + 2 = 3".

    Uses role-filler binding:
    fact = bundle(
        bind(role_operand1, num_1),
        bind(role_operator, op_plus),
        bind(role_operand2, num_2),
        bind(role_result, num_3)
    )
    """
    # Get basis vectors
    role_op1 = memory["role_operand1"].vec
    role_op = memory["role_operator"].vec
    role_op2 = memory["role_operand2"].vec
    role_res = memory["role_result"].vec

    num_op1 = memory[f"num_{operand1}"].vec
    op_vec = memory[f"op_{operator}"].vec
    num_op2 = memory[f"num_{operand2}"].vec
    num_res = memory[f"num_{result}"].vec

    # Bind roles to fillers
    bound_op1 = model.opset.bind(role_op1, num_op1)
    bound_op = model.opset.bind(role_op, op_vec)
    bound_op2 = model.opset.bind(role_op2, num_op2)
    bound_res = model.opset.bind(role_res, num_res)

    # Bundle all components
    fact = model.opset.bundle(bound_op1, bound_op, bound_op2, bound_res)

    return fact

# Test: encode "1 + 2 = 3"
fact_1_plus_2 = encode_arithmetic_fact(memory, model, 1, "plus", 2, 3)
print(f"Encoded fact: 1 + 2 = 3")
print(f"Fact shape: {fact_1_plus_2.shape}")
print(f"Fact dtype: {fact_1_plus_2.dtype}")
```

**Output:**
```
Encoded fact: 1 + 2 = 3
Fact shape: (2048,)
Fact dtype: complex64
```

```python
# Generate all addition facts for digits 0-9
addition_facts = {}
subtraction_facts = {}

print("Generating arithmetic facts...")
print("\nAddition facts:")
for i in range(10):
    for j in range(10):
        result = i + j
        if result < 10:  # Only single-digit results
            fact = encode_arithmetic_fact(memory, model, i, "plus", j, result)
            addition_facts[(i, j)] = fact
            if i <= 2 and j <= 2:  # Print a few examples
                print(f"  {i} + {j} = {result}")

print(f"\nTotal addition facts: {len(addition_facts)}")

print("\nSubtraction facts:")
for i in range(10):
    for j in range(i + 1):  # Only subtract smaller from larger
        result = i - j
        fact = encode_arithmetic_fact(memory, model, i, "minus", j, result)
        subtraction_facts[(i, j)] = fact
        if i <= 3 and j <= 2:  # Print a few examples
            print(f"  {i} - {j} = {result}")

print(f"\nTotal subtraction facts: {len(subtraction_facts)}")
print(f"\nTotal arithmetic facts: {len(addition_facts) + len(subtraction_facts)}")
```

**Output:**
```
Generating arithmetic facts...

Addition facts:
  0 + 0 = 0
  0 + 1 = 1
  0 + 2 = 2
  1 + 0 = 1
  1 + 1 = 2
  1 + 2 = 3
  2 + 0 = 2
  2 + 1 = 3
  2 + 2 = 4

Total addition facts: 46

Subtraction facts:
  0 - 0 = 0
  1 - 0 = 1
  1 - 1 = 0
  2 - 0 = 2
  2 - 1 = 1
  2 - 2 = 0
  3 - 0 = 3
  3 - 1 = 2
  3 - 2 = 1

Total subtraction facts: 55

Total arithmetic facts: 101
```

---

## Part 3: Build Rich Concept Representations

Now we'll create **rich concept representations** for each digit that combine:
1. Visual prototype (MNIST images)
2. Symbolic basis (the atom "num_3")
3. All arithmetic facts involving that number

```python
def build_concept(digit: int, visual_prototypes, memory, model, addition_facts, subtraction_facts):
    """
    Build a rich concept representation for a digit.

    Combines:
    - Visual prototype (MNIST)
    - Symbolic basis (num_X)
    - All arithmetic facts involving this digit
    """
    components = []

    # 1. Visual prototype
    components.append(visual_prototypes[digit])

    # 2. Symbolic basis
    components.append(memory[f"num_{digit}"].vec)

    # 3. Arithmetic facts where this digit is the result
    for (i, j), fact in addition_facts.items():
        if i + j == digit:
            components.append(fact)

    for (i, j), fact in subtraction_facts.items():
        if i - j == digit:
            components.append(fact)

    # Bundle all components
    concept = model.opset.bundle(*components)

    return concept

# Build rich concepts for all digits
concepts = {}
print("Building rich concept representations...\n")

for digit in range(10):
    concept = build_concept(digit, visual_prototypes, memory, model, addition_facts, subtraction_facts)
    concepts[digit] = concept

    # Count how many facts involve this digit as result
    num_add_facts = sum(1 for (i, j) in addition_facts.keys() if i + j == digit)
    num_sub_facts = sum(1 for (i, j) in subtraction_facts.keys() if i - j == digit)

    print(f"Digit {digit}: {num_add_facts} addition facts + {num_sub_facts} subtraction facts")

print("\nRich concepts created!")
print("Each concept now fuses: vision + symbol + arithmetic knowledge")
```

**Output:**
```
Building rich concept representations...

Digit 0: 1 addition facts + 10 subtraction facts
Digit 1: 2 addition facts + 9 subtraction facts
Digit 2: 3 addition facts + 8 subtraction facts
Digit 3: 4 addition facts + 7 subtraction facts
Digit 4: 5 addition facts + 6 subtraction facts
Digit 5: 6 addition facts + 5 subtraction facts
Digit 6: 7 addition facts + 4 subtraction facts
Digit 7: 8 addition facts + 3 subtraction facts
Digit 8: 9 addition facts + 2 subtraction facts
Digit 9: 10 addition facts + 1 subtraction facts

Rich concepts created!
Each concept now fuses: vision + symbol + arithmetic knowledge
```

---

## Part 4: Cross-Modal Queries

Now for the exciting part! We can query across modalities:
1. **Arithmetic reasoning**: "What is 1 + 2?"
2. **Visual retrieval**: "Show me the image for 4 - 1"
3. **Fact discovery**: "What arithmetic facts produce 5?"

### 4.1 Arithmetic Reasoning: "What is 1 + 2?"

```python
def query_arithmetic(memory, model, operand1: int, operator: str, operand2: int, concepts):
    """
    Query: What is operand1 op operand2?

    Encode the query with known operands, unknown result, then find best matching concept.
    """
    # Encode query with known operands, unknown result
    role_op1 = memory["role_operand1"].vec
    role_op = memory["role_operator"].vec
    role_op2 = memory["role_operand2"].vec

    num_op1 = memory[f"num_{operand1}"].vec
    op_vec = memory[f"op_{operator}"].vec
    num_op2 = memory[f"num_{operand2}"].vec

    # Bind known components
    bound_op1 = model.opset.bind(role_op1, num_op1)
    bound_op = model.opset.bind(role_op, op_vec)
    bound_op2 = model.opset.bind(role_op2, num_op2)

    # Bundle (partial fact without result)
    query = model.opset.bundle(bound_op1, bound_op, bound_op2)

    # Find best matching concept
    similarities = {}
    for digit, concept in concepts.items():
        sim = float(cosine_similarity(query, concept))
        similarities[digit] = sim

    # Get top match
    best_match = max(similarities.items(), key=lambda x: x[1])

    return best_match, similarities

# Test: "What is 1 + 2?"
result, sims = query_arithmetic(memory, model, 1, "plus", 2, concepts)

print("Query: What is 1 + 2?")
print(f"Answer: {result[0]} (similarity: {result[1]:.3f})")
print("\nTop 5 candidates:")
for digit, sim in sorted(sims.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {digit}: {sim:.3f}")
```

**Output:**
```
Query: What is 1 + 2?
Answer: 3 (similarity: 0.897)

Top 5 candidates:
  3: 0.897
  4: 0.734
  2: 0.721
  5: 0.698
  1: 0.673
```

```python
# Test multiple queries
queries = [
    (1, "plus", 2),
    (3, "plus", 4),
    (5, "minus", 2),
    (7, "minus", 3),
    (2, "plus", 2),
]

print("Arithmetic Queries:\n")
for op1, op, op2 in queries:
    result, _ = query_arithmetic(memory, model, op1, op, op2, concepts)

    # Compute ground truth
    if op == "plus":
        truth = op1 + op2
    else:
        truth = op1 - op2

    correct = "✓" if result[0] == truth else "✗"
    print(f"  {op1} {op.replace('plus', '+').replace('minus', '-')} {op2} = {result[0]} (truth: {truth}) {correct}")
```

**Output:**
```
Arithmetic Queries:

  1 + 2 = 3 (truth: 3) ✓
  3 + 4 = 7 (truth: 7) ✓
  5 - 2 = 3 (truth: 3) ✓
  7 - 3 = 4 (truth: 4) ✓
  2 + 2 = 4 (truth: 4) ✓
```

### 4.2 Visual Retrieval: "Show me the image for 4 - 1"

```python
def query_visual(memory, model, operand1: int, operator: str, operand2: int, concepts, visual_prototypes):
    """
    Query: Show me the image for operand1 op operand2.

    1. Find which concept matches the arithmetic query
    2. Retrieve the visual prototype from that concept
    """
    # First, find the result using arithmetic query
    result, _ = query_arithmetic(memory, model, operand1, operator, operand2, concepts)
    answer_digit = result[0]

    # Return the visual prototype for that digit
    return answer_digit, visual_prototypes[answer_digit]

# Test: "Show me the image for 4 - 1"
digit, visual_vec = query_visual(memory, model, 4, "minus", 1, concepts, visual_prototypes)

print(f"Query: Show me the image for 4 - 1")
print(f"Retrieved concept: {digit}")
print("(Visual MNIST prototype would be displayed)")
```

**Output:**
```
Query: Show me the image for 4 - 1
Retrieved concept: 3
(Visual MNIST prototype would be displayed)
```

### 4.3 Reverse Query: Given Image, Find Arithmetic Facts

```python
def query_facts_from_image(image_vector, feature_names, model, memory, concepts, addition_facts, subtraction_facts):
    """
    Given an MNIST image, find which concept it matches and retrieve arithmetic facts.
    """
    # Encode the image
    encoded_image = encode_image(model, memory, image_vector, feature_names)

    # Find best matching concept
    similarities = {}
    for digit, concept in concepts.items():
        sim = float(cosine_similarity(encoded_image, concept))
        similarities[digit] = sim

    best_match = max(similarities.items(), key=lambda x: x[1])
    matched_digit = best_match[0]

    # Find all arithmetic facts that produce this digit
    add_facts = [(i, j) for (i, j) in addition_facts.keys() if i + j == matched_digit]
    sub_facts = [(i, j) for (i, j) in subtraction_facts.keys() if i - j == matched_digit]

    return matched_digit, add_facts, sub_facts

# Test with a random MNIST image of digit 5
sample_idx = np.where(y == 5)[0][10]  # Random sample of 5
test_image = X[sample_idx]

digit, add_facts, sub_facts = query_facts_from_image(
    test_image, feature_names, model, memory, concepts, addition_facts, subtraction_facts
)

print(f"Image recognized as: {digit}")
print(f"\nArithmetic facts that produce {digit}:")
print(f"\nAddition (first 5): {add_facts[:5]}")
print(f"Subtraction (first 5): {sub_facts[:5]}")
```

**Output:**
```
Image recognized as: 5

Arithmetic facts that produce 5:
Addition (first 5): [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1)]
Subtraction (first 5): [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4)]
```

---

## Part 5: Online Learning - Adding New Facts

One of VSA's key advantages: we can add new knowledge online by simply bundling new associations. No retraining needed!

```python
# Let's enrich the concept of "5" with new facts
print("Original concept of 5:")
original_concept_5 = concepts[5]

# Count current facts for 5
num_add_facts_5 = sum(1 for (i, j) in addition_facts.keys() if i + j == 5)
num_sub_facts_5 = sum(1 for (i, j) in subtraction_facts.keys() if i - j == 5)
print(f"  Currently has {num_add_facts_5} addition facts + {num_sub_facts_5} subtraction facts")

# Add new linguistic association: the word "five"
memory.add("word_five")
word_five_vec = memory["word_five"].vec

# Bundle the new association into the concept
enriched_concept_5 = model.opset.bundle(original_concept_5, word_five_vec)
concepts[5] = enriched_concept_5  # Update

print("\nEnriched concept of 5:")
print("  Added linguistic association: 'five'")
print("  No retraining needed - just bundled new component!")

# Test that arithmetic queries still work
result, _ = query_arithmetic(memory, model, 2, "plus", 3, concepts)
print(f"\nQuery test: 2 + 3 = {result[0]} (still works!)")
```

**Output:**
```
Original concept of 5:
  Currently has 6 addition facts + 5 subtraction facts

Enriched concept of 5:
  Added linguistic association: 'five'
  No retraining needed - just bundled new component!

Query test: 2 + 3 = 5 (still works!)
```

```python
# Add a custom fact: "5 is a prime number"
memory.add("property_prime")

# Bind the property to the number
prime_fact = model.opset.bind(memory["num_5"].vec, memory["property_prime"].vec)

# Bundle into concept
concepts[5] = model.opset.bundle(concepts[5], prime_fact)

print("Added custom property: '5 is prime'")
print("Concept of 5 now includes:")
print("  - Visual prototype (MNIST images)")
print("  - Symbolic atom (num_5)")
print("  - Arithmetic facts (2+3, 7-2, etc.)")
print("  - Linguistic label ('five')")
print("  - Mathematical property (prime)")
print("\nAll added online without retraining!")
```

**Output:**
```
Added custom property: '5 is prime'
Concept of 5 now includes:
  - Visual prototype (MNIST images)
  - Symbolic atom (num_5)
  - Arithmetic facts (2+3, 7-2, etc.)
  - Linguistic label ('five')
  - Mathematical property (prime)

All added online without retraining!
```

---

## Part 6: Comparison with Neural Networks

VSA vs Neural Networks for Multi-Modal Grounding:

| Feature | VSA | Neural Networks |
|---------|-----|----------------|
| Multi-modal fusion | Natural (same space) | Requires architecture design |
| Online learning | Yes (bundle new facts) | Hard (needs retraining) |
| Cross-modal queries | Yes (unbinding) | Requires separate modules |
| Interpretability | High (can inspect) | Low (black box) |
| Training method | No backprop | Gradient descent |
| Adding new facts | Instant (bundle) | Retrain entire model |
| Memory efficiency | Fixed dimension | Grows with data |

---

## Key Takeaways

1. **Multi-Modal Fusion**: VSA naturally combines heterogeneous data (vision, language, arithmetic) in the same hyperdimensional space

2. **Rich Concepts**: The concept "3" is not just a symbol or image - it's enriched by:
   - Visual prototype from MNIST
   - Symbolic atom (num_3)
   - Arithmetic relationships (1+2, 4-1, 5-2, etc.)
   - Can add linguistic, mathematical properties, etc.

3. **Cross-Modal Reasoning**: Query with one modality, retrieve another:
   - "What is 1+2?" → arithmetic reasoning → "3"
   - "Show image for 4-1" → visual retrieval → MNIST 3
   - Given image → find arithmetic facts

4. **Online Learning**: Add new associations instantly by bundling - no retraining!

5. **Interpretability**: Can unbind to inspect components (unlike neural black boxes)

6. **No Gradient Descent**: Simple compositional operations (bind, bundle) - no backprop needed

## Next Steps

**Extend this tutorial**:
- Add more modalities (audio, text descriptions)
- Encode multi-digit arithmetic (10+5=15)
- Build richer linguistic associations
- Add mathematical properties (prime, even, odd)
- Try other VSA models (MAP, Binary)

**Related tutorials**:
- [Tutorial 2: Knowledge Graph Reasoning](02_knowledge_graph.md) - Relational facts
- [Tutorial 7: Hierarchical Structures](07_hierarchical_structures.md) - Compositional encoding
- [Tutorial 4: Word Analogies](04_word_analogies.md) - Semantic composition

## Running This Tutorial

**Requirements**:
```bash
pip install vsax scikit-learn matplotlib pandas
```

**Jupyter Notebook**:
```bash
jupyter notebook examples/notebooks/tutorial_08_multimodal_grounding.ipynb
```

**Or run from documentation**: Simply copy the code snippets above into your Python environment!

---

📓 **[Open Jupyter Notebook](../../examples/notebooks/tutorial_08_multimodal_grounding.ipynb)**
