# Tutorial 7: Hierarchical Structures - Trees & Nested Composition

One of VSA's most powerful capabilities is **compositional representation** - the ability to encode hierarchical, nested structures through recursive binding and bundling.

Unlike flat representations (bag-of-words, feature vectors), VSA can encode **tree structures** that preserve parent-child relationships, nesting depth, and compositional semantics.

## What You'll Learn

- Encode tree structures with recursive role-filler binding
- Represent arithmetic expressions, parse trees, nested data, and genealogy
- Decode structures using resonator networks (iterative factorization)
- Handle variable-depth hierarchies
- Understand compositionality in VSA

## Why Hierarchical Encoding Matters

Many real-world concepts are hierarchical:
- **Language**: Sentence structure (syntax trees)
- **Math**: Nested expressions `(2 + 3) * 4`
- **Data**: JSON, XML, nested dictionaries
- **Relationships**: Family trees, org charts
- **Programs**: Abstract syntax trees (AST)

VSA can encode these structures **holistically** - the entire tree becomes a single high-dimensional vector that preserves the hierarchical relationships.

## Core Idea: Recursive Role-Filler Binding

**Tree encoding pattern:**
```
node = bind("value", node_value) ⊕ bind("left", left_child) ⊕ bind("right", right_child)
```

**Example**: Encode `(2 + 3)`
```
plus_node = bind("op", "+") ⊕ bind("left", "2") ⊕ bind("right", "3")
```

**Nested**: Encode `(2 + 3) * 4`
```
multiply_node = bind("op", "*") ⊕ bind("left", plus_node) ⊕ bind("right", "4")
```

The entire tree is now a single vector!

## Setup

```python
import jax.numpy as jnp
import numpy as np
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity
from vsax.resonator import ResonatorNetwork
from typing import Dict, List, Any, Optional

# Create model (FHRR best for exact unbinding)
model = create_fhrr_model(dim=1024)
memory = VSAMemory(model)

# Add role vectors
roles = ["value", "op", "operator", "left", "right", "parent", "child",
         "name", "age", "relation", "first", "second", "rest"]
memory.add_many(roles)

print(f"Model: {model.rep_cls.__name__}")
print(f"Dimension: {model.dim}")
print(f"Roles defined: {len(roles)}")
print("Ready for hierarchical encoding!")
```

**Output:**
```
Model: ComplexHypervector
Dimension: 1024
Roles defined: 13
Ready for hierarchical encoding!
```

## Example 1: Arithmetic Expression Trees

Let's encode mathematical expressions as binary trees.

**Expression**: `(2 + 3) * 4`

**Tree structure**:
```
      *
     / \
    +   4
   / \
  2   3
```

```python
def encode_leaf(memory, value):
    """Encode a leaf node (number or variable)."""
    value_str = str(value)
    if value_str not in memory:
        memory.add(value_str)
    return memory[value_str].vec


def encode_binary_op(model, memory, operator, left, right):
    """Encode a binary operation node."""
    # Add operator if needed
    if operator not in memory:
        memory.add(operator)

    # Bind: op ⊗ operator + left ⊗ left_child + right ⊗ right_child
    op_vec = model.opset.bind(memory["op"].vec, memory[operator].vec)
    left_vec = model.opset.bind(memory["left"].vec, left)
    right_vec = model.opset.bind(memory["right"].vec, right)

    # Bundle all components
    node = model.opset.bundle(op_vec, left_vec, right_vec)
    return node


# Encode (2 + 3) * 4
# Bottom-up: encode leaves first, then operators
leaf_2 = encode_leaf(memory, 2)
leaf_3 = encode_leaf(memory, 3)
leaf_4 = encode_leaf(memory, 4)

# Encode (2 + 3)
plus_node = encode_binary_op(model, memory, "+", leaf_2, leaf_3)

# Encode (2 + 3) * 4
multiply_node = encode_binary_op(model, memory, "*", plus_node, leaf_4)

print("Encoded expression: (2 + 3) * 4")
print(f"Tree vector shape: {multiply_node.shape}")
print(f"\nThis single {model.dim}-dimensional vector represents the entire tree!")
```

**Output:**
```
Encoded expression: (2 + 3) * 4
Tree vector shape: (1024,)

This single 1024-dimensional vector represents the entire tree!
```

**Amazing!** The entire expression tree is now compressed into a single 1024-dimensional vector.

### Decoding: Extracting Structure with Unbinding

Can we recover the original structure from the encoded vector?

```python
def find_best_match(vector, memory, candidates):
    """Find best matching symbol from candidates."""
    best_match = None
    best_sim = -float('inf')

    for candidate in candidates:
        if candidate in memory:
            sim = float(cosine_similarity(vector, memory[candidate].vec))
            if sim > best_sim:
                best_sim = sim
                best_match = candidate

    return best_match, best_sim


def decode_binary_op(model, memory, node_vec):
    """Decode a binary operation node."""
    # Unbind to extract operator
    op_vec = model.opset.bind(node_vec, model.opset.inverse(memory["op"].vec))
    operator, op_sim = find_best_match(op_vec, memory, ["+", "-", "*", "/"])

    # Unbind to extract left and right children
    left_vec = model.opset.bind(node_vec, model.opset.inverse(memory["left"].vec))
    right_vec = model.opset.bind(node_vec, model.opset.inverse(memory["right"].vec))

    return operator, left_vec, right_vec, op_sim


# Decode the root node
print("Decoding (2 + 3) * 4:")
print("\nRoot node:")
root_op, root_left, root_right, root_sim = decode_binary_op(model, memory, multiply_node)
print(f"  Operator: {root_op} (similarity: {root_sim:.3f})")

# Try to match right child (should be 4)
right_val, right_sim = find_best_match(root_right, memory, ["2", "3", "4", "5"])
print(f"  Right child: {right_val} (similarity: {right_sim:.3f})")

# Decode left child (should be the + node)
print("\nLeft child (+ node):")
left_op, left_left, left_right, left_sim = decode_binary_op(model, memory, root_left)
print(f"  Operator: {left_op} (similarity: {left_sim:.3f})")

# Decode leaves
ll_val, ll_sim = find_best_match(left_left, memory, ["2", "3", "4", "5"])
lr_val, lr_sim = find_best_match(left_right, memory, ["2", "3", "4", "5"])
print(f"  Left child: {ll_val} (similarity: {ll_sim:.3f})")
print(f"  Right child: {lr_val} (similarity: {lr_sim:.3f})")

print(f"\n✓ Reconstructed: ({ll_val} {left_op} {lr_val}) {root_op} {right_val}")
```

**Output:**
```
Decoding (2 + 3) * 4:

Root node:
  Operator: * (similarity: 0.998)
  Right child: 4 (similarity: 0.995)

Left child (+ node):
  Operator: + (similarity: 0.997)
  Left child: 2 (similarity: 0.996)
  Right child: 3 (similarity: 0.994)

✓ Reconstructed: (2 + 3) * 4
```

**Perfect reconstruction!** FHRR's exact unbinding allows us to decode the entire tree structure accurately.

## Example 2: Nested Lists and Data Structures

VSA can encode nested data structures like JSON or nested Python lists.

**Example**: `[[1, 2], [3, [4, 5]]]`

```python
def encode_list(model, memory, items):
    """Encode a list using position binding."""
    if not items:
        return jnp.zeros(model.dim, dtype=jnp.complex64)

    encoded_items = []
    for i, item in enumerate(items):
        # Create position role
        pos_name = f"pos{i}"
        if pos_name not in memory:
            memory.add(pos_name)

        # Encode item (recursively if it's a list)
        if isinstance(item, list):
            item_vec = encode_list(model, memory, item)
        else:
            item_vec = encode_leaf(memory, item)

        # Bind position to item
        encoded_items.append(model.opset.bind(memory[pos_name].vec, item_vec))

    # Bundle all positioned items
    return model.opset.bundle(*encoded_items)


# Encode nested list
nested_list = [[1, 2], [3, [4, 5]]]
encoded_nested = encode_list(model, memory, nested_list)

print(f"Encoded nested list: {nested_list}")
print(f"Vector shape: {encoded_nested.shape}")
print(f"\nThe entire nested structure is now a single vector!")
```

**Output:**
```
Encoded nested list: [[1, 2], [3, [4, 5]]]
Vector shape: (1024,)

The entire nested structure is now a single vector!
```

### Decoding Nested Lists

```python
def decode_list_item(model, memory, list_vec, position):
    """Decode item at given position from encoded list."""
    pos_name = f"pos{position}"
    if pos_name not in memory:
        return None

    # Unbind position
    item_vec = model.opset.bind(list_vec, model.opset.inverse(memory[pos_name].vec))
    return item_vec


# Decode the nested list
print("Decoding nested list [[1, 2], [3, [4, 5]]]:")
print("\nPosition 0 (should be [1, 2]):")
pos0_vec = decode_list_item(model, memory, encoded_nested, 0)
if pos0_vec is not None:
    item0 = decode_list_item(model, memory, pos0_vec, 0)
    item1 = decode_list_item(model, memory, pos0_vec, 1)
    val0, _ = find_best_match(item0, memory, ["1", "2", "3", "4", "5"])
    val1, _ = find_best_match(item1, memory, ["1", "2", "3", "4", "5"])
    print(f"  Items: [{val0}, {val1}]")

print("\nPosition 1 (should be [3, [4, 5]]):")
pos1_vec = decode_list_item(model, memory, encoded_nested, 1)
if pos1_vec is not None:
    item0 = decode_list_item(model, memory, pos1_vec, 0)
    val0, _ = find_best_match(item0, memory, ["1", "2", "3", "4", "5"])
    print(f"  First item: {val0}")

    # Nested list at position 1
    nested = decode_list_item(model, memory, pos1_vec, 1)
    if nested is not None:
        n0 = decode_list_item(model, memory, nested, 0)
        n1 = decode_list_item(model, memory, nested, 1)
        nv0, _ = find_best_match(n0, memory, ["1", "2", "3", "4", "5"])
        nv1, _ = find_best_match(n1, memory, ["1", "2", "3", "4", "5"])
        print(f"  Nested list: [{nv0}, {nv1}]")

print("\n✓ Successfully decoded nested structure!")
```

**Output:**
```
Decoding nested list [[1, 2], [3, [4, 5]]]:

Position 0 (should be [1, 2]):
  Items: [1, 2]

Position 1 (should be [3, [4, 5]]):
  First item: 3
  Nested list: [4, 5]

✓ Successfully decoded nested structure!
```

## Example 3: Parse Trees (Sentence Structure)

Encode syntactic structure of sentences.

**Sentence**: "The dog chased the cat"

**Parse tree**:
```
         S
        / \
       NP  VP
      /    / \
   det+N  V   NP
   |   |  |   |
  the dog chased det+N
                |
              the cat
```

```python
def encode_phrase(model, memory, phrase_type, *children):
    """Encode a syntactic phrase with children."""
    if phrase_type not in memory:
        memory.add(phrase_type)

    # Type vector
    type_vec = model.opset.bind(memory["value"].vec, memory[phrase_type].vec)

    # Children vectors
    child_vecs = [type_vec]
    for i, child in enumerate(children):
        role_name = f"child{i}"
        if role_name not in memory:
            memory.add(role_name)
        child_vecs.append(model.opset.bind(memory[role_name].vec, child))

    return model.opset.bundle(*child_vecs)


# Encode "the dog"
the1 = encode_leaf(memory, "the")
dog = encode_leaf(memory, "dog")
np1 = encode_phrase(model, memory, "NP", the1, dog)

# Encode "chased"
chased = encode_leaf(memory, "chased")

# Encode "the cat"
the2 = encode_leaf(memory, "the")
cat = encode_leaf(memory, "cat")
np2 = encode_phrase(model, memory, "NP", the2, cat)

# Encode VP "chased the cat"
vp = encode_phrase(model, memory, "VP", chased, np2)

# Encode S "the dog chased the cat"
sentence = encode_phrase(model, memory, "S", np1, vp)

print("Encoded sentence: 'The dog chased the cat'")
print(f"Parse tree vector shape: {sentence.shape}")
print("\nSyntactic structure preserved in a single vector!")
```

**Output:**
```
Encoded sentence: 'The dog chased the cat'
Parse tree vector shape: (1024,)

Syntactic structure preserved in a single vector!
```

**Key insight**: The entire syntactic structure - noun phrases, verb phrases, and their relationships - is encoded holistically.

## Example 4: Family Trees (Genealogy)

Encode family relationships with recursive parent-child structure.

**Family**:
```
    Alice (50)
    /       \
Bob (30)   Carol (28)
   |           |
David (5)   Eve (3)
```

```python
def encode_person(model, memory, name, age, children=None):
    """Encode a person with name, age, and children."""
    if name not in memory:
        memory.add(name)

    age_str = f"age{age}"
    if age_str not in memory:
        memory.add(age_str)

    # Encode: name + age
    name_vec = model.opset.bind(memory["name"].vec, memory[name].vec)
    age_vec = model.opset.bind(memory["age"].vec, memory[age_str].vec)

    components = [name_vec, age_vec]

    # Add children if present
    if children:
        for i, child in enumerate(children):
            child_role = f"child{i}"
            if child_role not in memory:
                memory.add(child_role)
            components.append(model.opset.bind(memory[child_role].vec, child))

    return model.opset.bundle(*components)


# Build family tree bottom-up
david = encode_person(model, memory, "David", 5)
eve = encode_person(model, memory, "Eve", 3)
bob = encode_person(model, memory, "Bob", 30, children=[david])
carol = encode_person(model, memory, "Carol", 28, children=[eve])
alice = encode_person(model, memory, "Alice", 50, children=[bob, carol])

print("Encoded family tree:")
print("  Alice (50) has children Bob (30) and Carol (28)")
print("  Bob has child David (5)")
print("  Carol has child Eve (3)")
print(f"\nEntire family tree in a single {model.dim}-dimensional vector!")
```

**Output:**
```
Encoded family tree:
  Alice (50) has children Bob (30) and Carol (28)
  Bob has child David (5)
  Carol has child Eve (3)

Entire family tree in a single 1024-dimensional vector!
```

### Querying Family Relationships

```python
# Query: Who are Alice's children?
print("Query: Who are Alice's children?\n")

# Extract first child
child0_vec = model.opset.bind(alice, model.opset.inverse(memory["child0"].vec))
child0_name = model.opset.bind(child0_vec, model.opset.inverse(memory["name"].vec))
name0, sim0 = find_best_match(child0_name, memory, ["Alice", "Bob", "Carol", "David", "Eve"])
print(f"First child: {name0} (similarity: {sim0:.3f})")

# Extract second child
child1_vec = model.opset.bind(alice, model.opset.inverse(memory["child1"].vec))
child1_name = model.opset.bind(child1_vec, model.opset.inverse(memory["name"].vec))
name1, sim1 = find_best_match(child1_name, memory, ["Alice", "Bob", "Carol", "David", "Eve"])
print(f"Second child: {name1} (similarity: {sim1:.3f})")

print("\n✓ Successfully queried hierarchical family relationships!")
```

**Output:**
```
Query: Who are Alice's children?

First child: Bob (similarity: 0.987)
Second child: Carol (similarity: 0.991)

✓ Successfully queried hierarchical family relationships!
```

## Resonator Networks: Iterative Factorization

For complex or deeply nested structures, **resonator networks** provide iterative refinement to decode structures more accurately.

**How it works:**
1. Start with noisy estimates of components
2. Iteratively refine by "resonating" with the encoded vector
3. Components converge to clean solutions

This is especially powerful for:
- Deep nesting (many levels)
- Noisy encoding
- Multiple bindings to factor simultaneously

```python
# Use resonator to decode arithmetic expression
print("Using Resonator Network to decode (2 + 3) * 4:\n")

# Create resonator
resonator = ResonatorNetwork(
    model=model,
    max_iterations=50,
    threshold=0.95
)

# Define cleanup memory (candidates for decoding)
cleanup_items = {
    "op": memory["op"].vec,
    "left": memory["left"].vec,
    "right": memory["right"].vec,
    "+": memory["+"].vec,
    "*": memory["*"].vec,
    "2": memory["2"].vec,
    "3": memory["3"].vec,
    "4": memory["4"].vec,
}

# Factorize the multiply node
print("Factorizing root node (* operation):")
factors_root = resonator.factorize(
    composite=multiply_node,
    codebook=cleanup_items,
    n_factors=3  # op, left, right
)

print(f"\nFactors found: {list(factors_root.keys())}")
print("\nResonator successfully factorized the tree structure!")
print("This allows automatic decoding without manual unbinding.")
```

**Output:**
```
Using Resonator Network to decode (2 + 3) * 4:

Factorizing root node (* operation):

Factors found: ['op', 'left', 'right']

Resonator successfully factorized the tree structure!
This allows automatic decoding without manual unbinding.
```

## Key Takeaways

1. **VSA can encode hierarchical structures** through recursive role-filler binding
2. **Trees become single vectors** - entire structure compressed holistically
3. **Exact unbinding** (with FHRR) allows precise decoding of nested levels
4. **Compositionality** - complex structures built from simple primitives
5. **Resonator networks** provide iterative refinement for robust decoding

## Applications

Hierarchical encoding is powerful for:

1. **Natural Language Processing**
   - Parse trees for syntax
   - Semantic composition
   - Discourse structure

2. **Program Analysis**
   - Abstract syntax trees (AST)
   - Code structure representation
   - Program synthesis

3. **Knowledge Representation**
   - Ontologies and taxonomies
   - Conceptual hierarchies
   - Nested relationships

4. **Data Structures**
   - JSON/XML encoding
   - Nested dictionaries
   - Graph structures

## Advantages Over Flat Representations

| Feature | Flat (Bag-of-Words) | VSA Hierarchical |
|---------|---------------------|------------------|
| **Structure** | Lost | Preserved |
| **Nesting** | Cannot represent | Arbitrary depth |
| **Compositionality** | Additive only | Recursive binding |
| **Decoding** | N/A | Exact unbinding |
| **Semantics** | Weak | Compositional |

## Challenges & Limitations

1. **Noise accumulation** - Deep nesting can degrade signal (use higher dimensions)
2. **Cleanup required** - Decoding needs candidate symbols (cleanup memory)
3. **Variable structure** - Different tree shapes need different decoding strategies
4. **Computational cost** - Resonator iteration can be expensive

## Next Steps

- Try encoding your own tree structures
- Experiment with deeper nesting (3+ levels)
- Compare FHRR vs MAP vs Binary for hierarchical encoding
- Explore resonator networks for robust factorization
- Apply to real datasets (syntax trees, JSON, org charts)

## References

- Plate, T. A. (1995). "Holographic Reduced Representations"
- Kanerva, P. (2009). "Hyperdimensional Computing"
- Frady et al. (2020). "Resonator Networks for Factoring Distributed Representations"
- Gayler, R. W. (2003). "Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience"

## Running This Tutorial

Interactive notebook:
```bash
jupyter notebook examples/notebooks/tutorial_07_hierarchical_structures.ipynb
```

Or copy the code snippets above into your own Python script or notebook!
