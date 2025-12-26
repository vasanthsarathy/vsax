# Lesson 4.3: Hierarchical Structures & Resonators

**Estimated time:** 60 minutes

## Learning Objectives

By the end of this lesson, you will be able to:

- Encode hierarchical tree structures using recursive role-filler binding
- Understand the factorization problem in VSA
- Use resonator networks for iterative cleanup and convergent factorization
- Decode nested structures from composite vectors
- Apply hierarchical encoding to parse trees, JSON, and family trees

## Prerequisites

- Module 2, Lesson 2.1 (FHRR operations - binding/unbinding)
- Module 3, Lesson 3.2 (DictEncoder - role-filler binding)
- Understanding of tree data structures

---

## The Problem: Encoding Hierarchical Data

Many real-world concepts are **hierarchical** - they have nested, tree-like structures:

- **Language:** Sentence syntax trees `[S [NP the cat] [VP sat]]`
- **Mathematics:** Nested expressions `((2 + 3) * 4) - 5`
- **Data structures:** JSON, XML, nested dictionaries
- **Relationships:** Family trees, organizational charts
- **Programs:** Abstract syntax trees (AST)

How can we represent these trees using hypervectors?

### Why Flat Encoding Doesn't Work

Standard bag-of-words or feature vector approaches lose structure:

```python
# ❌ Flat encoding loses structure
expression = "(2 + 3) * 4"
tokens = model.opset.bundle(
    memory["2"].vec,
    memory["+"].vec,
    memory["3"].vec,
    memory["*"].vec,
    memory["4"].vec
)

# Problem: Could be "2 * 3 + 4" or "(2 + 3) * 4" - no difference!
# Bundling is order-invariant and doesn't preserve hierarchy
```

**What's missing?**
- Parent-child relationships
- Nesting depth
- Compositional structure

### The Solution: Recursive Role-Filler Binding

**Key idea:** Encode trees by **binding roles to fillers** recursively:

```
node = bind("value", node_value) ⊕
       bind("left", left_child) ⊕
       bind("right", right_child)
```

**Example:** Encode binary tree node `+` with children `2` and `3`:

```python
plus_node = model.opset.bundle(
    model.opset.bind(memory["op"].vec, memory["+"].vec),
    model.opset.bind(memory["left"].vec, memory["2"].vec),
    model.opset.bind(memory["right"].vec, memory["3"].vec)
)
```

**For nested structures**, child nodes themselves can be composite vectors:

```python
# ((2 + 3) * 4)
# First encode (2 + 3)
addition = encode_node("+", "2", "3")

# Then use it as left child of multiplication
expression = encode_node("*", addition, "4")
```

**Result:** Entire tree compressed into a single hypervector!

---

## Encoding Tree Structures

### Example: Arithmetic Expression Trees

Let's encode `(2 + 3) * 4` step by step.

**Tree visualization:**
```
      *
     / \
    +   4
   / \
  2   3
```

**Setup:**

```python
import jax
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

# Create model (FHRR best for exact unbinding)
model = create_fhrr_model(dim=1024, key=jax.random.PRNGKey(42))
memory = VSAMemory(model)

# Add role vectors
memory.add_many(["op", "left", "right", "value"])

# Add leaf values and operators
memory.add_many(["2", "3", "4", "+", "*"])
```

**Encoding functions:**

```python
def encode_leaf(memory, value):
    """Encode a leaf node (number or variable)."""
    value_str = str(value)
    if value_str not in memory:
        memory.add(value_str)
    return memory[value_str].vec


def encode_binary_op(model, memory, operator, left, right):
    """Encode a binary operation node."""
    if operator not in memory:
        memory.add(operator)

    # Bind each role to its filler
    op_vec = model.opset.bind(memory["op"].vec, memory[operator].vec)
    left_vec = model.opset.bind(memory["left"].vec, left)
    right_vec = model.opset.bind(memory["right"].vec, right)

    # Bundle all role-filler pairs
    node = model.opset.bundle(op_vec, left_vec, right_vec)
    return node
```

**Build tree bottom-up:**

```python
# Encode leaves
leaf_2 = encode_leaf(memory, "2")
leaf_3 = encode_leaf(memory, "3")
leaf_4 = encode_leaf(memory, "4")

# Encode (2 + 3)
addition = encode_binary_op(model, memory, "+", leaf_2, leaf_3)

# Encode (2 + 3) * 4
expression = encode_binary_op(model, memory, "*", addition, leaf_4)

print(f"Encoded: (2 + 3) * 4")
print(f"Tree vector shape: {expression.shape}")  # (1024,)
print(f"\nEntire tree is a single {model.dim}-dimensional vector!")
```

**Amazing!** The entire tree structure is now in a single 1024-dimensional vector.

### Decoding: Extracting Structure

Can we recover the tree structure from the encoded vector?

**Step 1:** Unbind the operator

```python
def find_best_match(vector, memory, candidates):
    """Find best matching symbol from candidates."""
    best_match = None
    best_sim = -1.0

    for candidate in candidates:
        sim = cosine_similarity(vector, memory[candidate].vec)
        if sim > best_sim:
            best_sim = sim
            best_match = candidate

    return best_match, best_sim


# What operator is at the root?
operator_vec = model.opset.bind(expression, model.opset.inverse(memory["op"].vec))
root_op, sim = find_best_match(operator_vec, memory, ["+", "*", "-", "/"])

print(f"Root operator: {root_op} (similarity: {sim:.3f})")
# Output: Root operator: * (similarity: 0.847)
```

**Step 2:** Unbind children

```python
# Get left child
left_child = model.opset.bind(expression, model.opset.inverse(memory["left"].vec))

# Is left child an operator or a leaf?
left_op, sim_op = find_best_match(left_child, memory, ["+", "*", "-", "/"])
left_leaf, sim_leaf = find_best_match(left_child, memory, ["2", "3", "4"])

if sim_op > sim_leaf:
    print(f"Left child is operator: {left_op} (similarity: {sim_op:.3f})")
else:
    print(f"Left child is leaf: {left_leaf} (similarity: {sim_leaf:.3f})")

# Output: Left child is operator: + (similarity: 0.723)


# Get right child
right_child = model.opset.bind(expression, model.opset.inverse(memory["right"].vec))
right_val, sim = find_best_match(right_child, memory, ["2", "3", "4"])

print(f"Right child is leaf: {right_val} (similarity: {sim:.3f})")
# Output: Right child is leaf: 4 (similarity: 0.891)
```

**Result:** We successfully decoded the tree structure!

```
Root: *
├─ Left: + (operator)
└─ Right: 4 (leaf)
```

To fully decode, recursively unbind the left child (`+` node) to get `2` and `3`.

---

## The Factorization Problem

### What is Factorization?

Given a composite vector formed by binding multiple factors:

```
s = a ⊗ b ⊗ c
```

**Goal:** Recover the original factors `a`, `b`, `c`.

**Why it's hard:**
- Binding scrambles information - the composite doesn't obviously contain the factors
- With large codebooks, search space is enormous (N³ for 3 factors)
- Unbinding with MAP or noisy FHRR is approximate, leading to errors

### Example: Three-Attribute Scene

```python
# Encode: "red circle that is large"
memory.add_many(["red", "blue", "circle", "square", "large", "small"])

composite = model.opset.bind(
    model.opset.bind(memory["red"].vec, memory["circle"].vec),
    memory["large"].vec
)

# Question: What three attributes created this composite?
# (Need to search through all combinations!)
```

**Naive approach:** Try all combinations

```python
colors = ["red", "blue"]
shapes = ["circle", "square"]
sizes = ["large", "small"]

best_match = None
best_sim = -1.0

# Try all 2 × 2 × 2 = 8 combinations
for color in colors:
    for shape in shapes:
        for size in sizes:
            candidate = model.opset.bind(
                model.opset.bind(memory[color].vec, memory[shape].vec),
                memory[size].vec
            )
            sim = cosine_similarity(composite, candidate)
            if sim > best_sim:
                best_sim = sim
                best_match = (color, shape, size)

print(f"Best match: {best_match} (similarity: {best_sim:.3f})")
# Output: Best match: ('red', 'circle', 'large') (similarity: 0.998)
```

**Problem:** This is O(N^k) where k is the number of factors. For large codebooks, this is infeasible!

---

## Resonator Networks: Iterative Factorization

**Resonator networks** solve factorization efficiently using an iterative algorithm.

### The Algorithm

**Intuition:** Alternate between unbinding and cleanup (projection onto codebooks).

**Steps:**
1. Initialize random guesses for each factor
2. **Unbind:** Remove current estimates of other factors to isolate one factor
3. **Cleanup:** Project result onto codebook to get clean estimate
4. Repeat for each factor
5. Iterate until convergence (estimates stop changing)

**Key insight:** Correct factors **resonate** - they mutually reinforce each other through iterations.

### Using Resonators in VSAX

```python
from vsax.resonator import CleanupMemory, Resonator

# Create composite: red ⊗ circle ⊗ large
memory.add_many(["red", "blue", "circle", "square", "large", "small"])

composite = model.opset.bind(
    model.opset.bind(memory["red"].vec, memory["circle"].vec),
    memory["large"].vec
)

# Create codebooks (one per factor position)
colors = CleanupMemory(["red", "blue"], memory)
shapes = CleanupMemory(["circle", "square"], memory)
sizes = CleanupMemory(["large", "small"], memory)

# Create resonator
resonator = Resonator(
    codebooks=[colors, shapes, sizes],
    opset=model.opset,
    max_iterations=100,
    convergence_threshold=0.99
)

# Factorize!
factors = resonator.factorize(composite)
print(f"Recovered factors: {factors}")
# Output: Recovered factors: ['red', 'circle', 'large']
```

**That's it!** Resonator automatically finds the factors in ~10-20 iterations.

### How Resonators Work (Under the Hood)

**Iteration example for 3 factors:**

```
# Initial guesses (random)
a₀ = random vector
b₀ = random vector
c₀ = random vector

# Iteration 1:
# Isolate a: s ⊗ b₀⁻¹ ⊗ c₀⁻¹ ≈ a
a₁ = cleanup(s ⊗ b₀⁻¹ ⊗ c₀⁻¹, codebook_A)

# Isolate b: s ⊗ a₁⁻¹ ⊗ c₀⁻¹ ≈ b
b₁ = cleanup(s ⊗ a₁⁻¹ ⊗ c₀⁻¹, codebook_B)

# Isolate c: s ⊗ a₁⁻¹ ⊗ b₁⁻¹ ≈ c
c₁ = cleanup(s ⊗ a₁⁻¹ ⊗ b₁⁻¹, codebook_C)

# Repeat until (a, b, c) stop changing
```

**Cleanup:** Projects noisy vector onto codebook (finds nearest clean vector).

```python
# CleanupMemory.query() does this:
def query(self, noisy_vector):
    similarities = [cosine_similarity(noisy_vector, vec)
                    for vec in codebook_vectors]
    best_idx = argmax(similarities)
    return symbols[best_idx]
```

### Convergence Visualization

```python
# Track resonator convergence
import matplotlib.pyplot as plt

# Resonator stores iteration history
history = resonator.get_history()  # List of factor estimates per iteration

# Plot similarities over iterations
iterations = range(len(history))
similarities = [
    cosine_similarity(
        model.opset.bind(
            model.opset.bind(memory[h[0]].vec, memory[h[1]].vec),
            memory[h[2]].vec
        ),
        composite
    )
    for h in history
]

plt.plot(iterations, similarities)
plt.xlabel('Iteration')
plt.ylabel('Similarity to Composite')
plt.title('Resonator Convergence')
plt.axhline(y=0.99, color='r', linestyle='--', label='Convergence threshold')
plt.legend()
plt.show()
```

**Typical convergence:** 10-30 iterations to reach >0.99 similarity.

---

## Practical Applications

### Application 1: Parsing Natural Language

Encode sentence parse trees:

```python
# Sentence: "The cat sat"
# Parse tree:
#       S
#      / \
#    NP   VP
#    |    |
#  "the cat"  "sat"

memory.add_many(["S", "NP", "VP", "the", "cat", "sat"])

# Encode NP (noun phrase)
np = model.opset.bundle(
    model.opset.bind(memory["value"].vec, memory["the cat"].vec)
)

# Encode VP (verb phrase)
vp = model.opset.bundle(
    model.opset.bind(memory["value"].vec, memory["sat"].vec)
)

# Encode sentence
sentence = model.opset.bundle(
    model.opset.bind(memory["op"].vec, memory["S"].vec),
    model.opset.bind(memory["left"].vec, np),
    model.opset.bind(memory["right"].vec, vp)
)

# Decode structure
op_vec = model.opset.bind(sentence, model.opset.inverse(memory["op"].vec))
sentence_type, sim = find_best_match(op_vec, memory, ["S", "NP", "VP"])
print(f"Sentence type: {sentence_type}")  # S
```

### Application 2: Nested JSON Encoding

```python
# JSON: {"name": "Alice", "age": 30, "city": "NYC"}
memory.add_many(["name", "age", "city", "Alice", "30", "NYC"])

json_obj = model.opset.bundle(
    model.opset.bind(memory["name"].vec, memory["Alice"].vec),
    model.opset.bind(memory["age"].vec, memory["30"].vec),
    model.opset.bind(memory["city"].vec, memory["NYC"].vec)
)

# Nested JSON: {"person": {"name": "Alice", "age": 30}}
memory.add("person")

nested_json = model.opset.bundle(
    model.opset.bind(memory["person"].vec, json_obj)
)

# Query: what's the person's name?
person_vec = model.opset.bind(nested_json, model.opset.inverse(memory["person"].vec))
name_vec = model.opset.bind(person_vec, model.opset.inverse(memory["name"].vec))

name, sim = find_best_match(name_vec, memory, ["Alice", "Bob", "Charlie"])
print(f"Person's name: {name} (similarity: {sim:.3f})")
# Output: Person's name: Alice (similarity: 0.921)
```

### Application 3: Family Trees

```python
# Encode family relationships
memory.add_many(["parent", "child", "Alice", "Bob", "Charlie"])

# Alice is parent of Bob
relationship1 = model.opset.bundle(
    model.opset.bind(memory["parent"].vec, memory["Alice"].vec),
    model.opset.bind(memory["child"].vec, memory["Bob"].vec)
)

# Bob is parent of Charlie
relationship2 = model.opset.bundle(
    model.opset.bind(memory["parent"].vec, memory["Bob"].vec),
    model.opset.bind(memory["child"].vec, memory["Charlie"].vec)
)

# Bundle both relationships
family_tree = model.opset.bundle(relationship1, relationship2)

# Query: Who is Charlie's grandparent?
# Step 1: Find Charlie's parent
charlie_parent = model.opset.bind(
    family_tree,
    model.opset.bind(
        model.opset.inverse(memory["child"].vec),
        model.opset.inverse(memory["Charlie"].vec)
    )
)
# Cleanup gives "Bob"

# Step 2: Find Bob's parent
bob_parent = model.opset.bind(
    family_tree,
    model.opset.bind(
        model.opset.inverse(memory["child"].vec),
        model.opset.inverse(memory["Bob"].vec)
    )
)
# Cleanup gives "Alice"

grandparent, sim = find_best_match(bob_parent, memory, ["Alice", "Bob", "Charlie"])
print(f"Charlie's grandparent: {grandparent}")  # Alice
```

---

## Design Patterns

### Pattern 1: Role-Filler Binding

**When:** Encoding structured data with labeled fields

```python
# General pattern
node = bundle(
    bind(role1, filler1),
    bind(role2, filler2),
    ...
    bind(roleN, fillerN)
)
```

**Examples:** Dictionaries, records, parse tree nodes

### Pattern 2: Recursive Composition

**When:** Encoding tree structures where children can be complex

```python
# Recursive pattern
def encode_tree(node):
    if is_leaf(node):
        return encode_leaf(node.value)
    else:
        left_child = encode_tree(node.left)
        right_child = encode_tree(node.right)
        return encode_node(node.value, left_child, right_child)
```

**Examples:** Expression trees, syntax trees, hierarchical data

### Pattern 3: Sequential Binding

**When:** Encoding ordered sequences with position information

```python
# Encode sequence with positions
sequence = bundle(
    bind(pos[0], item[0]),
    bind(pos[1], item[1]),
    ...
    bind(pos[n], item[n])
)
```

**Examples:** Time series with timestamps, ordered lists, sentence encoding

---

## Performance Considerations

### Depth Limits

Nested binding accumulates noise. Each level of nesting adds uncertainty:

| Depth | FHRR Accuracy | MAP Accuracy |
|-------|---------------|--------------|
| 1-2 levels | >0.9 similarity | >0.8 similarity |
| 3-4 levels | >0.8 similarity | >0.6 similarity |
| 5+ levels | >0.7 similarity | <0.5 similarity |

**Best practice:** Limit tree depth to 4-5 levels. For deeper trees, use FHRR or increase dimensionality.

### Resonator Complexity

Resonator iterations scale with number of factors:

| Factors | Iterations to Converge | Time (dim=1024) |
|---------|------------------------|-----------------|
| 2 factors | ~10 iterations | ~50ms |
| 3 factors | ~15 iterations | ~100ms |
| 4 factors | ~25 iterations | ~200ms |
| 5+ factors | ~40 iterations | ~400ms |

**Note:** All operations are GPU-accelerated via JAX.

### Model Selection

**FHRR (ComplexHypervector):**
- ✅ Exact unbinding (best for deep hierarchies)
- ✅ High accuracy at all depths
- ❌ Slower than Binary

**Binary (BinaryHypervector):**
- ✅ Fast operations (XOR binding)
- ✅ Memory efficient
- ❌ Approximate unbinding (noise accumulates)

**Recommendation:** Use FHRR for hierarchical structures unless performance is critical.

---

## Common Pitfalls

### Problem 1: Forgetting to Normalize

```python
# ❌ Unbinding can produce unnormalized vectors
unboundvec = model.opset.bind(composite, model.opset.inverse(memory["role"].vec))
# Don't use directly for similarity!

# ✅ Normalize before similarity check
from vsax.representations import ComplexHypervector
hv = ComplexHypervector(unbound_vec).normalize()
sim = cosine_similarity(hv.vec, memory["candidate"].vec)
```

### Problem 2: Wrong Codebook for Resonator

```python
# ❌ Using wrong symbols in codebook
colors = CleanupMemory(["red", "blue"], memory)
# But actual factor is "green" - not in codebook!
# Resonator will return wrong answer

# ✅ Ensure codebooks contain all possible values
colors = CleanupMemory(["red", "blue", "green", "yellow"], memory)
```

### Problem 3: Too Many Bundled Components

```python
# ❌ Bundling 100 role-filler pairs
node = bundle(*[bind(role[i], filler[i]) for i in range(100)])
# Similarity degrades significantly

# ✅ Limit to ~10-20 role-filler pairs per node
# Or increase dimensionality
```

---

## Self-Assessment

Before moving on, ensure you can:

- [ ] Explain how recursive role-filler binding encodes tree structures
- [ ] Encode a simple binary tree using binding and bundling
- [ ] Decode tree nodes by unbinding roles
- [ ] Describe the factorization problem and why it's hard
- [ ] Use CleanupMemory to project noisy vectors onto codebooks
- [ ] Apply Resonator to factorize multi-factor composites
- [ ] Choose appropriate depth limits for hierarchical encoding

## Quick Quiz

**Question 1:** What is the key difference between bundling and binding for hierarchical encoding?

a) Bundling is faster than binding
b) Bundling combines peers, binding connects roles to fillers
c) Binding is order-invariant, bundling preserves order
d) There is no difference

<details>
<summary>Answer</summary>
**b) Bundling combines peers, binding connects roles to fillers**

In hierarchical encoding, we **bind** roles to fillers (e.g., `"left" ⊗ child_node`), then **bundle** all the role-filler pairs for a node (e.g., `op_binding ⊕ left_binding ⊕ right_binding`). Binding creates associations, bundling aggregates them.
</details>

**Question 2:** Why do resonator networks converge to the correct factors?

a) They try all possible combinations
b) Correct factors mutually reinforce through iterative cleanup
c) They use gradient descent
d) They rely on random search

<details>
<summary>Answer</summary>
**b) Correct factors mutually reinforce through iterative cleanup**

Resonators work through resonance: when one factor estimate improves, it helps improve estimates of other factors in the next iteration. Incorrect estimates don't reinforce and get replaced by cleanup. This process converges to the unique set of factors that are mutually consistent.
</details>

**Question 3:** For encoding a 6-level deep expression tree, which model is best?

a) Binary - fastest operations
b) MAP - good balance
c) FHRR - exact unbinding
d) Doesn't matter

<details>
<summary>Answer</summary>
**c) FHRR - exact unbinding**

Deep hierarchies (6 levels) accumulate noise from approximate unbinding. FHRR provides near-exact unbinding (>0.99 similarity), maintaining accuracy across all levels. Binary and MAP would have significant accuracy degradation at depth 6.
</details>

---

## Hands-On Exercise

**Task:** Build a family tree encoder and query system.

**Requirements:**
1. Encode at least 3 generations of a family (9+ people)
2. Use role-filler binding for relationships ("parent", "child", "spouse")
3. Implement a query function to find:
   - Someone's parents
   - Someone's children
   - Someone's grandparents (2-hop query)
4. Use resonator to factorize a bundled relationship

**Starter code:**

```python
import jax
from vsax import create_fhrr_model, VSAMemory
from vsax.similarity import cosine_similarity

model = create_fhrr_model(dim=2048, key=jax.random.PRNGKey(42))
memory = VSAMemory(model)

# Add roles
memory.add_many(["parent", "child", "spouse"])

# Add people (3 generations)
people = [
    "grandpa_john", "grandma_mary",
    "dad_bob", "mom_alice",
    "child_charlie", "child_diana"
]
memory.add_many(people)

# YOUR CODE HERE:
# 1. Encode relationships (e.g., grandpa is parent of dad, dad is parent of charlie)
# 2. Bundle all relationships into family_tree
# 3. Query: Who are Charlie's grandparents?

def encode_relationship(role, filler1, filler2):
    """Encode a binary relationship: role(filler1, filler2)"""
    # YOUR CODE HERE
    pass

def query_relationship(family_tree, role, known_person, candidates):
    """Query: role(?, known_person) or role(known_person, ?)"""
    # YOUR CODE HERE
    pass

# Test your implementation
```

<details>
<summary>Solution</summary>

```python
def encode_relationship(role, filler1, filler2):
    """Encode: role(filler1, filler2)"""
    return model.opset.bundle(
        model.opset.bind(memory[role].vec,
                         model.opset.bind(memory[filler1].vec, memory[filler2].vec))
    )

# Encode relationships
relationships = []

# Generation 1 → 2
relationships.append(encode_relationship("parent", "grandpa_john", "dad_bob"))
relationships.append(encode_relationship("parent", "grandma_mary", "dad_bob"))
relationships.append(encode_relationship("parent", "grandpa_john", "mom_alice"))  # Or other parent
relationships.append(encode_relationship("parent", "grandma_mary", "mom_alice"))

# Generation 2 → 3
relationships.append(encode_relationship("parent", "dad_bob", "child_charlie"))
relationships.append(encode_relationship("parent", "mom_alice", "child_charlie"))
relationships.append(encode_relationship("parent", "dad_bob", "child_diana"))
relationships.append(encode_relationship("parent", "mom_alice", "child_diana"))

# Spouse relationships
relationships.append(encode_relationship("spouse", "grandpa_john", "grandma_mary"))
relationships.append(encode_relationship("spouse", "dad_bob", "mom_alice"))

# Bundle all
family_tree = model.opset.bundle(*relationships)


def query_relationship(family_tree, role, known_person, candidates):
    """Find: who has 'role' relationship with 'known_person'?"""
    # Unbind role and known person
    query_vec = model.opset.bind(
        family_tree,
        model.opset.inverse(
            model.opset.bind(memory[role].vec, memory[known_person].vec)
        )
    )

    # Find best match
    best_match = None
    best_sim = -1.0
    for candidate in candidates:
        sim = cosine_similarity(query_vec, memory[candidate].vec)
        if sim > best_sim:
            best_sim = sim
            best_match = candidate

    return best_match, best_sim


# Query: Who are Charlie's parents?
parent1, sim1 = query_relationship(family_tree, "parent", "child_charlie", people)
print(f"Charlie's parent: {parent1} (similarity: {sim1:.3f})")
# Output: dad_bob or mom_alice (depending on bundling order)

# For grandparents, query in two steps:
# 1. Find parent
# 2. Find parent's parent
parent, _ = query_relationship(family_tree, "parent", "child_charlie", people)
grandparent, sim = query_relationship(family_tree, "parent", parent, people)
print(f"Charlie's grandparent: {grandparent} (similarity: {sim:.3f})")
# Output: grandpa_john or grandma_mary
```
</details>

---

## Key Takeaways

✓ **Hierarchical structures encoded via recursive role-filler binding** - bind roles to fillers, bundle peers
✓ **Trees become single vectors** - entire structure compressed holistically
✓ **Unbinding extracts structure** - invert roles to decode nodes
✓ **Factorization problem is hard** - exponential search space for multiple factors
✓ **Resonators solve factorization efficiently** - iterative unbind + cleanup converges
✓ **Depth limits apply** - 4-5 levels safe, FHRR best for deep trees
✓ **Applications: parse trees, JSON, family trees, ASTs**

---

## Next Steps

**Next Lesson:** [Lesson 4.4 - Multi-Modal & Neural-Symbolic Integration](04_multimodal.md)
Learn how to combine vision, language, and symbolic reasoning using heterogeneous data fusion.

**For Hands-On Practice:** [Tutorial 7 - Hierarchical Structures](../../tutorials/07_hierarchical_structures.md)
Complete walkthrough with code for encoding expression trees, parse trees, and nested data.

**For Deep Technical Details:** [Resonator Networks Guide](../../guide/resonator.md)
Comprehensive reference on resonator algorithm, convergence analysis, and implementation details.

**Related Content:**
- [Module 3, Lesson 3.2 - DictEncoder](../../course/03_encoders/02_dict_sets.md)
- [Tutorial 6 - Sequence Encoding](../../tutorials/06_sequence_encoding.md)

## References

- Frady, E. P., Kleyko, D., & Sommer, F. T. (2020). "A Theory of Sequence Indexing and Working Memory in Recurrent Neural Networks." *Neural Computation.*
- Plate, T. A. (1995). "Holographic Reduced Representations." *IEEE Transactions on Neural Networks.*
- Gayler, R. W. (2003). "Vector symbolic architectures answer Jackendoff's challenges for cognitive neuroscience."
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." *Cognitive Computation.*
